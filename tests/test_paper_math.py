"""Numerical checks of the propositions in the paper (Props. 1, 2, 4, 5). Run: python -m pytest -q tests"""
import numpy as np
import torch

from damecl.data import cutmix
from damecl import losses as L


def test_prop1_unbiased_squared_resultant():
    rng = np.random.default_rng(0)
    d, n, reps = 16, 5, 20000
    mean_dir = np.ones(d) / np.sqrt(d)
    def draw(k):
        u = mean_dir + 0.6 * rng.normal(size=(k, d))
        return u / np.linalg.norm(u, axis=1, keepdims=True)
    rho2 = float(np.sum(draw(200000).mean(0) ** 2))
    s2 = np.array([np.sum(draw(n).sum(0) ** 2) for _ in range(reps)])
    assert abs(np.mean((s2 - n) / (n * (n - 1))) - rho2) < 0.01                 # unbiased
    assert abs(np.mean(s2 / n ** 2) - (rho2 + (1 - rho2) / n)) < 0.01            # naive bias = (1 - rho^2)/n


def test_prop2_ema_variance():
    rng = np.random.default_rng(1)
    gamma, e = 0.9, 30
    G = None
    for k in range(e):
        x = rng.normal(size=100000)
        G = x if G is None else gamma * G + (1 - gamma) * x
    pred = gamma ** (2 * (e - 1)) + (1 - gamma) ** 2 * (1 - gamma ** (2 * (e - 1))) / (1 - gamma ** 2)
    assert abs(G.var() - pred) < 0.01 and abs(G.mean()) < 0.01


def test_prop4_expected_label_mass():
    rng = np.random.default_rng(2)
    torch.manual_seed(2)
    cnt = np.array([4980, 2985, 1789, 1072, 643, 385, 231, 138, 83, 49])
    pi = cnt / cnt.sum()
    q = L.paste_probabilities(cnt.tolist(), "inv").numpy()
    m = 40000
    yb, yf = rng.choice(10, size=m, p=pi), rng.choice(10, size=m, p=q)
    _, lam = cutmix(torch.zeros(m, 1, 32, 32), torch.ones(m, 1, 32, 32))
    lam = lam.numpy()
    mass = np.zeros(10)
    np.add.at(mass, yb, lam)
    np.add.at(mass, yf, 1 - lam)
    pred = lam.mean() * pi + (1 - lam.mean()) * q
    assert np.abs(mass / m - pred).max() < 0.01


def test_prop5_bounded_schedule():
    rng = np.random.default_rng(3)
    a, C = 36.0, 10
    G = torch.tensor(rng.uniform(1, 6, size=(5000, C)), dtype=torch.float32)
    q = torch.stack([L.paste_probabilities([1] * C, "gaponly", g, gap_power=2.0) for g in G])
    assert q.min() >= 1 / (1 + a * (C - 1)) - 1e-6
    assert q.max() <= a / (a + C - 1) + 1e-6
    assert (q.max(1).values / q.min(1).values).max() <= a + 1e-3


def test_round3_data_variants_and_ink_label():
    import numpy as np
    from damecl import data as D
    from damecl.metrics import class_groups
    # step profile: equal minority classes, groups = majority / minority
    c = D.step_class_counts(4980, 10, 100, 2)
    assert c == [4980] * 8 + [49] * 2
    g = class_groups(c)
    assert g["few"] == [8, 9] and g["medium"] == [] and len(g["many"]) == 8
    # duplicates keep the count but reduce distinct images
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, size=(100, 4, 4, 3), dtype=np.uint8)
    y = np.repeat(np.arange(4), 25)
    xd = D.duplicate_classes(x, y, "1:5")
    assert len(np.unique(xd[y == 1].reshape(25, -1), axis=0)) == 5
    assert (xd[y != 1] == x[y != 1]).all()
    # ink label: an empty (black) background gives all label mass to the pasted content
    zeros = (torch.zeros(4, 3, 32, 32) - D.MEAN) / D.STD
    ones = (torch.ones(4, 3, 32, 32) - D.MEAN) / D.STD
    torch.manual_seed(5)
    _, lam = D.cutmix(zeros, ones, label="ink")
    torch.manual_seed(5)
    _, lam_area = D.cutmix(zeros, ones, label="area")   # same boxes
    assert torch.allclose(lam[lam_area < 1], torch.zeros_like(lam[lam_area < 1]), atol=1e-5)
