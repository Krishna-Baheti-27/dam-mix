"""Mathematical sanity tests. Run:  python -m pytest -q tests"""
import math
import torch
import torch.nn.functional as F

from damecl import losses as L
from damecl import data as D
from damecl.model import Net


def unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def test_cosine_geometry():
    a = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(a @ a.t(), torch.ones(1, 1))
    assert torch.allclose(a @ torch.tensor([[0.0, 1.0]]).t(), torch.zeros(1, 1))
    assert torch.allclose(a @ torch.tensor([[-1.0, 0.0]]).t(), -torch.ones(1, 1))


def test_psi_zero_margin_is_identity_and_monotone():
    s = torch.linspace(-0.99, 0.99, 50)
    assert torch.allclose(L.psi(s, torch.tensor(0.0)), s, atol=1e-5)
    out = L.psi(s, torch.tensor(0.3))
    assert torch.all(out[1:] >= out[:-1] - 1e-6)
    assert torch.all(out <= s + 1e-6)  # a margin can only make the similarity harder


def test_dam_with_zero_margins_equals_ecl():
    z = unit(32, 16)
    y = torch.randint(0, 4, (16,)).repeat(2)
    protos = unit(4, 16, seed=1)
    a = L.ecl_contrastive_loss(z, y, protos, 4, 0.1)
    b = L.ecl_contrastive_loss(z, y, protos, 4, 0.1, margins=torch.zeros(4))
    assert torch.allclose(a, b, atol=1e-5)
    c = L.ecl_contrastive_loss(z, y, protos, 4, 0.1, margins=torch.full((4,), 0.3))
    assert c > a  # margins make the task harder


def test_alc_reduces_to_logit_adjustment():
    logits = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    lp = torch.log(torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05]))
    assert torch.allclose(L.alc_loss(logits, y, lp, torch.zeros(5), 5.0), L.logit_adjusted_loss(logits, y, lp))


def test_maca_reduces_to_ccge_with_uniform_margins():
    W, M = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(L.alignment_loss(W, M), L.alignment_loss(W, M, torch.full((5,), 0.2)), atol=1e-6)


def test_bias_corrected_dispersion_is_not_fooled_by_small_classes():
    torch.manual_seed(0)
    base = F.normalize(torch.ones(128), dim=0)
    def sample(n):
        return F.normalize(base + 0.15 * torch.randn(n, 128), dim=1)
    est = {}
    for n in (5, 500):
        vals = []
        for _ in range(200):
            z = sample(n)
            S = z.sum(0)
            rho2 = ((S @ S) - n) / (n * (n - 1))
            vals.append(1 - float(rho2.clamp(0, 1).sqrt()))
        est[n] = sum(vals) / len(vals)
    assert abs(est[5] - est[500]) < 0.03
    naive5 = 1 - float(sample(5).sum(0).norm() / 5)
    assert naive5 < est[500] - 0.05  # the naive estimate makes small classes look tighter


def test_rarity_and_margin_ordering():
    ms = L.MarginState([1000, 1000, 10, 10], version="v2", device="cpu", total_epochs=10, proj_dim=2)
    assert ms.R[2] > ms.R[0] and ms.R[0] == 0
    ms.D = torch.tensor([0.1, 0.4, 0.1, 0.4])
    ms.has_D = True
    m = ms.margins(epoch=10)
    assert m[3] == m.max()      # rare + diffuse gets the largest margin
    assert m[0] == m.min()      # common + compact gets the smallest
    assert m.max() <= 0.4 + 1e-6


def test_rare_class_dispersion_floor():
    ms = L.MarginState([1000, 500, 10, 5], version="v2", device="cpu", total_epochs=10, proj_dim=2)
    ms.D = torch.tensor([0.4, 0.3, 0.05, 0.02])  # rare classes look artificially tight
    ms.has_D = True
    Deff = ms.effective_D()
    assert Deff[2] >= ms.D.median() and Deff[3] >= ms.D.median()
    ms_off = L.MarginState([1000, 500, 10, 5], version="v2", floor=False, device="cpu", total_epochs=10, proj_dim=2)
    ms_off.D, ms_off.has_D = ms.D.clone(), True
    assert ms.margins(10)[3] > ms_off.margins(10)[3]


def test_supcon_handles_singletons():
    z = unit(6, 8)
    y = torch.tensor([0, 0, 1, 2, 3, 3])
    assert torch.isfinite(L.supcon_loss(z, y))


def test_long_tail_counts():
    counts = D.lt_class_counts(5000, 10, 100)
    assert counts[0] == 5000 and counts[-1] == 50 and sum(counts) == 12406


def test_augmentations_shape_and_randomness():
    x = torch.rand(4, 3, 32, 32)
    a, b = D.strong_aug(x), D.strong_aug(x)
    assert a.shape == x.shape and not torch.allclose(a, b)
    assert D.weak_aug(x).shape == x.shape


def test_full_loss_backward():
    model = Net(4)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    h, logits, z = model(x)
    m = torch.tensor([0.05, 0.1, 0.2, 0.3])
    loss = (L.ecl_contrastive_loss(z, y, model.prototypes(), 4, 0.1, m)
            + 3 * L.alignment_loss(model.classifier.weight, torch.randn(4, 64), m)
            + L.alc_loss(logits, y, torch.log(torch.full((4,), 0.25)), m, 5.0))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.T.weight.grad is not None and model.backbone.stem[0].weight.grad.abs().sum() > 0


def test_heldout_dispersion_is_used_for_dam_ho():
    ms = L.MarginState([100, 100, 10, 10], version="v2", source="heldout", device="cpu", total_epochs=10, proj_dim=4)
    # training views: all classes look perfectly tight (memorised)
    z_tr = F.normalize(torch.eye(4)[torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])] + 0.0, dim=1)
    ms.accumulate(z_tr, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    # held-out probe: class 3 is very spread out
    zp = torch.cat([F.normalize(torch.eye(4)[[0, 0, 1, 1, 2, 2]], dim=1), unit(6, 4, seed=3)])
    yp = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    ms.end_epoch(0, zp, yp)
    h = ms.history[-1]
    assert h["D_train"][3] < 0.05 and h["D_heldout"][3] > 0.3
    assert abs(ms.D[3] - h["D_heldout"][3]) < 1e-6       # the margin uses the held-out value
    assert ms.margins(10)[3] == ms.margins(10).max()
    ms_tr = L.MarginState([100, 100, 10, 10], version="v2", source="train", floor=False, device="cpu", total_epochs=10, proj_dim=4)
    ms_tr.accumulate(z_tr, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    ms_tr.end_epoch(0, zp, yp)
    assert ms_tr.D[3] < 0.05                               # training-based version is fooled


def test_probe_split_disjoint_from_long_tail():
    x = torch.arange(1000).numpy().reshape(1000, 1, 1, 1).repeat(32, 1).repeat(32, 2).repeat(3, 3)
    y = (torch.arange(1000) % 10).numpy()
    xp, yp = D.split_probe(x, y, 10, 20)
    xt, yt, counts = D.make_long_tailed(x, y, 10, 10, n_max=100 - 20)
    assert len(set(xp[:, 0, 0, 0].tolist()) & set(xt[:, 0, 0, 0].tolist())) == 0
    assert (torch.bincount(torch.from_numpy(yp)) == 20).all()


def test_cutmix_label_share_matches_pixels():
    bg = torch.zeros(64, 3, 32, 32)
    fg = torch.ones(64, 3, 32, 32)
    mixed, lam_bg = D.cutmix(bg, fg)
    share_fg = mixed.mean(dim=(1, 2, 3))
    assert torch.allclose(share_fg, 1 - lam_bg, atol=1e-6)
    assert (lam_bg >= 0).all() and (lam_bg <= 1).all() and lam_bg.std() > 0


def test_paste_probabilities():
    counts = [1000, 100, 10]
    q = L.paste_probabilities(counts, "inv")
    assert torch.allclose(q.sum(), torch.tensor(1.0)) and q[2] > q[1] > q[0]
    assert torch.allclose(q, torch.tensor([0.01, 0.1, 1.0]) / 1.11, atol=1e-6)
    G = torch.tensor([1.0, 1.0, 3.0])
    q_gap = L.paste_probabilities(counts, "gap", G, rarity_power=0.5)
    q_nogap = L.paste_probabilities(counts, "sqrtinv")
    assert q_gap[2] > q_nogap[2]                      # a memorised class is pasted more often
    assert torch.allclose(L.paste_probabilities(counts, "gap", None, 0.5), q_nogap)


def test_gap_tracker():
    gt = L.GapTracker(2, "cpu")
    tight = F.normalize(torch.tensor([[1.0, 0.01], [1.0, -0.01], [1.0, 0.02]]), dim=1)
    spread = unit(30, 2, seed=5)
    h_tr = torch.cat([tight, tight]); y_tr = torch.tensor([0, 0, 0, 1, 1, 1])
    h_pr = torch.cat([tight, spread]); y_pr = torch.cat([torch.zeros(3, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    gt.update(0, h_tr, y_tr, h_pr, y_pr)
    assert gt.G[0] == 1.0 and gt.G[1] > 2.0           # class 1 looks tight on training data only


def test_posthoc_tau_recovers_bias():
    torch.manual_seed(0)
    y = torch.arange(4).repeat(50)
    log_prior = torch.log(torch.tensor([0.7, 0.2, 0.08, 0.02]))
    logits = F.one_hot(y, 4).float() * 2.0 + torch.randn(200, 4) * 0.5 + 1.0 * log_prior   # head-biased logits
    tau, bacc = L.posthoc_tau(logits, y, log_prior)
    assert 0.5 <= tau <= 1.5 and bacc > 0.9
