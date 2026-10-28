"""Figures that show REAL images (needs the datasets; run where they can be downloaded, e.g. Kaggle or Colab
with Internet on, or a laptop):

    python make_paper_images.py --out paper_assets/figures
    python make_paper_images.py --out paper_assets/figures --fake      # smoke test without data

Writes
  fig_teaser.pdf   CIFAR-10-LT: the long tail, the few training images of a rare class, and the same rare
                   objects pasted into head-class scenes by DAM-Mix (in the spirit of CMO's Fig. 1)
  fig_samples.pdf  for every dataset: one rare training image and mixed images built exactly as in training
                   (background drawn uniformly from the long-tailed set, rare foreground, CutMix box)
The long-tailed set and the probe are rebuilt with the same code and seeds as the experiments.
"""
import argparse
import os

import numpy as np

from damecl import data as D

CLASS_NAMES = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "fmnist": ["T-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"],
    "svhn": [str(i) for i in range(10)],
    "mnist": [str(i) for i in range(10)],
}
DS_NAME = {"cifar10": "CIFAR-10-LT", "fmnist": "Fashion-MNIST-LT", "svhn": "SVHN-LT", "mnist": "MNIST-LT"}


def long_tailed(name, root, fake=False, ir=100, k=20, cap=4980):
    if fake:
        xall, yall, _, _, C = D.fake_cifar(10, n_train=600, n_test=10)
    else:
        xall, yall, _, _, C = D.load_dataset(name, root)
    per_class = int(np.bincount(yall, minlength=C).min())
    xtr, ytr, counts = D.make_long_tailed(xall, yall, C, ir, n_max=min(per_class - k, cap))
    return xtr, ytr, np.array(counts)


def paste(bg, fg, rng, area=None):
    """CutMix geometry of damecl.data.cutmix on single uint8 images; returns image, box, background share."""
    H, W = bg.shape[:2]
    lam = rng.uniform(0, 1) if area is None else 1 - area
    cut = np.sqrt(1 - lam)
    bw, bh = int(W * cut), int(H * cut)
    cx, cy = rng.integers(0, W), rng.integers(0, H)
    if area is not None:   # display version: keep the box inside the image so the object stays visible
        cx = int(np.clip(cx, bw // 2, W - bw // 2))
        cy = int(np.clip(cy, bh // 2, H - bh // 2))
    x0, x1 = np.clip(cx - bw // 2, 0, W), np.clip(cx + bw // 2, 0, W)
    y0, y1 = np.clip(cy - bh // 2, 0, H), np.clip(cy + bh // 2, 0, H)
    out = bg.copy()
    out[y0:y1, x0:x1] = fg[y0:y1, x0:x1]
    lam_bg = 1 - (x1 - x0) * (y1 - y0) / (H * W)
    return out, (x0, y0, x1, y1), lam_bg


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                         "pdf.fonttype": 42})
    return plt


def show(ax, img, box=None, color="#e8590c"):
    from matplotlib.patches import Rectangle
    ax.imshow(img, interpolation="nearest")
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, lw=1.2, ec=color))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_teaser(out, root, fake=False, seed=0):
    plt = _plt()
    xtr, ytr, counts = long_tailed("cifar10", root, fake)
    names = CLASS_NAMES["cifar10"]
    rng = np.random.default_rng(seed)
    rare = int(np.argmin(counts))
    head = list(np.argsort(-counts)[:3])
    rare_idx = np.where(ytr == rare)[0]
    head_idx = np.where(np.isin(ytr, head))[0]

    fig = plt.figure(figsize=(7.0, 2.35))
    gs = fig.add_gridspec(2, 9, width_ratios=[2.6, 0.25, 1, 1, 1, 1, 0.12, 1, 1], hspace=0.12, wspace=0.08)
    ax = fig.add_subplot(gs[:, 0])
    order = np.argsort(-counts)
    cols = ["#1f6fd1" if i < 3 else ("#e8590c" if i >= len(order) - 3 else "#9aa5b1") for i in range(len(order))]
    ax.bar(range(len(order)), counts[order], color=cols, width=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[c] for c in order], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("training images")
    ax.set_title("CIFAR-10-LT (imbalance ratio 100)", fontsize=8)
    ax.text(0.97, 0.9, f"{names[rare]}: {counts[rare]} images", transform=ax.transAxes, ha="right", fontsize=7,
            color="#e8590c")
    # the rare class as seen in training
    for i in range(4):
        a = fig.add_subplot(gs[0, 2 + i])
        show(a, xtr[rare_idx[i % len(rare_idx)]])
        if i == 0:
            a.set_ylabel("training\nimages", fontsize=7)
    # DAM-Mix: the same objects in head-class scenes
    for i in range(4):
        a = fig.add_subplot(gs[1, 2 + i])
        img, box, _ = paste(xtr[rng.choice(head_idx)], xtr[rare_idx[i % len(rare_idx)]], rng, area=0.45)
        show(a, img, box)
        if i == 0:
            a.set_ylabel("pasted into\nhead scenes", fontsize=7)
    fig.text(0.535, 0.97, f"rare class '{names[rare]}'", ha="center", fontsize=8)
    # head-class backgrounds used
    for i in range(2):
        for j in range(2):
            a = fig.add_subplot(gs[i, 7 + j])
            show(a, xtr[rng.choice(head_idx)])
    fig.text(0.875, 0.97, "head-class scenes", ha="center", fontsize=8)
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/fig_teaser.pdf", dpi=300)
    plt.close(fig)


def fig_samples(out, root, datasets, fake=False, seed=0, n=7):
    plt = _plt()
    ok = []
    for ds in datasets:
        try:
            ok.append((ds,) + long_tailed(ds, root, fake))
        except Exception as e:
            print(f"skipping {ds}: {e}")
    if not ok:
        return
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(ok), n + 1, figsize=(7.0, 1.1 * len(ok) + 0.2), squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.55})
    for r, (ds, xtr, ytr, counts) in enumerate(ok):
        names = CLASS_NAMES.get(ds, [str(i) for i in range(len(counts))])
        rare = int(np.argmin(counts))
        fg = xtr[np.where(ytr == rare)[0][0]]
        show(axes[r][0], fg, color="#1f6fd1")
        axes[r][0].set_title(f"rare: {names[rare]}", fontsize=6.5, pad=2)
        axes[r][0].set_ylabel(DS_NAME.get(ds, ds), fontsize=7)
        for j in range(n):
            b = rng.integers(0, len(ytr))         # background: uniform over the long-tailed set
            img, box, lam = paste(xtr[b], fg, rng, area=rng.uniform(0.3, 0.6))
            show(axes[r][j + 1], img, box)
            axes[r][j + 1].set_title(f"on {names[ytr[b]]}", fontsize=6, pad=2)
            axes[r][j + 1].set_xlabel(f"{lam:.2f} / {1 - lam:.2f}", fontsize=5.5, labelpad=1)
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/fig_samples.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper_assets/figures")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--datasets", nargs="+", default=["cifar10", "fmnist", "svhn", "mnist"])
    p.add_argument("--fake", action="store_true", help="random images, to test the script without data")
    a = p.parse_args()
    fig_teaser(a.out, a.data_root, a.fake)
    fig_samples(a.out, a.data_root, a.datasets, a.fake)
    print("written:", sorted(f for f in os.listdir(a.out) if f.startswith(("fig_teaser", "fig_samples"))))
