"""Why does pasting fail on MNIST? Two dataset properties, measured on the long-tailed training sets
(no training needed; needs the datasets, e.g. on Kaggle):

  empty_share     share of near-black pixels (grey level < 0.05) -- how much of an image is "no context"
  label_mismatch  mean |lambda_area - lambda_ink| over random CutMix pairs: how far CMO's area-based label is
                  from the share of actual image content contributed by each image

    python make_dataset_stats.py --out paper_assets/dataset_stats.json
"""
import argparse
import json
import os

import numpy as np
import torch

from damecl import data as D


def stats(name, root, n_pairs=20000, seed=0):
    xall, yall, _, _, C = D.load_dataset(name, root)
    k = 20 if C == 10 else 10
    per_class = int(np.bincount(yall, minlength=C).min())
    xtr, ytr, counts = D.make_long_tailed(xall, yall, C, 100, n_max=min(per_class - k, 4980))
    if name not in ("cifar10", "cifar100"):
        D.set_normalisation(xtr)
    else:   # restore the CIFAR constants in case another dataset changed them
        D.MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        D.STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    grey = xtr.astype(np.float32).mean(-1) / 255.0
    empty = float((grey < 0.05).mean())
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    diffs = []
    for _ in range(n_pairs // 2000):
        b = rng.integers(0, len(ytr), 2000)
        f = rng.integers(0, len(ytr), 2000)
        xb = D.normalize(D.to_tensor(xtr[b], "cpu"))
        xf = D.normalize(D.to_tensor(xtr[f], "cpu"))
        state = torch.get_rng_state()
        _, lam_area = D.cutmix(xb, xf, label="area")
        torch.set_rng_state(state)          # identical boxes for both labels
        _, lam_ink = D.cutmix(xb, xf, label="ink")
        diffs.append((lam_area - lam_ink).abs().numpy())
    d = np.concatenate(diffs)
    return {"dataset": name, "empty_share": empty, "label_mismatch_mean": float(d.mean()),
            "label_mismatch_over_0.25": float((d > 0.25).mean()), "train_images": int(len(ytr))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper_assets/dataset_stats.json")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100", "svhn", "fmnist", "mnist"])
    a = p.parse_args()
    res = []
    for ds in a.datasets:
        try:
            res.append(stats(ds, a.data_root))
            print(res[-1], flush=True)
        except Exception as e:
            print(f"skipping {ds}: {e}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
