"""Evaluation: classification metrics (with rare-class focus) and simple geometry metrics."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, precision_score


def class_groups(counts):
    """Many / Medium / Few split.
    CIFAR-100 style (C > 10): Many > 100, Medium 20..100, Few < 20 training images.
    CIFAR-10 (C = 10): Many = 3 most frequent, Few = 3 rarest, Medium = the rest."""
    counts = np.asarray(counts)
    C = len(counts)
    order = np.argsort(-counts)
    if len(np.unique(counts)) == 2:   # step imbalance: majority vs minority classes
        return {"many": np.where(counts == counts.max())[0].tolist(), "medium": [],
                "few": np.where(counts == counts.min())[0].tolist()}
    if C <= 10:
        many, few = order[:3], order[-3:]
        med = order[3:-3]
    else:
        many = np.where(counts > 100)[0]
        few = np.where(counts < 20)[0]
        med = np.where((counts >= 20) & (counts <= 100))[0]
    return {"many": many.tolist(), "medium": med.tolist(), "few": few.tolist()}


def ece_score(probs, y, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y, logits, counts):
    """y: [N] ints, logits: [N,C] numpy; counts: training images per class."""
    C = logits.shape[1]
    probs = torch.softmax(torch.from_numpy(logits).float(), 1).numpy()
    pred = probs.argmax(1)
    recall = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(C)])
    f1 = f1_score(y, pred, average=None, labels=list(range(C)), zero_division=0)
    prec = precision_score(y, pred, average=None, labels=list(range(C)), zero_division=0)
    ap = average_precision_score(np.eye(C)[y], probs, average=None)
    groups = class_groups(counts)
    onehot = np.eye(C)[y]
    out = {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(f1.mean()),
        "macro_auprc": float(average_precision_score(onehot, probs, average="macro")),
        "worst_class_recall": float(np.nanmin(recall)),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(np.nanmean(recall)),
        # G-mean: geometric mean of the per-class recalls; it is 0 as soon as one class is never recognised
        "g_mean": float(np.exp(np.mean(np.log(np.clip(recall, 1e-12, 1.0))))) if np.all(recall > 0) else 0.0,
        "per_class_precision": prec.tolist(),
        "per_class_ap": np.asarray(ap).tolist(),
        "ece": ece_score(probs, y),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion": confusion_matrix(y, pred, labels=list(range(C))).tolist(),
    }
    for g, idx in groups.items():
        out[f"{g}_acc"] = float(np.nanmean(recall[idx])) if len(idx) else float("nan")
        out[f"{g}_f1"] = float(np.mean(f1[idx])) if len(idx) else float("nan")
        out[f"{g}_precision"] = float(np.mean(prec[idx])) if len(idx) else float("nan")
        out[f"{g}_auprc"] = float(np.mean(np.asarray(ap)[idx])) if len(idx) else float("nan")
    out["tail_f1"] = out["few_f1"]
    return out


@torch.no_grad()
def dispersion_per_class(feat, y, C):
    """Bias-corrected directional dispersion D_c = 1 - mean resultant length, per class.
    feat must be L2-normalised. Returns numpy [C]."""
    D = np.full(C, np.nan)
    for c in range(C):
        f = feat[y == c]
        n = f.shape[0]
        if n < 2:
            continue
        S = f.sum(0)
        rho2 = ((S @ S) - n) / (n * (n - 1))
        D[c] = 1 - float(rho2.clamp(0, 1).sqrt())
    return D


@torch.no_grad()
def geometry_metrics(feat, y, C, k=10, max_n=10000):
    """feat: [N,d] L2-normalised test features (torch, any device); y: [N] torch long.
    - dispersion D_c (lower = tighter class)
    - nearest centroid angle per class (higher = better separated), in degrees
    - kNN purity per class: share of the k nearest neighbours with the same label (higher = better);
      1 - purity is the 'neighbourhood contamination'."""
    if feat.shape[0] > max_n:   # keep every class: take the first max_n/C images of each class (test sets may be sorted by class)
        C_ = int(y.max()) + 1
        per = max_n // C_
        keep = torch.cat([torch.nonzero(y == c).flatten()[:per] for c in range(C_)])
        feat, y = feat[keep], y[keep]
    feat = feat.float()
    D = dispersion_per_class(feat, y, C)
    cent = F.normalize(torch.stack([feat[y == c].mean(0) for c in range(C)]), dim=1)
    cos = (cent @ cent.t()).clamp(-1, 1)
    cos.fill_diagonal_(-1)
    nearest_angle = torch.rad2deg(torch.arccos(cos.max(1).values)).cpu().numpy()
    iu = torch.triu_indices(C, C, 1)
    mean_angle = float(torch.rad2deg(torch.arccos((cent @ cent.t()).clamp(-1, 1)[iu[0], iu[1]])).mean())
    purity = torch.zeros(feat.shape[0], device=feat.device)
    for s in range(0, feat.shape[0], 1000):
        sim = feat[s:s + 1000] @ feat.t()
        idx = torch.arange(s, min(s + 1000, feat.shape[0]), device=feat.device)
        sim[torch.arange(len(idx)), idx] = -2  # exclude self
        nn_idx = sim.topk(k, dim=1).indices
        purity[s:s + 1000] = (y[nn_idx] == y[idx][:, None]).float().mean(1)
    purity_c = np.array([float(purity[y == c].mean()) for c in range(C)])
    return {
        "dispersion": D.tolist(),
        "nearest_centroid_angle": nearest_angle.tolist(),
        "mean_centroid_angle": mean_angle,
        "min_centroid_angle": float(nearest_angle.min()),
        "knn_purity": purity_c.tolist(),
        "mean_knn_purity": float(purity_c.mean()),
        "mean_dispersion": float(np.nanmean(D)),
    }
