"""Data: CIFAR-10/100-LT construction and fast GPU-side augmentation.

The whole (small) long-tailed training set is kept on the GPU as a tensor, and
augmentations are done on the GPU in batches. This avoids the slow CPU
DataLoader and is the main reason one run takes minutes instead of an hour.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
FLIP = True   # horizontal flips in augmentation (switched off for digit datasets, where a flip changes the meaning)

# name -> (number of classes, grayscale 28x28 source?, horizontal flip allowed?)
DATASETS = {
    "cifar10": (10, False, True),
    "cifar100": (100, False, True),
    "mnist": (10, True, False),
    "fmnist": (10, True, True),
    "svhn": (10, False, False),
}


def set_normalisation(x_uint8):
    """Per-dataset channel mean/std, computed on the long-tailed TRAINING images only
    (CIFAR keeps the standard published CIFAR-10 constants, so earlier runs are reproduced exactly)."""
    global MEAN, STD
    x = torch.from_numpy(x_uint8).float().div_(255.0)
    MEAN = x.mean(dim=(0, 1, 2)).view(1, 3, 1, 1)
    STD = x.std(dim=(0, 1, 2)).clamp(min=1e-3).view(1, 3, 1, 1)


def set_flip(allowed):
    global FLIP
    FLIP = bool(allowed)


# ----------------------------------------------------------------------------- loading
HF_URLS = {  # fast mirror of the official files (same images, same label ids)
    "cifar10": ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet", "label"),
    "cifar100": ("https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{split}-00000-of-00001.parquet", "fine_label"),
    # fallbacks only: the new datasets are loaded with torchvision first (official sources)
    "mnist": ("https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/{split}-00000-of-00001.parquet", "label"),
    "fmnist": ("https://huggingface.co/datasets/zalando-datasets/fashion_mnist/resolve/main/fashion_mnist/{split}-00000-of-00001.parquet", "label"),
    "svhn": ("https://huggingface.co/datasets/ufldl-stanford/svhn/resolve/main/cropped_digits/{split}-00000-of-00001.parquet", "label"),
}


def _canonical_order(x, y):
    """Sort images by a hash of their pixels. The long-tailed subset ('first n_c images of each class')
    and the probe ('last k images') then pick EXACTLY the same images whichever source the data came
    from (HuggingFace mirror, Kaggle copy, or the original Toronto download)."""
    import hashlib
    keys = [hashlib.md5(img.tobytes()).hexdigest() for img in x]
    order = np.array(sorted(range(len(y)), key=lambda i: keys[i]))
    return x[order], y[order]


def _load_hf(name, root):
    import io
    import os
    import urllib.request
    import pandas as pd
    from PIL import Image
    url, col = HF_URLS[name]
    out = []
    for split in ("train", "test"):
        path = os.path.join(root, f"{name}_{split}.parquet")
        if not os.path.exists(path):
            os.makedirs(root, exist_ok=True)
            print(f"downloading {name} {split} from the HuggingFace mirror ...", flush=True)
            urllib.request.urlretrieve(url.format(split=split), path + ".part")
            os.replace(path + ".part", path)
        df = pd.read_parquet(path)
        imgcol = "img" if "img" in df.columns else ("image" if "image" in df.columns else df.columns[0])
        x = np.stack([np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB")) for v in df[imgcol]])
        out += [x.astype(np.uint8), df[col].to_numpy().astype(np.int64)]
    return out


def _load_torchvision(name, root, download):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    tr = cls(root, train=True, download=download)
    te = cls(root, train=False, download=download)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets)


def load_cifar(name, root):
    """Returns (x_train uint8 [N,32,32,3], y_train, x_test, y_test, num_classes).
    Tries, in order: a cached copy in `root`, files already on disk (torchvision format),
    the fast HuggingFace mirror, and finally the slow official download."""
    import os
    C = 10 if name == "cifar10" else 100
    cache = os.path.join(root, f"{name}_canonical.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"], C
    try:
        xtr, ytr, xte, yte = _load_torchvision(name, root, download=False)   # already on disk
    except Exception:
        try:
            xtr, ytr, xte, yte = _load_hf(name, root)
        except Exception as e:
            print(f"HuggingFace mirror failed ({e}); falling back to the official (slow) download", flush=True)
            xtr, ytr, xte, yte = _load_torchvision(name, root, download=True)
    assert len(ytr) == 50000 and len(yte) == 10000, "unexpected CIFAR size"
    xtr, ytr = _canonical_order(xtr, ytr)
    os.makedirs(root, exist_ok=True)
    with open(cache + ".part", "wb") as f:      # write then rename, so a half-written cache never exists
        np.savez(f, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    os.replace(cache + ".part", cache)
    return xtr, ytr, xte, yte, C


def to_rgb32(x):
    """uint8 images [N,H,W] or [N,H,W,1|3] -> [N,32,32,3]. 28x28 grayscale digits/clothes are zero-padded
    by 2 pixels on every side (their background is already black) and copied into 3 channels, so the
    same ResNet-32 and the same augmentation code serve every dataset."""
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    h, w = x.shape[1], x.shape[2]
    if (h, w) != (32, 32):
        ph, pw = (32 - h) // 2, (32 - w) // 2
        x = np.pad(x, ((0, 0), (ph, 32 - h - ph), (pw, 32 - w - pw), (0, 0)))
    return x.astype(np.uint8)


def _load_torchvision_other(name, root):
    import torchvision
    if name == "svhn":
        tr = torchvision.datasets.SVHN(root, split="train", download=True)
        te = torchvision.datasets.SVHN(root, split="test", download=True)
        # torchvision stores SVHN as NCHW and already maps the digit '0' to label 0
        return (tr.data.transpose(0, 2, 3, 1), np.asarray(tr.labels),
                te.data.transpose(0, 2, 3, 1), np.asarray(te.labels))
    cls = torchvision.datasets.MNIST if name == "mnist" else torchvision.datasets.FashionMNIST
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()


def balance_test(x, y, C):
    """Keep the same number of test images for every class (the smallest class size), so that test
    accuracy equals balanced accuracy on every dataset (MNIST and SVHN test sets are not balanced)."""
    m = int(np.bincount(y, minlength=C).min())
    idx = np.concatenate([np.where(y == c)[0][:m] for c in range(C)])
    return x[idx], y[idx]


def load_dataset(name, root):
    """Any supported dataset as (x_train uint8 [N,32,32,3], y_train, x_test, y_test, num_classes)."""
    import os
    if name in ("cifar10", "cifar100"):
        return load_cifar(name, root)
    C = DATASETS[name][0]
    cache = os.path.join(root, f"{name}_canonical.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"], C
    try:
        xtr, ytr, xte, yte = _load_torchvision_other(name, root)
    except Exception as e:
        print(f"torchvision download of {name} failed ({e}); trying the HuggingFace mirror", flush=True)
        xtr, ytr, xte, yte = _load_hf(name, root)
    xtr, xte = to_rgb32(xtr), to_rgb32(xte)
    ytr, yte = np.asarray(ytr).astype(np.int64), np.asarray(yte).astype(np.int64)
    assert ytr.min() == 0 and ytr.max() == C - 1, f"unexpected labels for {name}"
    xtr, ytr = _canonical_order(xtr, ytr)
    xte, yte = _canonical_order(xte, yte)
    xte, yte = balance_test(xte, yte, C)
    os.makedirs(root, exist_ok=True)
    with open(cache + ".part", "wb") as f:
        np.savez(f, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    os.replace(cache + ".part", cache)
    return xtr, ytr, xte, yte, C


def fake_cifar(C=10, n_train=300, n_test=50, seed=0):
    """Random stand-in data with the same shapes (used for smoke tests only)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(C, 32, 32, 3))
    def make(n):
        y = np.repeat(np.arange(C), n)
        x = np.clip(base[y] + rng.normal(0, 60, size=(len(y), 32, 32, 3)), 0, 255).astype(np.uint8)
        return x, y
    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte, C


def lt_class_counts(n_max, C, ir):
    """Standard exponential long-tail profile (Cao et al., 2019): n_c = n_max * ir^(-c/(C-1))."""
    return [int(n_max * (1.0 / ir) ** (c / (C - 1))) for c in range(C)]


def step_class_counts(n_max, C, ir, n_minor):
    """Step imbalance: the last n_minor classes keep n_max/ir images, all others keep n_max."""
    n_min = int(n_max / ir)
    return [n_max] * (C - n_minor) + [n_min] * n_minor


def make_long_tailed(x, y, C, ir, n_max=None, profile="exp", n_minor=2):
    """Keep the first n_c images of class c (deterministic, the usual CIFAR-LT recipe).
    profile 'exp': exponential long tail (Cao et al.); 'step': n_minor rare classes of equal size."""
    if n_max is None:
        n_max = int(np.bincount(y, minlength=C).max())
    counts = lt_class_counts(n_max, C, ir) if profile == "exp" else step_class_counts(n_max, C, ir, n_minor)
    idx = np.concatenate([np.where(y == c)[0][:counts[c]] for c in range(C)])
    return x[idx], y[idx], counts


def split_probe(x, y, C, k):
    """Held-out PROBE set: the LAST k training images of every class. The long-tailed training set
    uses the FIRST n_c images of each class, with n_max = (images per class) - k, so the probe never
    overlaps the training set. The probe is balanced (k per class) and is never trained on
    (except in the data-fair control runs)."""
    idx = np.concatenate([np.where(y == c)[0][-k:] for c in range(C)])
    return x[idx], y[idx]


def duplicate_classes(x, y, spec):
    """'c:U,c2:U2' -> class c keeps its N_c images but only U of them are distinct (each repeated ~N_c/U times).
    The class count stays N_c, so count-based rules still see a large class; its real diversity is U images."""
    x = x.copy()
    for item in filter(None, spec.split(",")):
        c, u = (int(v) for v in item.split(":"))
        idx = np.where(y == c)[0]
        x[idx] = x[np.resize(idx[:u], len(idx))]
    return x


def to_tensor(x_uint8, device):
    """uint8 NHWC -> float NCHW in [0,1] on device (kept in [0,1]; normalised after augmentation)."""
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ----------------------------------------------------------------------------- GPU augmentation
def _affine_crop(x, scale_min, scale_max, ratio=(3 / 4, 4 / 3), translate_only=False):
    """Batched random-resized-crop + horizontal flip with one grid_sample call."""
    n = x.shape[0]
    dev = x.device
    if translate_only:  # pad-4 random crop equivalent: shift up to 4/32 of the image
        sx = torch.ones(n, device=dev)
        sy = torch.ones(n, device=dev)
        tx = (torch.randint(-4, 5, (n,), device=dev).float() * 2 / 32)
        ty = (torch.randint(-4, 5, (n,), device=dev).float() * 2 / 32)
    else:
        area = torch.empty(n, device=dev).uniform_(scale_min, scale_max)
        logr = torch.empty(n, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]))
        r = torch.exp(logr)
        sx = torch.sqrt(area * r).clamp(max=1.0)
        sy = torch.sqrt(area / r).clamp(max=1.0)
        tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - sx)
        ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - sy)
    flip = (torch.rand(n, device=dev) < 0.5).float() * 2 - 1 if FLIP else torch.ones(n, device=dev)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = sx * flip
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = sy
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _color_jitter(x, strength=0.4, p=0.8):
    n = x.shape[0]
    dev = x.device
    apply = (torch.rand(n, 1, 1, 1, device=dev) < p).float()
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * strength
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * strength
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * strength
    y = x * b
    m = y.mean(dim=(1, 2, 3), keepdim=True)
    y = (y - m) * c + m
    g = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])
    y = (y - g) * s + g
    y = y.clamp(0, 1)
    return apply * y + (1 - apply) * x


def _random_gray(x, p=0.2):
    n = x.shape[0]
    g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)
    m = (torch.rand(n, 1, 1, 1, device=x.device) < p).float()
    return m * g + (1 - m) * x


def weak_aug(x):
    """Standard CIFAR augmentation for CE-type methods: random crop (pad 4) + flip."""
    return normalize(_affine_crop(x, 1, 1, translate_only=True))


def strong_aug(x):
    """SimCLR-style augmentation for contrastive methods."""
    x = _affine_crop(x, 0.2, 1.0)
    x = _color_jitter(x)
    x = _random_gray(x)
    return normalize(x)


def cutmix(bg, fg, label="area"):
    """CutMix (Yun et al., 2019) as used by CMO (Park et al., 2022): paste a random box of each `fg`
    image onto the matching `bg` image. Box area = (1 - lam) of the image with lam ~ U(0, 1).
    Returns the mixed batch and lam_bg = the exact share of pixels still coming from the background."""
    n, _, H, W = bg.shape
    dev = bg.device
    lam = torch.rand(n, device=dev)
    cut = torch.sqrt(1 - lam)
    bw, bh = (W * cut).long(), (H * cut).long()
    cx = torch.randint(0, W, (n,), device=dev)
    cy = torch.randint(0, H, (n,), device=dev)
    x0, x1 = (cx - bw // 2).clamp(0, W), (cx + bw // 2).clamp(0, W)
    y0, y1 = (cy - bh // 2).clamp(0, H), (cy + bh // 2).clamp(0, H)
    xs = torch.arange(W, device=dev)[None, :]
    ys = torch.arange(H, device=dev)[None, :]
    mx = (xs >= x0[:, None]) & (xs < x1[:, None])            # [n, W]
    my = (ys >= y0[:, None]) & (ys < y1[:, None])            # [n, H]
    box = (my[:, :, None] & mx[:, None, :]).unsqueeze(1).to(bg.dtype)   # [n, 1, H, W]
    mixed = bg * (1 - box) + fg * box
    lam_bg = 1 - box.flatten(1).mean(1)
    if label == "ink":
        # label split by content mass instead of area: on black-background images (MNIST, Fashion-MNIST) an empty
        # region carries no evidence for its class. Mass = grey level of the un-normalised image.
        def mass(x):
            return (x * STD.to(x.device) + MEAN.to(x.device)).clamp(0, 1).mean(1, keepdim=True)
        m_bg = (mass(bg) * (1 - box)).flatten(1).sum(1)
        m_fg = (mass(fg) * box).flatten(1).sum(1)
        tot = m_bg + m_fg
        lam_bg = torch.where(tot > 1e-6, m_bg / tot.clamp(min=1e-6), lam_bg)
    return mixed, lam_bg
