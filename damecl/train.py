"""One training run. Example:
    python -m damecl.train --dataset cifar10 --ir 100 --method dam_ho --seed 0 --epochs 100

Methods (existing approaches first, ours last)
  ce         cross-entropy                                   (weak aug, 1 view)
  wce        class-weighted cross-entropy (1/N_c)            (weak aug, 1 view)
  la         logit-adjusted cross-entropy (Menon et al.)     (weak aug, 1 view)
  supcon     0.5*SupCon + 0.5*CE (Khosla et al.)             (strong aug, 2 views)
  ecl        0.5*BC-ECL + 3*CC-GE + 0.5*logit compensation   (strong aug, 2 views)
  dam_v1     DAM-ECL as first written (margin from rarity + TRAINING dispersion)
  dam_v2     DAM-ECL with small fixes + rare-class floor (dispersion still from TRAINING data)
  dam_ho     DAM-ECL with dispersion measured on a small HELD-OUT probe set (v2 of our idea)
Datasets: cifar10, cifar100, mnist, fmnist (Fashion-MNIST), svhn. 28x28 grayscale images are padded
to 32x32x3; flips are off for the digit datasets (mnist, svhn); test sets are class-balanced.
  ecl_cmo    ECL + context-rich minority oversampling (CMO: rare images pasted onto common ones, q_c ~ 1/N_c)
  dam_mix    ECL + minority pasting driven by the memorisation gap G_c = D_probe / D_train measured on the
             held-out probe. FINAL METHOD ("DAM-Mix"):  --method dam_mix --paste gaponly --gap_power 2
             Variants: --paste gap (rarity x gap, "DAM-Mix-R"), --paste sqrtinv (softened CMO, no probe)
             Control:  --paste gapshuffle (the measured gaps are randomly permuted inside the Many,
                       Medium and Few groups every epoch: keeps the coarse rarity trend of G, destroys
                       which class inside a group has the larger gap)
Every run also reports "metrics_posthoc": logits - tau*log(prior), tau tuned on the probe.

Data-fair control switches
  --probe_in_train   train on the probe images too, from the start (e.g. "ECL + probe data")
  --absorb           ours: use the probe for measuring until 80% of training, then freeze the
                     margins and train on the probe images for the last 20%
Ablations (for dam_ho / dam_v2): --mode fixed|rarity|dispersion, --no_floor, --lam_align 0, --no_alc
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import losses as L
from .metrics import classification_metrics, geometry_metrics, dispersion_per_class, class_groups
from .model import Net

CONTRASTIVE = {"supcon", "ecl", "dam_v1", "dam_v2", "dam_ho", "ecl_cmo", "dam_mix"}
MIX = {"ecl_cmo", "dam_mix"}
METHODS = ["ce", "wce", "la"] + sorted(CONTRASTIVE)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=sorted(D.DATASETS))
    p.add_argument("--ir", type=float, default=100)
    p.add_argument("--method", default="dam_ho", choices=METHODS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--tau", type=float, default=0.1)
    # probe (held-out) set
    p.add_argument("--probe_k", type=int, default=-1, help="probe images per class (-1: 20 for 10 classes, 10 for 100)")
    p.add_argument("--n_max_cap", type=int, default=4980,
                   help="largest class size of the long-tailed set (4980 = CIFAR-10-LT; same profile on every 10-class dataset)")
    p.add_argument("--probe_in_train", action="store_true")
    p.add_argument("--absorb", action="store_true")
    p.add_argument("--absorb_frac", type=float, default=0.8)
    # minority pasting (ecl_cmo / dam_mix)
    p.add_argument("--paste", default="", choices=["", "inv", "sqrtinv", "gap", "gaponly", "gapshuffle"],
                   help="foreground class sampling; default: inv for ecl_cmo, gap for dam_mix")
    p.add_argument("--rarity_power", type=float, default=0.5)
    p.add_argument("--gap_power", type=float, default=1.0)
    p.add_argument("--mix_weight", type=float, default=0.5)
    p.add_argument("--mix_label", default="area", choices=["area", "ink"],
                   help="CutMix label share by pixel area (CMO) or by content mass (for black-background images)")
    # dataset variants (round 3): when class counts do not describe the data
    p.add_argument("--profile", default="exp", choices=["exp", "step"], help="exponential long tail or step imbalance")
    p.add_argument("--n_minor", type=int, default=2, help="step profile: number of minority classes")
    p.add_argument("--dup", default="", help="e.g. '0:250,2:90': class 0 keeps its count but only 250 distinct images")
    # DAM-ECL
    p.add_argument("--m0", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=0.15)
    p.add_argument("--m_max", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--s_alc", type=float, default=5.0)
    p.add_argument("--mode", default="full", choices=["full", "fixed", "rarity", "dispersion"])
    p.add_argument("--no_floor", action="store_true")
    p.add_argument("--no_alc", action="store_true")
    p.add_argument("--lam_con", type=float, default=0.5)
    p.add_argument("--lam_align", type=float, default=3.0)
    p.add_argument("--lam_cls", type=float, default=0.5)
    # infra
    p.add_argument("--data_root", default="./data")
    p.add_argument("--fake_data", action="store_true", help="random data, for smoke tests only")
    p.add_argument("--out", default="results/runs/test.json")
    p.add_argument("--emb_out", default="")
    p.add_argument("--save_model", default="", help="path to save the trained weights (.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=0, help="debug: stop each epoch after N steps")
    return p.parse_args(argv)


def lr_at(progress, epochs, base, warm_epochs=5):
    """progress = fractional epoch. Linear warm-up, then cosine decay to 0."""
    if progress < warm_epochs:
        return base * (progress + 1e-3) / warm_epochs
    t = (progress - warm_epochs) / max(epochs - warm_epochs, 1e-6)
    return base * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))


@torch.no_grad()
def predict(model, x, device, bs=1000):
    """Returns normalised backbone features h, logits and projected embeddings z (eval mode, no aug)."""
    model.eval()
    hs, logits, zs = [], [], []
    for s in range(0, x.shape[0], bs):
        h, lg, z = model(D.normalize(x[s:s + bs].to(device)))
        hs.append(F.normalize(h.float(), dim=1))
        logits.append(lg.float())
        zs.append(z.float())
    model.train()
    return torch.cat(hs), torch.cat(logits), torch.cat(zs)


def run(args):
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    use_amp = dev.type == "cuda"
    torch.backends.cudnn.benchmark = True

    # ---------------- data: probe split, then long-tailed training set (identical for all methods)
    C_ds, gray, flip_ok = D.DATASETS[args.dataset]
    D.set_flip(flip_ok)
    if args.fake_data:
        xall, yall, xte, yte, C = D.fake_cifar(C_ds, n_train=200 if C_ds == 10 else 100, n_test=20)
        if gray:   # exercise the 28x28 grayscale path too
            xall, xte = D.to_rgb32(xall[:, 2:30, 2:30, 0]), D.to_rgb32(xte[:, 2:30, 2:30, 0])
    else:
        xall, yall, xte, yte, C = D.load_dataset(args.dataset, args.data_root)
    k = args.probe_k if args.probe_k >= 0 else (20 if C == 10 else 10)
    per_class = int(np.bincount(yall, minlength=C).min())
    xpr, ypr = D.split_probe(xall, yall, C, k)
    xtr, ytr, counts = D.make_long_tailed(xall, yall, C, args.ir, n_max=min(per_class - k, args.n_max_cap),
                                          profile=args.profile, n_minor=args.n_minor)
    if args.dup:
        xtr = D.duplicate_classes(xtr, ytr, args.dup)
    if args.dataset not in ("cifar10", "cifar100"):
        D.set_normalisation(xtr)   # CIFAR keeps its standard constants (reproduces the earlier runs)
    lt_counts = list(counts)                      # used for rarity and for Many/Medium/Few groups
    Xtr, Ytr = D.to_tensor(xtr, dev), torch.from_numpy(ytr).long().to(dev)
    Xpr, Ypr = D.to_tensor(xpr, dev), torch.from_numpy(ypr).long().to(dev)
    Xte, Yte = D.to_tensor(xte, dev), torch.from_numpy(yte).long().to(dev)
    train_counts = list(counts)
    if args.probe_in_train:
        Xtr, Ytr = torch.cat([Xtr, Xpr]), torch.cat([Ytr, Ypr])
        train_counts = [c + k for c in counts]

    def priors(cnts):
        cnt = torch.tensor(cnts, dtype=torch.float32, device=dev).clamp(min=1)
        w = 1.0 / cnt
        return torch.log(cnt / cnt.sum()), w / w.mean()
    log_prior, cls_w = priors(train_counts)

    # ---------------- model / optimiser
    model = Net(C).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    method = args.method
    contrastive = method in CONTRASTIVE
    mstate = None
    if method in ("dam_v1", "dam_v2", "dam_ho"):
        mstate = L.MarginState(lt_counts, version="v1" if method == "dam_v1" else "v2", mode=args.mode,
                               m0=args.m0, alpha=args.alpha, beta=args.beta, m_max=args.m_max,
                               gamma=args.gamma, floor=(method == "dam_v2" and not args.no_floor),
                               source="heldout" if method == "dam_ho" else "train",
                               device=dev, total_epochs=args.epochs)
    centroids = torch.zeros(C, 64, device=dev)
    seen = torch.zeros(C, dtype=torch.bool, device=dev)
    absorb_epoch = int(args.absorb_frac * args.epochs) if args.absorb else None
    is_mix = method in MIX
    paste_mode = args.paste or ("inv" if method == "ecl_cmo" else "gap")
    gap = L.GapTracker(C, dev) if (is_mix and paste_mode in ("gap", "gaponly", "gapshuffle")) else None
    groups_idx = [torch.as_tensor(g, device=dev, dtype=torch.long)
                  for g in class_groups(lt_counts).values() if len(g)]
    shuffle_gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)
    paste_hist = []

    curves = []
    for epoch in range(args.epochs):
        if absorb_epoch is not None and epoch == absorb_epoch:
            # data-fair variant of ours: stop measuring, freeze margins, then train on the probe too
            mstate.frozen = True
            Xtr, Ytr = torch.cat([Xtr, Xpr]), torch.cat([Ytr, Ypr])
            train_counts = [c + k for c in lt_counts]
            log_prior, cls_w = priors(train_counts)
        N = len(Ytr)
        steps = max(N // args.batch_size, 1)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        perm = torch.randperm(N, device=dev)
        margins = mstate.margins(epoch) if mstate is not None else None
        if is_mix:  # per-class paste probabilities for this epoch -> per-image sampling weights
            use_gap = gap is not None and epoch >= int(0.1 * args.epochs)
            G_used = gap.G if use_gap else None
            if use_gap and paste_mode == "gapshuffle":   # control: permute G inside Many / Medium / Few
                G_used = gap.G.clone()
                for g in groups_idx:
                    perm_g = g[torch.randperm(len(g), generator=shuffle_gen).to(dev)]
                    G_used[g] = gap.G[perm_g]
            q = L.paste_probabilities(lt_counts, "gaponly" if paste_mode == "gapshuffle" else paste_mode,
                                      G_used, args.rarity_power, args.gap_power).to(dev)
            cnt_now = torch.bincount(Ytr, minlength=C).float().clamp(min=1)
            img_w = (q / cnt_now)[Ytr]
            paste_hist.append(q.tolist())
        ep_loss = 0.0
        for b in range(steps):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch + b / steps, args.epochs, args.lr)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                if contrastive:
                    x2 = torch.cat([D.strong_aug(xb), D.strong_aug(xb)]).contiguous(memory_format=torch.channels_last)
                    y2 = torch.cat([yb, yb])
                else:
                    x2 = D.weak_aug(xb).contiguous(memory_format=torch.channels_last)
                    y2 = yb
                if is_mix:  # paste rare images (foreground) onto this batch's images (background)
                    fg = torch.multinomial(img_w, len(idx), replacement=True)
                    xmix, lam_bg = D.cutmix(D.weak_aug(xb), D.weak_aug(Xtr[fg]), label=args.mix_label)
                    y_fg = Ytr[fg]
                    nb2 = x2.shape[0]
                    h, logits, z = model(torch.cat([x2, xmix.contiguous(memory_format=torch.channels_last)]))
                    logits_mix = logits[nb2:].float()
                    h, logits, z = h[:nb2], logits[:nb2], z[:nb2]
                else:
                    h, logits, z = model(x2)
            logits = logits.float()

            if method == "ce":
                loss = L.ce_loss(logits, y2)
            elif method == "wce":
                loss = L.weighted_ce_loss(logits, y2, cls_w)
            elif method == "la":
                loss = L.logit_adjusted_loss(logits, y2, log_prior)
            elif method == "supcon":
                loss = 0.5 * L.supcon_loss(z, y2, args.tau) + 0.5 * L.ce_loss(logits, y2)
            else:  # ecl / dam_*
                with torch.no_grad():  # running class centres of normalised features (for CC-GE / MACA)
                    hn = F.normalize(h.float(), dim=1)
                    sums = torch.zeros_like(centroids).index_add_(0, y2, hn)
                    nc = torch.bincount(y2, minlength=C).float()
                    present = nc > 0
                    mean_c = sums / nc.clamp(min=1)[:, None]
                    upd = torch.where(seen[:, None], 0.9 * centroids + 0.1 * mean_c, mean_c)
                    centroids = torch.where(present[:, None], upd, centroids)
                    seen = seen | present
                protos = model.prototypes()
                W = model.classifier.weight.float()
                if method in ("ecl", "ecl_cmo", "dam_mix"):
                    l_con = L.ecl_contrastive_loss(z, y2, protos, C, args.tau)
                    l_align = L.alignment_loss(W, centroids)
                    l_cls = L.logit_adjusted_loss(logits, y2, log_prior)
                    if is_mix:
                        l_cls = l_cls + args.mix_weight / args.lam_cls * L.mixed_la_loss(
                            logits_mix, yb, y_fg, lam_bg, log_prior)
                else:
                    l_con = L.ecl_contrastive_loss(z, y2, protos, C, args.tau, margins=margins)
                    l_align = L.alignment_loss(W, centroids, margins, use_sqrt_m=(method == "dam_v1"))
                    if args.no_alc:
                        l_cls = L.ce_loss(logits, y2)
                    else:
                        l_cls = L.alc_loss(logits, y2, log_prior, margins, 1.0 if method == "dam_v1" else args.s_alc)
                    mstate.accumulate(z, y2)
                loss = args.lam_con * l_con + args.lam_align * l_align + args.lam_cls * l_cls

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {b}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        if mstate is not None:  # measure dispersion: training views (always) + held-out probe
            _, _, zp = predict(model, Xpr, dev)
            mstate.end_epoch(epoch, zp, Ypr)
        if gap is not None:      # memorisation gap: clean training images vs held-out probe
            n_lt0 = int(sum(lt_counts))
            htr_e, _, _ = predict(model, Xtr[:n_lt0], dev)
            hpr_e, _, _ = predict(model, Xpr, dev)
            gap.update(epoch, htr_e, Ytr[:n_lt0], hpr_e, Ypr)
        rec = {"epoch": epoch + 1, "loss": ep_loss / steps, "time_min": (time.time() - t0) / 60}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            _, lg, _ = predict(model, Xte, dev)
            pr = lg.argmax(1)
            rec["test_acc"] = float((pr == Yte).float().mean())
            few = np.array(class_groups(lt_counts)["few"])
            mask = torch.isin(Yte, torch.as_tensor(few, device=dev))
            rec["test_few_acc"] = float((pr[mask] == Yte[mask]).float().mean())
            print(f"[{method} s{args.seed}] ep {epoch + 1}/{args.epochs} loss {rec['loss']:.3f} "
                  f"acc {rec['test_acc']:.3f} few {rec['test_few_acc']:.3f} ({rec['time_min']:.1f} min)", flush=True)
        curves.append(rec)

    # ---------------- final evaluation
    hte, lte, _ = predict(model, Xte, dev)
    res = {"config": vars(args), "counts": lt_counts, "train_counts": train_counts, "probe_k": k,
           "num_classes": C, "train_size": int(len(Ytr))}
    res["metrics"] = classification_metrics(yte, lte.cpu().numpy(), lt_counts)
    # post-hoc logit adjustment, tau tuned on the held-out probe (applied identically to every method)
    _, lpr, _ = predict(model, Xpr, dev)
    lp_train = torch.log(torch.tensor(train_counts, dtype=torch.float32, device=dev).clamp(min=1) / sum(train_counts))
    tau, probe_bacc = L.posthoc_tau(lpr, Ypr, lp_train)
    res["probe_metrics"] = {kk: v for kk, v in classification_metrics(
        ypr, lpr.cpu().numpy(), lt_counts).items() if kk not in ("confusion",)}   # for choosing settings on the probe
    res["posthoc"] = {"tau": tau, "probe_balanced_acc": probe_bacc,
                      "probe_was_trained_on": bool(args.probe_in_train or args.absorb)}
    res["metrics_posthoc"] = classification_metrics(yte, (lte - tau * lp_train).cpu().numpy(), lt_counts)
    res["geometry"] = geometry_metrics(hte, Yte, C)
    n_lt = int(sum(lt_counts))
    htr, _, _ = predict(model, Xtr[:n_lt], dev)        # the long-tailed training images only
    hpr, _, _ = predict(model, Xpr, dev)
    res["geometry"]["train_dispersion"] = dispersion_per_class(htr, Ytr[:n_lt], C).tolist()
    res["geometry"]["probe_dispersion"] = dispersion_per_class(hpr, Ypr, C).tolist()
    res["curves"] = curves
    if mstate is not None:
        res["margin_state"] = {"R": mstate.R.tolist(), "D": mstate.D.tolist(),
                               "D_effective": mstate.effective_D().tolist(),
                               "m_final": mstate.margins(args.epochs).tolist(),
                               "source": mstate.source, "history": mstate.history}
    if is_mix:
        res["paste"] = {"mode": paste_mode, "q_per_epoch": paste_hist,
                        "gap_history": gap.history if gap is not None else None}
    res["time_min"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, args.out)   # a result file only appears once the run is completely finished
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "num_classes": C, "config": vars(args),
                    "train_counts": train_counts, "test_metrics": res["metrics"],
                    "posthoc_tau": tau}, args.save_model)
        print(f"saved weights to {args.save_model}", flush=True)
    if args.emb_out:
        per = 200 if C <= 10 else 20
        keep = np.concatenate([np.where(yte == c)[0][:per] for c in range(C)])
        os.makedirs(os.path.dirname(args.emb_out) or ".", exist_ok=True)
        np.savez_compressed(args.emb_out, h=hte[keep].cpu().numpy().astype(np.float16), y=yte[keep])
    m = res["metrics"]
    mp = res["metrics_posthoc"]
    print(f"DONE {method} seed {args.seed}: acc {m['accuracy']:.4f} few {m['few_acc']:.4f} "
          f"macroF1 {m['macro_f1']:.4f} | post-hoc (tau={tau:+.1f}) acc {mp['accuracy']:.4f} few {mp['few_acc']:.4f} "
          f"in {res['time_min']:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    run(get_args())
