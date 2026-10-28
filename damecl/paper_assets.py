"""Tables (LaTeX, booktabs) and figures (vector PDF) for the journal paper, built only from the result files.

    python make_paper_assets.py --results results [more result folders ...] --out paper_assets

Every number in the paper should come from here, so nothing is typed by hand. Each table is a bare
`tabular` (no caption) that main.tex pulls in with \\input{...}; captions are written in main.tex.
Works with partial results: datasets or methods without runs are skipped, never invented.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DS_ORDER = ["cifar10", "cifar100", "fmnist", "svhn", "mnist"]
DS_NAME = {"cifar10": "CIFAR-10-LT", "cifar100": "CIFAR-100-LT", "fmnist": "Fashion-MNIST-LT",
           "svhn": "SVHN-LT", "mnist": "MNIST-LT"}
METHODS = ["ce", "wce", "la", "ecl", "ecl_cmo", "dam_mix_nogap", "dam_mix_gaponly_p1"]
BASELINES = ["ecl", "ecl_cmo", "dam_mix_nogap"]          # the ones ours is tested against (paired)
OURS = "dam_mix_gaponly_p1"   # final configuration: gap power 1, selected on the probe
NAME = {"ce": "CE", "wce": "Weighted CE", "la": "Logit-adjusted CE", "supcon": "SupCon", "ecl": "ECL",
        "ecl_cmo": "ECL + CMO", "dam_mix_nogap": "ECL + CMO$_{r=1/2}$", "dam_mix": "DAM-Mix-R (rarity $\\times$ gap)",
        "dam_mix_gaponly": "DAM-Mix, $\\kappa=2$", "dam_v1": "DAM-ECL v1", "dam_v2": "DAM-ECL v2", "dam_ho": "DAM-ECL-HO",
        "dam_mix_gapshuffle": "DAM-Mix ($\\kappa=2$), gaps pooled within groups", "dam_mix_gaponly_p1": "DAM-Mix (ours)",
        "dam_mix_gaponly_p3": "DAM-Mix, $\\kappa=3$", "dam_mix_gapshuffle_p1": "DAM-Mix, gaps pooled within groups", "ecl_cmo_plus": "ECL + CMO + probe images",
        "dam_mix_nogap_plus": "ECL + CMO$_{r=1/2}$ + probe images", "ecl_plus": "ECL + probe images"}
PLAIN = {k: re.sub(r"\$_\{r=1/2\}\$", " (r=1/2)", v).replace("$\\times$", "x").replace("$", "")
         for k, v in NAME.items()}
COLOR = {"ce": "#9e9e9e", "wce": "#bdbdbd", "la": "#e69f00", "supcon": "#009e73", "ecl": "#d55e00",
         "ecl_cmo": "#cc79a7", "dam_mix_nogap": "#7f7f7f", "dam_mix": "#56b4e9", "dam_mix_gaponly": "#56b4e9",
         "dam_mix_gaponly_p1": "#0060c0", "dam_mix_gapshuffle_p1": "#8fb3e0"}

# (key, header, higher is better?)
MAIN_COLS = [("acc", "Acc.", True), ("many", "Many", True), ("medium", "Med.", True), ("few", "Few", True),
             ("macro_f1", "F1", True), ("g_mean", "G-Mean", True), ("precision", "Prec.", True),
             ("auprc", "AUPRC", True), ("ece", "ECE$\\downarrow$", False)]


# ----------------------------------------------------------------------------- loading
def load_runs(folders):
    """(dataset, ir, tag) -> {seed: run}. Later folders do not overwrite earlier identical runs."""
    runs = defaultdict(dict)
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(folder, "runs", "*.json"))):
            m = re.match(r"^([a-z0-9]+)_ir(\d+)_(.+)_s(\d+)\.json$", os.path.basename(f))
            if not m or m.group(1) not in DS_NAME:
                continue
            key = (m.group(1), int(m.group(2)), m.group(3))
            seed = int(m.group(4))
            if seed not in runs[key]:
                with open(f) as fh:
                    r = json.load(fh)
                r["_seed"], r["_emb"] = seed, os.path.join(folder, "emb", os.path.basename(f)[:-5] + ".npz")
                runs[key][seed] = r
    return runs


def get(runs, ds, tag, ir=100):
    return [runs[(ds, ir, tag)][s] for s in sorted(runs.get((ds, ir, tag), {}))]


def metrics_of(r, posthoc=False):
    """Flat metric dict in [0,1]; G-mean / precision are rebuilt from the confusion matrix for older runs."""
    m = r["metrics_posthoc" if posthoc else "metrics"]
    conf = np.array(m["confusion"], dtype=float)
    rec = np.array(m["per_class_recall"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.nan_to_num(np.diag(conf) / conf.sum(0))
    gmean = float(np.exp(np.mean(np.log(np.clip(rec, 1e-12, 1))))) if np.all(rec > 0) else 0.0
    return {"acc": m["accuracy"], "many": m["many_acc"], "medium": m["medium_acc"], "few": m["few_acc"],
            "macro_f1": m["macro_f1"], "few_f1": m.get("few_f1", m.get("tail_f1")),
            "g_mean": m.get("g_mean", gmean), "precision": m.get("macro_precision", float(prec.mean())),
            "recall": float(np.nanmean(rec)), "auprc": m["macro_auprc"], "ece": m["ece"],
            "worst": m["worst_class_recall"]}


def values(rs, key, posthoc=False):
    return np.array([metrics_of(r, posthoc)[key] for r in rs], dtype=float) * 100


def fmt(v, bold=False, digits=2):
    if len(v) == 0:
        return "--"
    s = f"{v.mean():.{digits}f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if len(v) > 1:
        s += f"{{\\scriptsize$\\pm${v.std(ddof=1):.{digits}f}}}"
    return s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# ----------------------------------------------------------------------------- tables
def table_datasets(runs, out):
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Dataset & Classes & Train (LT) & $N_{\\max}$ & $N_{\\min}$ & Probe / class & Test / class \\\\",
             "\\midrule"]
    for ds in DS_ORDER:
        rs = next((get(runs, ds, t) for t in METHODS + ["ecl"] if get(runs, ds, t)), [])
        if not rs:
            continue
        r = rs[0]
        cnt = r["counts"]
        test_pc = int(np.array(r["metrics"]["confusion"]).sum(1)[0])
        lines.append(f"{DS_NAME[ds]} & {len(cnt)} & {sum(cnt):,} & {max(cnt):,} & {min(cnt):,} & {r['probe_k']} & {test_pc:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_datasets.tex", "\n".join(lines) + "\n")


def cols_for(runs, ds):
    """G-Mean is 0 as soon as one class has zero recall, which always happens with 100 classes: 10-class only."""
    many_classes = any(len(r["counts"]) > 10 for t in METHODS for r in get(runs, ds, t))
    return [c for c in MAIN_COLS if not (many_classes and c[0] == "g_mean")]


def table_main(runs, ds, out, tags=None, posthoc=False):
    MAIN = cols_for(runs, ds)
    tags = [t for t in (tags or METHODS) if get(runs, ds, t)]
    if posthoc:   # the first round of runs did not store post-hoc metrics
        tags = [t for t in tags if all("metrics_posthoc" in r for r in get(runs, ds, t))]
    if not tags:
        return None
    best = {}
    for key, _, hib in MAIN:
        means = {t: values(get(runs, ds, t), key, posthoc).mean() for t in tags}
        best[key] = (max if hib else min)(means, key=means.get)
    head = " & ".join(h for _, h, _ in MAIN)
    lines = ["\\begin{tabular}{l c " + "c" * len(MAIN) + "}", "\\toprule",
             f"Method & Seeds & {head} \\\\", "\\midrule"]
    for t in tags:
        rs = get(runs, ds, t)
        cells = [fmt(values(rs, k, posthoc), bold=(best[k] == t)) for k, _, _ in MAIN]
        if t == OURS:
            lines.append("\\midrule")
        lines.append(f"{NAME[t]} & {len(rs)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    suffix = "_posthoc" if posthoc else ""
    write(f"{out}/tables/tab_main_{ds}{suffix}.tex", "\n".join(lines) + "\n")
    return tags


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p-values in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired(runs, ds, key, base, ours=OURS, posthoc=False):
    from scipy import stats
    a = {r["_seed"]: metrics_of(r, posthoc)[key] * 100 for r in get(runs, ds, ours)}
    b = {r["_seed"]: metrics_of(r, posthoc)[key] * 100 for r in get(runs, ds, base)}
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([a[s] - b[s] for s in seeds])
    p = float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue) if d.std() > 0 else 0.0
    return {"delta": d.mean(), "wins": int((d > 0).sum()), "n": len(seeds), "p": p}


def table_paired(runs, ds, out, keys=("acc", "few", "macro_f1", "g_mean")):
    keys = [k for k in keys if k in [c[0] for c in cols_for(runs, ds)] or k == "few"]
    bases = [b for b in BASELINES if get(runs, ds, b)]
    if not get(runs, ds, OURS) or not bases:
        return
    res = {k: {b: paired(runs, ds, k, b) for b in bases} for k in keys}
    for k in keys:   # Holm over the baselines, per metric
        ok = [b for b in bases if res[k][b] is not None]
        if ok:
            for b, pa in zip(ok, holm([res[k][b]["p"] for b in ok])):
                res[k][b]["p_holm"] = pa
    hdr = {"acc": "Acc.", "few": "Few", "macro_f1": "F1", "g_mean": "G-Mean"}
    lines = ["\\begin{tabular}{l" + "rcc" * len(keys) + "}", "\\toprule",
             "DAM-Mix vs. & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{hdr[k]}}}" for k in keys) + " \\\\",
             " & " + " & ".join("$\\Delta$ & wins & $p_{\\mathrm{Holm}}$" for _ in keys) + " \\\\", "\\midrule"]
    for b in bases:
        cells = []
        for k in keys:
            r = res[k][b]
            if r is None:
                cells += ["--", "--", "--"]
            else:
                d = f"{r['delta']:+.2f}"
                cells += [f"\\textbf{{{d}}}" if r.get("p_holm", 1) < 0.05 else d, f"{r['wins']}/{r['n']}",
                          ("$<$0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}")]
        lines.append(f"{NAME[b]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_paired_{ds}.tex", "\n".join(lines) + "\n")


def table_summary(runs, out):
    dss = [d for d in DS_ORDER if any(get(runs, d, t) for t in METHODS)]
    lines = ["\\begin{tabular}{l" + "cc" * len(dss) + "}", "\\toprule",
             "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_NAME[d]}}}" for d in dss) + " \\\\",
             " & " + " & ".join("Acc. & Few" for _ in dss) + " \\\\", "\\midrule"]
    best = {}
    for d in dss:
        for k in ("acc", "few"):
            means = {t: values(get(runs, d, t), k).mean() for t in METHODS if get(runs, d, t)}
            best[(d, k)] = max(means, key=means.get) if means else None
    for t in METHODS:
        cells = []
        for d in dss:
            for k in ("acc", "few"):
                rs = get(runs, d, t)
                cells.append(fmt(values(rs, k), bold=best[(d, k)] == t, digits=1) if rs else "--")
        if t == OURS:
            lines.append("\\midrule")
        lines.append(f"{NAME[t]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_summary.tex", "\n".join(lines) + "\n")


def rare_head_idx(r):
    cnt = np.array(r["counts"])
    order = np.argsort(-cnt)
    if len(cnt) <= 10:
        return order[-3:], order[:3]
    return np.where(cnt < 20)[0], np.where(cnt > 100)[0]


def memorisation_stats(rs):
    """Mean over seeds: rare D on train / probe / test, test-to-train ratios, and the error of each estimate."""
    out = defaultdict(list)
    for r in rs:
        g = r["geometry"]
        tr, pr, te = (np.array(g[k], dtype=float) for k in ("train_dispersion", "probe_dispersion", "dispersion"))
        if np.isnan(te).any():   # older runs: test features were subsampled without all classes -> use saved embeddings
            if not os.path.exists(r.get("_emb", "")):
                continue
            import torch
            from .metrics import dispersion_per_class
            d = np.load(r["_emb"])
            h = torch.nn.functional.normalize(torch.tensor(d["h"].astype(np.float32)), dim=1)
            te = dispersion_per_class(h, torch.tensor(d["y"]).long(), len(tr))
        rare, head = rare_head_idx(r)
        out["rare_train"].append(np.nanmean(tr[rare]))
        out["rare_probe"].append(np.nanmean(pr[rare]))
        out["rare_test"].append(np.nanmean(te[rare]))
        with np.errstate(invalid="ignore", divide="ignore"):   # mean of the per-class ratios D_test / D_train
            out["ratio_rare"].append(np.nanmean(te[rare] / tr[rare]))
            out["ratio_head"].append(np.nanmean(te[head] / tr[head]))
        out["err_train"].append(np.nanmean(np.abs(tr - te)))
        out["err_probe"].append(np.nanmean(np.abs(pr - te)))
    if not out:
        return None
    o = {k: float(np.mean(v)) for k, v in out.items()}
    o["better"] = o["err_train"] / max(o["err_probe"], 1e-9)
    return o


def table_memorisation(runs, out, tags=("ce", "la", "ecl", "ecl_cmo", "dam_mix_gaponly_p1")):
    lines = ["\\begin{tabular}{ll rrr rr r}", "\\toprule",
             "Dataset & Model & $\\bar D^{\\mathrm{train}}_{\\mathrm{rare}}$ & $\\bar D^{\\mathrm{probe}}_{\\mathrm{rare}}$ & "
             "$\\bar D^{\\mathrm{test}}_{\\mathrm{rare}}$ & test/train (rare) & test/train (head) & probe better by \\\\",
             "\\midrule"]
    allstats = []
    for ds in DS_ORDER:
        rows = [(t, memorisation_stats(get(runs, ds, t))) for t in tags if get(runs, ds, t)]
        rows = [(t, st) for t, st in rows if st is not None]
        for i, (t, s) in enumerate(rows):
            allstats.append(s)
            lines.append(f"{DS_NAME[ds] if i == 0 else ''} & {NAME[t]} & {s['rare_train']:.3f} & {s['rare_probe']:.3f} & "
                         f"{s['rare_test']:.3f} & {s['ratio_rare']:.1f}$\\times$ & {s['ratio_head']:.1f}$\\times$ & "
                         f"{s['better']:.1f}$\\times$ \\\\")
        if rows and ds != DS_ORDER[-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_memorisation.tex", "\n".join(lines) + "\n")
    return allstats


def table_controls(runs, out, ds="cifar10"):
    """Controls on CIFAR-10-LT. The gap power is chosen by PROBE balanced accuracy (never the test set)."""
    rows = [("dam_mix_gaponly_p1", "1"), ("dam_mix_gaponly", "2"), ("dam_mix_gaponly_p3", "3"),
            ("dam_mix_gapshuffle", "2"), ("dam_mix_nogap", "--"), ("dam_mix_nogap_plus", "--"),
            ("ecl_cmo", "--"), ("ecl_cmo_plus", "--"), ("ecl", "--"), ("ecl_plus", "--")]
    rows = [(t, pw) for t, pw in rows if get(runs, ds, t)]
    if len(rows) < 2:
        return
    lines = ["\\begin{tabular}{l c c c c c}", "\\toprule",
             "Variant & Power & Seeds & Probe bal. acc. & Test acc. & Test few \\\\", "\\midrule"]
    for t, pw in rows:
        rs = get(runs, ds, t)
        probe = np.array([r["posthoc"]["probe_balanced_acc"] for r in rs if "posthoc" in r]) * 100
        lines.append(f"{NAME[t]} & {pw} & {len(rs)} & {fmt(probe)} & {fmt(values(rs, 'acc'))} & {fmt(values(rs, 'few'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_controls.tex", "\n".join(lines) + "\n")


def cutmix_background_share(n=1_000_000, seed=0):
    """E[lambda_bg] of our CutMix implementation (box clipped at the border), by Monte Carlo."""
    import torch
    from .data import cutmix
    torch.manual_seed(seed)
    tot, cnt = 0.0, 0
    while cnt < n:
        m = min(50_000, n - cnt)
        _, lam = cutmix(torch.zeros(m, 1, 32, 32), torch.ones(m, 1, 32, 32))
        tot, cnt = tot + float(lam.sum()), cnt + m
    return tot / cnt


def table_label_mass(runs, out, ds="cifar10"):
    """Proposition 4 evaluated on real schedules: expected share of the mixed-label mass per class group,
    E[y~_c] = mu * pi_c + (1 - mu) * q_c, with mu = E[lambda_bg] and q the paste rates averaged over epochs 11-100."""
    tags = [t for t in ["ecl_cmo", "dam_mix_nogap", OURS] if get(runs, ds, t)]
    if not tags:
        return None
    r0 = get(runs, ds, tags[0])[0]
    cnt = np.array(r0["counts"], dtype=float)
    pi = cnt / cnt.sum()
    order = np.argsort(-cnt)
    groups = [order[:3], order[3:-3], order[-3:]] if len(cnt) <= 10 else None
    if groups is None:
        return None
    mu = cutmix_background_share()
    lines = ["\\begin{tabular}{l ccc ccc}", "\\toprule",
             " & \\multicolumn{3}{c}{paste rate $\\sum_{c\\in g} q_c$ (\\%)} & \\multicolumn{3}{c}{label mass $\\sum_{c\\in g}\\mathbb{E}[\\tilde y_c]$ (\\%)} \\\\",
             "Schedule & Many & Med. & Few & Many & Med. & Few \\\\", "\\midrule",
             "no pasting (prior $\\pi$) & -- & -- & -- & " + " & ".join(f"{pi[g].sum() * 100:.1f}" for g in groups) + " \\\\"]
    for t in tags:
        q = mean_schedule(get(runs, ds, t))
        mass = mu * pi + (1 - mu) * q
        lines.append(f"{NAME[t]} & " + " & ".join(f"{q[g].sum() * 100:.1f}" for g in groups) + " & "
                     + " & ".join(f"{mass[g].sum() * 100:.1f}" for g in groups) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_label_mass.tex", "\n".join(lines) + "\n")
    return mu


def table_ir(runs, out, ds="cifar10"):
    tags = [t for t in ["ce", "la", "ecl", "ecl_cmo", "dam_mix_nogap", "dam_mix_gaponly"] if get(runs, ds, t, 10)]
    if not tags:
        return
    lines = ["\\begin{tabular}{l" + "cc" * 3 + "}", "\\toprule",
             "Method & \\multicolumn{2}{c}{IR = 10} & \\multicolumn{2}{c}{IR = 50} & \\multicolumn{2}{c}{IR = 100} \\\\",
             " & Acc. & Few & Acc. & Few & Acc. & Few \\\\", "\\midrule"]
    for t in tags:
        cells = []
        for ir in (10, 50, 100):
            rs = get(runs, ds, t, ir)
            cells += [fmt(values(rs, "acc"), digits=1) if rs else "--", fmt(values(rs, "few"), digits=1) if rs else "--"]
        lines.append(f"{NAME[t]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_ir.tex", "\n".join(lines) + "\n")


def table_margin_history(runs, out, ds="cifar10"):
    tags = ["ecl", "dam_v1", "dam_v2", "dam_ho", "abl_fixed", "abl_rarity", "abl_dispersion", "abl_nomaca", "abl_noalc"]
    names = dict(NAME, abl_fixed="DAM-ECL-HO, fixed margin", abl_rarity="DAM-ECL-HO, rarity only",
                 abl_dispersion="DAM-ECL-HO, dispersion only", abl_nomaca="DAM-ECL-HO, no margin-aware alignment",
                 abl_noalc="DAM-ECL-HO, no logit adjustment")
    tags = [t for t in tags if get(runs, ds, t)]
    if not tags:
        return
    lines = ["\\begin{tabular}{l c c c}", "\\toprule", "Variant & Seeds & Acc. & Few \\\\", "\\midrule"]
    for t in tags:
        rs = get(runs, ds, t)
        lines.append(f"{names[t]} & {len(rs)} & {fmt(values(rs, 'acc'))} & {fmt(values(rs, 'few'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_margin_history.tex", "\n".join(lines) + "\n")


def table_geometry(runs, out):
    lines = ["\\begin{tabular}{ll ccc}", "\\toprule",
             "Dataset & Method & kNN purity & rare kNN purity & min. centre angle ($^\\circ$) \\\\", "\\midrule"]
    any_row = False
    for ds in [d for d in DS_ORDER if d != "svhn"]:   # SVHN: older runs subsampled the test features by class order
        tags = [t for t in ["ecl", "ecl_cmo", "dam_mix_nogap", OURS] if get(runs, ds, t)]
        for i, t in enumerate(tags):
            rs = get(runs, ds, t)
            pur = np.array([r["geometry"]["mean_knn_purity"] for r in rs]) * 100
            rare = np.array([np.mean(np.array(r["geometry"]["knn_purity"])[rare_head_idx(r)[0]]) for r in rs]) * 100
            ang = np.array([r["geometry"]["min_centroid_angle"] for r in rs])
            lines.append(f"{DS_NAME[ds] if i == 0 else ''} & {NAME[t]} & {fmt(pur)} & {fmt(rare)} & {fmt(ang, digits=1)} \\\\")
            any_row = True
        if tags and ds != DS_ORDER[-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if any_row:
        write(f"{out}/tables/tab_geometry.tex", "\n".join(lines) + "\n")


def export_csv(runs, out):
    import csv
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/all_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["acc", "many", "medium", "few", "macro_f1", "few_f1", "g_mean", "precision", "recall", "auprc", "ece", "worst"]
        w.writerow(["dataset", "ir", "tag", "seed"] + cols + ["posthoc_acc", "posthoc_few", "posthoc_tau", "minutes"])
        for (ds, ir, tag), d in sorted(runs.items()):
            for s, r in sorted(d.items()):
                m = metrics_of(r)
                mp = metrics_of(r, True) if "metrics_posthoc" in r else None
                w.writerow([ds, ir, tag, s] + [f"{m[c] * 100:.3f}" for c in cols] +
                           ([f"{mp['acc'] * 100:.3f}", f"{mp['few'] * 100:.3f}", r["posthoc"]["tau"]] if mp else ["", "", ""])
                           + [f"{r.get('time_min', 0):.2f}"])


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                         "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


def fig_class_distribution(runs, out, ds="cifar10"):
    rs = get(runs, ds, "ecl") or get(runs, ds, OURS)
    if not rs:
        return
    plt = _plt()
    r = rs[0]
    cnt = np.array(r["counts"])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(cnt))
    ax.bar(x, cnt, color="#0060c0", width=0.7, label="long-tailed training set")
    ax.bar(x, np.full(len(cnt), r["probe_k"]), bottom=cnt, color="#9e9e9e", width=0.7,
           label=f"held-out probe ({r['probe_k']} per class)")
    ax.set_yscale("log")
    ax.set_ylim(10, cnt.max() * 2.5)
    ax.set_xticks(x)
    ax.set_xlabel("class index $c$ (sorted by $N_c$)")
    ax.set_ylabel("images per class (log)")
    ax.legend(loc="upper right")
    fig.savefig(f"{out}/figures/fig_class_distribution.pdf")
    plt.close(fig)


def fig_memorisation(runs, out, tag="ecl"):
    dss = [d for d in DS_ORDER if d != "cifar100" and get(runs, d, tag)]
    if not dss:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, len(dss), figsize=(1.9 * len(dss) + 0.6, 2.3), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        rs = get(runs, ds, tag)
        tr = np.mean([r["geometry"]["train_dispersion"] for r in rs], 0)
        pr = np.mean([r["geometry"]["probe_dispersion"] for r in rs], 0)
        te = np.mean([r["geometry"]["dispersion"] for r in rs], 0)
        order = np.argsort(-np.array(rs[0]["counts"]))
        x = np.arange(len(order))
        ax.plot(x, tr[order], "o-", ms=3, lw=1.2, color="#9e9e9e", label="training images")
        ax.plot(x, pr[order], "s-", ms=3, lw=1.2, color="#0060c0", label="probe (20/class)")
        ax.plot(x, te[order], "--", lw=1.2, color="black", label="test images")
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("class (head $\\rightarrow$ tail)")
        ax.set_xticks([0, len(order) - 1])
    axes[0][0].set_ylabel("dispersion $D_c$")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{out}/figures/fig_memorisation.pdf")
    plt.close(fig)


def fig_main_bars(runs, out):
    dss = [d for d in DS_ORDER if any(get(runs, d, t) for t in METHODS)]
    if not dss:
        return
    plt = _plt()
    fig, axes = plt.subplots(2, len(dss), figsize=(1.55 * len(dss) + 1.0, 3.6), squeeze=False)
    for j, ds in enumerate(dss):
        tags = [t for t in METHODS if get(runs, ds, t)]
        for i, (key, lab) in enumerate([("acc", "accuracy (%)"), ("few", "few-shot acc. (%)")]):
            ax = axes[i][j]
            v = [values(get(runs, ds, t), key) for t in tags]
            mean = np.array([a.mean() for a in v])
            err = np.array([a.std(ddof=1) if len(a) > 1 else 0 for a in v])
            ax.bar(range(len(tags)), mean, yerr=err, color=[COLOR[t] for t in tags], width=0.75,
                   error_kw={"lw": 0.8, "capsize": 1.5})
            lo = max(0, mean.min() - 8)
            ax.set_ylim(lo, min(100, mean.max() + 4))
            ax.set_xticks([])
            if i == 0:
                ax.set_title(DS_NAME[ds])
            if j == 0:
                ax.set_ylabel(lab)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[t]) for t in METHODS]
    fig.legend(handles, [PLAIN[t] for t in METHODS], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(f"{out}/figures/fig_main_bars.pdf")
    plt.close(fig)


def fig_groups(runs, out, ds="cifar10"):
    tags = [t for t in METHODS if get(runs, ds, t)]
    if not tags:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        v = [values(get(runs, ds, t), k) for k in ("many", "medium", "few")]
        ax.bar(np.arange(3) + (i - len(tags) / 2 + 0.5) * w, [a.mean() for a in v], w * 0.92,
               yerr=[a.std(ddof=1) if len(a) > 1 else 0 for a in v], color=COLOR[t], label=PLAIN[t],
               error_kw={"lw": 0.7, "capsize": 1})
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Many", "Medium", "Few"])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(30, 100)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6.5)
    fig.savefig(f"{out}/figures/fig_groups_{ds}.pdf")
    plt.close(fig)


def mean_schedule(rs, start=10):
    q = [np.array(r["paste"]["q_per_epoch"])[start:].mean(0) for r in rs if "paste" in r]
    return np.mean(q, 0) if q else None


def fig_schedule(runs, out, ds="cifar10"):
    rs_ours = get(runs, ds, OURS)
    if not rs_ours:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    tags = [t for t in ["ecl_cmo", "dam_mix_nogap", OURS] if get(runs, ds, t)]
    w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        q = mean_schedule(get(runs, ds, t))
        ax.bar(np.arange(len(q)) + (i - len(tags) / 2 + 0.5) * w, q * 100, w * 0.92, color=COLOR[t], label=PLAIN[t])
    ax.set_xlabel("class $c$ (head $\\rightarrow$ tail)")
    ax.set_ylabel("share of pasted patches $q_c$ (%)")
    ax.set_xticks(range(len(q)))
    ax.legend(loc="upper left")
    ax = axes[1]
    G = np.mean([[h["G"] for h in r["paste"]["gap_history"]] for r in rs_ours], 0)   # [epochs, C]
    rare = set(rare_head_idx(rs_ours[0])[0].tolist())
    ep = np.arange(1, len(G) + 1)
    for c in range(G.shape[1]):   # grey first, the rare classes on top
        if c not in rare:
            ax.plot(ep, G[:, c], lw=0.9, color="#b0b0b0")
    for c in sorted(rare):
        ax.plot(ep, G[:, c], lw=1.8, color=COLOR[OURS])
    ax.plot([], [], lw=1.8, color=COLOR[OURS], label="3 rarest classes")
    ax.plot([], [], lw=0.9, color="#b0b0b0", label="other classes")
    ax.axvline(len(G) * 0.1, color="#666666", lw=0.7, ls=":")
    ax.text(len(G) * 0.1 + 1, ax.get_ylim()[1] * 0.97, "warm-up ends", fontsize=7, va="top", color="#555555")
    ax.set_xlabel("epoch")
    ax.set_ylabel("memorisation gap $G_c$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_schedule_{ds}.pdf")
    plt.close(fig)


def fig_per_class_recall(runs, out, ds="cifar10"):
    tags = [t for t in ["ce", "la", "ecl", "ecl_cmo", "dam_mix_nogap", OURS] if get(runs, ds, t)]
    if not tags:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for t in tags:
        rec = np.mean([r["metrics"]["per_class_recall"] for r in get(runs, ds, t)], 0) * 100
        ax.plot(range(len(rec)), rec, "o-", ms=3, lw=1.6 if t == OURS else 1.0, color=COLOR[t], label=PLAIN[t])
    ax.set_xlabel("class $c$ (head $\\rightarrow$ tail)")
    ax.set_ylabel("test recall (%)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5)
    fig.savefig(f"{out}/figures/fig_per_class_recall_{ds}.pdf")
    plt.close(fig)


def fig_curves(runs, out, ds="cifar10"):
    tags = [t for t in ["ecl", "ecl_cmo", "dam_mix_nogap", OURS] if get(runs, ds, t)]
    if not tags:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for t in tags:
        r = get(runs, ds, t)[0]
        pts = [c for c in r["curves"] if "test_acc" in c]
        ep = [c["epoch"] for c in pts]
        axes[0].plot(ep, [c["test_acc"] * 100 for c in pts], "o-", ms=2.5, lw=1.2, color=COLOR[t], label=PLAIN[t])
        axes[1].plot(ep, [c["test_few_acc"] * 100 for c in pts], "o-", ms=2.5, lw=1.2, color=COLOR[t])
    axes[0].set_ylabel("test accuracy (%)")
    axes[1].set_ylabel("few-shot accuracy (%)")
    for a in axes:
        a.set_xlabel("epoch")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_curves_{ds}.pdf")
    plt.close(fig)


def fig_confusion(runs, out, ds="cifar10"):
    tags = [t for t in ["ecl", "ecl_cmo", OURS] if get(runs, ds, t)]
    if len(tags) < 2:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, len(tags), figsize=(2.2 * len(tags), 2.2))
    for ax, t in zip(axes, tags):
        cm = np.mean([np.array(r["metrics"]["confusion"], float) for r in get(runs, ds, t)], 0)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(PLAIN[t])
        ax.set_xlabel("predicted")
        ax.grid(False)
        ax.set_xticks(range(0, len(cm), 3))
        ax.set_yticks(range(0, len(cm), 3))
    axes[0].set_ylabel("true class")
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_confusion_{ds}.pdf")
    plt.close(fig)


def _emb_for(runs, ds, tag):
    cands = [(ds, 100, tag)] + ([(ds, 100, "dam_mix_gaponly_final")] if tag == "dam_mix_gaponly" else [])
    for key in cands:
        for s, r in sorted(runs.get(key, {}).items()):
            if os.path.exists(r["_emb"]):
                return r["_emb"]
    return None


def fig_tsne(runs, out, tags=("ecl", "ecl_cmo", "dam_mix_gaponly"), max_points=2000):
    from sklearn.manifold import TSNE
    dss = [d for d in DS_ORDER if d != "cifar100" and all(_emb_for(runs, d, t) for t in tags)]
    missing = [d for d in DS_ORDER if d != "cifar100" and get(runs, d, "ecl") and d not in dss]
    if missing:
        print(f"t-SNE: no saved embeddings for {', '.join(missing)} (needs the seed-0 runs of the paper plan)")
    if not dss:
        return
    plt = _plt()
    fig, axes = plt.subplots(len(dss), len(tags), figsize=(2.1 * len(tags), 2.0 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        for j, t in enumerate(tags):
            d = np.load(_emb_for(runs, ds, t))
            h, y = d["h"].astype(np.float32)[:max_points], d["y"][:max_points]
            e = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(h)
            ax = axes[i][j]
            ax.scatter(e[:, 0], e[:, 1], c=y, cmap="tab10", s=2, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(PLAIN[t])
            if j == 0:
                ax.set_ylabel(DS_NAME[ds])
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_tsne.pdf", dpi=300)
    plt.close(fig)


def fig_schematic(runs, out):
    """Illustration with synthetic directions (clearly labelled as such): a head class looks the same on
    training and unseen images; a memorised tail class is tight on its training images only."""
    plt = _plt()
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, (title, k_tr, k_te, ntr) in zip(axes, [("head class", 6, 6, 60), ("rare class (memorised)", 90, 6, 12)]):
        ax.plot(np.cos(t), np.sin(t), color="#d0d0d0", lw=0.8)
        a_tr = rng.vonmises(np.pi / 2, k_tr, ntr)
        a_te = rng.vonmises(np.pi / 2, k_te, 40)
        ax.scatter(0.82 * np.cos(a_te), 0.82 * np.sin(a_te), s=9, facecolors="none", edgecolors="#e8590c", lw=0.7,
                   label="unseen images")
        ax.scatter(np.cos(a_tr), np.sin(a_tr), s=9, color="#1f6fd1", label="training images")
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[1].text(0, -1.35, "$D^{\\mathrm{train}}\\ll D^{\\mathrm{test}}\\approx D^{\\mathrm{probe}}$", ha="center", fontsize=7)
    axes[0].text(0, -1.35, "$D^{\\mathrm{train}}\\approx D^{\\mathrm{test}}$", ha="center", fontsize=7)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.42), ncol=2, fontsize=7, handletextpad=0.2)
    fig.text(0.5, -0.06, "illustration with synthetic directions", ha="center", fontsize=6, color="#777777")
    fig.savefig(f"{out}/figures/fig_schematic.pdf")
    plt.close(fig)


def fig_probe_scatter(runs, out, tags=("ecl", OURS)):
    """Estimated dispersion (training images vs probe) against the truth (test images), every class, mean over seeds."""
    dss = [d for d in DS_ORDER if d != "cifar100" and any(get(runs, d, t) for t in tags)]
    if not dss:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, len(dss), figsize=(1.75 * len(dss) + 0.4, 1.95), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        lo, hi = 1, 0
        for t, mk in zip(tags, ("o", "s")):
            rs = get(runs, ds, t)
            if not rs:
                continue
            tr, pr, te = (np.mean([r["geometry"][k] for r in rs], 0)
                          for k in ("train_dispersion", "probe_dispersion", "dispersion"))
            ax.scatter(te, tr, s=12, marker=mk, color="#9e9e9e", label=f"training images ({PLAIN[t]})")
            ax.scatter(te, pr, s=12, marker=mk, color=COLOR[OURS], label=f"probe ({PLAIN[t]})")
            lo, hi = min(lo, tr.min(), te.min(), pr.min()), max(hi, tr.max(), te.max(), pr.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.7, ls="--")
        ax.set_title(DS_NAME[ds])
        ax.set_xlabel("$D_c$ on test images")
    axes[0][0].set_ylabel("estimated $D_c$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.14), fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_probe_scatter.pdf")
    plt.close(fig)


def fig_forest(runs, out, keys=(("acc", "overall accuracy"), ("few", "few-shot accuracy"))):
    """Paired differences DAM-Mix minus each baseline, mean and 95% t-interval over seeds, every dataset."""
    from scipy import stats
    rows = []
    for ds in DS_ORDER:
        for b in BASELINES:
            if get(runs, ds, b) and get(runs, ds, OURS):
                rows.append((ds, b))
    if not rows:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, len(keys), figsize=(7.0, 0.22 * len(rows) + 0.9), sharey=True)
    for ax, (k, lab) in zip(np.atleast_1d(axes), keys):
        for i, (ds, b) in enumerate(rows):
            a = {r["_seed"]: metrics_of(r)[k] * 100 for r in get(runs, ds, OURS)}
            c = {r["_seed"]: metrics_of(r)[k] * 100 for r in get(runs, ds, b)}
            d = np.array([a[s] - c[s] for s in sorted(set(a) & set(c))])
            if len(d) == 0:
                continue
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
            y = len(rows) - 1 - i
            ax.errorbar(d.mean(), y, xerr=half, fmt="o", ms=3.5, color=COLOR[b], ecolor=COLOR[b], lw=1, capsize=2)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_xlabel(f"$\\Delta$ {lab} (points)")
        ax.grid(axis="y", visible=False)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(rows)))
    ax0.set_yticklabels([f"{DS_NAME[ds]} vs {PLAIN[b]}" for ds, b in rows][::-1], fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_forest.pdf")
    plt.close(fig)


def fig_power(runs, out, ds="cifar10"):
    """Gap power chosen on the probe: probe balanced accuracy (selection) next to test numbers (report only)."""
    tags = [(1, "dam_mix_gaponly_p1"), (2, "dam_mix_gaponly"), (3, "dam_mix_gaponly_p3")]
    tags = [(k, t) for k, t in tags if get(runs, ds, t)]
    if len(tags) < 2:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for key, lab, col, mk in [("probe", "probe balanced acc. (selection)", COLOR[OURS], "s"),
                              ("acc", "test accuracy", "#555555", "o"), ("few", "test few-shot acc.", "#e69f00", "^")]:
        m, e = [], []
        for _, t in tags:
            rs = get(runs, ds, t)
            v = (np.array([r["posthoc"]["probe_balanced_acc"] for r in rs if "posthoc" in r]) * 100 if key == "probe"
                 else values(rs, key))
            m.append(v.mean())
            e.append(v.std(ddof=1) if len(v) > 1 else 0)
        ax.errorbar([k for k, _ in tags], m, yerr=e, fmt=mk + "-", ms=4, lw=1.2, color=col, capsize=2, label=lab)
    ax.set_xticks([k for k, _ in tags])
    ax.set_xlabel("gap power $\\kappa$")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=6.5, loc="best")
    fig.savefig(f"{out}/figures/fig_power_{ds}.pdf")
    plt.close(fig)


MISLEAD = [("dup_", "Duplicated classes", "acc"), ("step2_", "2 minority classes", "few"), ("step5_", "5 minority classes", "few")]
MISLEAD_TAGS = ["ecl", "ecl_cmo", "dam_mix_nogap", "dam_mix_gaponly_p1"]


def table_mislead(runs, out, ds="cifar10"):
    """Settings in which class counts misdescribe the data (pre-registered round 3, extended to 5 seeds)."""
    if not get(runs, ds, "dup_" + OURS):
        return None
    lines = ["\\begin{tabular}{ll c ccc cc}", "\\toprule",
             "Setting & Method & Seeds & Acc. & Many & Few & $\\Delta$ vs.\\ CMO$_{r=1/2}$ & wins \\\\", "\\midrule"]
    res = {}
    for pref, label, prim in MISLEAD:
        tags = [t for t in MISLEAD_TAGS + (["dam_mix_gapshuffle_p1"] if pref == "dup_" else []) if get(runs, ds, pref + t)]
        best = {k: max(tags, key=lambda t: values(get(runs, ds, pref + t), k).mean()) for k in ("acc", "many", "few")}
        for i, t in enumerate(tags):
            rs = get(runs, ds, pref + t)
            cells = [fmt(values(rs, k), bold=(best[k] == t and t != "dam_mix_gapshuffle_p1")) for k in ("acc", "many", "few")]
            d, w = "", ""
            if t in (OURS, "dam_mix_gapshuffle_p1"):
                pr = paired(runs, ds, prim, pref + "dam_mix_nogap", ours=pref + t)
                d, w = f"{pr['delta']:+.2f} ({'Acc.' if prim == 'acc' else 'Few'})", f"{pr['wins']}/{pr['n']}"
                res[(pref, t)] = pr
            if t == OURS:
                lines.append("\\cmidrule(l){2-8}")
            lines.append(f"{label if i == 0 else ''} & {NAME[t]} & {len(rs)} & " + " & ".join(cells) + f" & {d} & {w} \\\\")
        if pref != MISLEAD[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_mislead.tex", "\n".join(lines) + "\n")
    # pre-registered primary tests, Holm over the three settings
    prim = [res.get((pref, OURS)) for pref, _, _ in MISLEAD]
    adj = holm([p["p"] for p in prim])
    with open(f"{out}/key_numbers.txt", "a") as f:
        f.write("== counts mislead (primary tests vs CMO r=1/2, Holm over 3)\n")
        for (pref, label, k), p, a in zip(MISLEAD, prim, adj):
            f.write(f"  {label:20s} {k}: {p['delta']:+.2f} wins {p['wins']}/{p['n']} p={p['p']:.4f} p_holm={a:.4f}\n")
        if ("dup_", "dam_mix_gapshuffle_p1") in res:
            p = paired(runs, ds, "acc", "dup_dam_mix_gapshuffle_p1", ours="dup_" + OURS)
            f.write(f"  ours - pooled-gap control (dup, acc): {p['delta']:+.2f} wins {p['wins']}/{p['n']} p={p['p']:.4f}\n")
    return prim, adj


def table_mislead_mass(runs, out, ds="cifar10"):
    """Mechanism: share of pastes given to the minority classes and the resulting label mass of the majority."""
    mu = cutmix_background_share(200_000)
    lines = ["\\begin{tabular}{ll ccc}", "\\toprule",
             "Setting & Schedule & pastes to minority (\\%) & majority label mass (\\%) & Many acc. (\\%) \\\\", "\\midrule"]
    ok = False
    for pref, label in (("step2_", "2 minority classes"), ("step5_", "5 minority classes")):
        for i, t in enumerate(["ecl_cmo", "dam_mix_nogap", OURS]):
            rs = get(runs, ds, pref + t)
            if not rs:
                continue
            cnt = np.array(rs[0]["counts"], float)
            pi = cnt / cnt.sum()
            q = mean_schedule(rs)
            maj = cnt == cnt.max()
            mass = mu * pi + (1 - mu) * q
            lines.append(f"{label if i == 0 else ''} & {NAME[t]} & {q[~maj].sum() * 100:.1f} & {mass[maj].sum() * 100:.1f} & "
                         f"{fmt(values(rs, 'many'))} \\\\")
            ok = True
        if pref == "step2_":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if ok:
        write(f"{out}/tables/tab_mislead_mass.tex", "\n".join(lines) + "\n")


def table_dup_schedule(runs, out, ds="cifar10"):
    lines = ["\\begin{tabular}{l cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{share of pastes (\\%)} & \\multicolumn{2}{c}{test recall (\\%)} \\\\",
             "Method & class 0 & class 2 & class 0 & class 2 \\\\", "\\midrule"]
    ok = False
    for t in MISLEAD_TAGS + ["dam_mix_gapshuffle_p1"]:
        rs = get(runs, ds, "dup_" + t)
        if not rs:
            continue
        rec = np.mean([r["metrics"]["per_class_recall"] for r in rs], 0) * 100
        q = mean_schedule(rs) if "paste" in rs[0] else None
        qs = f"{q[0] * 100:.1f} & {q[2] * 100:.1f}" if q is not None else "-- & --"
        lines.append(f"{NAME[t]} & {qs} & {rec[0]:.1f} & {rec[2]:.1f} \\\\")
        ok = True
    lines += ["\\bottomrule", "\\end{tabular}"]
    if ok:
        write(f"{out}/tables/tab_dup_schedule.tex", "\n".join(lines) + "\n")


def find_stats(folders):
    for fo in folders:
        for f in glob.glob(os.path.join(fo, "**", "dataset_stats.json"), recursive=True):
            with open(f) as fh:
                return {d["dataset"]: d for d in json.load(fh)}
    return None


def table_scope(runs, out, stats):
    """When does pasting help? Background emptiness and label mismatch vs the effect of pasting (minus ECL)."""
    if not stats:
        return None
    tags = ["ecl_cmo", "dam_mix_nogap", OURS]
    lines = ["\\begin{tabular}{l cc ccc}", "\\toprule",
             " & & & \\multicolumn{3}{c}{accuracy change from pasting (vs.\\ ECL, points)} \\\\",
             "Dataset & empty pixels (\\%) & label mismatch & " + " & ".join(NAME[t] for t in tags) + " \\\\", "\\midrule"]
    rows = []
    for ds in ["svhn", "cifar10", "cifar100", "fmnist", "mnist"]:
        if ds not in stats or not get(runs, ds, "ecl"):
            continue
        e = values(get(runs, ds, "ecl"), "acc").mean()
        eff = [values(get(runs, ds, t), "acc").mean() - e if get(runs, ds, t) else np.nan for t in tags]
        rows.append((ds, eff))
        lines.append(f"{DS_NAME[ds]} & {stats[ds]['empty_share'] * 100:.1f} & {stats[ds]['label_mismatch_mean']:.3f} & "
                     + " & ".join(f"{v:+.1f}" for v in eff) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{out}/tables/tab_scope.tex", "\n".join(lines) + "\n")
    return rows


def table_ink(runs, out):
    tags = ["ecl_cmo", "dam_mix_nogap", OURS]
    lines = ["\\begin{tabular}{ll ccc}", "\\toprule",
             "Dataset & Method & area label & content label & $\\Delta$ (wins) \\\\", "\\midrule"]
    ok = False
    for ds in ("mnist", "fmnist"):
        if not get(runs, ds, "ink_" + OURS):
            continue
        e = values(get(runs, ds, "ecl"), "acc")
        lines.append(f"{DS_NAME[ds]} & ECL (no pasting) & {fmt(e)} & -- & -- \\\\")
        for t in tags:
            a, b = get(runs, ds, t), get(runs, ds, "ink_" + t)
            pr = paired(runs, ds, "acc", t, ours="ink_" + t)
            lines.append(f" & {NAME[t]} & {fmt(values(a, 'acc'))} & {fmt(values(b, 'acc'))} & "
                         f"{pr['delta']:+.2f} ({pr['wins']}/{pr['n']}) \\\\")
        lines.append("\\midrule" if ds == "mnist" else "")
        ok = True
    lines = [l for l in lines if l != ""]
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}"]
    if ok:
        write(f"{out}/tables/tab_ink.tex", "\n".join(lines) + "\n")


def fig_mislead(runs, out, ds="cifar10"):
    if not get(runs, ds, "dup_" + OURS):
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for ax, (pref, label, k) in zip(axes, MISLEAD):
        tags = [t for t in MISLEAD_TAGS if get(runs, ds, pref + t)]
        v = [values(get(runs, ds, pref + t), k) for t in tags]
        m = np.array([a.mean() for a in v])
        ax.bar(range(len(tags)), m, yerr=[a.std(ddof=1) for a in v], color=[COLOR[t] for t in tags], width=0.75,
               error_kw={"lw": 0.8, "capsize": 1.5})
        ax.set_ylim(m.min() - 6, m.max() + 3)
        ax.set_xticks([])
        ax.set_title(label)
        ax.set_ylabel("overall accuracy (%)" if k == "acc" else "minority accuracy (%)")
    h = [plt.Rectangle((0, 0), 1, 1, color=COLOR[t]) for t in MISLEAD_TAGS]
    fig.legend(h, [PLAIN[t] for t in MISLEAD_TAGS], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/fig_mislead.pdf")
    plt.close(fig)


def fig_scope(runs, out, stats):
    if not stats:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for t, mk in (("ecl_cmo", "o"), ("dam_mix_nogap", "s"), (OURS, "D")):
        xs, ys, names = [], [], []
        for ds in ["svhn", "cifar10", "cifar100", "fmnist", "mnist"]:
            if ds in stats and get(runs, ds, t) and get(runs, ds, "ecl"):
                xs.append(stats[ds]["empty_share"] * 100)
                ys.append(values(get(runs, ds, t), "acc").mean() - values(get(runs, ds, "ecl"), "acc").mean())
                names.append(DS_NAME[ds])
        ax.plot(xs, ys, mk, ms=4, color=COLOR[t], label=PLAIN[t])
    offsets = {"fmnist": (-38, 9), "mnist": (-44, 6), "svhn": (6, 6), "cifar10": (6, -2), "cifar100": (6, -10)}
    for ds in ("fmnist", "mnist"):
        if ds in stats and get(runs, ds, OURS) and get(runs, ds, "ecl"):
            x = stats[ds]["empty_share"] * 100
            y = values(get(runs, ds, OURS), "acc").mean() - values(get(runs, ds, "ecl"), "acc").mean()
            ax.annotate(DS_NAME[ds], (x, y), textcoords="offset points", xytext=offsets[ds], fontsize=6.5,
                        color="#555555")
    ax.annotate("CIFAR-10/100-LT, SVHN-LT", (2, 4), textcoords="offset points", xytext=(8, 4), fontsize=6.5,
                color="#555555")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("near-black pixels in training images (%)")
    ax.set_ylabel("accuracy change vs. ECL (points)")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.savefig(f"{out}/figures/fig_scope.pdf")
    plt.close(fig)



# ----------------------------------------------------------------------------- numbers used in the text
def key_numbers(runs, out, memo_stats):
    """Plain-text list of the numbers quoted in the prose, so the text and the tables cannot disagree."""
    lines = []
    for ds in DS_ORDER:
        if not get(runs, ds, OURS):
            continue
        lines.append(f"== {DS_NAME[ds]}")
        for t in METHODS:
            rs = get(runs, ds, t)
            if rs:
                lines.append(f"  {PLAIN[t]:28s} acc {fmt(values(rs, 'acc'))}  few {fmt(values(rs, 'few'))}  "
                             f"F1 {fmt(values(rs, 'macro_f1'))}  G-mean {fmt(values(rs, 'g_mean'))}  (n={len(rs)})".replace("{\\scriptsize$\\pm$", " ± ").replace("}", ""))
        for b in BASELINES:
            for k in ("acc", "few"):
                p = paired(runs, ds, k, b)
                if p:
                    lines.append(f"  ours - {PLAIN[b]:22s} {k:4s} {p['delta']:+.2f} wins {p['wins']}/{p['n']} p={p['p']:.4f}")
        q = mean_schedule(get(runs, ds, OURS))
        if q is not None:
            lines.append("  DAM-Mix paste shares (%): " + ", ".join(f"{v * 100:.1f}" for v in q))
    if memo_stats:
        rr = [s["ratio_rare"] for s in memo_stats]
        bb = [s["better"] for s in memo_stats]
        lines.append(f"== memorisation: test/train rare ratio {min(rr):.1f}-{max(rr):.1f}x; probe better by {min(bb):.1f}-{max(bb):.1f}x")
    write(f"{out}/key_numbers.txt", "\n".join(lines) + "\n")
    try:
        with open(f"{out}/key_numbers.txt", "a") as f:
            f.write(f"== E[lambda_bg] of our CutMix (Monte Carlo, 1e6 boxes) = {cutmix_background_share():.4f}\n")
    except Exception as e:
        print("could not compute E[lambda_bg]:", e)


def build(folders, out):
    runs = load_runs(folders)
    os.makedirs(f"{out}/tables", exist_ok=True)
    os.makedirs(f"{out}/figures", exist_ok=True)
    print(f"{sum(len(v) for v in runs.values())} runs from {folders}")
    table_datasets(runs, out)
    for ds in DS_ORDER:
        table_main(runs, ds, out)
        table_main(runs, ds, out, posthoc=True)
        table_paired(runs, ds, out)
    table_summary(runs, out)
    memo = table_memorisation(runs, out)
    table_controls(runs, out)
    mu = table_label_mass(runs, out)
    if mu is not None:
        print(f"E[lambda_bg] of our CutMix = {mu:.4f}")
    table_ir(runs, out)
    table_margin_history(runs, out)
    table_geometry(runs, out)
    export_csv(runs, out)
    jobs = [(f, {}) for f in (fig_class_distribution, fig_memorisation, fig_main_bars, fig_tsne, fig_schematic,
                              fig_probe_scatter, fig_forest, fig_power)]
    for ds in DS_ORDER:   # per-dataset figures (10-class datasets; CIFAR-100 only gets the group bars)
        jobs.append((fig_groups, {"ds": ds}))
        if ds != "cifar100":
            jobs += [(f, {"ds": ds}) for f in (fig_schedule, fig_per_class_recall, fig_curves, fig_confusion)]
    for f, kw in jobs:
        try:
            f(runs, out, **kw)
        except Exception as e:   # a missing piece of data should not stop the other figures
            print(f"skipped {f.__name__} {kw}: {e}")
    key_numbers(runs, out, memo)
    stats = find_stats(folders)
    for f, args in ((table_mislead, ()), (table_mislead_mass, ()), (table_dup_schedule, ()), (table_scope, (stats,)),
                    (table_ink, ()), (fig_mislead, ()), (fig_scope, (stats,))):
        try:
            f(runs, out, *args)
        except Exception as e:
            print(f"skipped {f.__name__}: {e}")
    print("tables :", sorted(os.listdir(f"{out}/tables")))
    print("figures:", sorted(os.listdir(f"{out}/figures")))
