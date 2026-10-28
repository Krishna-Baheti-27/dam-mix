"""Runs the whole experiment list: resumable, and shardable across GPUs and Kaggle accounts.

    python run_all.py --list                                        # print the run list
    python run_all.py                                               # everything on one GPU
    python run_all.py --shard 0 --num_shards 4 --gpu 0              # one of 4 parts (2 accounts x 2 GPUs)

A run's result file appears only when the run has fully finished, and finished runs are skipped,
so you can stop at any time and continue later (copy the old results/ folder back first).
"""
import argparse
import os
import subprocess
import sys
import time

SEEDS = [0, 1, 2]
MAIN = ["ce", "wce", "la", "supcon", "ecl", "dam_v1", "dam_v2", "dam_ho"]
ABLATIONS = {  # all on top of dam_ho (ours)
    "abl_fixed": "--mode fixed",
    "abl_rarity": "--mode rarity",
    "abl_dispersion": "--mode dispersion",
    "abl_nomaca": "--lam_align 0",
    "abl_noalc": "--no_alc",
}
FAIR = {  # data-fair check: both runs end up training on the probe images
    "ecl_plus": ("ecl", "--probe_in_train"),
    "dam_ho_absorb": ("dam_ho", "--absorb"),
}


def R(dataset, ir, method, seed, tag=None, extra="", emb=False):
    tag = tag or method
    return dict(name=f"{dataset}_ir{int(ir)}_{tag}_s{seed}",
                args=f"--dataset {dataset} --ir {ir} --method {method} --seed {seed} {extra}".strip(), emb=emb)


NEW_DATASETS = ["fmnist", "svhn", "mnist"]      # generalisation study asked for by the professor
PAPER_METHODS = [  # (tag, method, extra args): the baselines + ours, identical recipe on every dataset
    ("ce", "ce", ""), ("wce", "wce", ""), ("la", "la", ""), ("ecl", "ecl", ""), ("ecl_cmo", "ecl_cmo", ""),
    ("dam_mix_nogap", "dam_mix", "--paste sqrtinv"),
    ("dam_mix_gaponly", "dam_mix", "--paste gaponly --gap_power 2"),
]


def paper_plan():
    """New datasets (3 seeds x 7 methods each) + a third seed on CIFAR-100-LT, in priority order:
    a complete seed-0 table first, so that a partial run still gives a usable table."""
    runs = [dict(name="cifar10_ir100_dam_mix_gaponly_final_s0",   # DAM-Mix weights + t-SNE embeddings
                 args="--dataset cifar10 --ir 100 --method dam_mix --paste gaponly --gap_power 2 --seed 0 "
                      "--save_model final_model/dam_mix_cifar10lt_ir100_s0.pt", emb=True)]
    c100 = [R("cifar100", 100, m, 2, tag=t, extra=e) for t, m, e in PAPER_METHODS if t not in ("ce", "wce", "la")]
    c100 += [R("cifar100", 100, m, s, tag=t, extra=e) for s in (1, 2) for t, m, e in PAPER_METHODS
             if t in ("ce", "la")]
    c100 += [R("cifar100", 100, "wce", s) for s in (0, 1, 2)]
    for seed in (0, 1, 2):
        runs += [R(ds, 100, m, seed, tag=t, extra=e, emb=(seed == 0)) for ds in NEW_DATASETS
                 for t, m, e in PAPER_METHODS]
        if seed == 0:
            runs += c100[:4]
    runs += c100[4:]
    return runs


def controls_plan():
    """Controls on CIFAR-10-LT (IR 100) that close the open questions of the first study:
    shuffled gap, the gap power chosen on the probe, and the probe images given to the baselines too."""
    runs = []
    for seed in range(5):
        runs.append(R("cifar10", 100, "dam_mix", seed, tag="dam_mix_gapshuffle", extra="--paste gapshuffle --gap_power 2"))
        if seed < 3:
            for pw in (1, 3):
                runs.append(R("cifar10", 100, "dam_mix", seed, tag=f"dam_mix_gaponly_p{pw}",
                              extra=f"--paste gaponly --gap_power {pw}"))
            runs.append(R("cifar10", 100, "ecl_cmo", seed, tag="ecl_cmo_plus", extra="--probe_in_train"))
            runs.append(R("cifar10", 100, "dam_mix", seed, tag="dam_mix_nogap_plus", extra="--paste sqrtinv --probe_in_train"))
    return runs


ROUND3_METHODS = [  # main configuration from round 3 on: DAM-Mix with the probe-selected power 1
    ("ecl", "ecl", ""), ("ecl_cmo", "ecl_cmo", ""), ("dam_mix_nogap", "dam_mix", "--paste sqrtinv"),
    ("dam_mix_gaponly_p1", "dam_mix", "--paste gaponly --gap_power 1"),
]


def round3_plan():
    """Pre-registered round 3 (see docs/PREREGISTRATION_round3.md), in priority order.
    A  duplicates: classes 0 and 2 keep their counts (4980, 1789) but have only 250 / 90 distinct images.
    B  step imbalance: 2 or 5 minority classes of 49 images, all other classes 4980.
    C  MNIST/Fashion-MNIST with the CutMix label split by content mass ('ink') instead of area.
    D  DAM-Mix with power 1 on every dataset (CIFAR-10 seeds 3-4 complete its 5 seeds)."""
    runs = []
    for seed in (0, 1, 2):
        runs += [R("cifar10", 100, m, seed, tag="dup_" + t, extra=f"{e} --dup 0:250,2:90".strip()) for t, m, e in ROUND3_METHODS]
        for nm in (2, 5):
            runs += [R("cifar10", 100, m, seed, tag=f"step{nm}_" + t, extra=f"{e} --profile step --n_minor {nm}".strip())
                     for t, m, e in ROUND3_METHODS]
        for ds in ("mnist", "fmnist"):
            runs += [R(ds, 100, m, seed, tag="ink_" + t, extra=f"{e} --mix_label ink".strip())
                     for t, m, e in ROUND3_METHODS if t != "ecl"]
        runs += [R(ds, 100, "dam_mix", seed, tag="dam_mix_gaponly_p1", extra="--paste gaponly --gap_power 1")
                 for ds in ("fmnist", "svhn", "mnist", "cifar100")]
    runs += [R("cifar10", 100, "dam_mix", s, tag="dam_mix_gaponly_p1", extra="--paste gaponly --gap_power 1") for s in (3, 4)]
    return runs


def round4_plan():
    """Round 4 (addendum to the round-3 pre-registration): seeds 3-4 for A, B2 and B5 with the same four methods,
    and the shuffled-gap control in the duplicates setting (seeds 0-4)."""
    runs = []
    for seed in (3, 4):
        runs += [R("cifar10", 100, m, seed, tag="dup_" + t, extra=f"{e} --dup 0:250,2:90".strip()) for t, m, e in ROUND3_METHODS]
        for nm in (2, 5):
            runs += [R("cifar10", 100, m, seed, tag=f"step{nm}_" + t, extra=f"{e} --profile step --n_minor {nm}".strip())
                     for t, m, e in ROUND3_METHODS]
    runs += [R("cifar10", 100, "dam_mix", seed, tag="dup_dam_mix_gapshuffle_p1",
               extra="--paste gapshuffle --gap_power 1 --dup 0:250,2:90") for seed in range(5)]
    return runs


def build_plan(plans):
    runs = []
    if "round4" in plans:
        runs += round4_plan()
    if "round3" in plans:
        runs += round3_plan()
    if "paper" in plans:
        runs += paper_plan()
    if "controls" in plans:
        runs += controls_plan()
    if "final" in plans and "paper" not in plans:
        # the final model: DAM-Mix (probe gap only), trained once more with weights + embeddings saved
        runs.append(dict(name="cifar10_ir100_dam_mix_gaponly_final_s0",
                         args="--dataset cifar10 --ir 100 --method dam_mix --paste gaponly --gap_power 2 --seed 0 "
                              "--save_model final_model/dam_mix_cifar10lt_ir100_s0.pt", emb=True))
    if "v4" in plans:
        # v4: can the measured gap REPLACE class counts? + complete the no-gap ablation
        for seed in range(5):
            runs.append(R("cifar10", 100, "dam_mix", seed, tag="dam_mix_gaponly", extra="--paste gaponly --gap_power 2"))
            if seed >= 3:
                runs.append(R("cifar10", 100, "dam_mix", seed, tag="dam_mix_nogap", extra="--paste sqrtinv"))
        for seed in (0, 1):
            runs.append(R("cifar100", 100, "dam_mix", seed, tag="dam_mix_nogap", extra="--paste sqrtinv"))
            runs.append(R("cifar100", 100, "dam_mix", seed, tag="dam_mix_gaponly", extra="--paste gaponly --gap_power 2"))
        for ir in (10, 50):
            runs.append(R("cifar10", ir, "dam_mix", 0, tag="dam_mix_nogap", extra="--paste sqrtinv"))
            runs.append(R("cifar10", ir, "dam_mix", 0, tag="dam_mix_gaponly", extra="--paste gaponly --gap_power 2"))
    if "v3" in plans:
        # v3: ours (dam_mix) vs ECL vs ECL+CMO, 5 seeds; ablation without the probe gap; robustness
        for seed in range(5):
            runs += [R("cifar10", 100, m, seed, emb=(seed == 0), extra=("--paste gap" if m == "dam_mix" else ""))
                     for m in ["ecl", "ecl_cmo", "dam_mix"]]
            if seed < 3:
                runs.append(R("cifar10", 100, "dam_mix", seed, tag="dam_mix_nogap", extra="--paste sqrtinv"))
        for seed in (0, 1):
            runs += [R("cifar100", 100, m, seed, extra=("--paste gap" if m == "dam_mix" else "")) for m in ["ecl", "ecl_cmo", "dam_mix"]]
        for ir in (10, 50):
            runs += [R("cifar10", ir, m, 0, extra=("--paste gap" if m == "dam_mix" else "")) for m in ["ecl_cmo", "dam_mix"]]
    # Priority order: a complete seed-0 picture first, then more seeds, then the extras.
    for seed in SEEDS:
        if "main" in plans:
            runs += [R("cifar10", 100, m, seed, emb=(seed == 0)) for m in MAIN]
        if "ablation" in plans and seed in (0, 1):
            runs += [R("cifar10", 100, "dam_ho", seed, tag=t, extra=e) for t, e in ABLATIONS.items()]
        if "fair" in plans and seed in (0, 1):
            runs += [R("cifar10", 100, m, seed, tag=t, extra=e) for t, (m, e) in FAIR.items()]
    if "ir" in plans:
        for ir in (10, 50):
            runs += [R("cifar10", ir, m, 0) for m in ["ce", "la", "ecl", "dam_ho"]]
    if "c100" in plans:
        runs += [R("cifar100", 100, m, 0) for m in ["ce", "la", "ecl", "dam_v2", "dam_ho"]]
    if "sens" in plans:
        for a in (0.05, 0.3):
            runs.append(R("cifar10", 100, "dam_ho", 0, tag=f"sens_alpha{a}", extra=f"--alpha {a}"))
        for b in (0.05, 0.3):
            runs.append(R("cifar10", 100, "dam_ho", 0, tag=f"sens_beta{b}", extra=f"--beta {b}"))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plans", default="final", help="paper,controls (new) | final | earlier sets: v4,v3,main,ablation,fair,ir,c100,sens")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--gpu", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--results", default="results")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--budget_hours", type=float, default=11.0, help="stop starting new runs after this")
    p.add_argument("--extra", default="", help="extra args for every run, e.g. --fake_data")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    runs = build_plan(a.plans.split(","))
    mine = [r for i, r in enumerate(runs) if i % a.num_shards == a.shard]
    if a.list:
        for i, r in enumerate(runs):
            print(f"part {i % a.num_shards}  {r['name']:40s} {r['args']}")
        print(f"{len(runs)} runs total; this part ({a.shard}/{a.num_shards}) has {len(mine)}")
        return
    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.makedirs(f"{a.results}/runs", exist_ok=True)
    os.makedirs(f"{a.results}/logs", exist_ok=True)
    t0 = time.time()
    times = []
    todo = [r for r in mine if not os.path.exists(f"{a.results}/runs/{r['name']}.json")]
    print(f"part {a.shard}/{a.num_shards}: {len(mine)} runs, {len(mine) - len(todo)} already done, {len(todo)} to do",
          flush=True)
    for k, r in enumerate(todo):
        if (time.time() - t0) / 3600 > a.budget_hours:
            print(f"time budget of {a.budget_hours} h reached - stopping (re-run later to continue)")
            break
        out = f"{a.results}/runs/{r['name']}.json"
        emb = f"--emb_out {a.results}/emb/{r['name']}.npz" if r["emb"] else ""
        cmd = (f"{sys.executable} -m damecl.train {r['args']} --epochs {a.epochs} --data_root {a.data_root} "
               f"--out {out} {emb} {a.extra}")
        print(f"[{k + 1}/{len(todo)}] {r['name']}", flush=True)
        ts = time.time()
        with open(f"{a.results}/logs/{r['name']}.log", "w") as log:
            ret = subprocess.call(cmd.split(), stdout=log, stderr=subprocess.STDOUT, env=env)
        times.append((time.time() - ts) / 60)
        left = len(todo) - k - 1
        print(f"    {'ok' if ret == 0 else 'FAILED - see ' + a.results + '/logs/' + r['name'] + '.log'} "
              f"in {times[-1]:.1f} min | about {sum(times) / len(times) * left / 60:.1f} h left in this part", flush=True)


if __name__ == "__main__":
    main()
