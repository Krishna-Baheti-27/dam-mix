#!/bin/bash
# Runs this account's part of the experiment list on Kaggle (Accelerator "GPU T4 x2", Internet ON).
#
#   ACCOUNT=0 NUM_ACCOUNTS=2 bash kaggle_run.sh     <- you
#   ACCOUNT=1 NUM_ACCOUNTS=2 bash kaggle_run.sh     <- your friend
#   (one person alone: NUM_ACCOUNTS=1 bash kaggle_run.sh)
# Optional: EPOCHS=60 (if the GPU is slow), BUDGET_HOURS=10.5 (stop starting new runs after this)
#           PLANS=final (default: train + save the final DAM-Mix model, ~10 min)
#           PLANS=round4  (extra seeds for round 3 A/B + shuffled gaps with duplicates; 29 runs)
#           PLANS=round3  (pre-registered round 3: duplicates, step imbalance, ink labels, power 1; 68 runs)
#           PLANS=paper,controls  (journal paper: Fashion-MNIST/SVHN/MNIST-LT + CIFAR-100 seed 3 + controls,
#                                  92 runs, about 10-12 GPU-hours = about 3 hours with 2 accounts x 2 GPUs)
#           PLANS=main,ablation,fair,ir,c100,sens,v3,v4  (reproduce every experiment, ~12 GPU-hours)
#
# PAUSE / RESUME: if you add a previous results_*.zip to the notebook as input data, its finished
# runs are unpacked first and skipped. Unfinished runs simply start again.
set -e
ACCOUNT=${ACCOUNT:-0}
NUM_ACCOUNTS=${NUM_ACCOUNTS:-1}
PLANS=${PLANS:-final}
ZIPNAME=results_$(echo "$PLANS" | tr ',' '_' | cut -c1-40)_account$ACCOUNT.zip
pip -q install scikit-learn matplotlib > /dev/null 2>&1 || true

# 0) resume: bring in finished runs from an earlier session (results_* zips/folders attached as input)
for z in $(find /kaggle/input -name "results_*.zip" 2>/dev/null); do
  echo "Resuming from $z"; python merge_results.py "$z" --out results || true
done
for d in $(find /kaggle/input -path "*results_*" -type d -name runs 2>/dev/null); do
  echo "Resuming from $d"; python merge_results.py "$(dirname "$d")" --out results || true
done

# 1) get CIFAR once from the fast mirror (so the GPU processes don't download it at the same time)
python - <<'PYEOF'
from damecl.data import load_dataset
for name in ("cifar10", "cifar100", "fmnist", "svhn", "mnist"):
    try:
        xtr, ytr, xte, yte, C = load_dataset(name, "./data")
        print(f"{name} ready: {len(ytr)} train / {len(yte)} test images, {C} classes", flush=True)
    except Exception as e:
        print(f"WARNING: could not load {name}: {e}", flush=True)
PYEOF

# 2) one process per GPU
NGPU=$(nvidia-smi -L | wc -l)
NSHARDS=$((NUM_ACCOUNTS * NGPU))
echo "Account $ACCOUNT of $NUM_ACCOUNTS, $NGPU GPU(s) -> parts $((ACCOUNT * NGPU))..$((ACCOUNT * NGPU + NGPU - 1)) of $NSHARDS"
for i in $(seq 0 $((NGPU - 1))); do
  SHARD=$((ACCOUNT * NGPU + i))
  nohup python run_all.py --plans "$PLANS" --shard $SHARD --num_shards $NSHARDS --gpu $i \
      --budget_hours ${BUDGET_HOURS:-10.5} --epochs ${EPOCHS:-100} > progress_part$SHARD.txt 2>&1 &
done
sleep 5; tail -n 3 progress_part*.txt
wait

# 3) zip this account's results (download it from the Output tab)
if [ -d final_model ]; then python evaluate_model.py final_model/*.pt || true; fi
if [ "$ACCOUNT" = "0" ] && [ "$PLANS" = "round3" ]; then   # cheap extras on one account: dataset statistics and the real-image figures
  python make_dataset_stats.py --out results/dataset_stats.json || true
  python make_paper_images.py --out results/figures || true
fi
zip -qr "$ZIPNAME" results $( [ -d final_model ] && echo final_model )
echo "Done. Download $ZIPNAME"
