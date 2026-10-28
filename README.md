# DAM-Mix: memorisation-aware minority pasting for long-tailed image classification

Code and complete results for the paper *DAM-Mix: Memorisation-Aware Minority Pasting for Long-Tailed Image
Classification* (B.Tech. Machine Learning project, SVNIT Surat).

DAM-Mix extends ECL (balanced contrastive learning with logit adjustment) with CMO-style minority pasting
(CutMix with a rare foreground). How often each class is pasted is set by its **memorisation gap**, measured on a
small held-out probe after every epoch, instead of by its class count:

```
G_c = dispersion on a 20-image held-out probe / dispersion on the class's own training images   (clipped to [1, 6], EMA 0.9)
q_c = G_c^kappa / sum_k G_k^kappa        kappa = 1 (selected on the probe)
```

## Main results (ResNet-32, 100 epochs, imbalance ratio 100, overall accuracy %)

| Dataset | ECL | ECL + CMO | ECL + CMO (r=1/2) | **DAM-Mix** |
|---|---|---|---|---|
| CIFAR-10-LT (5 seeds) | 76.45 | 77.97 | 79.93 | **80.49** |
| CIFAR-100-LT (3 seeds) | 41.63 | 43.00 | 44.62 | **44.71** |
| Fashion-MNIST-LT (3 seeds) | 90.13 | 88.92 | 91.08 | **91.39** |
| SVHN-LT (3 seeds) | 87.27 | 90.50 | 90.87 | **91.36** |
| MNIST-LT (3 seeds) | **98.90** | 79.98 | 88.37 | 86.73 (97.36 with content labels) |

Settings where class counts misdescribe the data (CIFAR-10, 5 seeds, pre-registered in
`docs/PREREGISTRATION.md`): duplicated classes +1.46 overall accuracy vs. CMO (r=1/2), Holm p = 0.002;
five minority classes +2.85 minority accuracy, Holm p = 0.013.

Every number in the paper is in `paper_assets/key_numbers.txt` and can be regenerated from `results/`.

## Repository layout

| Path | Content |
|---|---|
| `damecl/` | Code: `data.py` (datasets, long-tailed splits, duplicates/step profiles, CutMix with area or content labels), `model.py`, `losses.py` (gap tracker, paste schedules, ECL losses), `train.py`, `metrics.py`, `paper_assets.py` (all tables and figures) |
| `run_all.py`, `kaggle_run.sh`, `kaggle_train.ipynb` | Experiment runner (resumable, shardable across GPUs / Kaggle accounts); plans `final`, `paper`, `controls`, `round3`, `round4` |
| `make_paper_assets.py` | Builds every table (LaTeX) and figure (PDF) of the paper from the result folders |
| `make_paper_images.py`, `make_dataset_stats.py` | Figures with real images; empty-pixel share and label mismatch per dataset (needs the datasets) |
| `evaluate_model.py` | Loads saved weights and prints test metrics |
| `tests/` | Unit tests, including numerical checks of Propositions 1, 2, 4 and 5 (`pytest tests`) |
| `results/runs/` | One JSON file per training run, 283 in total (metrics, confusion matrices, paste schedules, gap histories, probe metrics) |
| `results/emb/`, `results/logs/` | Saved test embeddings (used for the geometry table and t-SNE) and training logs |
| `results/dataset_stats.json` | Empty-pixel share and label mismatch per dataset (Section 10 of the paper) |
| `paper_assets/` | Generated tables, figures, `all_runs.csv` and `key_numbers.txt` used in the paper |
| `docs/PREREGISTRATION.md` | Hypotheses, primary tests and decision rule written before the count-misdescription and content-label experiments |

## Reproduce

```bash
pip install -r requirements.txt
pytest tests                                   # 23 tests
python run_all.py --plans paper,controls,round3,round4 --gpu 0   # or open kaggle_train.ipynb on Kaggle (GPU T4 x2, Internet on)
python make_paper_assets.py --results results --out paper_assets   # rebuilds every table, figure and key_numbers.txt
```

One DAM-Mix run on CIFAR-10-LT takes about 9.4 minutes on an NVIDIA T4; the full study used 38.8 GPU-hours.

## Method tags in the result files

| Tag | Method |
|---|---|
| `ce`, `wce`, `la` | cross-entropy, weighted CE, logit-adjusted CE |
| `ecl` | ECL |
| `ecl_cmo` | ECL + CMO (q ∝ 1/N) |
| `dam_mix_nogap` | ECL + CMO with r = 1/2 (q ∝ 1/√N) |
| `dam_mix_gaponly_p1` | **DAM-Mix, κ = 1 (final)** |
| `dam_mix_gaponly`, `dam_mix_gaponly_p3` | DAM-Mix, κ = 2 and κ = 3 |
| `dam_mix_gapshuffle*` | gaps permuted within Many/Medium/Few groups (pooling control) |
| `*_plus` | baseline trained with the probe images added |
| `*_dup_*`, `*_step2_*`, `*_step5_*`, `*_ink_*` (in the file name) | duplicated classes, step imbalance, content-based labels |
