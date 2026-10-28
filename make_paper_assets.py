"""Build every table (LaTeX) and figure (PDF) of the paper from one or more result folders.

    python make_paper_assets.py --results results_old results_paper_a results_paper_b --out paper_assets

Then copy paper_assets/tables and paper_assets/figures into the Overleaf project (same folder names).
"""
import argparse

from damecl.paper_assets import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results"])
    p.add_argument("--out", default="paper_assets")
    a = p.parse_args()
    build(a.results, a.out)
