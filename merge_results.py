"""Combine results from several accounts / sessions into one results/ folder.

    python merge_results.py results_account0.zip results_account1.zip --out results

Accepts .zip files or folders. Existing files are never overwritten (a run is only saved when finished).
"""
import argparse
import os
import shutil
import tempfile
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="+")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    added = 0
    for src in a.sources:
        tmp = None
        if src.endswith(".zip"):
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            root = tmp
        else:
            root = src
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not (f.endswith(".json") or f.endswith(".npz") or f.endswith(".log")):
                    continue
                sub = os.path.basename(dirpath)
                if sub not in ("runs", "emb", "logs"):
                    continue
                dst_dir = os.path.join(a.out, sub)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(dirpath, f), dst)
                    added += f.endswith(".json")
        if tmp:
            shutil.rmtree(tmp)
    n = len([f for f in os.listdir(os.path.join(a.out, "runs")) if f.endswith(".json")])
    print(f"added {added} finished runs; {a.out}/runs now has {n}")


if __name__ == "__main__":
    main()
