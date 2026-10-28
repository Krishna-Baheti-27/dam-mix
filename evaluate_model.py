"""Load the saved DAM-Mix weights and evaluate them on the CIFAR-10 test set.
    python evaluate_model.py final_model/dam_mix_cifar10lt_ir100_s0.pt
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from damecl import data as D
from damecl.model import Net
from damecl.metrics import classification_metrics

NAMES10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main(path, data_root="./data"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=dev, weights_only=False)
    C = ck["num_classes"]
    model = Net(C).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    name = ck["config"]["dataset"]
    _, _, xte, yte, _ = D.load_cifar(name, data_root)
    X = D.to_tensor(xte, dev)
    with torch.no_grad():
        logits = torch.cat([model(D.normalize(X[i:i + 1000]))[1].float() for i in range(0, len(X), 1000)])
    m = classification_metrics(yte, logits.cpu().numpy(), ck["train_counts"])
    print(f"model: {path}  ({ck['config']['method']}, paste={ck['config'].get('paste')}, seed {ck['config']['seed']})")
    print(f"accuracy {m['accuracy']*100:.2f}% | many {m['many_acc']*100:.2f}% | medium {m['medium_acc']*100:.2f}% | "
          f"few (rare) {m['few_acc']*100:.2f}% | macro-F1 {m['macro_f1']*100:.2f}%")
    if C == 10:
        for c in range(10):
            print(f"  {c} {NAMES10[c]:10s} train images {ck['train_counts'][c]:5d}  recall {m['per_class_recall'][c]*100:5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
