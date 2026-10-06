"""Evaluate a checkpoint on the ChestMNIST test split.

Writes results.json and prints the headline metric as the last line:
    score: <mean test ROC-AUC over 14 findings>   (higher is better)
"""
import argparse
import json

import numpy as np
import torch

from common import LABELS, XrayDataset, build_model, load_split, mean_auc, pick_device, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results.json")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    device = pick_device()
    ck = torch.load(a.ckpt, map_location="cpu")
    size = ck["size"]
    model = build_model(size)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    ds = XrayDataset(*load_split(a.split, size, a.smoke))
    dl = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=a.workers)
    prob, y = predict(model, dl, device)
    auc, per = mean_auc(y, prob)
    acc = float(((prob > 0.5) == (y > 0.5)).mean())

    res = {
        "metric": "mean_roc_auc",
        "score": round(auc, 4),
        "accuracy": round(acc, 4),
        "split": a.split,
        "image_size": size,
        "per_finding_auc": {k: (None if np.isnan(v) else round(v, 4)) for k, v in zip(LABELS, per)},
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"score: {auc:.4f}")


if __name__ == "__main__":
    main()
