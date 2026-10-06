"""Train a ChestMNIST multi-label classifier. Saves the best-val checkpoint to checkpoints/best.pt."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import XrayDataset, build_model, load_split, mean_auc, pick_device, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/best.pt")
    p.add_argument("--smoke", action="store_true", help="synthetic data, 1 epoch, for a quick pipeline check")
    a = p.parse_args()
    if a.smoke:
        a.epochs, a.size = 1, 28

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device()

    tr = XrayDataset(*load_split("train", a.size, a.smoke), augment=True)
    va = XrayDataset(*load_split("val", a.size, a.smoke))
    tl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True,
                                     num_workers=a.workers, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=512, num_workers=a.workers)

    model = build_model(a.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        prob, yv = predict(model, vl, device)
        auc, _ = mean_auc(yv, prob)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.4f} val_auc {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best = auc
            torch.save({"state_dict": model.state_dict(), "size": a.size, "val_auc": auc, "args": vars(a)}, a.out)

    print(json.dumps({"best_val_auc": round(best, 4), "checkpoint": a.out}))


if __name__ == "__main__":
    main()
