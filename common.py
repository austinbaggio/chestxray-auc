"""Shared data loading and model definition for ChestMNIST."""
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

NUM_CLASSES = 14
LABELS = [
    "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
    "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
    "emphysema", "fibrosis", "pleural", "hernia",
]
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))


def load_split(split, size, smoke=False):
    """Return (images uint8 [N,H,W], labels float32 [N,14])."""
    if smoke:
        n = 256 if split == "train" else 128
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        x = rng.integers(0, 256, size=(n, size, size), dtype=np.uint8)
        y = (rng.random((n, NUM_CLASSES)) < 0.1).astype(np.float32)
        return x, y
    from medmnist import ChestMNIST
    os.makedirs(DATA_ROOT, exist_ok=True)
    ds = ChestMNIST(split=split, size=size, root=DATA_ROOT, download=True)
    return ds.imgs, ds.labels.astype(np.float32)


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, augment=False):
        self.imgs = torch.from_numpy(imgs).float().div_(255.0).unsqueeze(1)  # [N,1,H,W]
        self.labels = torch.from_numpy(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.imgs[i]
        if self.augment:
            # small random shift (+-4 px) and brightness/contrast jitter
            dx, dy = np.random.randint(-4, 5, size=2)
            x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(1, 2))
            x = (x - 0.5) * np.random.uniform(0.9, 1.1) + 0.5 + np.random.uniform(-0.05, 0.05)
        x = (x - 0.5) / 0.25
        return x, self.labels[i]


def build_model(size):
    """ResNet-18 adapted to single-channel input and 14 sigmoid outputs."""
    m = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    if size <= 64:
        # small images: keep spatial resolution in the stem
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    else:
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return m


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def mean_auc(y_true, y_prob):
    """Mean ROC-AUC over the 14 labels (MedMNIST's headline metric)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for k in range(y_true.shape[1]):
        if 0 < y_true[:, k].sum() < len(y_true):
            aucs.append(roc_auc_score(y_true[:, k], y_prob[:, k]))
        else:
            aucs.append(float("nan"))
    return float(np.nanmean(aucs)), aucs
