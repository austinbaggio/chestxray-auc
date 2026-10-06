# chestxray-auc

Multi-label chest X-ray classifier. Fourteen findings per image (effusion, pneumothorax, cardiomegaly, pneumonia and ten more). Built as a test target for Ensue's research loop.

## Task

- **Data:** ChestMNIST from MedMNIST v2, derived from NIH ChestX-ray14. 112,120 frontal X-rays from 30,805 patients. Split 78,468 train / 11,219 val / 22,433 test. Downloads automatically on first run.
- **Metric:** mean ROC-AUC across the 14 findings on the test split. Higher is better.
- **Baseline model:** ResNet-18, single-channel input, 64×64 images, BCE loss, AdamW, one-cycle LR, 20 epochs.

## Run

```bash
pip install -r requirements.txt
python train.py            # writes checkpoints/best.pt (best val AUC)
python eval.py             # writes results.json, last line is "score: 0.xxxx"
```

Useful flags: `--size {28,64,128,224}`, `--epochs`, `--lr`, `--batch-size`. `--smoke` runs both scripts on synthetic data in seconds to check the pipeline.

## Numbers to beat (test mean AUC, MedMNIST v2 paper)

| Method | AUC |
|---|---|
| auto-sklearn | 0.649 |
| AutoKeras | 0.742 |
| ResNet-18 (28px) | 0.768 |
| ResNet-50 (28px) | 0.769 |
| ResNet-18 (224px) | 0.773 |
| ResNet-50 (224px) | 0.773 |
| Google AutoML Vision | 0.778 |

The interesting target is Google AutoML Vision at 0.778.

## Why this task

Chest X-ray triage is a live commercial category. Every hospital produces these images, and the same multi-label setup carries over to any proprietary imaging set a client brings. Experiment cycles are short (a 64px run takes minutes on one GPU), so a research loop gets many iterations per dollar.

## Sources

- [MedMNIST v2 paper (Yang et al., Scientific Data 2023)](https://arxiv.org/abs/2110.14795): dataset splits and baseline table
- [MedMNIST project and license (CC BY 4.0)](https://medmnist.com/)
- [NIH ChestX-ray14 (Wang et al., CVPR 2017)](https://arxiv.org/abs/1705.02315)
