"""
classifier.py
Part A — Few-Shot Classification with Transfer Learning
Group 8: SVHN (source) → MNIST (target)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import MNIST, SVHN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────
# 1.  Preprocessing (shared for both domains)
# ─────────────────────────────────────────
# ImageNet statistics applied to BOTH domains (as instructed)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # MNIST/SVHN → 3-ch
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─────────────────────────────────────────
# 2.  Dataset helpers
# ─────────────────────────────────────────
NUM_CLASSES   = 10   # digits 0-9
FEW_SHOT_K    = 50   # images per class for training


def get_few_shot_indices(dataset, k: int = FEW_SHOT_K, seed: int = 42) -> list:
    """Return indices with exactly k examples per class."""
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)
    indices = []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(targets == cls)[0]
        chosen  = rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    return indices


def load_svhn(data_root: str = "./data", seed: int = 42):
    """Load SVHN; returns (train_few_shot, val, test) subsets."""
    full_train = SVHN(root=data_root, split='train', download=True, transform=transform_train)
    full_test  = SVHN(root=data_root, split='test',  download=True, transform=transform_eval)

    # SVHN uses .labels instead of .targets
    full_train.targets = full_train.labels
    full_test.targets  = full_test.labels

    few_idx = get_few_shot_indices(full_train, k=FEW_SHOT_K, seed=seed)

    # validation: another 50 per class from the remaining training data
    all_idx  = set(range(len(full_train)))
    rest_idx = list(all_idx - set(few_idx))
    rest_sub = Subset(full_train, rest_idx)
    rest_sub.targets = np.array(full_train.targets)[rest_idx]
    val_idx  = get_few_shot_indices(rest_sub, k=50, seed=seed + 1)
    val_idx_global = [rest_idx[i] for i in val_idx]

    train_ds = Subset(full_train, few_idx)
    val_ds   = Subset(full_train, val_idx_global)
    # apply eval transform to val/test
    full_val_base = SVHN(root=data_root, split='train', download=True, transform=transform_eval)
    full_val_base.targets = full_val_base.labels
    val_ds = Subset(full_val_base, val_idx_global)

    return train_ds, val_ds, full_test


def load_mnist(data_root: str = "./data"):
    """Load MNIST test split (target domain evaluation only)."""
    test_ds = MNIST(root=data_root, train=False, download=True, transform=transform_eval)
    return test_ds


# ─────────────────────────────────────────
# 3.  Model builders
# ─────────────────────────────────────────
def _load_resnet50():
    """Load ResNet-50 with pretrained weights; falls back gracefully if offline."""
    for w in [models.ResNet50_Weights.IMAGENET1K_V2, models.ResNet50_Weights.IMAGENET1K_V1]:
        try:
            return models.resnet50(weights=w)
        except Exception:
            pass
    print("[WARNING] Could not download pretrained weights — using random init.")
    print("          Run in an environment with internet access (e.g., Google Colab) for full results.")
    return models.resnet50(weights=None)


def build_feature_extractor(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with frozen backbone; only the fc head is trainable."""
    model = _load_resnet50()
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_finetuned(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with layer3, layer4, and fc unfrozen."""
    model = _load_resnet50()
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ('layer3', 'layer4', 'fc')):
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_from_scratch(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with random weights — no pretraining (baseline)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────
# 4.  Training loop
# ─────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return total_loss / total, correct / total, torch.cat(all_preds), torch.cat(all_labels)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device,
    label: str = "model",
    figures_dir: str = "figures",
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{label}] Epoch {epoch:3d}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)

    # Plot curves
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title(f"{label} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title(f"{label} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"curves_{label.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved training curves → figures/curves_{label.replace(' ', '_')}.png")

    return model, history


# ─────────────────────────────────────────
# 5.  Main: run Part A over 3 seeds
# ─────────────────────────────────────────
def run_part_a(
    data_root:   str  = "./data",
    figures_dir: str  = "figures",
    ckpt_dir:    str  = "checkpoints",
    seeds:       list = [42, 123, 7],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Part A — Few-Shot Classification  |  device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(ckpt_dir, exist_ok=True)

    # Configs: (label, builder, epochs, lr)
    configs = [
        ("Frozen Backbone",  build_feature_extractor, 25, 1e-3),
        ("Fine-Tuned",       build_finetuned,         35, 1e-4),
        ("From Scratch",     build_from_scratch,      35, 1e-3),
    ]

    # Accumulate results across seeds
    results = {cfg[0]: {"src_acc": [], "tgt_acc": []} for cfg in configs}

    best_model_overall = None
    best_acc_overall   = -1.0
    best_label_overall = ""

    for seed in seeds:
        print(f"\n──── SEED {seed} ────")
        set_seed(seed)

        train_ds, val_ds, test_svhn = load_svhn(data_root, seed=seed)
        test_mnist = load_mnist(data_root)

        train_loader = DataLoader(train_ds,  batch_size=32, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,    batch_size=64, shuffle=False, num_workers=0)
        src_loader   = DataLoader(test_svhn, batch_size=64, shuffle=False, num_workers=0)
        tgt_loader   = DataLoader(test_mnist,batch_size=64, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()

        for (label, builder, epochs, lr) in configs:
            print(f"\n  >> {label}  (trainable params: {count_trainable(builder()):,})")
            set_seed(seed)   # reset before each model for fair comparison
            model = builder().to(device)

            model, _ = train_model(
                model, train_loader, val_loader, epochs, lr, device,
                label=f"{label}_seed{seed}", figures_dir=figures_dir,
            )

            _, src_acc, _, _ = evaluate(model, src_loader, criterion, device)
            _, tgt_acc, _, _ = evaluate(model, tgt_loader, criterion, device)

            results[label]["src_acc"].append(src_acc)
            results[label]["tgt_acc"].append(tgt_acc)
            print(f"  Source (SVHN) acc: {src_acc:.4f} | Target (MNIST) acc: {tgt_acc:.4f} | "
                  f"Δshift: {src_acc - tgt_acc:.4f}")

            # Track best overall (fine-tuned usually wins)
            if label == "Fine-Tuned" and src_acc > best_acc_overall:
                best_acc_overall   = src_acc
                best_label_overall = f"{label}_seed{seed}"
                best_model_overall = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("PART A SUMMARY (mean ± std over 3 seeds)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Src Acc':>10} {'Tgt Acc':>10} {'Δshift':>10}")
    print("-" * 55)
    for label in results:
        s = np.array(results[label]["src_acc"])
        t = np.array(results[label]["tgt_acc"])
        d = s - t
        print(f"{label:<20} {s.mean():.4f}±{s.std():.4f} "
              f"{t.mean():.4f}±{t.std():.4f} {d.mean():.4f}±{d.std():.4f}")

    # Save best fine-tuned checkpoint
    if best_model_overall:
        ckpt_path = os.path.join(ckpt_dir, "best_partA_finetuned.pt")
        torch.save(best_model_overall, ckpt_path)
        print(f"\nBest model saved → {ckpt_path}")

    return results, best_model_overall


if __name__ == "__main__":
    run_part_a()
