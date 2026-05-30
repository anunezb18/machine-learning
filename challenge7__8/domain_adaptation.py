"""
domain_adaptation.py
Part C — Domain Shift Measurement and Adaptation
Group 8: SVHN → MNIST

Strategies compared:
  1. Baseline (no adaptation)
  2. Target fine-tuning (50 labelled MNIST images per class)
  3. Style-transfer augmentation (synthetic images from Part B)
  4. [Optional] DANN — Domain-Adversarial Neural Networks
"""

import os
import random
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import SVHN, MNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Function

# Local
from classifier import (
    set_seed, transform_train, transform_eval,
    load_svhn, load_mnist,
    build_finetuned, evaluate, train_one_epoch,
    count_trainable, NUM_CLASSES, FEW_SHOT_K,
    get_few_shot_indices,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────
# 1.  Synthetic dataset loader
# ─────────────────────────────────────────
class SyntheticDataset(Dataset):
    """Loads NST-generated images saved by style_transfer.py."""
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples = []
        for label in range(NUM_CLASSES):
            cls_dir = os.path.join(root, str(label))
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────
# 2.  DANN components (optional)
# ─────────────────────────────────────────
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class DANNClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int = NUM_CLASSES):
        super().__init__()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()          # remove original head
        self.backbone   = backbone
        self.class_head = nn.Linear(in_features, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2),               # 0=source, 1=target
        )

    def forward(self, x, alpha: float = 1.0):
        feat    = self.backbone(x)
        cls_out = self.class_head(feat)
        rev     = GradientReversal.apply(feat, alpha)
        dom_out = self.domain_head(rev)
        return cls_out, dom_out


# ─────────────────────────────────────────
# 3.  Adaptation training loops
# ─────────────────────────────────────────
def finetune_target(
    model: nn.Module,
    target_train_loader,
    target_val_loader,
    epochs: int = 15,
    lr: float = 1e-4,
    device = None,
    figures_dir: str = "figures",
    label: str = "target_finetune",
):
    """Fine-tune last conv block + head on a small labelled MNIST set."""
    model = copy.deepcopy(model).to(device)
    # Unfreeze layer3, layer4, fc
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ('layer3', 'layer4', 'fc')):
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, target_train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, target_val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc  = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"    [{label}] Epoch {epoch:3d} | tr {tr_acc:.4f} | val {vl_acc:.4f}")

    model.load_state_dict(best_state)

    # Save curve
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key_pair, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"]
    ):
        ax.plot(history[key_pair[0]], label="train")
        ax.plot(history[key_pair[1]], label="val")
        ax.set_title(f"{label} — {title}")
        ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, f"curves_{label}.png"), dpi=150)
    plt.close(fig)

    return model


def train_dann(
    source_loader,
    target_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 1e-4,
    lambda_d: float = 0.5,
    device = None,
    figures_dir: str = "figures",
):
    """DANN with linear alpha schedule."""
    backbone = build_finetuned().to(device)
    model    = DANNClassifier(backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"cls_loss": [], "dom_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None

    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)

    total_steps = epochs * max(len(source_loader), len(target_loader))
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_cls, epoch_dom = 0.0, 0.0
        n_batches = max(len(source_loader), len(target_loader))

        for _ in range(n_batches):
            # Linear alpha: 0 → 1 over training
            p     = step / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            step += 1

            # Source batch
            try:
                src_imgs, src_labels = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                src_imgs, src_labels = next(src_iter)

            # Target batch (no class labels needed)
            try:
                tgt_imgs, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                tgt_imgs, _ = next(tgt_iter)

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            # Build batch with domain labels
            combined = torch.cat([src_imgs, tgt_imgs], dim=0)
            dom_labels = torch.cat([
                torch.zeros(src_imgs.size(0), dtype=torch.long),
                torch.ones(tgt_imgs.size(0),  dtype=torch.long),
            ]).to(device)

            optimizer.zero_grad()
            cls_out, dom_out = model(combined, alpha=alpha)

            # Class loss only on source
            cls_loss = criterion(cls_out[:src_imgs.size(0)], src_labels)
            dom_loss = criterion(dom_out, dom_labels)
            loss     = cls_loss + lambda_d * dom_loss

            loss.backward()
            optimizer.step()

            epoch_cls += cls_loss.item()
            epoch_dom += dom_loss.item()

        # Validate on MNIST (target)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                cls_out, _ = model(imgs)
                correct += (cls_out.argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total

        history["cls_loss"].append(epoch_cls / n_batches)
        history["dom_loss"].append(epoch_dom / n_batches)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc  = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"    [DANN] Epoch {epoch:3d} | cls {epoch_cls/n_batches:.4f} | "
                  f"dom {epoch_dom/n_batches:.4f} | val {val_acc:.4f} | alpha {alpha:.3f}")

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────
# 4.  t-SNE visualisation
# ─────────────────────────────────────────
@torch.no_grad()
def extract_features(model: nn.Module, loader, device, max_samples: int = 1000):
    """Extract penultimate-layer features."""
    # Hook on the avgpool (before fc)
    feats_list, labels_list = [], []
    hook_out = {}

    def hook_fn(module, input, output):
        hook_out['feat'] = output.view(output.size(0), -1).cpu()

    handle = model.avgpool.register_forward_hook(hook_fn)
    model.eval()
    count = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        _ = model(imgs)
        feats_list.append(hook_out['feat'])
        labels_list.append(labels)
        count += imgs.size(0)
        if count >= max_samples:
            break
    handle.remove()
    return torch.cat(feats_list)[:max_samples], torch.cat(labels_list)[:max_samples]


def plot_tsne(
    src_feats, src_labels, tgt_feats, tgt_labels,
    save_path: str, title: str = "t-SNE",
):
    from sklearn.manifold import TSNE
    import matplotlib.cm as cm

    all_feats  = torch.cat([src_feats, tgt_feats]).numpy()
    all_labels = torch.cat([src_labels, tgt_labels]).numpy()
    domain     = np.array([0] * len(src_labels) + [1] * len(tgt_labels))

    print("    Running t-SNE …", end=" ", flush=True)
    proj = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500).fit_transform(all_feats)
    print("done.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = cm.tab10(np.linspace(0, 1, 10))

    # By class
    for cls in range(10):
        mask = all_labels == cls
        axes[0].scatter(proj[mask, 0], proj[mask, 1], c=[colors[cls]],
                        s=10, alpha=0.6, label=str(cls))
    axes[0].set_title(f"{title}\nColoured by class")
    axes[0].legend(fontsize=7, ncol=2, markerscale=2)

    # By domain
    for d, name, c in [(0, "SVHN (source)", "steelblue"), (1, "MNIST (target)", "tomato")]:
        mask = domain == d
        axes[1].scatter(proj[mask, 0], proj[mask, 1], c=c, s=10, alpha=0.5, label=name)
    axes[1].set_title(f"{title}\nColoured by domain")
    axes[1].legend(markerscale=2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    t-SNE saved → {save_path}")


# ─────────────────────────────────────────
# 5.  Grad-CAM visualisation
# ─────────────────────────────────────────
def plot_gradcam(model: nn.Module, loader, device,
                 save_path: str, n_correct: int = 2, n_wrong: int = 2):
    """Generate Grad-CAM maps for correct and incorrect predictions."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("    [WARNING] pytorch-grad-cam not installed — skipping Grad-CAM.")
        return

    target_layer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)

    correct_imgs, correct_cams = [], []
    wrong_imgs,   wrong_cams   = [], []

    denorm = transforms.Compose([
        transforms.Normalize([-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                             [1/s for s in [0.229, 0.224, 0.225]]),
    ])

    model.eval()
    for imgs, labels in loader:
        imgs_dev = imgs.to(device)
        with torch.no_grad():
            preds = model(imgs_dev).argmax(1).cpu()

        for i in range(imgs.size(0)):
            if len(correct_imgs) < n_correct and preds[i] == labels[i]:
                grayscale_cam = cam(input_tensor=imgs_dev[i:i+1])[0]
                raw = denorm(imgs[i]).permute(1, 2, 0).clamp(0, 1).numpy()
                vis = show_cam_on_image(raw, grayscale_cam, use_rgb=True)
                correct_imgs.append(raw)
                correct_cams.append(vis)

            if len(wrong_imgs) < n_wrong and preds[i] != labels[i]:
                grayscale_cam = cam(input_tensor=imgs_dev[i:i+1])[0]
                raw = denorm(imgs[i]).permute(1, 2, 0).clamp(0, 1).numpy()
                vis = show_cam_on_image(raw, grayscale_cam, use_rgb=True)
                wrong_imgs.append(raw)
                wrong_cams.append(vis)

            if len(correct_imgs) >= n_correct and len(wrong_imgs) >= n_wrong:
                break

        if len(correct_imgs) >= n_correct and len(wrong_imgs) >= n_wrong:
            break

    n_rows = max(n_correct, n_wrong)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    titles = ["Correct — Input", "Correct — Grad-CAM", "Wrong — Input", "Wrong — Grad-CAM"]
    for col, t in enumerate(titles):
        axes[0][col].set_title(t, fontsize=10)

    for row in range(n_rows):
        for col, (imgs_list, col_idx) in enumerate(
            [(correct_imgs, 0), (correct_cams, 1), (wrong_imgs, 2), (wrong_cams, 3)]
        ):
            ax = axes[row][col_idx]
            if row < len(imgs_list):
                ax.imshow(imgs_list[row])
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Grad-CAM saved → {save_path}")


# ─────────────────────────────────────────
# 6.  Per-class accuracy
# ─────────────────────────────────────────
@torch.no_grad()
def per_class_accuracy(model, loader, device):
    model.eval()
    correct = np.zeros(NUM_CLASSES)
    total   = np.zeros(NUM_CLASSES)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for c in range(NUM_CLASSES):
            mask = labels == c
            total[c]   += mask.sum().item()
            correct[c] += (preds[mask] == c).sum().item()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.where(total > 0, correct / total, 0.0)
    return acc


# ─────────────────────────────────────────
# 7.  Main: Part C
# ─────────────────────────────────────────
def run_part_c(
    best_partA_state: dict = None,    # loaded from checkpoint if None
    data_root:    str = "./data",
    synthetic_dir:str = "data/synthetic_target",
    figures_dir:  str = "figures",
    ckpt_dir:     str = "checkpoints",
    seeds:        list = [42, 123, 7],
    run_dann:     bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Part C — Domain Shift & Adaptation  |  device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load best Part A checkpoint if not provided
    if best_partA_state is None:
        ckpt_path = os.path.join(ckpt_dir, "best_partA_finetuned.pt")
        if os.path.exists(ckpt_path):
            best_partA_state = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded Part A checkpoint from {ckpt_path}")
        else:
            print("[WARNING] No Part A checkpoint found — using random init for adaptation.")

    results = {
        "Baseline (no adapt)":   {"src_acc": [], "tgt_acc": []},
        "Target Fine-Tune":      {"src_acc": [], "tgt_acc": []},
        "Style-Aug":             {"src_acc": [], "tgt_acc": []},
    }
    if run_dann:
        results["DANN"] = {"src_acc": [], "tgt_acc": []}

    best_adapted_model = None
    best_adapted_acc   = -1.0

    # ── Check synthetic data availability ──
    synth_available = (
        os.path.isdir(synthetic_dir) and
        any(os.listdir(os.path.join(synthetic_dir, str(c)))
            for c in range(NUM_CLASSES)
            if os.path.isdir(os.path.join(synthetic_dir, str(c))))
    )
    if not synth_available:
        print(f"[WARNING] Synthetic images not found in {synthetic_dir}. "
              "Style-Aug strategy will be skipped or use empty set.")

    for seed in seeds:
        print(f"\n──── SEED {seed} ────")
        set_seed(seed)

        # Data
        train_svhn, val_svhn, test_svhn = load_svhn(data_root, seed=seed)
        test_mnist = load_mnist(data_root)

        # Small labelled MNIST set (50 per class)
        mnist_train_full = MNIST(root=data_root, train=True, download=True,
                                 transform=transform_train)
        mnist_val_full   = MNIST(root=data_root, train=True, download=True,
                                 transform=transform_eval)

        tgt_train_idx = get_few_shot_indices(mnist_train_full, k=FEW_SHOT_K, seed=seed)
        all_mnist_idx = set(range(len(mnist_train_full)))
        rest_mnist    = list(all_mnist_idx - set(tgt_train_idx))
        rest_sub      = Subset(mnist_val_full, rest_mnist)
        rest_sub.targets = np.array(mnist_train_full.targets)[rest_mnist]
        val_mnist_idx = get_few_shot_indices(rest_sub, k=50, seed=seed + 99)
        val_mnist_global = [rest_mnist[i] for i in val_mnist_idx]

        tgt_train_ds = Subset(mnist_train_full, tgt_train_idx)
        tgt_val_ds   = Subset(mnist_val_full, val_mnist_global)

        src_loader = DataLoader(test_svhn, batch_size=64, shuffle=False, num_workers=0)
        tgt_loader = DataLoader(test_mnist, batch_size=64, shuffle=False, num_workers=0)
        tgt_train_loader = DataLoader(tgt_train_ds, batch_size=32, shuffle=True, num_workers=0)
        tgt_val_loader   = DataLoader(tgt_val_ds,   batch_size=64, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()

        # ── Strategy 1: Baseline ──
        model_baseline = build_finetuned().to(device)
        if best_partA_state:
            model_baseline.load_state_dict(best_partA_state)
        _, src_acc, _, _ = evaluate(model_baseline, src_loader, criterion, device)
        _, tgt_acc, _, _ = evaluate(model_baseline, tgt_loader, criterion, device)
        results["Baseline (no adapt)"]["src_acc"].append(src_acc)
        results["Baseline (no adapt)"]["tgt_acc"].append(tgt_acc)
        print(f"  Baseline    | src {src_acc:.4f} | tgt {tgt_acc:.4f} | Δ {src_acc-tgt_acc:.4f}")

        # ── Strategy 2: Target Fine-Tune ──
        model_ft = build_finetuned().to(device)
        if best_partA_state:
            model_ft.load_state_dict(best_partA_state)
        model_ft = finetune_target(
            model_ft, tgt_train_loader, tgt_val_loader,
            epochs=15, lr=1e-4, device=device,
            figures_dir=figures_dir, label=f"target_ft_seed{seed}",
        )
        _, src_acc_ft, _, _ = evaluate(model_ft, src_loader, criterion, device)
        _, tgt_acc_ft, _, _ = evaluate(model_ft, tgt_loader, criterion, device)
        results["Target Fine-Tune"]["src_acc"].append(src_acc_ft)
        results["Target Fine-Tune"]["tgt_acc"].append(tgt_acc_ft)
        print(f"  TgtFineTune | src {src_acc_ft:.4f} | tgt {tgt_acc_ft:.4f} | Δ {src_acc_ft-tgt_acc_ft:.4f}")

        if tgt_acc_ft > best_adapted_acc:
            best_adapted_acc   = tgt_acc_ft
            best_adapted_model = {k: v.cpu().clone() for k, v in model_ft.state_dict().items()}

        # ── Strategy 3: Style-Transfer Augmentation ──
        if synth_available:
            synth_ds = SyntheticDataset(synthetic_dir, transform=transform_train)
            combined_train = ConcatDataset([train_svhn, synth_ds])
            combined_loader = DataLoader(combined_train, batch_size=32, shuffle=True, num_workers=0)

            model_aug = build_finetuned().to(device)
            if best_partA_state:
                model_aug.load_state_dict(best_partA_state)

            opt_aug  = optim.Adam(
                filter(lambda p: p.requires_grad, model_aug.parameters()), lr=1e-4)
            sched    = optim.lr_scheduler.CosineAnnealingLR(opt_aug, T_max=20)
            crit     = nn.CrossEntropyLoss()
            best_aug_acc, best_aug_state = 0.0, None

            for epoch in range(1, 21):
                tr_loss, tr_acc = train_one_epoch(model_aug, combined_loader, opt_aug, crit, device)
                _, vl_acc, _, _ = evaluate(model_aug, tgt_val_loader, crit, device)
                sched.step()
                if vl_acc > best_aug_acc:
                    best_aug_acc  = vl_acc
                    best_aug_state = {k: v.cpu().clone()
                                      for k, v in model_aug.state_dict().items()}
                if epoch % 5 == 0:
                    print(f"    [StyleAug seed{seed}] Epoch {epoch:3d} | "
                          f"tr {tr_acc:.4f} | val {vl_acc:.4f}")

            model_aug.load_state_dict(best_aug_state)
            _, src_acc_aug, _, _ = evaluate(model_aug, src_loader, crit, device)
            _, tgt_acc_aug, _, _ = evaluate(model_aug, tgt_loader, crit, device)
            results["Style-Aug"]["src_acc"].append(src_acc_aug)
            results["Style-Aug"]["tgt_acc"].append(tgt_acc_aug)
            print(f"  StyleAug    | src {src_acc_aug:.4f} | tgt {tgt_acc_aug:.4f} | Δ {src_acc_aug-tgt_acc_aug:.4f}")

            if tgt_acc_aug > best_adapted_acc:
                best_adapted_acc   = tgt_acc_aug
                best_adapted_model = {k: v.cpu().clone()
                                      for k, v in model_aug.state_dict().items()}
        else:
            print("  StyleAug    | SKIPPED (no synthetic images)")

        # ── Strategy 4: DANN (optional) ──
        if run_dann:
            svhn_loader = DataLoader(train_svhn, batch_size=32, shuffle=True, num_workers=0)
            # Unlabelled MNIST for DANN
            mnist_unlabelled = MNIST(root=data_root, train=True, download=True,
                                     transform=transform_train)
            tgt_unlabelled_loader = DataLoader(mnist_unlabelled,
                                               batch_size=32, shuffle=True, num_workers=0)
            set_seed(seed)
            model_dann = train_dann(
                svhn_loader, tgt_unlabelled_loader, tgt_val_loader,
                epochs=20, lr=1e-4, lambda_d=0.5, device=device,
                figures_dir=figures_dir,
            )
            model_dann_cls = model_dann  # forward returns (cls, dom)

            # Evaluate: use only cls_out
            correct_src, total_src = 0, 0
            correct_tgt, total_tgt = 0, 0
            model_dann.eval()
            with torch.no_grad():
                for imgs, labels in src_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    cls_out, _ = model_dann(imgs)
                    correct_src += (cls_out.argmax(1) == labels).sum().item()
                    total_src   += labels.size(0)
                for imgs, labels in tgt_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    cls_out, _ = model_dann(imgs)
                    correct_tgt += (cls_out.argmax(1) == labels).sum().item()
                    total_tgt   += labels.size(0)

            src_acc_dann = correct_src / total_src
            tgt_acc_dann = correct_tgt / total_tgt
            results["DANN"]["src_acc"].append(src_acc_dann)
            results["DANN"]["tgt_acc"].append(tgt_acc_dann)
            print(f"  DANN        | src {src_acc_dann:.4f} | tgt {tgt_acc_dann:.4f} | Δ {src_acc_dann-tgt_acc_dann:.4f}")

    # ── Save best adapted model ──
    if best_adapted_model:
        ckpt_path = os.path.join(ckpt_dir, "best_partC_adapted.pt")
        torch.save(best_adapted_model, ckpt_path)
        print(f"\nBest adapted model saved → {ckpt_path}")

    # ── Summary table ──
    print(f"\n{'='*65}")
    print("PART C SUMMARY (mean ± std over 3 seeds)")
    print(f"{'='*65}")
    print(f"{'Strategy':<25} {'Src Acc':>12} {'Tgt Acc':>12} {'Δshift':>12}")
    print("-" * 65)
    for label, vals in results.items():
        if not vals["src_acc"]:
            print(f"{label:<25} {'SKIPPED':>38}")
            continue
        s = np.array(vals["src_acc"])
        t = np.array(vals["tgt_acc"])
        d = s - t
        print(f"{label:<25} {s.mean():.4f}±{s.std():.4f}  "
              f"{t.mean():.4f}±{t.std():.4f}  {d.mean():.4f}±{d.std():.4f}")

    # ── t-SNE before/after adaptation ──
    print("\n  Generating t-SNE visualisations …")
    _run_tsne_plots(
        best_partA_state, best_adapted_model,
        test_svhn, test_mnist, device, figures_dir,
    )

    # ── Grad-CAM ──
    print("  Generating Grad-CAM maps …")
    model_gc = build_finetuned().to(device)
    if best_adapted_model:
        model_gc.load_state_dict(best_adapted_model)
    tgt_loader_gc = DataLoader(test_mnist, batch_size=16, shuffle=True, num_workers=0)
    plot_gradcam(model_gc, tgt_loader_gc, device,
                 save_path=os.path.join(figures_dir, "gradcam_target.png"))

    return results


def _run_tsne_plots(state_before, state_after,
                    src_dataset, tgt_dataset, device, figures_dir):
    from torch.utils.data import DataLoader as DL
    sl = DL(src_dataset, batch_size=64, shuffle=False, num_workers=0)
    tl = DL(tgt_dataset, batch_size=64, shuffle=False, num_workers=0)

    for tag, state in [("before_adapt", state_before), ("after_adapt", state_after)]:
        if state is None:
            continue
        m = build_finetuned().to(device)
        m.load_state_dict(state)
        sf, sl_ = extract_features(m, sl, device, max_samples=500)
        tf, tl_ = extract_features(m, tl, device, max_samples=500)
        plot_tsne(sf, sl_, tf, tl_,
                  save_path=os.path.join(figures_dir, f"tsne_{tag}.png"),
                  title=f"t-SNE {tag.replace('_', ' ').title()}")


if __name__ == "__main__":
    run_part_c()
