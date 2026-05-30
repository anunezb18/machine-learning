"""
style_transfer.py
Part B — Gatys-style Neural Style Transfer
Group 8: SVHN content images → styled with MNIST target domain images
Produces ≥30 style-transferred images per digit class (300 total).
"""

import os
import copy
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import SVHN, MNIST
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# 0.  Constants
# ─────────────────────────────────────────
IMAGE_SIZE   = 256          # px (balance quality vs. speed)
ALPHA        = 1.0          # content weight
BETA         = 1e4          # style weight
NST_STEPS    = 300          # L-BFGS steps
PER_CLASS    = 30           # synthetic images per class
NUM_CLASSES  = 10

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# VGG-19 layer names used
CONTENT_LAYERS = {"21"}                         # relu4_2
STYLE_LAYERS   = {"1", "6", "11", "20", "29"}   # relu1_1 … relu5_1


# ─────────────────────────────────────────
# 1.  Image I/O helpers
# ─────────────────────────────────────────
def _to_tensor(img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """PIL → normalised tensor [1, 3, H, W]."""
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """Denormalise tensor → PIL image."""
    t = tensor.squeeze(0).cpu().detach().clone()
    t = t * IMAGENET_STD + IMAGENET_MEAN
    t = t.clamp(0.0, 1.0)
    return transforms.ToPILImage()(t)


def _dataset_image(dataset, idx: int) -> Image.Image:
    """Return a PIL image from a torchvision dataset (raw, no transform)."""
    img, _ = dataset[idx]
    if isinstance(img, torch.Tensor):
        return transforms.ToPILImage()(img)
    return img


# ─────────────────────────────────────────
# 2.  Gram matrix
# ─────────────────────────────────────────
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Normalised Gram matrix: G_ij = (1 / C·H·W) Σ_k F_ik F_jk."""
    b, c, h, w = feat.size()
    feat = feat.view(b, c, h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)


# ─────────────────────────────────────────
# 3.  VGG-19 feature extractor
# ─────────────────────────────────────────
def _build_vgg19(device):
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


def _extract_features(vgg, x: torch.Tensor, layers: set) -> dict:
    """Run x through VGG-19 up to the deepest required layer, collecting named activations."""
    feats = {}
    for name, layer in vgg.named_children():
        x = layer(x)
        if name in layers:
            feats[name] = x
        if layers and all(k in feats for k in layers):
            break
    return feats


# ─────────────────────────────────────────
# 4.  Style-Content loss module
# ─────────────────────────────────────────
class StyleContentModel(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg

    def forward(self, generated, content_targets: dict, style_targets: dict):
        all_layers = CONTENT_LAYERS | STYLE_LAYERS
        feats = _extract_features(self.vgg, generated, all_layers)

        content_loss = sum(
            nn.functional.mse_loss(feats[l], content_targets[l])
            for l in CONTENT_LAYERS if l in feats
        )
        style_loss = sum(
            nn.functional.mse_loss(gram_matrix(feats[l]), gram_matrix(style_targets[l]))
            for l in STYLE_LAYERS if l in feats
        )
        return content_loss, style_loss


# ─────────────────────────────────────────
# 5.  Single NST run
# ─────────────────────────────────────────
def run_nst(
    content_img: Image.Image,
    style_img:   Image.Image,
    device,
    steps: int = NST_STEPS,
    alpha: float = ALPHA,
    beta:  float = BETA,
) -> Image.Image:
    content_t = _to_tensor(content_img).to(device)
    style_t   = _to_tensor(style_img).to(device)
    generated = content_t.clone().requires_grad_(True)

    vgg   = _build_vgg19(device)
    model = StyleContentModel(vgg)

    all_layers     = CONTENT_LAYERS | STYLE_LAYERS
    content_feats  = _extract_features(vgg, content_t, all_layers)
    style_feats    = _extract_features(vgg, style_t,   all_layers)

    optimizer = optim.LBFGS([generated], lr=1.0, max_iter=20)

    step_counter = [0]
    def closure():
        with torch.no_grad():
            generated.clamp_(-3.0, 3.0)
        optimizer.zero_grad()
        c_loss, s_loss = model(generated, content_feats, style_feats)
        loss = alpha * c_loss + beta * s_loss
        loss.backward()
        step_counter[0] += 1
        return loss

    for _ in range(steps // 20):   # each LBFGS call does up to 20 inner iters
        optimizer.step(closure)

    return _to_pil(generated)


# ─────────────────────────────────────────
# 6.  Batch generation for all classes
# ─────────────────────────────────────────
def generate_synthetic_images(
    out_dir:     str  = "data/synthetic_target",
    figures_dir: str  = "figures",
    data_root:   str  = "./data",
    per_class:   int  = PER_CLASS,
    seed:        int  = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Part B — Neural Style Transfer  |  device: {device}")
    print(f"Generating {per_class} synthetic images per class × {NUM_CLASSES} classes")
    print(f"{'='*60}\n")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Raw (no transform) datasets for content and style sources
    raw_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])

    svhn_raw = SVHN(root=data_root, split='train', download=True, transform=None)
    mnist_raw = MNIST(root=data_root, train=True,  download=True, transform=None)

    # Index by class
    svhn_by_class  = {c: [] for c in range(NUM_CLASSES)}
    mnist_by_class = {c: [] for c in range(NUM_CLASSES)}

    for idx, (_, label) in enumerate(svhn_raw):
        svhn_by_class[int(label)].append(idx)
    for idx, (_, label) in enumerate(mnist_raw):
        mnist_by_class[int(label)].append(idx)

    gallery_content, gallery_style, gallery_gen = [], [], []   # for figure

    for cls in range(NUM_CLASSES):
        cls_dir = os.path.join(out_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)

        content_pool = random.sample(svhn_by_class[cls],  min(per_class * 3, len(svhn_by_class[cls])))
        style_pool   = random.sample(mnist_by_class[cls], min(per_class * 3, len(mnist_by_class[cls])))

        generated_count = 0
        for i in range(per_class):
            content_raw, _ = svhn_raw[content_pool[i % len(content_pool)]]
            style_raw,   _ = mnist_raw[style_pool[i % len(style_pool)]]

            # Convert to PIL
            if isinstance(content_raw, np.ndarray):
                content_pil = Image.fromarray(content_raw)
            else:
                content_pil = content_raw
            if isinstance(style_raw, Image.Image):
                style_pil = style_raw
            else:
                style_pil = transforms.ToPILImage()(style_raw)

            try:
                result = run_nst(content_pil, style_pil, device)
                out_path = os.path.join(cls_dir, f"nst_{cls}_{i:04d}.png")
                result.save(out_path)
                generated_count += 1

                # Save one example per class for the gallery figure
                if i == 0:
                    gallery_content.append(content_pil)
                    gallery_style.append(style_pil)
                    gallery_gen.append(result)

            except Exception as e:
                print(f"  [WARNING] Class {cls}, image {i}: {e}")

        print(f"  Class {cls}: {generated_count}/{per_class} images saved → {cls_dir}")

    # ── Gallery figure (required) ──
    _save_gallery(gallery_content, gallery_style, gallery_gen,
                  os.path.join(figures_dir, "nst_gallery.png"))

    print(f"\nDone. Synthetic images in: {out_dir}")
    print(f"Gallery figure saved → {figures_dir}/nst_gallery.png")


def _save_gallery(contents, styles, generated, save_path):
    """Side-by-side gallery: content | style | generated (one row per class)."""
    n = len(contents)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = [axes]
    col_titles = ["Content (SVHN)", "Style (MNIST)", "Generated"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight='bold')

    for row in range(n):
        imgs = [contents[row], styles[row], generated[row]]
        for col, img in enumerate(imgs):
            ax = axes[row][col]
            pil = img.convert("RGB") if img.mode != "RGB" else img
            ax.imshow(pil)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"Digit {row}", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    generate_synthetic_images()
