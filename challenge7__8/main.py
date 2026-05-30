"""
main.py — Orchestrator for Challenge 7 (Group 8: SVHN → MNIST)

Usage:
    python main.py --parts ABC        # run all three parts
    python main.py --parts A          # only Part A
    python main.py --parts BC         # skip Part A, load checkpoint
    python main.py --parts ABC --fast # fewer epochs/seeds for quick test
"""

import argparse
import os
import sys

# ── Reproducibility seed list ──
SEEDS = [42, 123, 7]


def main():
    parser = argparse.ArgumentParser(description="Challenge 7 — SVHN → MNIST Transfer Learning")
    parser.add_argument("--parts",      default="ABC", help="Which parts to run: A, B, C, AB, ABC …")
    parser.add_argument("--data_root",  default="./data",                help="Dataset download root")
    parser.add_argument("--synth_dir",  default="./data/synthetic_target", help="NST output directory")
    parser.add_argument("--figures_dir",default="./figures",             help="Figures output directory")
    parser.add_argument("--ckpt_dir",   default="./checkpoints",         help="Checkpoint directory")
    parser.add_argument("--fast",       action="store_true",
                        help="Quick test: 1 seed, fewer epochs")
    parser.add_argument("--no_dann",    action="store_true",
                        help="Skip DANN (optional strategy)")
    args = parser.parse_args()

    seeds = [42] if args.fast else SEEDS
    parts = args.parts.upper()

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.data_root,   exist_ok=True)

    best_partA_state = None

    # ─────────────────────────────────────────
    # Part A — Few-Shot Classification
    # ─────────────────────────────────────────
    if "A" in parts:
        from classifier import run_part_a
        results_a, best_partA_state = run_part_a(
            data_root=args.data_root,
            figures_dir=args.figures_dir,
            ckpt_dir=args.ckpt_dir,
            seeds=seeds,
        )
    else:
        # Try to load from disk
        import torch
        ckpt = os.path.join(args.ckpt_dir, "best_partA_finetuned.pt")
        if os.path.exists(ckpt):
            best_partA_state = torch.load(ckpt, map_location="cpu")
            print(f"[main] Loaded Part A checkpoint from {ckpt}")

    # ─────────────────────────────────────────
    # Part B — Neural Style Transfer
    # ─────────────────────────────────────────
    if "B" in parts:
        from style_transfer import generate_synthetic_images
        per_class = 5 if args.fast else 30    # 5 for quick test
        generate_synthetic_images(
            out_dir=args.synth_dir,
            figures_dir=args.figures_dir,
            data_root=args.data_root,
            per_class=per_class,
        )

    # ─────────────────────────────────────────
    # Part C — Domain Adaptation
    # ─────────────────────────────────────────
    if "C" in parts:
        from domain_adaptation import run_part_c
        results_c = run_part_c(
            best_partA_state=best_partA_state,
            data_root=args.data_root,
            synthetic_dir=args.synth_dir,
            figures_dir=args.figures_dir,
            ckpt_dir=args.ckpt_dir,
            seeds=seeds,
            run_dann=not args.no_dann,
        )

    print("\n" + "="*60)
    print("Challenge 7 complete.")
    print(f"  Figures   → {args.figures_dir}/")
    print(f"  Checkpoints → {args.ckpt_dir}/")
    print(f"  Synthetic images → {args.synth_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
