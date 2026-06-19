"""
Full training pipeline: download datasets, train both models, copy weights.

Usage:
    python -m training.pipeline --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env)")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip dataset download"
    )
    parser.add_argument(
        "--skip-detection", action="store_true", help="Skip player detection training"
    )
    parser.add_argument(
        "--skip-keypoints", action="store_true", help="Skip keypoint training"
    )
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Error: No Roboflow API key found.")
        print("Set ROBOFLOW_API_KEY in .env or pass --api-key")
        sys.exit(1)
    return key


def download_datasets(api_key: str) -> None:
    from training.download_datasets import DATASETS, download_dataset

    for name in DATASETS:
        download_dataset(api_key, name, DATASETS[name])


def train_detection() -> Path:
    from ultralytics import YOLO

    import torch

    config_path = REPO_ROOT / "training" / "configs" / "detection.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training player detection (device={device})...")
    print(f"{'='*60}")

    model = YOLO("yolov8n.pt")
    results = model.train(cfg=str(config_path), device=device)
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"Detection training complete. Best model: {best_pt}")
    return best_pt


def train_keypoints() -> Path:
    from ultralytics import YOLO

    import torch

    config_path = REPO_ROOT / "training" / "configs" / "keypoints.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training pitch keypoints (device={device})...")
    print(f"{'='*60}")

    model = YOLO("yolov8n-pose.pt")
    results = model.train(cfg=str(config_path), device=device)
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"Keypoint training complete. Best model: {best_pt}")
    return best_pt


def copy_models(detection_best: Path, keypoints_best: Path) -> None:
    models_dir = REPO_ROOT / "models"

    dst_detection = models_dir / "best_model_players.pt"
    dst_keypoints = models_dir / "best_model_keypoints.pt"

    shutil.copy2(detection_best, dst_detection)
    print(f"  Copied detection model: {detection_best} -> {dst_detection}")

    shutil.copy2(keypoints_best, dst_keypoints)
    print(f"  Copied keypoint model:  {keypoints_best} -> {dst_keypoints}")


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args)

    if not args.skip_download:
        print("\n=== Step 1: Download datasets ===")
        download_datasets(api_key)
    else:
        print("\n=== Step 1: Skipping download ===")

    detection_best = None
    keypoints_best = None

    if not args.skip_detection:
        print("\n=== Step 2: Train player detection ===")
        detection_best = train_detection()
    else:
        print("\n=== Step 2: Skipping detection training ===")

    if not args.skip_keypoints:
        print("\n=== Step 3: Train pitch keypoints ===")
        keypoints_best = train_keypoints()
    else:
        print("\n=== Step 3: Skipping keypoint training ===")

    if detection_best and keypoints_best:
        print("\n=== Step 4: Copy models ===")
        copy_models(detection_best, keypoints_best)

    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    main()
