"""
Train YOLOv8 player detection model.

Downloads the dataset from Roboflow if not present, then trains.
Usage:
    python -m training.train_detection --api-key YOUR_KEY

The ROBOFLOW_API_KEY environment variable is also read automatically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "players"


def ensure_dataset(api_key: str) -> None:
    from training.download_datasets import download_dataset

    if DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.jpg")):
        print(f"Dataset found at {DATASET_DIR}")
    else:
        print("Downloading player detection dataset from Roboflow...")
        download_dataset(
            api_key,
            "players",
            {
                "project": "football-players-detection-3zvbc",
                "version": 14,
                "output": DATASET_DIR,
            },
        )

    # Always ensure data.yaml exists with correct paths
    from training.download_datasets import _write_players_yaml
    _write_players_yaml(DATASET_DIR)


def train() -> None:
    import torch

    config_path = REPO_ROOT / "training" / "configs" / "detection.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()} — using device '{device}'")

    model = YOLO("yolov8x.pt")
    results = model.train(cfg=str(config_path), device=device)
    print(f"Training complete. Best model saved to: {results.save_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train player detection model")
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: No Roboflow API key found.")
        print("Set ROBOFLOW_API_KEY in .env or pass --api-key")
        sys.exit(1)

    ensure_dataset(api_key)
    train()


if __name__ == "__main__":
    main()
