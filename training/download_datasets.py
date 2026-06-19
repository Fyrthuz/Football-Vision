"""
Download Roboflow datasets for training.

Usage:
    python -m training.download_datasets --api-key YOUR_KEY

You need a free Roboflow API key from https://roboflow.com
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from roboflow import Roboflow

ROBOFLOW_WORKSPACE = "roboflow-jvuqo"

DATASETS = {
    "players": {
        "project": "football-players-detection-3zvbc",
        "version": 14,
        "output": Path("datasets/players"),
    },
    "keypoints": {
        "project": "football-field-detection-f07vi",
        "version": 15,
        "output": Path("datasets/keypoints"),
    },
}


def download_dataset(api_key: str, name: str, config: dict) -> Path:
    print(f"\nDownloading {name} dataset...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(config["project"])
    version = project.version(config["version"])

    output_dir = config["output"]

    # If the dir exists but has no images (stale from a previous failed download),
    # remove it so Roboflow's unzip doesn't conflict with stale metadata.
    downloaded = False
    if output_dir.exists():
        if any(output_dir.rglob("*.jpg")):
            print(f"  Dataset already has images at {output_dir}, skipping.")
        else:
            print(f"  Removing stale directory {output_dir} (no images found) ...")
            import shutil
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            version.download("yolov8", location=str(output_dir), overwrite=True)
            downloaded = True
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        version.download("yolov8", location=str(output_dir), overwrite=True)
        downloaded = True

    # Overwrite data.yaml with correct relative paths (Roboflow often writes wrong ones).
    if name == "players":
        _write_players_yaml(output_dir)
    elif name == "keypoints":
        _write_keypoints_yaml(output_dir)

    if downloaded:
        print(f"  -> {output_dir}")
    return output_dir


def _write_players_yaml(data_dir: Path) -> None:
    classes = ["ball", "goalkeeper", "player", "referee"]
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  Wrote {yaml_path}")


def _write_keypoints_yaml(data_dir: Path) -> None:
    classes = ["pitch"]
    kpt_shape = [32, 3]
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names: {classes}
kpt_shape: {kpt_shape}
"""
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  Wrote {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow datasets")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key (get one free at https://roboflow.com)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Datasets to download (default: all)",
    )
    args = parser.parse_args()

    for name in args.datasets:
        download_dataset(args.api_key, name, DATASETS[name])

    print("\nAll datasets downloaded!")


if __name__ == "__main__":
    main()
