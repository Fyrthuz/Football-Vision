from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

KPT_SIZE = 0.01
KEYPOINT_NAMES = [f"kp{i+1}" for i in range(32)]


def convert_dataset(src_base: Path, dst_base: Path) -> None:
    dst_base.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        src_img = src_base / split / "images"
        src_lbl = src_base / split / "labels"
        if not src_img.exists():
            continue
        dst_split = dst_base / split
        (dst_split / "images").mkdir(parents=True, exist_ok=True)
        (dst_split / "labels").mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_img.iterdir()):
            shutil.copy2(img_path, dst_split / "images" / img_path.name)

        for lbl_path in sorted(src_lbl.glob("*.txt")):
            lines = lbl_path.read_text().strip().splitlines()
            out_lines: list[str] = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                kpts = np.array(parts[5:], dtype=np.float32).reshape(-1, 3)
                for kp_idx, (kx, ky, kv) in enumerate(kpts):
                    if kv == 0:
                        continue
                    hs = KPT_SIZE
                    cx = float(kx)
                    cy = float(ky)
                    x1 = max(0.0, cx - hs)
                    y1 = max(0.0, cy - hs)
                    x2 = min(1.0, cx + hs)
                    y2 = min(1.0, cy + hs)
                    bw = x2 - x1
                    bh = y2 - y1
                    nx = x1 + bw / 2.0
                    ny = y1 + bh / 2.0
                    out_lines.append(f"{kp_idx} {nx:.6f} {ny:.6f} {bw:.6f} {bh:.6f}")
            out_path = dst_split / "labels" / lbl_path.name
            out_path.write_text("\n".join(out_lines))

        n_imgs = len(list((dst_split / "images").iterdir()))
        n_lbls = len(list((dst_split / "labels").iterdir()))
        print(f"  {split}: {n_imgs} images, {n_lbls} labels")

    yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: 32
names: {KEYPOINT_NAMES}
"""
    (dst_base / "data.yaml").write_text(yaml_content)
    print(f"  Wrote {dst_base / 'data.yaml'}")


def main() -> None:
    src = Path("datasets/keypoints")
    dst = Path("keypoints_detection")
    convert_dataset(src, dst)


if __name__ == "__main__":
    main()
