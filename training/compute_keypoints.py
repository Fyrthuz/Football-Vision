"""
Compute accurate keypoint positions for the field template based on
standard football pitch proportions mapped to 422×288 pixel space.

Run after adjusting margins/scale to regenerate sample.json:
    python -m training.compute_keypoints

Then regenerate field images:
    python -m training.field_template
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_JSON = REPO_ROOT / "sample.json"

# Real-world keypoint coordinates (cm) from Roboflow football-field-detection
# Field: 12000 x 7000 cm
KEYPOINTS_REAL: dict[str, tuple[float, float]] = {
    "1":  (0, 0),
    "2":  (0, 1450),
    "3":  (0, 2584),
    "4":  (0, 4416),
    "5":  (0, 5550),
    "6":  (0, 7000),
    "7":  (550, 2584),
    "8":  (550, 4416),
    "9":  (1100, 3500),
    "10": (2015, 1450),
    "11": (2015, 2584),
    "12": (2015, 4416),
    "13": (2015, 5550),
    "14": (6000, 0),
    "15": (6000, 2585),
    "16": (6000, 4415),
    "17": (6000, 7000),
    "18": (9985, 1450),
    "19": (9985, 2584),
    "20": (9985, 4416),
    "21": (9985, 5550),
    "22": (10900, 3500),
    "23": (11450, 2584),
    "24": (11450, 4416),
    "25": (12000, 0),
    "26": (12000, 1450),
    "27": (12000, 2584),
    "28": (12000, 4416),
    "29": (12000, 5550),
    "30": (12000, 7000),
    "31": (5085, 3500),
    "32": (6915, 3500),
}

# Canvas dimensions
CANVAS_W = 422
CANVAS_H = 288


def compute_template_keypoints(
    left_margin: int = 17,
    top_margin: int = 10,
    right_margin: int = 409,
    bottom_margin: int = 279,
) -> dict[str, tuple[int, int]]:
    """Map real-world cm coordinates to pixel space."""
    field_w = right_margin - left_margin   # 392
    field_h = bottom_margin - top_margin   # 269
    real_w = 12000.0
    real_h = 7000.0

    kps: dict[str, tuple[int, int]] = {}
    for idx, (rx, ry) in KEYPOINTS_REAL.items():
        px = left_margin + int(round(rx / real_w * field_w))
        py = top_margin + int(round(ry / real_h * field_h))
        kps[idx] = (px, py)
    return kps


def save_keypoints(kps: dict, path: Path) -> None:
    data = {"keypoints": {k: list(v) for k, v in kps.items()}, "height": CANVAS_H, "width": CANVAS_W}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Saved {len(kps)} keypoints to {path}")


def print_comparison(new_kps: dict) -> None:
    """Compare with current sample.json values."""
    try:
        with open(SAMPLE_JSON) as f:
            cur = json.load(f)["keypoints"]
    except (FileNotFoundError, KeyError):
        cur = {}

    print(f"{'KP':>4} {'Current':>12} {'Computed':>12} {'Diff':>8}")
    print("-" * 40)
    for idx in sorted(new_kps.keys(), key=int):
        c = tuple(cur.get(idx, (0, 0)))
        n = new_kps[idx]
        d = ((c[0] - n[0]) ** 2 + (c[1] - n[1]) ** 2) ** 0.5
        print(f"  {idx:>2}  ({c[0]:>3},{c[1]:>3})  ({n[0]:>3},{n[1]:>3})  {d:>5.1f}px")


def main():
    kps = compute_template_keypoints()
    print_comparison(kps)
    print()
    save_keypoints(kps, SAMPLE_JSON)


if __name__ == "__main__":
    main()
