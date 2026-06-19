"""
Generate the football field template image (football_field.png).

Draws all lines based on the 32 keypoints in sample.json so pitch
markings align exactly with the homography template.  Includes:
boundary, halfway line, centre circle + spot, penalty areas,
goal areas, penalty spots, penalty arcs ("D"), goals with net
hatching, and corner arcs.

Also generates football_field_keypoints.png — the field image with
all 32 numbered keypoints overlaid (for reference).

Usage:
    python -m training.field_template
"""

from __future__ import annotations

import json
from math import atan2, cos, pi, sin
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_JSON = REPO_ROOT / "sample.json"


def _load_kps(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: (int(v[0]), int(v[1])) for k, v in data["keypoints"].items()}


def _draw_penalty_arc(img, center, r, x_line, side, color, thickness=1):
    """Draw a penalty-arc polyline.

    side='right' — arc protrudes RIGHT of x_line (left penalty area)
    side='left'  — arc protrudes LEFT  of x_line (right penalty area)
    """
    cx, cy = center
    dx = x_line - cx
    if abs(dx) >= r:
        return
    dy = (r ** 2 - dx ** 2) ** 0.5

    if side == "right":
        a_start = atan2(-dy, dx)   # top intersection
        a_end = atan2(dy, dx)      # bottom intersection
    else:
        a_start = atan2(dy, dx)    # bottom intersection
        a_end = atan2(-dy, dx)     # top intersection

    if a_end <= a_start:
        a_end += 2 * pi

    pts = []
    n = 60
    for i in range(n + 1):
        theta = a_start + (a_end - a_start) * i / n
        x = int(cx + r * cos(theta))
        y = int(cy + r * sin(theta))
        pts.append([x, y])
    cv2.polylines(img, [np.array(pts)], False, color, thickness)


def generate_field_image(
    width: int = 422,
    height: int = 288,
    output_path: str | Path | None = None,
    kp_overlay_path: str | Path | None = None,
) -> np.ndarray:
    kps = _load_kps(SAMPLE_JSON)

    # --- Grass background ---
    grass = (50, 120, 50)
    img = np.full((height, width, 3), grass, dtype=np.uint8)

    # Field boundaries from keypoints
    left = kps["1"][0]
    right = kps["25"][0]
    top = kps["1"][1]
    bottom = kps["6"][1]
    field_w = right - left

    # --- Alternating grass stripes ---
    stripe_w = field_w // 12
    for i in range(12):
        if i % 2 == 0:
            x1 = left + i * stripe_w
            x2 = min(left + (i + 1) * stripe_w, right)
            overlay = img[top:bottom, x1:x2]
            lighter = cv2.addWeighted(overlay, 0.85, np.full_like(overlay, 15), 0.15, 0)
            img[top:bottom, x1:x2] = lighter

    line = (220, 220, 220)
    thick = 1

    # --- Pitch boundary ---
    cv2.rectangle(img, (left, top), (right, bottom), line, thick)

    # --- Halfway line ---
    cx = kps["14"][0]
    cv2.line(img, (cx, top), (cx, bottom), line, thick)

    # --- Centre circle ---
    centre = (cx, (kps["14"][1] + kps["17"][1]) // 2)
    r_circle = int(np.sqrt(
        (kps["31"][0] - centre[0]) ** 2 + (kps["31"][1] - centre[1]) ** 2
    ))
    cv2.circle(img, centre, r_circle, line, thick)
    cv2.circle(img, centre, 2, line, -1)  # centre spot

    # --- Left penalty area ---
    cv2.rectangle(
        img,
        (kps["2"][0], kps["2"][1]),
        (kps["10"][0], kps["13"][1]),
        line, thick,
    )

    # --- Right penalty area ---
    cv2.rectangle(
        img,
        (kps["18"][0], kps["18"][1]),
        (kps["25"][0], kps["29"][1]),
        line, thick,
    )

    # --- Left goal area ---
    cv2.rectangle(
        img,
        (kps["3"][0], kps["3"][1]),
        (kps["7"][0], kps["8"][1]),
        line, thick,
    )

    # --- Right goal area ---
    cv2.rectangle(
        img,
        (kps["23"][0], kps["23"][1]),
        (kps["27"][0], kps["28"][1]),
        line, thick,
    )

    # --- Penalty spots ---
    cv2.circle(img, kps["9"], 2, line, -1)
    cv2.circle(img, kps["22"], 2, line, -1)

    # --- Penalty arcs (the "D"), drawn as polylines ---
    _draw_penalty_arc(img, kps["9"], r_circle, kps["10"][0], "right", line, thick)
    _draw_penalty_arc(img, kps["22"], r_circle, kps["18"][0], "left", line, thick)

    # --- Goals ---
    goal_depth = left // 3
    g_top = kps["3"][1]
    g_bot = kps["4"][1]

    # Left goal
    cv2.rectangle(img, (left - goal_depth, g_top), (left, g_bot), line, thick)
    for y in range(g_top + 4, g_bot, 6):
        cv2.line(img, (left - goal_depth + 2, y), (left - 2, y), line, 1)
    for x in range(left - goal_depth + 4, left, 6):
        cv2.line(img, (x, g_top + 2), (x, g_bot - 2), line, 1)

    # Right goal
    goal_depth_r = (width - 1 - right) // 3
    cv2.rectangle(img, (right, g_top), (right + goal_depth_r, g_bot), line, thick)
    for y in range(g_top + 4, g_bot, 6):
        cv2.line(img, (right + 2, y), (right + goal_depth_r - 2, y), line, 1)
    for x in range(right + 4, right + goal_depth_r, 6):
        cv2.line(img, (x, g_top + 2), (x, g_bot - 2), line, 1)

    # --- Corner arcs ---
    for corner in [(left, top), (right, top), (left, bottom), (right, bottom)]:
        cv2.ellipse(img, corner, (6, 6), 0, 0, 90, line, thick)

    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"Field template saved to {output_path}")

    # --- Keypoint overlay image (for reference) ---
    if kp_overlay_path:
        kp_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        for idx_str, (x, y) in kps.items():
            is_left = int(idx_str) <= 13 or int(idx_str) == 31
            color = (50, 220, 50) if is_left else (50, 180, 255)
            cv2.circle(kp_img, (x, y), 4, color, -1)
            cv2.circle(kp_img, (x, y), 4, (255, 255, 255), 1)
            cv2.putText(
                kp_img, idx_str, (x + 6, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.imwrite(str(kp_overlay_path), cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR))
        print(f"Keypoints overlay saved to {kp_overlay_path}")

    return img


def main() -> None:
    root = REPO_ROOT
    generate_field_image(
        output_path=root / "football_field.png",
        kp_overlay_path=root / "football_field_keypoints.png",
    )


if __name__ == "__main__":
    main()
