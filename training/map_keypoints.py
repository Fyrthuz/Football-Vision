"""
Interactive keypoint mapping tool for the football field template.

Displays the 422×288 field image with all 32 keypoints overlaid,
letting you verify, reposition, and save the mapping in sample.json.

Modes:
  python -m training.map_keypoints              ← interactive edit
  python -m training.map_keypoints --image <path>  ← use a custom image
  python -m training.map_keypoints --help          ← full help

Controls (interactive mode):
  Left-click near a keypoint  → select it (yellow highlight)
  Left-click on empty area    → move selected keypoint there
  Right-click on empty area   → place a new point for the current index
  Mouse-wheel / N / P         → cycle selected keypoint index
  S                           → save to sample.json
  R                           → reset selected keypoint to default
  D                           → deselect
  ESC / Q                     → quit (prompts save if dirty)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = REPO_ROOT / "sample.json"
DEFAULT_IMAGE = REPO_ROOT / "football_field.png"

# Semantic labels for each keypoint (0-indexed → 1-indexed in file)
KP_LABELS: dict[str, str] = {
    "1":  "Top-left corner",
    "2":  "Left touchline, top of penalty box",
    "3":  "Left touchline, top of goal box",
    "4":  "Left touchline, bottom of goal box",
    "5":  "Left touchline, bottom of penalty box",
    "6":  "Bottom-left corner",
    "7":  "Left goal box, front top",
    "8":  "Left goal box, front bottom",
    "9":  "Left penalty spot",
    "10": "Left penalty area, top-left",
    "11": "Left penalty area, goal-box top",
    "12": "Left penalty area, goal-box bottom",
    "13": "Left penalty area, bottom-left",
    "14": "Halfway line, top touchline",
    "15": "Halfway line, top of centre circle",
    "16": "Halfway line, bottom of centre circle",
    "17": "Halfway line, bottom touchline",
    "18": "Right penalty area, top-right",
    "19": "Right penalty area, goal-box top",
    "20": "Right penalty area, goal-box bottom",
    "21": "Right penalty area, bottom-right",
    "22": "Right penalty spot",
    "23": "Right goal box, front top",
    "24": "Right goal box, front bottom",
    "25": "Top-right corner",
    "26": "Right touchline, top of penalty box",
    "27": "Right touchline, top of goal box",
    "28": "Right touchline, bottom of goal box",
    "29": "Right touchline, bottom of penalty box",
    "30": "Bottom-right corner",
    "31": "Centre circle, left edge",
    "32": "Centre circle, right edge",
}

SELECT_RADIUS = 12


def load_keypoints(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: (int(v[0]), int(v[1])) for k, v in data["keypoints"].items()}


def save_keypoints(kps: dict, path: Path) -> None:
    data = {"keypoints": {k: list(v) for k, v in kps.items()}, "height": 288, "width": 422}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Saved {len(kps)} keypoints to {path}")


def draw_field(
    img: np.ndarray,
    kps: dict,
    selected: str | None = None,
) -> np.ndarray:
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img.copy()
    h, w = disp.shape[:2]

    for idx_str, (x, y) in kps.items():
        if not (0 <= x < w and 0 <= y < h):
            continue
        is_sel = idx_str == selected
        is_left = int(idx_str) <= 13 or int(idx_str) == 31
        color = (50, 220, 50) if is_left else (50, 180, 255)  # green=left, orange=right
        if is_sel:
            color = (50, 230, 255)  # yellow
            cv2.circle(disp, (x, y), SELECT_RADIUS + 2, (255, 255, 255), 1)
        cv2.circle(disp, (x, y), 4, color, -1)
        cv2.circle(disp, (x, y), 4, (255, 255, 255), 1)
        label = idx_str
        cv2.putText(
            disp, label, (x + 6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Legend
    legend_y = 14
    cv2.putText(disp, "LEFT", (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 220, 50), 1, cv2.LINE_AA)
    cx = w // 2
    cv2.putText(disp, "RIGHT", (cx + 8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 180, 255), 1, cv2.LINE_AA)
    return disp


def interactive_edit(img: np.ndarray, kps: dict) -> dict:
    """OpenCV interactive window to edit keypoints."""
    window = "Football Field Keypoints — ESC=exit  S=save  N/P=next/prev  R=reset"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 844, 576)

    selected: str | None = "1"
    dirty = False
    orig = {k: v for k, v in kps.items()}
    order = sorted(kps.keys(), key=int)

    def _redraw():
        disp = draw_field(img, kps, selected)
        # Bottom bar with instructions
        sel_idx = selected or "—"
        if selected:
            info = f"KP {sel_idx}: {kps[selected]}  —  {KP_LABELS.get(sel_idx, '')}"
        else:
            info = "Click a keypoint to select"
        cv2.putText(disp, info, (8, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(disp, "[S]ave  [R]eset  [N]/[P]  [D]eselect  ESC=quit",
                    (disp.shape[1] // 2, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(window, disp)

    _redraw()

    while True:
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        # ESC or Q
        if key in (27, ord("q"), ord("Q")):
            if dirty:
                print("\nUnsaved changes.", end=" ")
                k = input("Save before quitting? [Y/n]: ").strip().lower()
                if k != "n":
                    return kps
            break

        # S = save
        if key in (ord("s"), ord("S")):
            save_keypoints(kps, DEFAULT_SAMPLE)
            dirty = False
            orig = {k: v for k, v in kps.items()}
            _redraw()
            continue

        # R = reset
        if key in (ord("r"), ord("R")) and selected:
            kps[selected] = orig[selected]
            dirty = True
            _redraw()
            continue

        # D = deselect
        if key in (ord("d"), ord("D")):
            selected = None
            _redraw()
            continue

        # N = next, P = prev
        if key in (ord("n"), ord("N"), ord("p"), ord("P")):
            if not selected:
                selected = order[0]
            else:
                i = order.index(selected)
                if key in (ord("n"), ord("N")):
                    selected = order[(i + 1) % len(order)]
                else:
                    selected = order[(i - 1) % len(order)]
            _redraw()
            continue

        # Left mouse button (OpenCV: 1 = down)
        flags_btn = cv2.EVENT_FLAG_LBUTTON
        # We use a callback approach via mouse callback
        # Actually, let's handle mouse via setMouseCallback

    cv2.destroyWindow(window)
    return kps


def mouse_callback_factory(img, kps, orig):
    """Create a mouse callback closure for interactive editing."""
    state = {
        "selected": "1",
        "dirty": False,
        "order": sorted(kps.keys(), key=int),
        "img": img,
        "kps": kps,
        "orig": orig,
        "window": "Football Field Keypoints",
        "redraw": None,
    }

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is near an existing keypoint
            best_dist = SELECT_RADIUS
            best_kp = None
            for idx_str, (kx, ky) in state["kps"].items():
                d = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_kp = idx_str
            if best_kp is not None:
                state["selected"] = best_kp
            elif state["selected"] is not None:
                # Move selected keypoint
                state["kps"][state["selected"]] = (x, y)
                state["dirty"] = True
            if callable(state["redraw"]):
                state["redraw"]()

    return callback, state


def main():
    parser = argparse.ArgumentParser(
        description="Interactive keypoint mapping for the football field template.",
    )
    parser.add_argument(
        "--image", type=str, default=str(DEFAULT_IMAGE),
        help=f"Field image path (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--sample", type=str, default=str(DEFAULT_SAMPLE),
        help=f"Keypoint JSON path (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Just show the current mapping without editing",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    sample_path = Path(args.sample)

    if not img_path.exists():
        print(f"Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)
    if not sample_path.exists():
        print(f"Keypoint file not found: {sample_path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}", file=sys.stderr)
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kps = load_keypoints(sample_path)
    orig = {k: v for k, v in kps.items()}

    if args.visualize:
        disp = draw_field(img, kps)
        cv2.namedWindow("Football Field Keypoints [read-only]", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Football Field Keypoints [read-only]", 844, 576)
        cv2.imshow("Football Field Keypoints [read-only]", disp)
        print("Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("Interactive keypoint editor")
    print("===========================")
    print("Left-click near a keypoint   → select it")
    print("Left-click on empty area      → move selected keypoint")
    print("N / P                         → next / previous keypoint")
    print("S                             → save to sample.json")
    print("R                             → reset selected keypoint")
    print("D                             → deselect")
    print("ESC / Q                       → quit")
    print()

    window = "Football Field Keypoints — ESC=exit  S=save  N/P=next/prev  R=reset"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 844, 576)

    selected: str | None = "1"
    dirty = False
    order = sorted(kps.keys(), key=int)

    def redraw():
        nonlocal selected
        disp = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        # Draw keypoints
        for idx_str, (x, y) in kps.items():
            is_sel = idx_str == selected
            is_left = int(idx_str) <= 13 or int(idx_str) == 31
            color = (50, 220, 50) if is_left else (50, 180, 255)
            if is_sel:
                color = (50, 230, 255)
                cv2.circle(disp, (x, y), SELECT_RADIUS + 2, (255, 255, 255), 1)
            cv2.circle(disp, (x, y), 5, color, -1)
            cv2.circle(disp, (x, y), 5, (255, 255, 255), 1)
            label = idx_str
            cv2.putText(
                disp, label, (x + 7, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Legend
        cv2.putText(disp, "LEFT SIDE", (8, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 220, 50), 1, cv2.LINE_AA)
        cv2.putText(disp, "RIGHT SIDE", (220, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 180, 255), 1, cv2.LINE_AA)

        # Bottom info bar
        sel_idx = selected or "—"
        kp_x, kp_y = kps.get(selected, (0, 0)) if selected else (0, 0)
        cx_line = kps["14"][0]
        info = f"KP {sel_idx}"
        if selected:
            info += f"  ({kp_x}, {kp_y})  —  {KP_LABELS.get(selected, '')}"
            # Show symmetry hint
            order_list = sorted(kps.keys(), key=int)
            i = order_list.index(selected)
            mirror_idx = order_list[len(order_list) - 1 - i]
            if mirror_idx != selected:
                info += f"  |  Mirror: KP {mirror_idx}"
        else:
            info += "  —  Click a keypoint to select"
        cv2.putText(disp, info, (8, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(disp, "[S]ave  [R]eset  [N]/[P]  [D]eselect  ESC=quit",
                    (disp.shape[1] // 2 + 20, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(window, disp)

    def mouse_handler(event, x, y, flags, param):
        nonlocal selected, dirty
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is near an existing keypoint
            best_dist = SELECT_RADIUS
            best_kp = None
            for idx_str, (kx, ky) in kps.items():
                d = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_kp = idx_str
            if best_kp is not None:
                selected = best_kp
            elif selected is not None:
                # Move selected keypoint to click position
                kps[selected] = (x, y)
                dirty = True
            redraw()

    cv2.setMouseCallback(window, mouse_handler)
    redraw()

    while True:
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        # ESC or Q
        if key in (27, ord("q"), ord("Q")):
            if dirty:
                print("\nUnsaved changes.", end=" ")
                k = input("Save before quitting? [Y/n]: ").strip().lower()
                if k != "n":
                    save_keypoints(kps, sample_path)
            break

        # S = save
        if key in (ord("s"), ord("S")):
            save_keypoints(kps, sample_path)
            dirty = False
            orig.clear()
            orig.update(kps)
            redraw()
            continue

        # R = reset
        if key in (ord("r"), ord("R")) and selected:
            if selected in orig:
                kps[selected] = orig[selected]
                dirty = True
                redraw()
            continue

        # D = deselect
        if key in (ord("d"), ord("D")):
            selected = None
            redraw()
            continue

        # N = next, P = prev
        if key in (ord("n"), ord("N"), ord("p"), ord("P")):
            if not selected:
                selected = order[0]
            else:
                i = order.index(selected)
                if key in (ord("n"), ord("N")):
                    selected = order[(i + 1) % len(order)]
                else:
                    selected = order[(i - 1) % len(order)]
            redraw()
            continue

        # Left/Right arrow keys cycle as well
        if key in (65363,):  # Right arrow
            if not selected:
                selected = order[0]
            else:
                i = order.index(selected)
                selected = order[(i + 1) % len(order)]
            redraw()
            continue
        if key in (65361,):  # Left arrow
            if not selected:
                selected = order[0]
            else:
                i = order.index(selected)
                selected = order[(i - 1) % len(order)]
            redraw()
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
