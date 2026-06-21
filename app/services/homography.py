from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.schemas import Detection, DetectionLabel, Keypoint, ProjectedPosition

_HOMOGRAPHY_SMOOTHING_WINDOW = 5


def _load_field_keypoints(path: str | Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    field_kps = {}
    for k, v in data["keypoints"].items():
        field_kps[k] = (int(v[0]), int(v[1]))
    return field_kps


_field_keypoints_cache: dict | None = None


def get_field_keypoints() -> dict:
    global _field_keypoints_cache
    if _field_keypoints_cache is None:
        _field_keypoints_cache = _load_field_keypoints(settings.field_keypoints_path)
    return _field_keypoints_cache


def load_field_image(path: str | Path | None = None) -> np.ndarray | None:
    p = path or settings.field_image_path
    img = cv2.imread(str(p))
    return img


def _get_field_bounds() -> tuple[int, int]:
    kps = get_field_keypoints()
    xs = [v[0] for v in kps.values()]
    ys = [v[1] for v in kps.values()]
    return max(xs) + 10, max(ys) + 10


class HomographyEstimator:
    def __init__(self, window_size: int = _HOMOGRAPHY_SMOOTHING_WINDOW, alpha: float = 0.3) -> None:
        self.window_size = window_size
        self.alpha = alpha
        self._history: deque[np.ndarray] = deque(maxlen=window_size)
        self._current: np.ndarray | None = None
        self._last_inlier_labels: set[str] = set()

    def update(
        self,
        detected_keypoints: list[Keypoint],
        field_keypoints: dict,
        conf_threshold: float = 0.95,
        min_matches: int = 4,
    ) -> np.ndarray | None:
        self._last_inlier_labels = set()
        kp_map = {}
        for kp in detected_keypoints:
            if kp.confidence >= conf_threshold:
                kp_map[str(kp.index + 1)] = (kp.x, kp.y)

        src_pts: list[tuple[float, float]] = []
        dst_pts: list[tuple[int, int]] = []
        matched_labels: list[str] = []

        for label, pos in field_keypoints.items():
            if label in kp_map:
                src_pts.append(kp_map[label])
                dst_pts.append(pos)
                matched_labels.append(label)

        if len(src_pts) < min_matches:
            return self._current

        H, mask = cv2.findHomography(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if H is None:
            return self._current

        if mask is not None:
            inlier_ratio = float(np.sum(mask)) / len(mask)
            if inlier_ratio < 0.60:
                return self._current
            self._last_inlier_labels = {
                matched_labels[i] for i in range(len(matched_labels)) if mask[i][0] == 1
            }

        # Filtro de Inercia: Rechazar cambios si la proyección del centro salta bruscamente
        if self._current is not None:
            pt_test = np.array([[[960.0, 540.0]]], dtype=np.float32)
            p_ant = cv2.perspectiveTransform(pt_test, self._current)[0][0]
            p_nue = cv2.perspectiveTransform(pt_test, H)[0][0]
            
            if np.linalg.norm(p_ant - p_nue) > 25.0:
                return self._current

        # EMA sobre la matriz H para evitar saltos frame a frame
        if self._current is not None:
            H = self.alpha * H + (1.0 - self.alpha) * self._current
        self._history.append(H)
        self._current = H 
        return self._current

    @property
    def current(self) -> np.ndarray | None:
        return self._current

    @property
    def inlier_labels(self) -> set[str]:
        return self._last_inlier_labels

    def reset(self) -> None:
        self._history.clear()
        self._current = None


def compute_homography(
    detected_keypoints: list[Keypoint],
    field_keypoints: dict,
    conf_threshold: float = 0.5,
    min_matches: int = 6,
) -> np.ndarray | None:
    kp_map = {}
    for kp in detected_keypoints:
        if kp.confidence >= conf_threshold:
            kp_map[str(kp.index + 1)] = (kp.x, kp.y)

    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[int, int]] = []

    for label, pos in field_keypoints.items():
        if label in kp_map:
            src_pts.append(kp_map[label])
            dst_pts.append(pos)

    if len(src_pts) < min_matches:
        return None

    H, mask = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if H is None:
        return None

    if mask is not None and float(np.sum(mask)) / len(mask) < 0.4:
        return None

    return H


def project_positions(
    detections: list[Detection],
    homography_matrix: np.ndarray,
) -> list[tuple[float, float, int | None, str, int | None]]:
    projected: list[tuple[float, float, int | None, str, int | None]] = []
    for det in detections:
        # CORRECCIÓN DE PARALAJE:
        # Si es el balón, usamos el centro. 
        # Si es un jugador, usamos la base (los pies) ya que es el punto en contacto con el plano del campo.
        cx = (det.bbox.x1 + det.bbox.x2) / 2
        if det.label == DetectionLabel.ball:
            cy = (det.bbox.y1 + det.bbox.y2) / 2
        else:
            cy = det.bbox.y2  # Base inferior de la bounding box
            
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, homography_matrix)
        fx, fy = warped[0][0]
        projected.append((float(fx), float(fy), det.team, det.label.value, det.tracking_id))
    return projected


def project_point(
    x: float, y: float, homography_matrix: np.ndarray
) -> tuple[float, float]:
    pt = np.array([[[x, y]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, homography_matrix)
    return (float(warped[0][0][0]), float(warped[0][0][1]))


class KeypointFilter:
    """Temporal filter for pitch keypoints to reduce jitter.

    Applies per-keypoint EMA smoothing and rejects inter-frame outliers.
    Missing keypoints are held at their last known position with low confidence.
    Stale entries (not seen for N frames) are forgotten so a re-detected
    keypoint after a camera cut is accepted immediately.
    """

    def __init__(self, alpha: float = 0.4, max_jump_px: float = 60.0, max_age: int = 15) -> None:
        self.alpha = alpha
        self.max_jump_px = max_jump_px
        self.max_age = max_age
        self._smoothed: dict[str, tuple[float, float]] = {}
        self._age: dict[str, int] = {}
        self._frame = 0

    def update(self, keypoints: list[Keypoint]) -> list[Keypoint]:
        self._frame += 1
        current: dict[str, tuple[float, float]] = {}
        for kp in keypoints:
            current[str(kp.index + 1)] = (kp.x, kp.y)

        out: list[Keypoint] = []
        seen: set[str] = set()

        for label, pos in current.items():
            seen.add(label)
            self._age[label] = self._frame
            if label in self._smoothed:
                px, py = self._smoothed[label]
                dist = np.hypot(pos[0] - px, pos[1] - py)
                if dist > self.max_jump_px:
                    pos = (px, py)
                    conf = 0.3
                else:
                    pos = (
                        self.alpha * pos[0] + (1.0 - self.alpha) * px,
                        self.alpha * pos[1] + (1.0 - self.alpha) * py,
                    )
                    conf = 1.0
            else:
                conf = 1.0
            self._smoothed[label] = pos
            out.append(Keypoint(index=int(label) - 1, x=pos[0], y=pos[1], confidence=conf))

        # Hold missing keypoints at last known position (low confidence),
        # but only if they haven't gone stale
        stale_labels = [
            label for label, last_seen in self._age.items()
            if self._frame - last_seen > self.max_age
        ]
        for label in stale_labels:
            self._smoothed.pop(label, None)
            self._age.pop(label, None)

        for label, pos in self._smoothed.items():
            if label not in seen:
                out.append(Keypoint(index=int(label) - 1, x=pos[0], y=pos[1], confidence=0.1))

        return out

    def reset(self) -> None:
        self._smoothed.clear()
        self._age.clear()
        self._frame = 0


def make_projected_positions(
    projected_raw: list[tuple[float, float, int | None, str, int | None]],
) -> list[ProjectedPosition]:
    return [
        ProjectedPosition(x=fx, y=fy, team=team, label=label, tracking_id=tid)
        for fx, fy, team, label, tid in projected_raw
    ]


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    team_1_color: list[float] | None,
    team_2_color: list[float] | None,
    keypoints: list[Keypoint] | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is not None:
        np.copyto(out, frame)
        annotated = out
    else:
        annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        if det.label == DetectionLabel.ball:
            color = (0, 255, 0)
        elif det.team is not None and team_1_color is not None and team_2_color is not None:
            tc = team_1_color if det.team == 0 else team_2_color
            color = tuple(int(c) for c in tc)
        else:
            color = (255, 0, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_parts = [f"{det.label.value}", f"{det.confidence:.2f}"]
        if det.tracking_id is not None:
            label_parts.insert(0, f"#{det.tracking_id}")
        label_text = " ".join(label_parts)
        cv2.putText(annotated, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if keypoints:
        for kp in keypoints:
            if kp.confidence < 0.5:
                continue
            center = (int(kp.x), int(kp.y))
            cv2.circle(annotated, center, 4, (0, 255, 255), -1)
            cv2.putText(
                annotated, str(kp.index + 1),
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
            )
    return annotated


def create_field_overlay(
    field_image: np.ndarray,
    projected: list[tuple[float, float, int | None, str, int | None]],
    team_1_color: list[float] | None,
    team_2_color: list[float] | None,
    active_kp_labels: set[str] | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is not None:
        np.copyto(out, field_image)
        overlay = out
    else:
        overlay = field_image.copy()

    # Draw field template keypoints as reference grid
    field_kps = get_field_keypoints()
    for label, (fx, fy) in field_kps.items():
        is_active = active_kp_labels is not None and label in active_kp_labels
        color = (0, 255, 255) if is_active else (60, 60, 60)
        cv2.circle(overlay, (fx, fy), 3, color, -1 if is_active else 1)
        cv2.putText(
            overlay, label, (fx + 4, fy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
        )

    for fx, fy, team, label, tid in projected:
        if team is not None and team_1_color is not None and team_2_color is not None:
            color = team_1_color if team == 0 else team_2_color
        elif label == "ball":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.circle(overlay, (int(fx), int(fy)), 5, tuple(int(c) for c in color), -1)
        if tid is not None:
            cv2.putText(
                overlay, f"#{tid}", (int(fx) + 6, int(fy) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
    return overlay