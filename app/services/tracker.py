from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from app.schemas import Detection, DetectionLabel, PlayerStats, ProjectedPosition

_BALL_PROXIMITY_THRESHOLD_PX = 50.0
_HEATMAP_HISTORY = 300
_PIXELS_TO_METERS = 0.05


class _PlayerState:
    def __init__(self, tracking_id: int, label: DetectionLabel, team: int | None = None) -> None:
        self.tracking_id = tracking_id
        self.label = label
        self.team = team
        self.positions: list[tuple[float, float]] = []
        self.field_positions: list[tuple[float, float]] = []
        self.timestamps: list[float] = []
        self.total_distance = 0.0
        self.top_speed = 0.0
        self.touches = 0
        self.last_touch_frame = -10
        self.last_position: tuple[float, float] | None = None
        self.last_seen_frame = 0


class PlayerTracker:
    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self.dt = 1.0 / fps
        self._players: dict[int, _PlayerState] = {}
        self._ball_positions: list[tuple[float, float]] = []
        self._ball_field_positions: list[tuple[float, float]] = []
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def update(
        self,
        detections: list[Detection],
        projected_positions: list[ProjectedPosition],
        ball_center: tuple[float, float] | None = None,
    ) -> None:
        self._frame_count += 1

        # Ensure all tracked players exist in the state dict,
        # even when projected_positions is empty (homography failed).
        for det in detections:
            tid = det.tracking_id
            if tid is None:
                continue
            if tid not in self._players:
                self._players[tid] = _PlayerState(tid, det.label, det.team)
            self._players[tid].last_seen_frame = self._frame_count

        detected_ids: set[int] = set()
        for det, proj in zip(detections, projected_positions):
            tid = det.tracking_id
            if tid is None:
                continue
            detected_ids.add(tid)
            cx = (det.bbox.x1 + det.bbox.x2) / 2.0
            cy = (det.bbox.y1 + det.bbox.y2) / 2.0

            state = self._players[tid]
            state.label = det.label
            state.team = det.team

            if state.last_position is not None:
                dx = cx - state.last_position[0]
                dy = cy - state.last_position[1]
                dist_px = np.hypot(dx, dy)
                dist_m = dist_px * _PIXELS_TO_METERS
                state.total_distance += dist_m

                speed = dist_m / self.dt
                if speed > state.top_speed:
                    state.top_speed = speed

            state.last_position = (cx, cy)
            state.positions.append((cx, cy))
            if len(state.positions) > _HEATMAP_HISTORY:
                state.positions.pop(0)

            fx, fy = proj.x, proj.y
            state.field_positions.append((fx, fy))
            if len(state.field_positions) > _HEATMAP_HISTORY:
                state.field_positions.pop(0)

        if ball_center is not None:
            self._ball_positions.append(ball_center)
            if len(self._ball_positions) > _HEATMAP_HISTORY:
                self._ball_positions.pop(0)

            for det in detections:
                if det.tracking_id is None:
                    continue
                if det.label not in (DetectionLabel.player, DetectionLabel.goalkeeper):
                    continue
                bx, by = ball_center
                cx = (det.bbox.x1 + det.bbox.x2) / 2.0
                cy = det.bbox.y2
                dist = np.hypot(cx - bx, cy - by)
                if dist < _BALL_PROXIMITY_THRESHOLD_PX:
                    state = self._players[det.tracking_id]
                    if self._frame_count - state.last_touch_frame > 5:
                        state.touches += 1
                        state.last_touch_frame = self._frame_count

        stale_ids = [pid for pid in self._players if pid not in detected_ids]
        for pid in stale_ids:
            state = self._players[pid]
            if self._frame_count - state.last_seen_frame > 30:
                del self._players[pid]

    def get_stats(self) -> list[PlayerStats]:
        stats: list[PlayerStats] = []
        for state in self._players.values():
            elapsed = self._frame_count / self.fps if self._frame_count > 0 else 0.0
            avg_speed = state.total_distance / elapsed if elapsed > 0 else 0.0
            stats.append(
                PlayerStats(
                    tracking_id=state.tracking_id,
                    label=state.label,
                    team=state.team,
                    total_distance=round(state.total_distance, 2),
                    avg_speed=round(avg_speed, 2),
                    top_speed=round(state.top_speed, 2),
                    touches=state.touches,
                    heatmap_positions=list(state.field_positions),
                )
            )
        return stats

    def get_player_possession(self) -> dict[int, float]:
        if not self._players:
            return {}
        total_touches = sum(s.touches for s in self._players.values())
        if total_touches == 0:
            return {}
        return {pid: round(s.touches / total_touches * 100, 1) for pid, s in self._players.items()}

    def get_team_possession(self) -> dict[int, float]:
        team_touches: dict[int, int] = {}
        for s in self._players.values():
            if s.team is not None:
                team_touches[s.team] = team_touches.get(s.team, 0) + s.touches
        total = sum(team_touches.values())
        if total == 0:
            return {}
        return {t: round(c / total * 100, 1) for t, c in team_touches.items()}

    def reset(self) -> None:
        self._players.clear()
        self._ball_positions.clear()
        self._ball_field_positions.clear()
        self._frame_count = 0


class BallTracker:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self.positions: list[tuple[int, int] | None] = []

    def update(self, center: tuple[int, int] | None) -> tuple[int, int] | None:
        self.positions.append(center)
        if len(self.positions) > self.window_size:
            self.positions.pop(0)

        if center is not None:
            return center

        valid = [p for p in self.positions if p is not None]
        if not valid:
            return None

        return self._interpolate(valid)

    @staticmethod
    def _interpolate(positions: list[tuple[int, int]]) -> tuple[int, int]:
        arr = np.array(positions)
        mean_x = int(np.mean(arr[:, 0]))
        mean_y = int(np.mean(arr[:, 1]))
        return (mean_x, mean_y)

    def reset(self) -> None:
        self.positions.clear()


def draw_ball_triangle(
    frame: np.ndarray, center: tuple[int, int], color: tuple[int, int, int] = (0, 255, 0)
) -> None:
    offset_y = 30
    points = np.array(
        [
            [center[0], center[1] + 20 - offset_y],
            [center[0] - 10, center[1] - 10 - offset_y],
            [center[0] + 10, center[1] - 10 - offset_y],
        ],
        dtype=np.int32,
    )
    cv2.drawContours(frame, [points], 0, color, -1)
