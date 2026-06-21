from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.schemas import BBox, Detection, DetectionLabel, Keypoint


class Detector:
    _instance: Detector | None = None

    def __init__(self) -> None:
        self.player_model: YOLO | None = None
        self.keypoint_model: YOLO | None = None
        self._player_classes: dict[int, str] | None = None
        self._track_history: dict[int, list[tuple[float, float]]] = {}

    def load_models(self) -> None:
        self.player_model = YOLO(settings.model_players_path, task='detect')
        self.keypoint_model = YOLO(settings.model_keypoints_path, task='detect')
        self.keypoint_model.overrides['imgsz'] = 1280
        self._player_classes = self.player_model.names

    @property
    def models_loaded(self) -> dict[str, bool]:
        return {
            "players": self.player_model is not None,
            "keypoints": self.keypoint_model is not None,
        }

    def is_loaded(self) -> bool:
        return self.player_model is not None and self.keypoint_model is not None

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], list[Keypoint]]:
        if not self.is_loaded():
            self.load_models()

        player_results = self.player_model.predict(frame, verbose=False)[0]
        keypoint_results = self.keypoint_model.predict(frame, verbose=False)[0]

        detections = self._parse_player_detections(player_results)
        keypoints = self._parse_keypoints(keypoint_results)
        return detections, keypoints

    def track(
        self, frame: np.ndarray, persist: bool = True
    ) -> tuple[list[Detection], list[Keypoint]]:
        if not self.is_loaded():
            self.load_models()

        player_results = self.player_model.predict(frame, tracker='botsort.yaml', verbose=False)[0]
        keypoint_results = self.keypoint_model.predict(frame, verbose=False)[0]

        detections = self._parse_player_detections(player_results, parse_tracking=True)
        keypoints = self._parse_keypoints(keypoint_results)
        return detections, keypoints

    def track_players(
        self, frame: np.ndarray, persist: bool = True
    ) -> list[Detection]:
        if not self.is_loaded():
            self.load_models()

        player_results = self.player_model.predict(frame, tracker='botsort.yaml', verbose=False)[0]
        return self._parse_player_detections(player_results, parse_tracking=True)

    def _parse_player_detections(
        self, result, parse_tracking: bool = False
    ) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0].item())
            label_str = result.names[cls_id]
            try:
                label = DetectionLabel(label_str)
            except ValueError:
                continue

            tracking_id = None
            if parse_tracking and box.id is not None:
                tracking_id = int(box.id[0].item())

            detections.append(
                Detection(
                    bbox=BBox(
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)
                    ),
                    label=label,
                    confidence=round(box.conf[0].item(), 4),
                    tracking_id=tracking_id,
                )
            )
        return detections

    def _parse_keypoints(self, result) -> list[Keypoint]:
        keypoints: list[Keypoint] = []

        # Try pose keypoint format first
        if result.keypoints is not None and result.keypoints.data is not None:
            kp_data = result.keypoints.data[0]
            for i, kp in enumerate(kp_data):
                x, y, conf = kp
                keypoints.append(
                    Keypoint(
                        index=i,
                        x=float(x),
                        y=float(y),
                        confidence=float(conf),
                    )
                )
            return keypoints

        # Fallback: 32-class detection format
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id >= 32:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                conf = float(box.conf[0].item())
                keypoints.append(
                    Keypoint(index=cls_id, x=cx, y=cy, confidence=conf)
                )
        return keypoints
