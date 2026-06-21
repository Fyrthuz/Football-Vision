from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from app.config import RESULTS_DIR
from app.schemas import DetectionLabel, ProjectedPosition
from app.services.classifier import TeamClassifier, get_player_team_color
from app.services.detector import Detector
from app.services.homography import (
    HomographyEstimator,
    KeypointFilter,
    annotate_frame,
    create_field_overlay,
    get_field_keypoints,
    load_field_image,
    project_positions,
)
from app.services.tracker import BallTracker, PlayerTracker
from batch_processor.celery_app import celery_app

_BATCH_SIZE = 8

# Per-job colour cache (tracking_id -> LAB colour)
_player_color_cache: dict[int, list[float]] = {}
_player_color_last_frame: dict[int, int] = {}


class FfmpegWriter:
    def __init__(self, path: Path, fps: int, size: tuple[int, int]):
        detect = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5
        )
        encoder = "h264_nvenc" if "h264_nvenc" in detect.stdout else "libx264"
        enc_opts = ["-preset", "p1", "-cq", "23"] if encoder == "h264_nvenc" else ["-preset", "fast"]

        self.cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{size[0]}x{size[1]}",
            "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-c:v", encoder, *enc_opts,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE)
        self.queue: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.event = threading.Event()
        self._stop = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while not self._stop or self.queue:
            self.event.wait(timeout=0.1)
            self.event.clear()
            with self.lock:
                frames = self.queue
                self.queue = []
            for f in frames:
                self.proc.stdin.write(f.tobytes())

    def write(self, frame: np.ndarray) -> None:
        with self.lock:
            self.queue.append(frame)
            self.event.set()

    def release(self) -> None:
        self._stop = True
        self.event.set()
        self.thread.join(timeout=10)
        self.proc.stdin.close()
        self.proc.wait()


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, video_path: str, job_id: str, track_enabled: bool = True) -> dict:
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = result_dir / "output.mp4"
    output_json_path = result_dir / "data.json"

    detector = Detector()
    detector.load_models()

    # Warmup ONNX models (required before first track() call for dynamic batch)
    detector.player_model.predict(np.zeros((720, 1280, 3), dtype=np.uint8), verbose=False)
    detector.keypoint_model.predict(np.zeros((720, 1280, 3), dtype=np.uint8), verbose=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    field_height = 288
    out = FfmpegWriter(
        output_video_path,
        fps,
        (frame_width, frame_height + field_height),
    )

    field_keypoints = get_field_keypoints()
    field_image = load_field_image()
    if field_image is None:
        field_image = np.zeros((field_height, frame_width, 3), dtype=np.uint8)

    # Pre-allocate concat buffer (reused per frame, avoids allocation churn)
    concat = np.empty((frame_height + field_height, frame_width, 3), dtype=np.uint8)
    overlay_slice = concat[frame_height:]
    field_overlay_pad_x = (frame_width - field_image.shape[1]) // 2

    ball_tracker = BallTracker()
    homography_estimator = HomographyEstimator()
    player_tracker = PlayerTracker(fps=float(fps))
    team_clf = TeamClassifier()
    kp_filter = KeypointFilter()

    frames_jsonl_path = result_dir / "frames.jsonl"
    frames_file = open(frames_jsonl_path, "w")
    frame_count = 0

    try:
        while True:
            batch_frames: list[np.ndarray] = []
            batch_indices: list[int] = []
            for _ in range(_BATCH_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_count)
                frame_count += 1

            if not batch_frames:
                break

            # Batched keypoint inference
            kp_results = detector.keypoint_model.predict(batch_frames, verbose=False)

            # Batched player detection (+ tracking opcional)
            if track_enabled:
                player_results = detector.player_model.track(batch_frames, persist=True, batch=_BATCH_SIZE, verbose=False)
            else:
                player_results = detector.player_model.predict(batch_frames, verbose=False)

            for i, frame in enumerate(batch_frames):
                detections = detector._parse_player_detections(player_results[i], parse_tracking=track_enabled)

                # Extract keypoints from batched result
                raw_kps = detector._parse_keypoints(kp_results[i])

                # Temporal smoothing + outlier rejection
                filtered_kps = kp_filter.update(raw_kps)

                player_colors_this_frame: list[list[float]] = []
                for det in detections:
                    if det.label in (DetectionLabel.player, DetectionLabel.goalkeeper):
                        tid = det.tracking_id
                        if tid is not None:
                            cached = _player_color_cache.get(tid)
                            last_seen = _player_color_last_frame.get(tid, -1)
                            if cached is not None and (frame_count - last_seen) < 30:
                                det.player_color = cached
                            else:
                                crop = frame[det.bbox.y1 : det.bbox.y2, det.bbox.x1 : det.bbox.x2]
                                color = get_player_team_color(crop)
                                if color is not None:
                                    cl = color.tolist()
                                    _player_color_cache[tid] = cl
                                    _player_color_last_frame[tid] = frame_count
                                    det.player_color = cl
                        else:
                            # Without tracking, cache by position hash to avoid repeat extraction
                            pos_key = (det.bbox.x1 // 20, det.bbox.y1 // 20)
                            if not hasattr(process_video_task, '_pos_color_cache'):
                                process_video_task._pos_color_cache = {}
                                process_video_task._pos_color_frame = {}
                            cached_color = process_video_task._pos_color_cache.get(pos_key)
                            last_seen_c = process_video_task._pos_color_frame.get(pos_key, -1)
                            if cached_color is not None and (frame_count - last_seen_c) < 15:
                                det.player_color = cached_color
                            else:
                                crop = frame[det.bbox.y1 : det.bbox.y2, det.bbox.x1 : det.bbox.x2]
                                color = get_player_team_color(crop)
                                if color is not None:
                                    cl = color.tolist()
                                    process_video_task._pos_color_cache[pos_key] = cl
                                    process_video_task._pos_color_frame[pos_key] = frame_count
                                    det.player_color = cl
                        if det.player_color is not None:
                            player_colors_this_frame.append(det.player_color)

                team_clf.update(player_colors_this_frame)

                for det in detections:
                    if det.player_color is not None and team_clf.colors_ready:
                        det.team = team_clf.assign(det.player_color)

                ball_center = None
                for det in detections:
                    if det.label == DetectionLabel.ball:
                        cx = (det.bbox.x1 + det.bbox.x2) // 2
                        cy = (det.bbox.y1 + det.bbox.y2) // 2
                        ball_center = (cx, cy)

                ball_center_px = ball_tracker.update(ball_center)

                H = homography_estimator.update(filtered_kps, field_keypoints, conf_threshold=0.5)

                # Only draw keypoints used as inliers in the homography
                inlier_labels = homography_estimator.inlier_labels
                draw_kps = [kp for kp in filtered_kps if str(kp.index + 1) in inlier_labels]

                projected_positions: list[ProjectedPosition] = []
                projected_raw: list = []

                if H is not None and detections:
                    projected_raw = project_positions(detections, H)
                    projected_positions = [
                        ProjectedPosition(x=fx, y=fy, team=team, label=label, tracking_id=tid)
                        for fx, fy, team, label, tid in projected_raw
                    ]

                if track_enabled:
                    player_tracker.update(
                        detections,
                        projected_positions,
                        (float(ball_center_px[0]), float(ball_center_px[1])) if ball_center_px else None,
                    )

                # Draw directly into pre-allocated concat buffer (saves frame.copy + concat copy)
                annotate_frame(frame, detections, team_clf.team_1_color, team_clf.team_2_color,
                               keypoints=draw_kps or None, out=concat[:frame_height])

                field_overlay = create_field_overlay(
                    field_image, projected_raw, team_clf.team_1_color, team_clf.team_2_color,
                    active_kp_labels=inlier_labels or None,
                )
                if field_overlay_pad_x > 0:
                    padded = cv2.copyMakeBorder(
                        field_overlay, 0, 0, field_overlay_pad_x, field_overlay_pad_x,
                        cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    )
                    overlay_slice[:] = padded
                else:
                    overlay_slice[:, :field_overlay.shape[1]] = field_overlay

                out.write(concat)

                fidx = batch_indices[i]
                frame_data = {
                    "frame": fidx,
                    "detections": [
                        {
                            "bbox": d.bbox.model_dump(),
                            "label": d.label.value,
                            "confidence": d.confidence,
                            "tracking_id": d.tracking_id,
                            "player_color": d.player_color,
                            "team": d.team,
                        }
                        for d in detections
                    ],
                    "keypoints": [
                        {"index": k.index, "x": k.x, "y": k.y, "confidence": k.confidence}
                        for k in filtered_kps
                    ],
                    "team_1_color": team_clf.team_1_color,
                    "team_2_color": team_clf.team_2_color,
                    "projected_positions": [
                        {"x": px, "y": py, "team": pt, "label": pl, "tracking_id": ptid}
                        for px, py, pt, pl, ptid in projected_raw
                    ],
                    "ball_center": ball_center_px,
                    "homography_matrix": H.tolist() if H is not None else None,
                }
                frames_file.write(json.dumps(frame_data) + "\n")

                if frame_count % 5 == 0:
                    progress = min(frame_count / max(total_frames, 1), 1.0)
                    self.update_state(
                        state="PROGRESS",
                        meta={"progress": progress, "current_frame": frame_count, "total_frames": total_frames},
                    )

    finally:
        cap.release()
        out.release()
        frames_file.close()
        cv2.destroyAllWindows()

    if track_enabled:
        final_stats = player_tracker.get_stats()
        team_possession = player_tracker.get_team_possession()
    else:
        final_stats = []
        team_possession = {}

    frames_data: list[dict] = []
    with open(frames_jsonl_path) as f:
        for line in f:
            frames_data.append(json.loads(line))
    frames_jsonl_path.unlink(missing_ok=True)

    output_data = {
        "metadata": {
            "total_frames": total_frames,
            "frames_file": "frames.jsonl",
            "fps": fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "track_enabled": track_enabled,
        },
        "frames": frames_data,
        "player_stats": [
            {
                "tracking_id": s.tracking_id,
                "label": s.label,
                "team": s.team,
                "total_distance": s.total_distance,
                "avg_speed": s.avg_speed,
                "top_speed": s.top_speed,
                "touches": s.touches,
                "heatmap_positions": s.heatmap_positions,
            }
            for s in final_stats
        ],
        "team_possession": team_possession,
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return {
        "job_id": job_id,
        "total_frames": total_frames,
        "output_video": str(output_video_path),
        "output_json": str(output_json_path),
    }
