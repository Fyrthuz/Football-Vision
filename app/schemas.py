from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DetectionLabel(str, Enum):
    player = "player"
    goalkeeper = "goalkeeper"
    referee = "referee"
    ball = "ball"


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    bbox: BBox
    label: DetectionLabel
    confidence: float
    tracking_id: int | None = None
    player_color: list[float] | None = None
    team: int | None = None


class Keypoint(BaseModel):
    index: int
    x: float
    y: float
    confidence: float


class FrameDetections(BaseModel):
    frame_width: int
    frame_height: int
    detections: list[Detection]
    keypoints: list[Keypoint]
    team_1_color: list[float] | None = None
    team_2_color: list[float] | None = None


class ProjectedPosition(BaseModel):
    x: float
    y: float
    team: int | None = None
    label: DetectionLabel
    tracking_id: int | None = None


class ProjectedFrame(BaseModel):
    positions: list[ProjectedPosition]
    homography_matrix: list[list[float]] | None = None


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class BatchJobOut(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    filename: str = ""
    duration_sec: float = 0.0
    result_video: str | None = None
    result_json: str | None = None
    error: str | None = None


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    filename: str = ""
    duration_sec: float = 0.0
    tracked_players: int = 0
    frame_count: int = 0
    error: str | None = None


class PlayerStats(BaseModel):
    tracking_id: int
    label: DetectionLabel
    team: int | None = None
    total_distance: float = 0.0
    avg_speed: float = 0.0
    top_speed: float = 0.0
    touches: int = 0
    heatmap_positions: list[tuple[float, float]] = []


class TrackedFrame(BaseModel):
    frame_width: int
    frame_height: int
    detections: list[Detection]
    keypoints: list[Keypoint]
    team_1_color: list[float] | None = None
    team_2_color: list[float] | None = None
    homography_matrix: list[list[float]] | None = None
    projected_positions: list[ProjectedPosition] = []
    ball_position: ProjectedPosition | None = None
    player_stats: list[PlayerStats] = []


class BatchStats(BaseModel):
    total_frames: int
    fps: int
    frame_width: int
    frame_height: int
    duration_sec: float = 0.0
    player_stats: list[PlayerStats] = []
    team_possession: dict[str, float] = {}
    total_distance: float = 0.0
    avg_speed_all: float = 0.0
    total_touches: int = 0
    tracked_players: int = 0


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str | None = None
    models_loaded: dict[str, bool]
