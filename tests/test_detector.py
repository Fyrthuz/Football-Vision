from __future__ import annotations

import numpy as np
import pytest

from app.schemas import BBox, Detection, DetectionLabel, Keypoint, PlayerStats, TrackedFrame


def test_bbox_model() -> None:
    bbox = BBox(x1=10, y1=20, x2=100, y2=200)
    assert bbox.x1 == 10
    assert bbox.x2 == 100
    assert bbox.y1 == 20
    assert bbox.y2 == 200


def test_detection_model() -> None:
    det = Detection(
        bbox=BBox(x1=0, y1=0, x2=50, y2=100),
        label=DetectionLabel.player,
        confidence=0.95,
        tracking_id=1,
        player_color=[128.0, 64.0, 32.0],
        team=0,
    )
    assert det.label == DetectionLabel.player
    assert det.confidence == 0.95
    assert det.tracking_id == 1
    assert det.team == 0


def test_detection_without_tracking_id() -> None:
    det = Detection(
        bbox=BBox(x1=10, y1=20, x2=100, y2=200),
        label=DetectionLabel.ball,
        confidence=0.85,
    )
    assert det.tracking_id is None
    assert det.player_color is None
    assert det.team is None


def test_keypoint_model() -> None:
    kp = Keypoint(index=5, x=320.0, y=240.0, confidence=0.85)
    assert kp.index == 5
    assert kp.x == 320.0
    assert kp.confidence == 0.85


def test_detection_labels() -> None:
    assert DetectionLabel.player.value == "player"
    assert DetectionLabel.goalkeeper.value == "goalkeeper"
    assert DetectionLabel.referee.value == "referee"
    assert DetectionLabel.ball.value == "ball"


def test_detection_serialization() -> None:
    det = Detection(
        bbox=BBox(x1=10, y1=20, x2=100, y2=200),
        label=DetectionLabel.player,
        confidence=0.95,
    )
    data = det.model_dump()
    assert data["label"] == "player"
    assert data["confidence"] == 0.95
    assert data["bbox"]["x1"] == 10
    assert data["tracking_id"] is None


def test_player_stats_model() -> None:
    stats = PlayerStats(
        tracking_id=1,
        label=DetectionLabel.player,
        team=0,
        total_distance=42.5,
        avg_speed=2.1,
        top_speed=5.3,
        touches=3,
    )
    assert stats.tracking_id == 1
    assert stats.total_distance == 42.5
    assert stats.avg_speed == 2.1
    assert stats.top_speed == 5.3
    assert stats.touches == 3


def test_player_stats_serialization() -> None:
    stats = PlayerStats(
        tracking_id=1,
        label=DetectionLabel.player,
        team=0,
        total_distance=42.5,
        avg_speed=2.1,
        top_speed=5.3,
        touches=3,
    )
    data = stats.model_dump()
    assert data["tracking_id"] == 1
    assert data["total_distance"] == 42.5
    assert data["heatmap_positions"] == []


def test_tracked_frame_model() -> None:
    frame = TrackedFrame(
        frame_width=640,
        frame_height=480,
        detections=[],
        keypoints=[],
    )
    assert frame.frame_width == 640
    assert frame.homography_matrix is None
    assert frame.ball_position is None
    assert frame.player_stats == []


def test_tracked_frame_with_full_data() -> None:
    det = Detection(
        bbox=BBox(x1=10, y1=20, x2=100, y2=200),
        label=DetectionLabel.player,
        confidence=0.95,
        tracking_id=1,
        team=0,
    )
    kp = Keypoint(index=0, x=100.0, y=200.0, confidence=0.9)
    stats = PlayerStats(tracking_id=1, label=DetectionLabel.player, team=0)
    frame = TrackedFrame(
        frame_width=640,
        frame_height=480,
        detections=[det],
        keypoints=[kp],
        team_1_color=[66.0, 133.0, 244.0],
        team_2_color=[234.0, 67.0, 53.0],
        homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        player_stats=[stats],
    )
    assert len(frame.detections) == 1
    assert len(frame.keypoints) == 1
    assert len(frame.player_stats) == 1
    assert frame.team_1_color == [66.0, 133.0, 244.0]
