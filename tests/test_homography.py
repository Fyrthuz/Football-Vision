from __future__ import annotations

import numpy as np
import pytest

from app.schemas import BBox, Detection, DetectionLabel, Keypoint
from app.services.homography import (
    HomographyEstimator,
    compute_homography,
    make_projected_positions,
    project_point,
    project_positions,
)


class TestHomographyEstimator:
    @pytest.fixture
    def field_keypoints(self) -> dict:
        return {
            "1": (18, 9),
            "2": (17, 49),
            "3": (18, 103),
            "4": (19, 191),
            "5": (18, 243),
            "6": (17, 279),
        }

    @pytest.fixture
    def detected_keypoints(self) -> list[Keypoint]:
        return [
            Keypoint(index=0, x=100, y=200, confidence=0.9),
            Keypoint(index=1, x=600, y=50, confidence=0.85),
            Keypoint(index=2, x=350, y=400, confidence=0.8),
            Keypoint(index=3, x=200, y=50, confidence=0.75),
            Keypoint(index=4, x=500, y=450, confidence=0.7),
            Keypoint(index=5, x=700, y=300, confidence=0.65),
        ]

    def test_update_returns_matrix_with_enough_points(
        self, field_keypoints: dict, detected_keypoints: list[Keypoint]
    ) -> None:
        estimator = HomographyEstimator(window_size=3)
        H = estimator.update(detected_keypoints, field_keypoints)
        assert H is not None
        assert H.shape == (3, 3)

    def test_update_returns_none_with_few_points(
        self, field_keypoints: dict
    ) -> None:
        estimator = HomographyEstimator(window_size=3)
        kps = [Keypoint(index=0, x=100, y=100, confidence=0.9)]
        H = estimator.update(kps, field_keypoints)
        assert H is None

    def test_smoothing_uses_median(
        self, field_keypoints: dict
    ) -> None:
        estimator = HomographyEstimator(window_size=3)
        kps_a = [
            Keypoint(index=0, x=100.0, y=200.0, confidence=0.9),
            Keypoint(index=1, x=600.0, y=50.0, confidence=0.85),
            Keypoint(index=2, x=350.0, y=400.0, confidence=0.8),
            Keypoint(index=3, x=200.0, y=50.0, confidence=0.75),
            Keypoint(index=4, x=500.0, y=450.0, confidence=0.7),
            Keypoint(index=5, x=700.0, y=300.0, confidence=0.65),
        ]
        kps_b = [
            Keypoint(index=0, x=150.0, y=220.0, confidence=0.9),
            Keypoint(index=1, x=650.0, y=70.0, confidence=0.85),
            Keypoint(index=2, x=300.0, y=420.0, confidence=0.8),
            Keypoint(index=3, x=160.0, y=80.0, confidence=0.75),
            Keypoint(index=4, x=520.0, y=430.0, confidence=0.7),
            Keypoint(index=5, x=680.0, y=320.0, confidence=0.65),
        ]
        kps_c = [
            Keypoint(index=0, x=130.0, y=210.0, confidence=0.9),
            Keypoint(index=1, x=620.0, y=60.0, confidence=0.85),
            Keypoint(index=2, x=320.0, y=410.0, confidence=0.8),
            Keypoint(index=3, x=180.0, y=70.0, confidence=0.75),
            Keypoint(index=4, x=510.0, y=440.0, confidence=0.7),
            Keypoint(index=5, x=690.0, y=310.0, confidence=0.65),
        ]
        H1 = estimator.update(kps_a, field_keypoints)
        H2 = estimator.update(kps_b, field_keypoints)
        H3 = estimator.update(kps_c, field_keypoints)
        assert H1 is not None
        assert H2 is not None
        assert H3 is not None
        # Median of [Ha, Hb, Hc] should differ from Ha
        assert not np.allclose(H1, H3)

    def test_low_confidence_keypoints_filtered(
        self, field_keypoints: dict
    ) -> None:
        estimator = HomographyEstimator(window_size=3)
        kps = [
            Keypoint(index=i, x=100 + i * 50, y=100 + i * 50, confidence=0.3)
            for i in range(10)
        ]
        H = estimator.update(kps, field_keypoints, conf_threshold=0.5)
        assert H is None

    def test_reset_clears_history(self) -> None:
        estimator = HomographyEstimator(window_size=3)
        estimator.reset()
        assert estimator.current is None


class TestComputeHomography:
    def test_returns_matrix_with_enough_points(self) -> None:
        field_kps = {"1": (0, 0), "2": (100, 0), "3": (100, 100), "4": (0, 100)}
        detected = [
            Keypoint(index=0, x=50, y=50, confidence=0.9),
            Keypoint(index=1, x=150, y=50, confidence=0.85),
            Keypoint(index=2, x=150, y=150, confidence=0.8),
            Keypoint(index=3, x=50, y=150, confidence=0.75),
        ]
        H = compute_homography(detected, field_kps, min_matches=4)
        assert H is not None
        assert H.shape == (3, 3)

    def test_returns_none_with_less_than_4_points(self) -> None:
        field_kps = {"1": (0, 0), "2": (100, 0), "3": (100, 100)}
        detected = [
            Keypoint(index=0, x=50, y=50, confidence=0.9),
            Keypoint(index=1, x=150, y=50, confidence=0.85),
            Keypoint(index=2, x=150, y=150, confidence=0.8),
        ]
        H = compute_homography(detected, field_kps)
        assert H is None


class TestProjectPositions:
    def make_detection(self, x1: int, y1: int, x2: int, y2: int, label: str = "player") -> Detection:
        return Detection(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            label=DetectionLabel(label),
            confidence=0.95,
            tracking_id=1,
            team=0,
        )

    def test_projects_center_of_bbox(self) -> None:
        H = np.eye(3, dtype=np.float32)
        detections = [self.make_detection(0, 0, 100, 100)]
        projected = project_positions(detections, H)
        assert len(projected) == 1
        fx, fy, team, label, tid = projected[0]
        assert fx == pytest.approx(50.0, abs=0.1)
        # Player uses bottom of bbox (feet) for parallax correction
        assert fy == pytest.approx(100.0, abs=0.1)
        assert team == 0
        assert label == "player"
        assert tid == 1

    def test_multiple_detections(self) -> None:
        H = np.eye(3, dtype=np.float32)
        dets = [
            self.make_detection(0, 0, 50, 50, "player"),
            self.make_detection(100, 100, 200, 200, "ball"),
        ]
        projected = project_positions(dets, H)
        assert len(projected) == 2
        assert projected[0][3] == "player"
        assert projected[1][3] == "ball"


class TestProjectPoint:
    def test_projects_point(self) -> None:
        H = np.eye(3, dtype=np.float32)
        x, y = project_point(100, 200, H)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_affine_transform(self) -> None:
        H = np.array([[2, 0, 10], [0, 2, 20], [0, 0, 1]], dtype=np.float32)
        x, y = project_point(100, 200, H)
        assert x == pytest.approx(210.0)
        assert y == pytest.approx(420.0)


class TestMakeProjectedPositions:
    def test_converts_raw_to_schema(self) -> None:
        raw = [(10.0, 20.0, 0, "player", 1), (30.0, 40.0, 1, "ball", None)]
        projected = make_projected_positions(raw)
        assert len(projected) == 2
        assert projected[0].x == 10.0
        assert projected[0].y == 20.0
        assert projected[0].team == 0
        assert projected[0].label == "player"
        assert projected[1].x == 30.0
        assert projected[1].y == 40.0
        assert projected[1].team == 1
        assert projected[1].label == "ball"
