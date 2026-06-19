from __future__ import annotations

import numpy as np
import pytest

from app.schemas import Detection, DetectionLabel, PlayerStats, ProjectedPosition
from app.services.tracker import BallTracker, PlayerTracker


class TestBallTracker:
    def test_update_returns_center_when_provided(self) -> None:
        tracker = BallTracker(window_size=3)
        result = tracker.update((100, 200))
        assert result == (100, 200)

    def test_update_returns_none_when_no_history(self) -> None:
        tracker = BallTracker(window_size=3)
        result = tracker.update(None)
        assert result is None

    def test_interpolates_missing_position(self) -> None:
        tracker = BallTracker(window_size=5)
        tracker.update((100, 200))
        tracker.update((110, 210))
        tracker.update(None)
        result = tracker.update(None)
        assert result is not None
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_reset_clears_history(self) -> None:
        tracker = BallTracker(window_size=3)
        tracker.update((100, 200))
        tracker.reset()
        assert len(tracker.positions) == 0


class TestPlayerTracker:
    def make_detection(
        self, tracking_id: int, label: DetectionLabel = DetectionLabel.player,
        team: int = 0, x1: int = 0, y1: int = 0, x2: int = 50, y2: int = 100,
    ) -> Detection:
        from app.schemas import BBox
        return Detection(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            label=label,
            confidence=0.95,
            tracking_id=tracking_id,
            team=team,
        )

    def make_projected(self, x: float, y: float, team: int = 0, label: str = "player") -> ProjectedPosition:
        return ProjectedPosition(x=x, y=y, team=team, label=label)

    def test_tracks_player_distance(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1, x1=0, y1=0, x2=50, y2=100)
        proj = self.make_projected(x=10, y=10)

        tracker.update([det], [proj])

        det.bbox.x1 = 10
        det.bbox.x2 = 60
        tracker.update([det], [self.make_projected(x=20, y=10)])

        stats = tracker.get_stats()
        assert len(stats) == 1
        assert stats[0].tracking_id == 1
        assert stats[0].total_distance > 0

    def test_ball_touches_increment_on_proximity(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1, x1=0, y1=80, x2=50, y2=100)
        proj = self.make_projected(x=10, y=90)

        ball_center = (25.0, 105.0)
        tracker.update([det], [proj], ball_center=ball_center)

        stats = tracker.get_stats()
        assert len(stats) == 1
        assert stats[0].touches == 1

    def test_ball_touches_dedup_within_5_frames(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1, x1=0, y1=80, x2=50, y2=100)
        proj = self.make_projected(x=10, y=90)
        ball = (25.0, 105.0)

        tracker.update([det], [proj], ball_center=ball)
        tracker.update([det], [proj], ball_center=ball)

        stats = tracker.get_stats()
        assert stats[0].touches == 1

    def test_multiple_players_tracked_separately(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det1 = self.make_detection(tracking_id=1, x1=0, y1=0, x2=50, y2=100)
        det2 = self.make_detection(tracking_id=2, x1=100, y1=0, x2=150, y2=100, team=1)
        proj1 = self.make_projected(x=10, y=10)
        proj2 = self.make_projected(x=110, y=10, team=1)

        tracker.update([det1, det2], [proj1, proj2])
        stats = tracker.get_stats()
        assert len(stats) == 2
        ids = {s.tracking_id for s in stats}
        assert ids == {1, 2}

    def test_stale_players_removed(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1)
        proj = self.make_projected(x=10, y=10)

        tracker.update([det], [proj])
        assert len(tracker.get_stats()) == 1

        for _ in range(35):
            tracker.update([], [])

        assert len(tracker.get_stats()) == 0

    def test_get_stats_returns_stats_after_single_frame(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1)
        proj = self.make_projected(x=10, y=10)

        tracker.update([det], [proj])
        stats = tracker.get_stats()
        assert len(stats) == 1
        assert stats[0].tracking_id == 1
        assert stats[0].total_distance == 0.0

    def test_top_speed_tracked(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1, x1=0, y1=0, x2=50, y2=100)
        proj = self.make_projected(x=10, y=10)

        tracker.update([det], [proj])
        det.bbox.x1 = 100
        det.bbox.x2 = 150
        tracker.update([det], [self.make_projected(x=110, y=10)])

        stats = tracker.get_stats()
        assert stats[0].top_speed > 0

    def test_team_possession(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        ball = (25.0, 105.0)

        d1 = self.make_detection(tracking_id=1, team=0, x1=0, y1=80, x2=50, y2=100)
        d2 = self.make_detection(tracking_id=2, team=1, x1=200, y1=80, x2=250, y2=100)
        p1 = self.make_projected(x=10, y=90, team=0)
        p2 = self.make_projected(x=210, y=90, team=1)

        tracker.update([d1, d2], [p1, p2], ball_center=ball)
        d1.bbox.x1 = 5
        d1.bbox.x2 = 55
        tracker.update([d1], [self.make_projected(x=15, y=90, team=0)], ball_center=ball)

        poss = tracker.get_team_possession()
        assert 0 in poss
        assert poss[0] == 100.0

    def test_reset_clears_everything(self) -> None:
        tracker = PlayerTracker(fps=30.0)
        det = self.make_detection(tracking_id=1)
        proj = self.make_projected(x=10, y=10)

        tracker.update([det], [proj])
        assert len(tracker.get_stats()) == 1

        tracker.reset()
        assert len(tracker.get_stats()) == 0
        assert tracker.frame_count == 0
