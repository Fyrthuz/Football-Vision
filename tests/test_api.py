from __future__ import annotations

from fastapi.testclient import TestClient
from app.schemas import TrackedFrame


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "models_loaded" in data


def test_tracked_frame_schema_roundtrip() -> None:
    payload = {
        "frame_width": 640,
        "frame_height": 480,
        "detections": [
            {
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
                "label": "player",
                "confidence": 0.95,
                "tracking_id": 1,
                "team": 0,
            }
        ],
        "keypoints": [{"index": 0, "x": 100.0, "y": 200.0, "confidence": 0.9}],
        "team_1_color": [66.0, 133.0, 244.0],
        "team_2_color": [234.0, 67.0, 53.0],
        "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "projected_positions": [{"x": 50.0, "y": 100.0, "team": 0, "label": "player"}],
        "player_stats": [
            {
                "tracking_id": 1,
                "label": "player",
                "team": 0,
                "total_distance": 10.5,
                "avg_speed": 1.2,
                "top_speed": 3.0,
                "touches": 2,
                "heatmap_positions": [],
            }
        ],
    }
    frame = TrackedFrame(**payload)
    dumped = frame.model_dump()
    assert dumped["frame_width"] == 640
    assert dumped["detections"][0]["tracking_id"] == 1
    assert dumped["projected_positions"][0]["x"] == 50.0
    assert dumped["player_stats"][0]["total_distance"] == 10.5
    assert dumped["homography_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
