from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_json_path(tmp_path: Path) -> str:
    import json

    data = {
        "keypoints": {
            "1": [18, 9],
            "2": [17, 49],
            "3": [18, 103],
        },
        "height": 288,
        "width": 422,
    }
    p = tmp_path / "sample.json"
    with open(p, "w") as f:
        json.dump(data, f)
    return str(p)
