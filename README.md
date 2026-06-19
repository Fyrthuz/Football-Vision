# Football Vision

Batch football video analytics powered by YOLOv8. Detects and **tracks**
players, referees, goalkeepers, and the ball, estimates 32 pitch keypoints,
classifies team colours, projects player positions onto a 2D field, and
computes per-player statistics (distance, speed, touches, heatmaps).

## Features

- **Web UI** — Browser-based dashboard at `http://localhost:8000`
  (Batch processing + Health check tabs)
- **Player/Referee/Goalkeeper Detection + Tracking** — YOLOv8x with ByteTrack
- **Pitch Keypoint Detection** — 32-class YOLOv8s trained on Roboflow
  football-field-detection dataset (detection format, mAP50=0.974)
- **Ball Tracking** — Interpolation-based tracking across frames
- **Team Colour Classification** — LAB-space K-Means on shirt crop, EMA-stabilised
- **Per-Player Stats** — Distance, avg/top speed, ball touches, possession, heatmaps
- **Homography Projection** — RANSAC with per-keypoint temporal filtering (EMA + jump rejection)
- **Batch Processing** — Upload video via REST API, receive annotated video +
  player stats JSON (Celery + Redis)
- **GPU Acceleration** — CUDA 13 support (RTX 50-series Blackwell sm_120)
- **Fully Containerized** — Docker Compose (api + worker + redis)

## GPU Compatibility

| GPU Family | PyTorch | CUDA |
|---|---|---|
| Turing (RTX 20-series) … Ada (RTX 40-series) | ≥ 2.0 | ≥ 11.8 |
| Blackwell (RTX 50-series, **sm_120**) | **≥ 2.7.0** | **≥ 12.8** |

Uses `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime`.

```bash
# Start with GPU
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA Container Toolkit (optional, for GPU)

### 1. Configure

```bash
cp .env.example .env
```

### 2. Generate field template (if missing)

```bash
docker compose -f docker/docker-compose.yml run --rm train training.field_template
```

### 3. Train models (optional)

Requires `ROBOFLOW_API_KEY` in `.env`:

```bash
# Player detection (YOLOv8x, 100 epochs)
docker compose -f docker/docker-compose.yml run --rm train training.train_detection

# Pitch keypoints (YOLOv8s, 100 epochs, detection format)
docker compose -f docker/docker-compose.yml run --rm train training.train_keypoints
```

### 4. Run

```bash
# Without GPU
docker compose -f docker/docker-compose.yml up --build

# With GPU
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build
```

### Access

| Service | Port | URL |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000 |
| Swagger UI | 8000 | http://localhost:8000/docs |
| Health API | 8000 | http://localhost:8000/health |

## Web UI

- **Batch** tab — Upload video, track progress (live frame count), view:
  - H.264 annotated video with bboxes + tracking IDs
  - Match summary (duration, resolution, frames, total distance, speed, touches, possession)
  - Per-player stats table with heatmap on field overlay
  - Job history with status, progress bar, delete
- **Health** tab — System status, GPU availability, model info

## API

```bash
# Upload
curl -X POST http://localhost:8000/batch/upload -F "file=@match.mp4;type=video/mp4"

# Status (includes live frame count)
curl http://localhost:8000/batch/status/{job_id}

# Stats
curl http://localhost:8000/batch/stats/{job_id}

# Video
curl http://localhost:8000/batch/video/{job_id}

# List jobs
curl http://localhost:8000/batch/jobs

# Delete job
curl -X DELETE http://localhost:8000/batch/jobs/{job_id}
```

## Project Structure

```
app/                      # FastAPI application
├── main.py               # App entry point
├── config.py             # Pydantic settings
├── schemas.py            # Request/response models
├── static/               # Frontend (HTML, JS, CSS, images)
├── routers/
│   ├── health.py         # GET /health
│   └── batch.py          # /batch/* endpoints
└── services/
    ├── detector.py       # YOLOv8 inference + ByteTrack + keypoint parsing
    ├── classifier.py     # Team colour classification
    ├── tracker.py        # PlayerTracker + BallTracker
    └── homography.py     # HomographyEstimator + KeypointFilter + projection

batch_processor/          # Celery worker
├── celery_app.py
└── worker.py             # Batch video processing pipeline

training/                 # Training & tools
├── train_keypoints.py
├── train_detection.py
├── field_template.py     # Generates football_field.png
├── map_keypoints.py      # Interactive keypoint mapping tool
├── convert_pose_to_detection.py
├── download_datasets.py
└── configs/keypoints.yaml

docker/
├── Dockerfile
├── docker-compose.yml
└── docker-compose.gpu.yml

tests/
├── conftest.py
├── test_api.py
├── test_detector.py
├── test_tracker.py
└── test_homography.py
```

## Keypoint Mapping

The 32 keypoints follow Roboflow `football-field-detection-f07vi` schema.
Template coordinates in `sample.json` (422 × 288 px space).

To interactively verify / adjust keypoints:
```bash
docker compose -f docker/docker-compose.yml run --rm api python -m training.map_keypoints
```

The `football_field_keypoints.png` image shows the current mapping for
quick reference. Regenerate after edits:
```bash
docker compose -f docker/docker-compose.yml run --rm train training.field_template
```

## Tests

```bash
docker compose -f docker/docker-compose.yml run --rm api pytest tests/ -v
```

## Tech Stack

Python 3.12+, FastAPI, Celery + Redis, YOLOv8 (Ultralytics), PyTorch 2.12,
OpenCV, scikit-learn, Roboflow, Docker + NVIDIA CUDA 13.
