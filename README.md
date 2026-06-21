# Football Vision

Batch football video analytics powered by YOLOv8. Detects and **tracks**
players, referees, goalkeepers, and the ball, estimates 32 pitch keypoints,
classifies team colours, projects player positions onto a 2D field, and
computes per-player statistics (distance, speed, touches, heatmaps).

## Features

- **Web UI** — Browser-based dashboard at `http://localhost:8000`
  (Batch processing + Health check tabs)
- **Player Detection + Tracking** — YOLOv8x (640px, 5 classes) with ByteTrack
- **Pitch Keypoint Detection** — YOLOv8m (1280px, 32-class detection format,
  52 MB, mAP50=0.974)
- **ONNX Runtime** — Both models exported to ONNX FP16 (opset 20, dynamic batch)
  via `CUDAExecutionProvider`. Optimal batch size = 8 (**112.5 fps** combined).
- **Ball Tracking** — Interpolation-based tracking across frames
- **Team Colour Classification** — LAB-space K-Means on shirt crop, EMA-stabilised
- **Per-Player Stats** — Distance, avg/top speed, ball touches, possession, heatmaps
- **Homography Projection** — RANSAC with per-keypoint temporal filtering (EMA + jump rejection);
  only RANSAC-inlier keypoints drawn on output
- **Batch Processing** — Upload video via REST API, receive annotated video +
  player stats JSON (Celery + Redis)
- **GPU Acceleration** — CUDA 13 support (RTX 50-series Blackwell sm_120)
- **Fully Containerized** — Docker Compose (api + worker + redis)

## Models

| Model | Task | Arch | Input | Params | Size | Runtime | Speed (8 frames) |
|---|---|---|---|---|---|---|---|
| **Player** | Detection + tracking (player, referee, gk, ball) | YOLOv8x | 640×640 | 68.2M | 130 MB | ONNX CUDA | 178 fps |
| **Keypoint** | 32-class keypoint detection (pitch registration) | YOLOv8m | 1280×1280 | 25.9M | 50 MB | ONNX CUDA | 179 fps |

Both models are exported to ONNX FP16 (opset 20, dynamic batch) and run via
ONNX Runtime with `CUDAExecutionProvider`. Loading the `.onnx` files instead
of `.pt` gives **1.23×** (keypoint) and **2.59×** (player) speedup vs PyTorch FP16.

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
# Player model: YOLOv8x detection + tracking (5 classes, 640px)
docker compose -f docker/docker-compose.yml run --rm train training.train_detection

# Keypoint model: YOLOv8m 32-class detection format (1280px, AdamW)
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

- **Batch** tab — Upload video, track progress (auto-refreshes every 4s), view:
  - H.264 annotated video with bboxes + tracking IDs
  - Match summary (duration, resolution, frames, total distance, speed, touches, possession)
  - Per-player stats table with heatmap on field overlay
  - Job history with status, progress bar, delete
- **Health** tab — System status, GPU availability, model info

## API

### Batch

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
│   ├── batch.py          # /batch/* endpoints
│   └── health.py         # GET /health
└── services/
    ├── detector.py       # ONNX CUDA predict/inference (player=640px, keypoint=1280px) + ByteTrack via track(batch=N)
    ├── classifier.py     # Team colour classification
    ├── tracker.py        # PlayerTracker + BallTracker
    └── homography.py     # HomographyEstimator + KeypointFilter + projection

batch_processor/          # Celery worker
├── celery_app.py
└── worker.py             # Batch pipeline: ONNX inference, ffmpeg pipe, JSONL output

training/                 # Training & tools
├── train_keypoints.py
├── train_detection.py
├── field_template.py     # Generates football_field.png + keypoints overlay
├── map_keypoints.py      # Interactive keypoint mapping tool
├── compute_keypoints.py  # Computes keypoint positions from pitch proportions
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
Positions are computed proportionally from real pitch dimensions (12000×7000 cm)
mapped to 422×288 px. Template coordinates in `sample.json`.

To interactively verify / adjust keypoints:
```bash
docker compose -f docker/docker-compose.yml run --rm api python -m training.map_keypoints
```

The `football_field_keypoints.png` image shows the current mapping for
quick reference. Regenerate after edits:
```bash
docker compose -f docker/docker-compose.yml run --rm train training.field_template
```

## Optimizations

| Optimization | Before | After | Gain |
|---|---|---|---|
| **Player inference** | PyTorch FP16 (44 fps) | ONNX CUDA (114 fps) | **2.59×** |
| **Keypoint inference** | PyTorch FP16 (145 fps) | ONNX CUDA (179 fps) | **1.23×** |
| **Player tracking batch** | `track()` forces `batch=1` (26 fps) | `track(batch=8)` batches n frames (178 fps) | **6.8×** |
| **Batch size tuning** | Batch=16 (98 fps) | Batch=8 (**112.5 fps** combined) | **+15%** |
| **Tests passing** | 2 homography tests failing | All **37 tests pass** | Fixed confidence threshold + inertia filter test |
| **Video encoding** | mp4v + ffmpeg transcode | ffmpeg pipe (libx264 direct) | Eliminates final transcode |
| **Color extraction** | Per-frame KMeans on every player crop | Cached per track ID (recalc every 30 frames) | ~90% fewer KMeans calls |
| **Memory allocation** | `np.zeros` + 2 full copies per frame | Pre-allocated concat buffer, draw in-place | ~12 MB saved per frame |
| **Frame JSON output** | Accumulated in RAM (`all_frames_data`) | Written to disk incrementally (JSONL) | Eliminates OOM risk |
| **Per-frame JSON size** | Included redundant `player_stats` | Only final stats at end | ~50% smaller frame data |
| **Frontend polling** | `setInterval` + `setTimeout` dual polling | Single polling, cleared during active job | Reduced server load |
| **Row click** | `viewJob` triggered twice (row + button) | `event.stopPropagation()` on button | Prevents double fetch |
| **Unused imports** | `OrderedDict`, `typing.Any`, `Field` | Removed | Cleaner code |

## Benchmarks

All benchmarks on RTX 5080 with CUDA 13.0, ONNX Runtime 1.27.0.

| Scenario | FPS | Detail |
|---|---|---|
| Keypoint model (1280px) | 179 fps | batch=8 single model |
| Player model track (640px) | **178 fps** | batch=8 (was 26 fps with `track()` default) |
| Combined throughput (keypoint + player) | **112.5 fps** | batch=8, both models |
| Combined throughput (keypoint + player) | 98.3 fps | batch=16 |
| Combined throughput (keypoint + player) | 80.6 fps | batch=4 |
| Combined throughput (keypoint + player) | 53.7 fps | batch=2 |
| H.264 encoding (ffmpeg libx264, software) | ~40 fps | single-pass |

## Architecture

```
┌──────────────┐   ┌─────────────────────────────────────────────────┐
│  User upload │   │                Worker (Celery)                  │
│  (FastAPI)   │   │                                                 │
│              │   │  while True:                                    │
│  POST /batch │   │    batch = cap.read(8)                          │
│  /upload     │   │    kp_results  = kp_model.predict(batch) ← ONNX│
│              │   │    pl_results  = pl_model.track(batch, ← ONNX│
│              │   │                    persist=True, batch=8)      │
│  GET /batch  │   │    for i, frame in enumerate(batch):            │
│  /status/:id │   │      detections = pl_results[i]                 │
│              │   │      color = cache.get(track_id)  ← cached 30fr│
│  GET /batch  │   │      H = homography(kps)                        │
│  /video/:id  │   │      annotate_frame(out=concat[:h]) ← in-place │
│              │   │      frames.jsonl += json_line      ← incr.    │
│  GET /batch  │   │      ffmpeg_pipe.write(concat)     ← H.264     │
│  /stats/:id  │   │                                                │
└──────────────┘   └─────────────────────────────────────────────────┘
```


## Tests

```bash
docker compose -f docker/docker-compose.yml run --rm api pytest tests/ -v
```

## Tech Stack

Python 3.12+, FastAPI, Celery + Redis, YOLOv8 (Ultralytics), PyTorch 2.12,
OpenCV, scikit-learn, Roboflow, Docker + NVIDIA CUDA 13.
