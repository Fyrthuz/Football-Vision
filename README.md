# Football Vision

Batch football video analytics powered by YOLOv8. Detects and **tracks**
players, referees, goalkeepers, and the ball, estimates 32 pitch keypoints,
classifies team colours, projects player positions onto a 2D field, and
computes per-player statistics (distance, speed, touches, heatmaps).

## Features

- **Web UI** — Browser-based dashboard at `http://localhost:8000`
  (Batch processing + Health check tabs)
- **Player Detection + Tracking** — YOLOv8m (640px, 5 classes) with ByteTrack
- **Pitch Keypoint Detection** — YOLOv8m (1280px, 32-class detection format,
   50 MB, mAP50=0.974)
- **ONNX Runtime** — Both models exported to ONNX FP16 (opset 20, dynamic batch)
  via `CUDAExecutionProvider`. Optimal batch size = 2 (**34 fps** combined model-only).
- **Ball Tracking** — Interpolation-based tracking across frames
- **Team Colour Classification** — LAB-space K-Means on shirt crop, EMA-stabilised
- **Per-Player Stats** — Distance, avg/top speed, ball touches, possession, heatmaps
  (toggle tracking on/off in the Web UI; stats only when tracking enabled)
- **Homography Projection** — RANSAC with per-keypoint temporal filtering (EMA + jump rejection);
  only RANSAC-inlier keypoints drawn on output
- **Batch Processing** — Upload video via REST API, receive annotated video +
  player stats JSON (Celery + Redis)
- **GPU Acceleration** — CUDA 13 support (RTX 50-series Blackwell sm_120)
- **Fully Containerized** — Docker Compose (api + worker + redis)

## Models

| Model | Task | Arch | Input | Params | Size | Runtime |
|---|---|---|---|---|---|---|---|
| **Player** | Detection + tracking (player, referee, gk, ball) | YOLOv8m | 640×640 | 25.8M | 50 MB | ONNX CUDA |
| **Keypoint** | 32-class keypoint detection (pitch registration) | YOLOv8m | 1280×1280 | 25.9M | 50 MB | ONNX CUDA |

Both models are pre-exported to ONNX FP16 (opset 20, dynamic batch) and run via
ONNX Runtime with `CUDAExecutionProvider`. This eliminates the PyTorch CPU
bottleneck and enables true batched GPU inference.

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
# Player model: YOLOv8m detection + tracking (5 classes, 640px)
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

- **Batch** tab — Upload video (toggle tracking on/off), track progress (auto-refreshes every 4s), view:
  - H.264 annotated video with bboxes + tracking IDs
  - Match summary (duration, resolution, frames, total distance, speed, touches, possession)
  - Per-player stats table with heatmap on field overlay
  - Job history with status, progress bar, delete
- **Health** tab — System status, GPU availability, model info

## API

### Batch

```bash
# Upload (with tracking enabled)
curl -X POST http://localhost:8000/batch/upload \
  -F "file=@match.mp4;type=video/mp4" \
  -F "track_enabled=true"

# Upload (detection only, no tracking)
curl -X POST http://localhost:8000/batch/upload \
  -F "file=@match.mp4;type=video/mp4" \
  -F "track_enabled=false"

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
|---|---|---|---|---|
| **Model runtime** | PyTorch CPU (FP32) | **ONNX CUDA** | Automatic |
| **Player model swap** | YOLOv8x (131 MB, 68.2M params) | YOLOv8m (50 MB, 25.8M params) | — |
| **Player inference** | PyTorch CPU batch=1 (27 fps) | ONNX CUDA batch=8 (70 fps) | **2.6×** |
| **Keypoint inference (1280px)** | ONNX default 640px (374 fps, 0 detections) | ONNX CUDA 1280px (89 fps, **10–12 kp/frame**) | **Correct pitch registration** |
| **Combined pipeline** | PyTorch CPU (— fps) | ONNX CUDA batch=2 (**34 fps** model-only) | — |
| **Player tracking batch** | `track()` forces `batch=1` (26 fps) | `track(batch=8)` batches n frames (70 fps) | **2.7×** |
| **Optimal batch size** | Batch=1 (32 fps combined) | Batch=2 (**34 fps** combined) | **+5%** |
| **Tracking overhead (real video)** | — | `track(batch=N)` vs `predict()`: **0–1%** overhead | Negligible |
| **Tests passing** | 2 homography tests failing | All **37 tests pass** | Fixed confidence threshold + inertia filter test |
| **Video encoding** | mp4v + ffmpeg transcode | ffmpeg pipe (libx264 direct) | Eliminates final transcode |
| **Color extraction** | Per-frame KMeans on every player crop | Cached per track ID (recalc every 30 frames) + position-hash cache for non-tracking mode | ~90% fewer KMeans calls |
| **Memory allocation** | `np.zeros` + 2 full copies per frame | Pre-allocated concat buffer, draw in-place | ~12 MB saved per frame |
| **Frame JSON output** | Accumulated in RAM (`all_frames_data`) | Written to disk incrementally (JSONL) | Eliminates OOM risk |
| **Per-frame JSON size** | Included redundant `player_stats` | Only final stats at end | ~50% smaller frame data |
| **Frontend polling** | `setInterval` + `setTimeout` dual polling | Single polling, cleared during active job | Reduced server load |
| **Row click** | `viewJob` triggered twice (row + button) | `event.stopPropagation()` on button | Prevents double fetch |
| **Unused imports** | `OrderedDict`, `typing.Any`, `Field` | Removed | Cleaner code |
| **Video encoding** | libx264 software (~25 fps) | h264_nvenc GPU hardware (**~25 fps**) | Negligible (bottleneck is model + processing) |
| **Encoder auto-detect** | Hardcoded libx264 | Auto-detects `h264_nvenc` / falls back to `libx264` | Works on CPU-only hosts |
| **ONNX warmup** | — | `predict()` before first `track()` | Fixes dynamic batch Reshape error |
| **Keypoint model imgsz fix** | Default 640px (0 keypoints, H=None) | `overrides['imgsz'] = 1280` in `detector.py:22` | **10–12 keypoints/frame, homography works** |

## Benchmarks

All benchmarks on RTX 5080 with CUDA 13.0, ONNX Runtime 1.27.0.
Video: 1080p real footage (bundesliga_sample.mp4), models at native resolution (player=640px, keypoint=1280px).
Player model: YOLOv8m (trained on local dataset, 100 epochs, mAP50=0.903 @1280px train / 0.811 @640px val).

> **Important note on keypoint resolution:** The keypoint model was originally benchmarked at the ONNX
> default of 640×640 (which gave 374 fps but **0 detections** because the model is trained at 1280×1280).
> The numbers below reflect the **correct 1280×1280** resolution after the imgsz override fix.

### Player model (`best_model_players.onnx` — YOLOv8m)

| Batch | `track(batch=N)` | `predict()` | Overhead |
|-------|-----------------|-------------|----------|
| 1     | 60.5 fps        | 60.8 fps    | ~0%      |
| 2     | 64.0 fps        | 64.5 fps    | ~0%      |
| 4     | 67.5 fps        | 67.8 fps    | ~0%      |
| 8     | 70.2 fps        | 70.7 fps    | ~0%      |
| 16    | 68.0 fps        | 68.5 fps    | ~0%      |

ByteTrack adds negligible post-processing overhead. **The speedup comes entirely from using `track(batch=N)` instead of `track()`** (which forces `batch=1`). Enabling/disabling tracking in the UI does not affect throughput.

> **Notes:**
> - **Player model warmup:** The YOLOv8m ONNX player model requires a warmup `predict()` call
>   before the first `track()` call, otherwise the internal Reshape node in opset 20 fails on
>   dynamic batch. The worker handles this automatically after `load_models()`.
> - **Keypoint model imgsz:** The keypoint model ONNX has dynamic height/width dims and defaults
>   to 640×640. Must set `model.overrides['imgsz'] = 1280` after loading (done in `detector.py:22`),
>   otherwise **0 keypoints are detected** and homography is always `None`.

### Keypoint model (`best_model_keypoints.onnx`)

| Batch | FPS |
|-------|-----|
| 1     | **89** |
| 2     | 56 |
| 4     | 51 |
| 8     | 70 |

### Combined pipeline (keypoint + player, sequential, as in worker)

| Batch | Combined FPS | Frame time |
|-------|-------------|------------|
| 1     | 32.0 fps    | 31.2 ms    |
| 2     | **33.6 fps** | **29.8 ms** |
| 4     | 31.2 fps    | 32.1 ms    |
| 8     | 30.8 fps    | 32.5 ms    |
| 16    | 29.3 fps    | 34.1 ms    |

**Optimal: batch=2 → ~34 fps combined.** The keypoint model at 1280×1280 is now the bottleneck
(~89 fps at batch=1, comparable to player model at ~70 fps). Batch beyond 2 does not improve
throughput — both models run sequentially on the same GPU, and the 1280px keypoint model saturates
memory bandwidth. End-to-end pipeline (annotation, homography, JSON serialization, H.264 encoding)
adds ~25% overhead, yielding **~25 fps** final output (≈ real-time for 25 fps video).

### H.264 encoding

| Encoder | End-to-end throughput | Speed vs real-time |
|---------|----------------------|-------------------|
| libx264 (software) | ~25 fps | 0.98× |
| **h264_nvenc** (GPU hardware) | **~25 fps** | **1.0×** |

The encoder is auto-detected at startup: `h264_nvenc` if available (GPU), `libx264` if not (CPU).
The bottleneck is model inference + per-frame Python processing, not the encoder — NVENC and
libx264 are nearly identical here.

## Architecture

```
┌──────────────┐   ┌─────────────────────────────────────────────────┐
│  User upload │   │                Worker (Celery)                  │
│  (FastAPI)   │   │                                                 │
│              │   │  while True:                                    │
│  POST /batch │   │    batch = cap.read(8)                          │
│  /upload     │   │    kp_results  = kp_model.predict(batch) ← ONNX│
│  +track_flag │   │    if track_enabled:                            │
│              │   │      pl_results = pl_model.track(batch, ← ONNX  │
│              │   │                        persist=True, batch=8)   │
│              │   │    else:                                        │
│              │   │      pl_results = pl_model.predict(batch) ← ONNX│
│  GET /batch  │   │    for i, frame in enumerate(batch):            │
│  /status/:id │   │      detections = pl_results[i]                 │
│              │   │      color = cache.get(track_id)  ← cached 30fr│
│  GET /batch  │   │      H = homography(kps)                        │
│  /video/:id  │   │      annotate_frame(out=concat[:h]) ← in-place │
│              │   │      frames.jsonl += json_line      ← incr.    │
│  GET /batch  │   │      ffmpeg_pipe.write(concat)     ← H.264     │
│  /stats/:id  │   │      (h264_nvenc if GPU, else libx264)         │
└──────────────┘   └─────────────────────────────────────────────────┘
```


## Future Enhancements

- **Parallel model inference** — Run player and keypoint models on separate CUDA streams to overlap GPU computation instead of executing sequentially.
- **Live / streaming mode** — Adapt the pipeline for real-time processing with WebSocket output, reducing latency from batch completion to frame-by-frame streaming.
- **Multi-camera support** — Fuse projections from multiple angles for full-pitch coverage and player re-identification across views.
- **Player re-identification** — Replace ByteTrack with appearance-based re-ID (OSNet, BoT) to maintain identity across occlusions and camera cuts.
- **Event detection** — Recognise football events (goals, corners, free kicks, offsides) from projected positions and ball trajectory.
- **WebRTC preview** — Stream intermediate results to the browser during processing instead of waiting for full completion.
- **Model distillation** — Distill the 1280px keypoint model into a smaller student network (e.g., YOLOv8n) to recover the throughput lost at higher resolution.
- **Configurable output** — Allow users to select output resolution, bitrate, and overlay options (e.g., field-only, no bounding boxes).

## Tests

```bash
docker compose -f docker/docker-compose.yml run --rm api pytest tests/ -v
```

## Tech Stack

Python 3.12+, FastAPI, Celery + Redis, YOLOv8 (Ultralytics), PyTorch 2.12,
OpenCV, scikit-learn, Roboflow, Docker + NVIDIA CUDA 13.
