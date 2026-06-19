# Football Vision Architecture

## Overview

Football Vision is a containerized computer vision service for football (soccer) video analytics. It provides batch processing for detecting and **tracking** players, referees, goalkeepers, and the ball, as well as pitch keypoint estimation with homography-based field projection and per-player statistics.

## System Architecture

```
                         REST (upload, status, stats, video, jobs)
                                    │
         ┌──────────────────────────▼────────────────────────────┐
         │                    FastAPI API                         │
         │  Port 8000                                            │
         │  ┌──────────────────────────────────────────────────┐ │
         │  │ /health /batch/upload /batch/status/{id}         │ │
         │  │ /batch/stats/{id} /batch/video/{id}              │ │
         │  │ /batch/jobs /batch/jobs/{id}                     │ │
         │  └──────────────────────────────────────────────────┘ │
         └──────────────────────────┬────────────────────────────┘
                                    │
         ┌──────────────────────────▼────────────────────────────┐
         │              Redis (Broker/Backend)                    │
         │              Port 6379                                 │
         └──────────────────────────┬────────────────────────────┘
                                    │
         ┌──────────────────────────▼────────────────────────────┐
         │              Celery Worker (GPU-accelerated)           │
         │  Uses Detector.track + PlayerTracker + Homography      │
         │  Produces annotated video + JSON with per-player stats │
         └───────────────────────────────────────────────────────┘
```

## Services

| Service  | Description                                   | Port  |
|----------|-----------------------------------------------|-------|
| api      | FastAPI app (REST + static files)             | 8000  |
| worker   | Celery worker for batch video processing      | —     |
| redis    | Message broker and result backend             | 6379  |
| train    | One-shot training container (profile: train)  | —     |

## Data Flow (Batch)

1. User uploads a video file via `POST /batch/upload`
2. API saves the file and enqueues a Celery task using `send_task(task_id=job_id)` so the Celery task ID equals the application `job_id`
3. Celery worker picks up the task:
   - Opens the video with OpenCV and reads frames sequentially
   - For each frame:
     - `Detector.track(frame)` — YOLOv8x `model.track()` (ByteTrack) → detections with persistent `tracking_id`
     - `YOLOv8n-pose(frame)` — 32 pitch keypoints
     - `TeamClassifier` — EMA-stabilized K-Means team color classification on player crops
     - `HomographyEstimator.update()` — Smoothed perspective transform (rolling window of 5)
     - `PlayerTracker.update()` — Accumulates distance, speed, touches, field positions
   - Writes annotated frame to output video
   - Every frame: calls `self.update_state(state="PROGRESS", meta={"progress": progress, "current_frame": frame_count, "total_frames": total_frames})` for live progress tracking
   - After all frames: writes per-frame data and final aggregated stats as JSON
4. Frontend polls `GET /batch/status/{job_id}` every 2s for progress (0.0–1.0 + `current_frame`/`total_frames`)
5. On completion, displays `GET /batch/stats/{job_id}` for match summary + per-player stats
6. Output video is streamed via `GET /batch/video/{job_id}`
7. If Celery result expires, the status endpoint falls back to checking result file existence (`output.mp4` + `data.json`)

## API Endpoints

### Health
- `GET /health` — Service health check with GPU and model status

### Batch (REST)
- `POST /batch/upload` — Upload video file, returns `job_id`
- `GET /batch/status/{job_id}` — Query job processing status + progress
- `GET /batch/stats/{job_id}` — Aggregated match stats (duration, distance, speed, touches, possession, per-player stats)
- `GET /batch/video/{job_id}` — Stream the annotated output video
- `GET /batch/jobs` — List all jobs with status and metadata
- `DELETE /batch/jobs/{job_id}` — Cancel and delete a job

## Detection + Tracking Pipeline

```
Frame ─► Detector.track() ─► ByteTrack ─► Detections + tracking_ids
  │                              │
  │        ┌─────────────────────┘
  │        ▼
  ├─ Player boxes → TeamClassifier (EMA + K-Means) → team per player
  ├─ Ball center → BallTracker (interpolation across gaps)
  ├─ Keypoints  → HomographyEstimator (rolling avg) → H matrix
  └─ Detections + H → project_positions → field (x, y)
       │
       ▼
  PlayerTracker.update(detections, projected, ball):
    ├─ Per-player: positions[] + field_positions[] (heatmap history)
    ├─ Distance = Σ ||center_i - center_{i-1}|| * pixels_to_meters
    ├─ Speed = distance / dt (top_speed tracked)
    ├─ Touches = count of frames where ball center within 50px of player feet
    └─ Stale players removed after 30 frames of absence
```

## Project Structure

```
football-vision/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Pydantic Settings (environment variables)
│   ├── schemas.py           # Pydantic request/response models
│   ├── static/              # Frontend assets
│   ├── routers/
│   │   ├── health.py        # Health check endpoint
│   │   └── batch.py         # REST batch processing endpoints (upload, status, stats, video, jobs, delete)
│   └── services/
│       ├── detector.py      # YOLOv8 detection + ByteTrack (track method)
│       ├── classifier.py    # EMA-stabilized K-Means team color classification
│       ├── tracker.py       # PlayerTracker (stats) + BallTracker
│       └── homography.py    # HomographyEstimator + projection
├── batch_processor/
│   ├── celery_app.py        # Celery configuration
│   └── worker.py            # Celery task (tracking pipeline)
├── docker/
│   ├── Dockerfile           # Multi-stage build with CUDA support
│   ├── docker-compose.yml   # Service orchestration (api, worker, redis, train)
│   └── docker-compose.gpu.yml  # GPU override
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_api.py          # API + schema tests
│   ├── test_detector.py     # Detection schema + tracking tests
│   ├── test_tracker.py      # PlayerTracker + BallTracker unit tests
│   └── test_homography.py   # HomographyEstimator + projection tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Design Decisions

### ByteTrack for Player IDs
Ultralytics `model.track(persist=True)` uses ByteTrack internally, providing stable cross-frame IDs. Used throughout the batch pipeline.

### TeamClassifier with EMA Stabilization
Instead of per-frame K-Means (which flickers with lighting changes), `TeamClassifier` maintains an exponential moving average (α=0.3) of team centroid colors and sorts by luminance to ensure consistent Team 1 / Team 2 assignment across frames.

### PlayerTracker Class
Maintains per-`tracking_id` state:
- `positions[]` — Image-coordinate history (for distance/speed)
- `field_positions[]` — Field-coordinate history (for heatmaps)
- Ball possession detection via proximity threshold (50px to player's feet)
- Stale player cleanup after 30 frames of absence

### Homography Smoothing
`HomographyEstimator` maintains a rolling window (5 frames) of homography matrices and returns their element-wise mean. This prevents jitter when keypoint detections fluctuate.

### Frontend served by FastAPI
The frontend static files (HTML, JS, CSS) are mounted in the API container and served directly by FastAPI at `/` → `/static/index.html`. No separate web server is needed.

### Celery Task ID = Job ID
The API uses `celery_app.send_task("process_video", args=[...], task_id=job_id)` so the Celery task ID matches the application job_id. This allows `AsyncResult(job_id)` to find the correct task for status polling. Tasks queued with `delay()` (which auto-generates a random task ID) are not supported — only `send_task` with explicit `task_id`.

### Celery Task Registration
`celery_app.py` explicitly imports `batch_processor.worker` to ensure the `process_video` task appears in the worker's registered task list. The video file path is passed as an argument rather than embedded in the task name.

### Status Fallback to File Existence
If Celery result backend data expires or is lost (e.g., after a Redis restart), the `_get_job_status` function in `batch.py` falls back to checking for result files:
- Both `output.mp4` and `data.json` exist → `done`
- Only `output.mp4` exists → `failed` (worker crashed mid-write)
- Neither exists → `pending`

## Configuration

All configuration is managed via environment variables (see `.env.example`):

| Variable                   | Description                    | Default                              |
|---------------------------|--------------------------------|--------------------------------------|
| `MODEL_PLAYERS_PATH`      | Path to player detection model | `/app/models/best_model_players.pt`  |
| `MODEL_KEYPOINTS_PATH`    | Path to keypoint model         | `/app/models/best_model_keypoints.pt`|
| `REDIS_URL`               | Redis connection URL           | `redis://redis:6379/0`               |
| `UPLOAD_DIR`              | Video upload directory         | `/app/data/uploads`                  |
| `RESULTS_DIR`             | Processing results directory   | `/app/data/results`                  |
| `LOG_LEVEL`               | Logging level                  | `info`                               |
