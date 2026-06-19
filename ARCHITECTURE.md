# Architecture

## Overview

Football Vision is a batch video analytics pipeline for football matches.
A user uploads a video via the web UI, a Celery worker processes it
(predictions, tracking, homography → stats), and the results are served
via FastAPI.

## System Diagram

```
  Browser  ──HTTP──▶  FastAPI (api)  ──Celery──▶  Worker  ──Redis──▶  api
       │                  │                            │
       │            ┌─────┴─────┐                ┌─────┴─────┐
       │            │  Health   │                │ Detector  │
       │            │  Batch    │                │ Classifier│
       │            │  REST API │                │ Homography│
       │            └───────────┘                │ Tracker   │
       │                                         └───────────┘
  (static files)
```

## Data Flow (batch processing)

```
Upload video → save to /data/uploads/{job_id}.mp4
           → Celery task with task_id = job_id
           → Worker reads video frame by frame (batch of 8)

For each batch of frames:
  1. Player detection model → detections (players, GK, referee, ball)
  2. ByteTrack assigns tracking IDs
  3. Keypoint detection model (32-class detection) → per-frame keypoints
  4. KeypointFilter smooths per-keypoint (EMA, jump rejection)
  5. HomographyEstimator computes H via RANSAC between detected ↔ template
  6. PlayerTracker accumulates per-player stats (distance, speed, touches)
  7. TeamClassifier assigns team colours (EMA-stabilised K-Means, LAB space)
  8. Project player positions onto 2D field overlay
  9. Write annotated frame → output video

After all frames:
  - Transcode to H.264 (libx264 + faststart) for browser playback
  - Build match/player stats JSON → /data/results/{job_id}/data.json
  - Polling UI detects "done" status via file-existence fallback
```

## Models

### Player Detection
- **Architecture**: YOLOv8x (detection)
- **Classes**: ball, goalkeeper, player, referee
- **Weights**: `models/best_model_players.pt`
- **Source**: Roboflow `football-players-detection-3zvbc` v14
- **Training**: 100 epochs, 640px

### Pitch Keypoints (Detection format)
- **Architecture**: YOLOv8s (detection, 32 independent classes)
- **Classes**: kp1 … kp32 (each keypoint is its own class with a small bbox)
- **Weights**: `models/best_model_keypoints.pt`
- **Source**: Roboflow `football-field-detection-f07vi` v10 (converted from pose to detection)
- **Training**: 100 epochs, 1280px, AdamW, batch 8
- **Validation mAP50**: 0.974
- **Fallback**: `detector.py` parses either pose or detection format automatically

Previous approach used YOLOv8n-pose (pose estimation with 32 keypoints per
instance) but was switched to detection format for better per-keypoint
accuracy.

## Key Services

### `app/services/detector.py`
- Loads both YOLO models (players + keypoints)
- `track(batch_frames)` → runs both models + ByteTrack for each frame
- `track_players(batch_frames)` → runs only player model (used in batched
  worker pipeline to avoid redundant keypoint inference)

### `app/services/classifier.py`
- `get_player_team_color()` crops top 40% of bbox (shirt area), clusters in
  LAB space, picks minority cluster (shirt colour vs background)
- `TeamClassifier` sorts centroids by luminance, applies EMA with stability
  check to avoid colour flips

### `app/services/homography.py`
- `HomographyEstimator`: uses np.median of keypoint matches, cv2.RANSAC with
  reprojThreshold=5.0, rejects if inlier_ratio < 0.4, requires min_matches=6
- `KeypointFilter`: per-keypoint EMA (alpha=0.6), rejects inter-frame jumps
  > 80 px, forgets stale entries after 15 frames
- `project_positions`: uses bbox.y2 (feet) for players, centre for ball
- Template keypoints loaded from `sample.json` (422 × 288 pixel space)

### `app/services/tracker.py`
- `PlayerTracker`: accumulates distance, speed, touches per tracking ID
- `BallTracker`: interpolates ball position across frames with gaps

## Frontend

Single-page HTML/JS/CSS served by FastAPI at `/`:

| Tab | Contents |
|-----|----------|
| **Batch** | Upload form, job history table, video player, match summary, per-player stats with heatmap on field overlay |
| **Health** | System health JSON (GPU, models, memory) |

Features:
- Live frame-count polling (`current_frame / total_frames`)
- H.264 video playback via `<video>` element
- Heatmap canvas draws on football field background

## Training

All training scripts live in `training/` and run inside a Docker container
via the `train` service profile:

```bash
docker compose run --rm train training.train_keypoints
```

Key scripts:
| Script | Purpose |
|--------|---------|
| `train_keypoints.py` | Trains keypoint detection model from YAML config |
| `train_detection.py` | Trains player detection model (YOLOv8x) |
| `field_template.py` | Generates `football_field.png` + `football_field_keypoints.png` |
| `map_keypoints.py` | Interactive tool to verify/adjust keypoint positions |
| `convert_pose_to_detection.py` | Converts pose-annotated dataset to 32-class detection format |
| `download_datasets.py` | Downloads datasets from Roboflow |

## Keypoint Mapping

The 32 keypoints follow the Roboflow `football-field-detection-f07vi` schema.
Template coordinates are stored in `sample.json` (422 × 288 px space):

```
KP1:  ( 18,   9)  Top-left corner
KP2:  ( 17,  49)  Left touchline, top of penalty box
… (see sample.json for full list)
KP14: (217,  11)  Halfway line, top touchline
…
KP31: (173, 136)  Centre circle, left edge
KP32: (260, 135)  Centre circle, right edge
```

The mapping can be interactively edited with:
```bash
python -m training.map_keypoints
```

## Field Template

- `football_field.png` (422 × 288 px) — used as heatmap overlay background
- `football_field_keypoints.png` — same image with numbered keypoints (reference)
- Regenerate after adjusting `sample.json`:
  ```bash
  python -m training.field_template
  ```

## Configuration

All config via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PLAYERS_PATH` | `/app/models/best_model_players.pt` | Player detection weights |
| `MODEL_KEYPOINTS_PATH` | `/app/models/best_model_keypoints.pt` | Keypoint detection weights |
| `FIELD_KEYPOINTS_PATH` | `/app/sample.json` | Ground-truth 2D field keypoints |
| `FIELD_IMAGE_PATH` | `/app/football_field.png` | Field template for heatmap overlay |
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker + result backend |
| `UPLOAD_DIR` | `/app/data/uploads` | Video upload directory |
| `RESULTS_DIR` | `/app/data/results` | Processing results directory |
| `MAX_UPLOAD_SIZE` | `524288000` | Max upload size (500 MB) |
| `LOG_LEVEL` | `info` | Logging verbosity |

## Tests

```bash
docker compose run --rm api pytest tests/ -v
```

Coverage:
- API endpoints (health, batch upload/status/stats/video/jobs)
- Pydantic schemas
- PlayerTracker (distance, speed, touches, possession, stale cleanup)
- BallTracker (interpolation)
- HomographyEstimator (smoothing, confidence filtering, reset)
- Detector model loading
