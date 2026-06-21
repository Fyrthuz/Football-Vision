from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.config import RESULTS_DIR, UPLOAD_DIR, settings
from app.schemas import BatchJobOut, BatchStats, JobInfo, JobStatus, PlayerStats

router = APIRouter(prefix="/batch", tags=["batch"])


def _get_job_status(job_id: str) -> tuple[JobStatus, float, int, int, str | None]:
    from batch_processor.celery_app import celery_app
    video_file, json_file = _get_result_files(job_id)

    result = celery_app.AsyncResult(job_id)
    meta = result.info or {}
    current = meta.get("current_frame", 0) if isinstance(meta, dict) else 0
    total = meta.get("total_frames", 0) if isinstance(meta, dict) else 0

    if result.state in ("PROGRESS", "STARTED"):
        return JobStatus.processing, meta.get("progress", 0.0), current, total, None

    if result.state == "SUCCESS":
        return JobStatus.done, 1.0, total, total, None

    if result.state == "FAILURE":
        err = meta.get("error", None) if isinstance(meta, dict) else str(meta)
        if not err:
            try:
                err = str(result.traceback) if result.traceback else "Unknown error"
            except Exception:
                err = "Unknown error"
        return JobStatus.failed, 0.0, current, total, err

    if json_file and video_file:
        return JobStatus.done, 1.0, current, total, None

    if video_file and not json_file:
        return JobStatus.failed, 0.0, 0, 0, "Output video incomplete (no data.json)"

    return JobStatus.pending, 0.0, 0, 0, None


def _get_result_files(job_id: str) -> tuple[Path | None, Path | None]:
    result_dir = RESULTS_DIR / job_id
    video = result_dir / "output.mp4"
    json_f = result_dir / "data.json"
    return video if video.exists() else None, json_f if json_f.exists() else None


@router.post("/upload", response_model=BatchJobOut)
async def upload_video(file: UploadFile, track_enabled: bool = Form(True)) -> BatchJobOut:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="File too large")

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "input.mp4"

    with open(video_path, "wb") as f:
        f.write(content)

    from batch_processor.celery_app import celery_app
    task = celery_app.send_task(
        "process_video",
        args=[str(video_path), job_id, track_enabled],
        task_id=job_id,
    )

    return BatchJobOut(
        job_id=job_id,
        status=JobStatus.pending,
        filename=file.filename or "unknown.mp4",
    )


@router.get("/status/{job_id}", response_model=BatchJobOut)
async def get_job_status(job_id: str) -> BatchJobOut:
    status, progress, current_frame, total_frames, error = _get_job_status(job_id)
    video_file, json_file = _get_result_files(job_id)

    duration = 0.0
    filename = ""
    if json_file:
        with open(json_file) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        duration = meta.get("total_frames", 0) / max(meta.get("fps", 1), 1)
        total_frames = meta.get("total_frames", total_frames)
        if status in (JobStatus.done, JobStatus.failed):
            current_frame = total_frames
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.exists():
            files = list(upload_dir.glob("*.mp4"))
            if files:
                filename = files[0].name

    return BatchJobOut(
        job_id=job_id,
        status=status,
        progress=progress,
        current_frame=current_frame,
        total_frames=total_frames,
        filename=filename,
        duration_sec=round(duration, 1),
        result_video=str(video_file) if video_file else None,
        result_json=str(json_file) if json_file else None,
        error=error,
    )


@router.get("/video/{job_id}")
async def stream_video(job_id: str):
    video_file, _ = _get_result_files(job_id)
    if not video_file:
        raise HTTPException(status_code=404, detail="Output video not found or still processing")
    return FileResponse(str(video_file), media_type="video/mp4")


@router.get("/jobs", response_model=list[JobInfo])
async def list_jobs() -> list[JobInfo]:
    jobs: list[JobInfo] = []
    if not RESULTS_DIR.exists():
        return jobs
    for job_dir in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        status, progress, current_frame, total_frames, error = _get_job_status(job_id)
        json_file = job_dir / "data.json"
        filename = ""
        duration = 0.0
        tracked = 0
        frames = 0
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            duration = meta.get("total_frames", 0) / max(meta.get("fps", 1), 1)
            frames = meta.get("total_frames", 0)
            tracked = len(data.get("player_stats", []))
            if status in (JobStatus.done, JobStatus.failed):
                current_frame = frames
                total_frames = frames
            upload_dir = UPLOAD_DIR / job_id
            if upload_dir.exists():
                files = list(upload_dir.glob("*.mp4"))
                if files:
                    filename = files[0].name
        jobs.append(JobInfo(
            job_id=job_id,
            status=status,
            progress=progress,
            current_frame=current_frame,
            total_frames=total_frames,
            filename=filename,
            duration_sec=round(duration, 1),
            tracked_players=tracked,
            frame_count=frames,
            error=error,
        ))
    return jobs


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    from batch_processor.celery_app import celery_app
    celery_app.control.revoke(job_id, terminate=True, signal="SIGKILL")
    for d in [RESULTS_DIR / job_id, UPLOAD_DIR / job_id]:
        if d.exists():
            shutil.rmtree(d)
    return {"status": "deleted", "job_id": job_id}


@router.get("/stats/{job_id}", response_model=BatchStats)
async def get_batch_stats(job_id: str) -> BatchStats:
    result_dir = RESULTS_DIR / job_id
    json_path = result_dir / "data.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Stats not found — job may still be processing")

    with open(json_path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    player_stats_raw = data.get("player_stats", [])
    team_possession = data.get("team_possession", {})

    player_stats = [
        PlayerStats(
            tracking_id=s["tracking_id"],
            label=s["label"],
            team=s.get("team"),
            total_distance=s.get("total_distance", 0.0),
            avg_speed=s.get("avg_speed", 0.0),
            top_speed=s.get("top_speed", 0.0),
            touches=s.get("touches", 0),
            heatmap_positions=s.get("heatmap_positions", [])[-300:],
        )
        for s in player_stats_raw
    ]

    total_distance = sum(s.total_distance for s in player_stats)
    total_touches = sum(s.touches for s in player_stats)
    tracked_players = len(player_stats)
    avg_speed_all = (
        sum(s.avg_speed for s in player_stats) / tracked_players
        if tracked_players > 0
        else 0.0
    )

    return BatchStats(
        total_frames=meta.get("total_frames", 0),
        fps=meta.get("fps", 0),
        frame_width=meta.get("frame_width", 0),
        frame_height=meta.get("frame_height", 0),
        duration_sec=meta.get("total_frames", 0) / max(meta.get("fps", 1), 1),
        player_stats=player_stats,
        team_possession={str(k): v for k, v in team_possession.items()},
        total_distance=round(total_distance, 2),
        avg_speed_all=round(avg_speed_all, 2),
        total_touches=total_touches,
        tracked_players=tracked_players,
    )
