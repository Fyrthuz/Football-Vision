from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_players_path: str = "/app/models/best_model_players.pt"
    model_keypoints_path: str = "/app/models/best_model_keypoints.pt"
    field_keypoints_path: str = "/app/sample.json"
    field_image_path: str = "/app/football_field.png"

    redis_url: str = "redis://redis:6379/0"
    upload_dir: str = "/app/data/uploads"
    results_dir: str = "/app/data/results"
    max_upload_size: int = 500 * 1024 * 1024

    log_level: str = "info"


settings = Settings()

UPLOAD_DIR = Path(settings.upload_dir)
RESULTS_DIR = Path(settings.results_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
