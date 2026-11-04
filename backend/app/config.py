"""Settings read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    job_backend: str  # "inline" runs jobs in a background thread; "rq" sends them to Redis workers
    redis_url: str
    cors_origins: list[str]
    max_upload_mb: int
    static_dir: Path | None


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("QUEST_DATA_DIR", "./data")).resolve()
    static = os.environ.get("QUEST_STATIC_DIR", "")
    return Settings(
        data_dir=data_dir,
        db_path=Path(os.environ.get("QUEST_DB_PATH", data_dir / "quest.db")),
        job_backend=os.environ.get("QUEST_JOB_BACKEND", "inline").lower(),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        cors_origins=[o.strip() for o in os.environ.get("QUEST_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
        max_upload_mb=int(os.environ.get("QUEST_MAX_UPLOAD_MB", "500")),
        static_dir=Path(static) if static else None,
    )
