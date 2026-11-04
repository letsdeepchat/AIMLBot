"""RQ worker entry point: `python -m app.worker` (docker compose runs this)."""
from __future__ import annotations

import logging

from redis import Redis
from rq import Queue, Worker

from .config import load_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = load_settings()
    connection = Redis.from_url(settings.redis_url)
    Worker([Queue("quest", connection=connection)], connection=connection).work()


if __name__ == "__main__":
    main()
