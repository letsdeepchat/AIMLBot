"""Background jobs for the analysis and baseline stages.

The same `execute_job` runs either in a thread inside the API process (QUEST_JOB_BACKEND=inline,
handy for local use) or in an RQ worker (QUEST_JOB_BACKEND=rq, used by docker compose).
Progress is written to the jobs table, which the UI polls.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path

from quest_core import ChallengeSpec, run_analysis, run_baselines, run_experiment
from quest_core.advisor import run_advice
from quest_core.endgame import run_blend

from .config import Settings, load_settings
from .db import Store

log = logging.getLogger("quest.jobs")

JOB_KINDS = {"analysis": run_analysis, "baselines": run_baselines, "experiment": run_experiment, "advice": run_advice,
             "blend": run_blend}
REF_KINDS = ("experiment", "advice", "blend")  # jobs that act on one experiment or advice session
_inline_slot = threading.Semaphore(1)  # one training job at a time; they compete for the same cores


def challenge_dir(settings: Settings, challenge_id: str) -> Path:
    return settings.data_dir / "challenges" / challenge_id


def execute_job(job_id: str) -> None:
    settings = load_settings()
    store = Store(settings.db_path)
    job = store.get_job(job_id)
    challenge = store.get_challenge(job["challenge_id"]) if job else None
    if not job or not challenge:
        log.warning("Job %s or its challenge no longer exists; skipping.", job_id)
        return

    store.update_job(job_id, status="running", started_at=time.time(), stage="Starting")
    try:
        spec = ChallengeSpec(**challenge["spec"])
        args = (challenge_dir(settings, challenge["id"]), spec) + ((job["ref"],) if job["kind"] in REF_KINDS else ())
        JOB_KINDS[job["kind"]](*args, on_stage=lambda stage: store.update_job(job_id, stage=stage))
        store.update_job(job_id, status="done", stage=None, finished_at=time.time())
    except Exception as exc:
        log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
        store.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())


def _run_inline(job_id: str) -> None:
    with _inline_slot:
        execute_job(job_id)


class JobRejected(ValueError):
    """The request is valid but can't run yet; the message tells the user what to do."""


def enqueue(settings: Settings, store: Store, challenge_id: str, kind: str, ref: str | None = None) -> dict:
    if kind not in JOB_KINDS:
        raise JobRejected(f"Unknown job '{kind}'. Use one of: {', '.join(JOB_KINDS)}.")
    if kind in REF_KINDS and not ref:
        raise JobRejected(f"A {kind} job needs a reference id.")
    existing = store.active_job(challenge_id, kind, ref)
    if existing:
        return existing  # idempotent: a double click doesn't start a second run

    root = challenge_dir(settings, challenge_id)
    if kind in ("analysis", "baselines"):
        # These rewrite the reports experiments are compared against, so nothing else may be running.
        busy = store.active_job(challenge_id)
        if busy:
            raise JobRejected(f"Wait for the running {busy['kind']} to finish first.")
    if kind == "baselines" and not (root / "analysis.json").exists():
        raise JobRejected("Run the analysis first. Baselines use its validation plan.")
    if kind in REF_KINDS:
        if store.active_job(challenge_id, "analysis") or store.active_job(challenge_id, "baselines"):
            raise JobRejected("Wait for the analysis and baselines to finish first.")
        if not (root / "baselines.json").exists():
            raise JobRejected("Run the baselines first. Experiments and advice are measured against them.")

    job = store.create_job(challenge_id, kind, ref)
    if settings.job_backend == "rq":
        from redis import Redis
        from rq import Queue

        Queue("quest", connection=Redis.from_url(settings.redis_url)).enqueue(
            execute_job, job["id"], job_id=job["id"], job_timeout=6 * 3600,
        )
    else:
        threading.Thread(target=_run_inline, args=(job["id"],), daemon=True).start()
    return job
