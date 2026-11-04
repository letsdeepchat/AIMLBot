"""SQLite store for challenges and jobs.

Plain sqlite3 keeps v1 dependency-free. The API and RQ workers share the file through a
Docker volume, which is fine for one user; move to Postgres when more people use it.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS challenges (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    spec TEXT NOT NULL,
    task TEXT NOT NULL,
    submission_limit INTEGER,
    deadline TEXT,
    train_file TEXT NOT NULL,
    test_file TEXT NOT NULL,
    n_train INTEGER,
    n_test INTEGER,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    ref TEXT,
    status TEXT NOT NULL,
    stage TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_by_challenge ON jobs(challenge_id, created_at);
"""

ACTIVE = ("queued", "running")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            # Week-1 databases lack jobs.ref; add it in place.
            if "ref" not in {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}:
                conn.execute("ALTER TABLE jobs ADD COLUMN ref TEXT")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ challenges

    def create_challenge(self, **fields) -> dict:
        row = {
            "id": fields.pop("id", None) or uuid.uuid4().hex[:12],
            "created_at": time.time(),
            **fields,
            "spec": json.dumps(fields["spec"]),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._conn() as conn:
            conn.execute(f"INSERT INTO challenges ({cols}) VALUES ({marks})", list(row.values()))
        return self.get_challenge(row["id"])

    def get_challenge(self, challenge_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
        return _challenge(row) if row else None

    def list_challenges(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM challenges ORDER BY created_at DESC").fetchall()
        return [_challenge(r) for r in rows]

    def update_spec(self, challenge_id: str, spec: dict) -> dict:
        with self._conn() as conn:
            conn.execute("UPDATE challenges SET spec = ? WHERE id = ?", (json.dumps(spec), challenge_id))
        return self.get_challenge(challenge_id)

    def delete_challenge(self, challenge_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM challenges WHERE id = ?", (challenge_id,))

    # ------------------------------------------------------------------ jobs

    def create_job(self, challenge_id: str, kind: str, ref: str | None = None) -> dict:
        job_id = uuid.uuid4().hex[:12]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, challenge_id, kind, ref, status, created_at) VALUES (?, ?, ?, ?, 'queued', ?)",
                (job_id, challenge_id, kind, ref, time.time()),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update_job(self, job_id: str, **fields) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*fields.values(), job_id])

    def jobs_for(self, challenge_id: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE challenge_id = ? ORDER BY created_at DESC LIMIT ?", (challenge_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def active_job(self, challenge_id: str, kind: str | None = None, ref: str | None = None) -> dict | None:
        """Most recent queued or running job, optionally of one kind and for one experiment."""
        sql = "SELECT * FROM jobs WHERE challenge_id = ? AND status IN (?, ?)"
        args: list = [challenge_id, *ACTIVE]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if ref:
            sql += " AND ref = ?"
            args.append(ref)
        with self._conn() as conn:
            row = conn.execute(sql + " ORDER BY created_at DESC LIMIT 1", args).fetchone()
        return dict(row) if row else None

    def latest_job_by_ref(self, challenge_id: str) -> dict[str, dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE challenge_id = ? AND ref IS NOT NULL ORDER BY created_at ASC", (challenge_id,)
            ).fetchall()
        return {r["ref"]: dict(r) for r in rows}  # later rows overwrite earlier ones

    def fail_interrupted_jobs(self) -> int:
        """Inline jobs die with the process; mark leftovers failed so the UI doesn't wait forever."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = 'failed', error = 'Interrupted by a server restart. Run it again.', "
                "finished_at = ? WHERE status IN (?, ?)",
                (time.time(), *ACTIVE),
            )
            return cur.rowcount


def _challenge(row: sqlite3.Row) -> dict:
    out = dict(row)
    out["spec"] = json.loads(out["spec"])
    return out
