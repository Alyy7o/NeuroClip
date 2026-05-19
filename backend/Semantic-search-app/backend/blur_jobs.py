"""In-memory blur job queue (one active job at a time via blur_service lock)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def create_job(job_id: str) -> None:
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "created_at": time.time(),
        }


def update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def start_blur_job(
    job_id: str,
    worker: Callable[[], None],
) -> None:
    create_job(job_id)

    def _run() -> None:
        update_job(job_id, status="processing", message="Starting", progress=1)
        try:
            worker()
        except Exception as exc:
            update_job(
                job_id,
                status="failed",
                message="Failed",
                error=str(exc),
                progress=0,
            )

    threading.Thread(target=_run, daemon=True).start()
