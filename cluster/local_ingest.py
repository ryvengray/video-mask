"""Local-directory task ingestion for single-worker cluster testing."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from cluster.store import ClusterStore


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".m2ts"}
DEFAULT_ARGS = ["--no-card", "--face-size", "960", "--face-int", "5", "--frame-skip", "3"]


class LocalIngestor:
    """Turns files in a shared test directory into idempotent tasks.

    This mode is intentionally only for a controller and Worker on the same
    machine. Production Workers use object-storage URLs instead.
    """

    def __init__(self, store: ClusterStore, source_dir: Path, output_dir: Path):
        self.store = store
        self.source_dir = source_dir.resolve()
        self.output_dir = output_dir.resolve()

    def scan(self) -> int:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        for source in sorted(self.source_dir.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES or source.name.startswith("masked_"):
                continue
            relative = source.relative_to(self.source_dir)
            output = self.output_dir / relative.parent / f"masked_{source.stem}.mp4"
            # Existing output is a completed local result and must not be queued again.
            if output.is_file() and output.stat().st_size > 1024:
                continue
            stat = source.stat()
            task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"local:{source}:{stat.st_size}:{stat.st_mtime_ns}"))
            try:
                self.store.create_task({
                    "task_id": task_id,
                    "source_url": source.as_uri(),
                    "output_upload_url": output.as_uri(),
                    "source_object_key": str(relative),
                    "source_size_bytes": stat.st_size,
                    "algorithm": "video_mask_batch_skip.py",
                    "arguments": DEFAULT_ARGS,
                    "output_object_key": str(output.relative_to(self.output_dir)),
                })
                created += 1
            except sqlite3.IntegrityError:
                pass  # unchanged file was already queued or completed
        return created
