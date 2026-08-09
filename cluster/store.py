"""Persistent task store for the video-mask cluster controller.

The controller is a single API process and SQLite WAL keeps task claiming
atomic while remaining very easy to deploy on the controller machine.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any


TERMINAL = {"completed", "failed"}
ACTIVE = {"assigned", "downloading", "processing", "uploading"}


def now() -> float:
    return time.time()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class ClusterStore:
    def __init__(self, database: str | Path):
        self.path = str(database)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS workers (
            worker_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            capabilities_json TEXT NOT NULL DEFAULT '{}',
            current_task_id TEXT,
            last_seen_at REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            source_object_key TEXT,
            source_size_bytes INTEGER,
            source_sha256 TEXT,
            source_duration_seconds REAL,
            algorithm TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            output_upload_url TEXT NOT NULL,
            output_object_key TEXT,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            assigned_worker_id TEXT,
            progress_json TEXT NOT NULL DEFAULT '{}',
            output_sha256 TEXT,
            output_duration_seconds REAL,
            error_message TEXT,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_tasks_worker ON tasks(assigned_worker_id, status);
        """)
        self.conn.commit()

    @synchronized
    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        for name in ("arguments_json", "progress_json", "capabilities_json"):
            if name in value:
                value[name.removesuffix("_json")] = json.loads(value.pop(name) or "{}")
        return value

    @synchronized
    def provision_worker(self, worker_id: str, token: str) -> dict[str, Any]:
        """Create or rotate a Worker credential; called only by an admin API."""
        stamp = now()
        self.conn.execute("""
            INSERT INTO workers(worker_id, token_hash, status, last_seen_at, created_at, updated_at)
            VALUES(?, ?, 'offline', ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
              token_hash=excluded.token_hash, status='offline', current_task_id=NULL,
              updated_at=excluded.updated_at
        """, (worker_id, token_hash(token), stamp, stamp, stamp))
        self.conn.commit()
        return self.worker(worker_id) or {}

    @synchronized
    def register_worker(self, worker_id: str, token: str,
                        capabilities: dict[str, Any]) -> dict[str, Any]:
        stamp = now()
        existing = self.conn.execute(
            "SELECT token_hash FROM workers WHERE worker_id=?", (worker_id,)
        ).fetchone()
        if existing is None or existing[0] != token_hash(token):
            raise PermissionError("worker is not provisioned or token is invalid")
        self.conn.execute("""
            UPDATE workers SET status='ready', capabilities_json=?, last_seen_at=?, updated_at=? WHERE worker_id=?
        """, (json.dumps(capabilities), stamp, stamp, worker_id))
        self.conn.commit()
        return self.worker(worker_id) or {}

    @synchronized
    def authenticate_worker(self, worker_id: str, token: str) -> None:
        row = self.conn.execute("SELECT token_hash FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        if row is None or row[0] != token_hash(token):
            raise PermissionError("unknown worker or invalid token")

    @synchronized
    def worker(self, worker_id: str) -> dict[str, Any] | None:
        return self._row(self.conn.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone())

    @synchronized
    def heartbeat(self, worker_id: str, token: str, status: str,
                  capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
        self.authenticate_worker(worker_id, token)
        stamp = now()
        values: list[Any] = [status, stamp, stamp, worker_id]
        update = "status=?, last_seen_at=?, updated_at=?"
        if capabilities is not None:
            update += ", capabilities_json=?"
            values.insert(3, json.dumps(capabilities))
        self.conn.execute(f"UPDATE workers SET {update} WHERE worker_id=?", values)
        self.conn.commit()
        return self.worker(worker_id) or {}

    @synchronized
    def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        import uuid
        task_id = payload.get("task_id") or str(uuid.uuid4())
        args = payload.get("arguments") or []
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            raise ValueError("arguments must be a list of strings")
        required = ("source_url", "output_upload_url")
        missing = [name for name in required if not payload.get(name)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        stamp = now()
        self.conn.execute("""
            INSERT INTO tasks(task_id, source_url, source_object_key, source_size_bytes, source_sha256,
              source_duration_seconds, algorithm, arguments_json, output_upload_url, output_object_key,
              status, max_attempts, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """, (task_id, payload["source_url"], payload.get("source_object_key"),
              payload.get("source_size_bytes"), payload.get("source_sha256"),
              payload.get("source_duration_seconds"), payload.get("algorithm", "video_mask_batch_skip.py"),
              json.dumps(args), payload["output_upload_url"], payload.get("output_object_key"),
              int(payload.get("max_attempts", 3)), stamp, stamp))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def task(self, task_id: str) -> dict[str, Any] | None:
        return self._row(self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    @synchronized
    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) or {} for row in rows]

    @synchronized
    def claim(self, worker_id: str, token: str) -> dict[str, Any] | None:
        self.authenticate_worker(worker_id, token)
        stamp = now()
        # BEGIN IMMEDIATE serializes claims across concurrent API requests.
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            worker = self.conn.execute("SELECT current_task_id FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
            if worker is None:
                raise PermissionError("unknown worker")
            if worker[0]:
                current = self.task(worker[0])
                if current and current["status"] in ACTIVE:
                    self.conn.execute("COMMIT")
                    return current
            row = self.conn.execute("""
                SELECT task_id FROM tasks WHERE status='pending' AND attempt_count < max_attempts
                ORDER BY created_at LIMIT 1
            """).fetchone()
            if row is None:
                self.conn.execute("UPDATE workers SET status='ready', current_task_id=NULL, last_seen_at=?, updated_at=? WHERE worker_id=?",
                                  (stamp, stamp, worker_id))
                self.conn.execute("COMMIT")
                return None
            task_id = row[0]
            self.conn.execute("""
                UPDATE tasks SET status='assigned', assigned_worker_id=?, attempt_count=attempt_count+1,
                  started_at=COALESCE(started_at, ?), updated_at=? WHERE task_id=?
            """, (worker_id, stamp, stamp, task_id))
            self.conn.execute("UPDATE workers SET status='busy', current_task_id=?, last_seen_at=?, updated_at=? WHERE worker_id=?",
                              (task_id, stamp, stamp, worker_id))
            self.conn.execute("COMMIT")
            return self.task(task_id)
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @synchronized
    def progress(self, worker_id: str, token: str, task_id: str,
                 status: str, progress: dict[str, Any]) -> dict[str, Any]:
        self.authenticate_worker(worker_id, token)
        if status not in {"downloading", "processing", "uploading"}:
            raise ValueError("invalid active task status")
        stamp = now()
        cursor = self.conn.execute("""
            UPDATE tasks SET status=?, progress_json=?, updated_at=?
            WHERE task_id=? AND assigned_worker_id=? AND status IN ('assigned','downloading','processing','uploading')
        """, (status, json.dumps(progress), stamp, task_id, worker_id))
        if cursor.rowcount != 1:
            raise ValueError("task is not assigned to this worker")
        self.conn.execute("UPDATE workers SET last_seen_at=?, updated_at=? WHERE worker_id=?", (stamp, stamp, worker_id))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def finish(self, worker_id: str, token: str, task_id: str,
               success: bool, payload: dict[str, Any]) -> dict[str, Any]:
        self.authenticate_worker(worker_id, token)
        stamp = now()
        current = self.task(task_id)
        if current is None or current["assigned_worker_id"] != worker_id or current["status"] not in ACTIVE:
            raise ValueError("task is not assigned to this worker")
        status = "completed" if success else ("pending" if current["attempt_count"] < current["max_attempts"] else "failed")
        cursor = self.conn.execute("""
            UPDATE tasks SET status=?, output_sha256=?, output_duration_seconds=?, error_message=?,
              progress_json=?, finished_at=?, updated_at=?
            WHERE task_id=? AND assigned_worker_id=? AND status IN ('assigned','downloading','processing','uploading')
        """, (status, payload.get("output_sha256"), payload.get("output_duration_seconds"),
              payload.get("error_message"), json.dumps(payload.get("progress") or {}), stamp, stamp,
              task_id, worker_id))
        self.conn.execute("UPDATE workers SET status='ready', current_task_id=NULL, last_seen_at=?, updated_at=? WHERE worker_id=?",
                          (stamp, stamp, worker_id))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def requeue_stale(self, stale_after_seconds: int) -> int:
        threshold = now() - stale_after_seconds
        rows = self.conn.execute("SELECT worker_id, current_task_id FROM workers WHERE current_task_id IS NOT NULL AND last_seen_at < ?", (threshold,)).fetchall()
        stamp = now()
        for row in rows:
            self.conn.execute("""
                UPDATE tasks SET status=CASE WHEN attempt_count < max_attempts THEN 'pending' ELSE 'failed' END,
                  assigned_worker_id=NULL, error_message='worker heartbeat timeout', updated_at=?
                WHERE task_id=? AND status IN ('assigned','downloading','processing','uploading')
            """, (stamp, row[1]))
            self.conn.execute("UPDATE workers SET status='offline', current_task_id=NULL, updated_at=? WHERE worker_id=?", (stamp, row[0]))
        self.conn.commit()
        return len(rows)
