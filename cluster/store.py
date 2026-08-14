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


TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE = {"assigned", "downloading", "processing", "uploading", "cancelling"}


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
        CREATE TABLE IF NOT EXISTS controller_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """)
        self.conn.commit()

    @synchronized
    def close(self) -> None:
        self.conn.close()

    @synchronized
    def boolean_setting(self, key: str, default: bool) -> bool:
        """Return a persistent Controller feature switch, or its safe default."""
        row = self.conn.execute(
            "SELECT setting_value FROM controller_settings WHERE setting_key=?", (key,)
        ).fetchone()
        if row is None:
            return default
        return str(row[0]).strip().lower() in {"1", "true", "yes", "on"}

    @synchronized
    def set_boolean_setting(self, key: str, value: bool) -> bool:
        """Persist a Controller feature switch across Controller restarts."""
        self.conn.execute("""
            INSERT INTO controller_settings(setting_key, setting_value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET
              setting_value=excluded.setting_value, updated_at=excluded.updated_at
        """, (key, "true" if value else "false", now()))
        self.conn.commit()
        return value

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
        """Create or rotate a Worker credential; called only by an admin API.

        Ansible calls this on every deployment.  Re-provisioning with the same
        token must preserve an active lease, otherwise a routine deployment
        briefly makes every slot look offline and hides its current task.
        """
        stamp = now()
        digest = token_hash(token)
        existing = self.conn.execute(
            "SELECT token_hash FROM workers WHERE worker_id=?", (worker_id,)
        ).fetchone()
        if existing is not None and existing[0] == digest:
            return self.worker(worker_id) or {}
        self.conn.execute("""
            INSERT INTO workers(worker_id, token_hash, status, last_seen_at, created_at, updated_at)
            VALUES(?, ?, 'offline', ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
              token_hash=excluded.token_hash, status='offline', current_task_id=NULL,
              updated_at=excluded.updated_at
        """, (worker_id, digest, stamp, stamp, stamp))
        self.conn.commit()
        return self.worker(worker_id) or {}

    @synchronized
    def retire_worker(self, worker_id: str) -> dict[str, Any]:
        """Remove a deliberately disabled idle Worker slot registration.

        This is an administrator action used after Ansible has stopped a slot
        while reducing a host's configured concurrency.  It never removes a
        slot that still has a task lease, so an accidental slot reduction
        cannot hide an active task from stale-worker recovery.
        """
        worker = self.worker(worker_id)
        if worker is None:
            return {"worker_id": worker_id, "retired": False}
        if worker.get("current_task_id") or worker.get("status") not in {"ready", "offline"}:
            raise ValueError("only an idle ready or offline worker slot can be retired")
        self.conn.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))
        self.conn.commit()
        return {"worker_id": worker_id, "retired": True}

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
    def list_workers(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute("""
            SELECT * FROM workers
            ORDER BY CASE status WHEN 'busy' THEN 0 WHEN 'cancelling' THEN 1 WHEN 'ready' THEN 2 ELSE 3 END,
                     last_seen_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [self._row(row) or {} for row in rows]

    @synchronized
    def heartbeat(self, worker_id: str, token: str, status: str,
                  capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
        self.authenticate_worker(worker_id, token)
        stamp = now()
        current_task = self.conn.execute(
            "SELECT current_task_id FROM workers WHERE worker_id=?", (worker_id,)
        ).fetchone()
        if current_task and current_task[0]:
            task = self.conn.execute("SELECT status FROM tasks WHERE task_id=?", (current_task[0],)).fetchone()
            if task and task[0] == "cancelling":
                status = "cancelling"
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
        required = ("source_url",)
        missing = [name for name in required if not payload.get(name)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        stamp = now()
        try:
            self.conn.execute("""
                INSERT INTO tasks(task_id, source_url, source_object_key, source_size_bytes, source_sha256,
                  source_duration_seconds, algorithm, arguments_json, output_upload_url, output_object_key,
                  status, max_attempts, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """, (task_id, payload["source_url"], payload.get("source_object_key"),
                  payload.get("source_size_bytes"), payload.get("source_sha256"),
                  payload.get("source_duration_seconds"), payload.get("algorithm", "video_mask_batch_skip.py"),
                  json.dumps(args), payload.get("output_upload_url") or "", payload.get("output_object_key"),
                  int(payload.get("max_attempts", 3)), stamp, stamp))
            self.conn.commit()
        except Exception:
            # A duplicate local-ingest task is expected and is handled by the
            # caller.  SQLite keeps the failed INSERT transaction open unless
            # it is rolled back, which would otherwise block the next claim.
            self.conn.rollback()
            raise
        return self.task(task_id) or {}

    @synchronized
    def retry_task(self, task_id: str) -> dict[str, Any]:
        """Return a failed task to the queue with a fresh retry budget."""
        current = self.task(task_id)
        if current is None:
            raise ValueError("task does not exist")
        if current["status"] != "failed":
            raise ValueError("only failed tasks can be retried")
        stamp = now()
        self.conn.execute("""
            UPDATE tasks SET status='pending', attempt_count=0, assigned_worker_id=NULL,
              progress_json='{}', error_message=NULL, started_at=NULL, finished_at=NULL, updated_at=?
            WHERE task_id=?
        """, (stamp, task_id))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def restart_task(self, task_id: str) -> dict[str, Any]:
        """Queue a terminal task again, including a completed task on request."""
        current = self.task(task_id)
        if current is None:
            raise ValueError("task does not exist")
        if current["status"] not in TERMINAL:
            raise ValueError("only completed, failed, or cancelled tasks can be restarted")
        stamp = now()
        self.conn.execute("""
            UPDATE tasks SET status='pending', attempt_count=0, assigned_worker_id=NULL,
              progress_json='{}', output_sha256=NULL, output_duration_seconds=NULL,
              error_message=NULL, started_at=NULL, finished_at=NULL, updated_at=?
            WHERE task_id=?
        """, (stamp, task_id))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a queued task immediately or request cancellation of an active task."""
        current = self.task(task_id)
        if current is None:
            raise ValueError("task does not exist")
        status = str(current["status"])
        stamp = now()
        if status == "pending":
            self.conn.execute("""
                UPDATE tasks SET status='cancelled', error_message='cancelled by administrator',
                  finished_at=?, updated_at=? WHERE task_id=?
            """, (stamp, stamp, task_id))
        elif status in ACTIVE - {"cancelling"}:
            self.conn.execute("""
                UPDATE tasks SET status='cancelling', error_message='cancellation requested by administrator',
                  updated_at=? WHERE task_id=?
            """, (stamp, task_id))
            self.conn.execute("""
                UPDATE workers SET status='cancelling', updated_at=?
                WHERE current_task_id=?
            """, (stamp, task_id))
        elif status == "cancelling":
            return current
        else:
            raise ValueError("task is already terminal")
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def purge_tasks(self) -> dict[str, int]:
        """Permanently remove all terminal and queued task records.

        Active leases are deliberately protected: deleting an assigned or
        processing task would leave its Worker unable to report progress or
        completion.  Cancel those tasks and wait for them to finish first.
        """
        active_count = int(self.conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status IN (?, ?, ?, ?, ?)",
            tuple(sorted(ACTIVE)),
        ).fetchone()[0])
        if active_count:
            raise ValueError(
                f"cannot delete all tasks while {active_count} task(s) are active; "
                "cancel them and wait for Workers to finish first"
            )
        count = int(self.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        self.conn.execute("DELETE FROM tasks")
        # A Controller restarted during a completed lease can leave a stale
        # Worker reference.  With no active task left, clear only such stale
        # busy/cancelling state before production work begins.
        self.conn.execute("""
            UPDATE workers
            SET current_task_id=NULL,
                status=CASE WHEN status IN ('busy', 'cancelling') THEN 'ready' ELSE status END,
                updated_at=?
            WHERE current_task_id IS NOT NULL OR status IN ('busy', 'cancelling')
        """, (now(),))
        self.conn.commit()
        return {"deleted_tasks": count}

    @synchronized
    def task(self, task_id: str) -> dict[str, Any] | None:
        return self._row(self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    @synchronized
    def active_task_for_worker(self, worker_id: str, token: str, task_id: str) -> dict[str, Any]:
        """Return a task only while its lease belongs to this Worker."""
        self.authenticate_worker(worker_id, token)
        task = self.task(task_id)
        if task is None or task["assigned_worker_id"] != worker_id or task["status"] not in ACTIVE:
            raise ValueError("task is not actively assigned to this worker")
        return task

    @synchronized
    def count_tasks(self, statuses: tuple[str, ...] | None = None) -> int:
        if not statuses:
            return int(self.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        placeholders = ", ".join("?" for _ in statuses)
        return int(self.conn.execute(
            f"SELECT COUNT(*) FROM tasks WHERE status IN ({placeholders})", statuses
        ).fetchone()[0])

    @synchronized
    def oldest_task_created_at(self, statuses: tuple[str, ...]) -> float | None:
        """Return the oldest queued task timestamp for autoscaling decisions."""
        if not statuses:
            return None
        placeholders = ", ".join("?" for _ in statuses)
        value = self.conn.execute(
            f"SELECT MIN(created_at) FROM tasks WHERE status IN ({placeholders})", statuses
        ).fetchone()[0]
        return float(value) if value is not None else None

    @synchronized
    def list_tasks(self, limit: int = 100, offset: int = 0,
                   statuses: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        where = ""
        values: list[Any] = []
        if statuses:
            where = " WHERE status IN (" + ", ".join("?" for _ in statuses) + ")"
            values.extend(statuses)
        values.extend((limit, max(0, offset)))
        rows = self.conn.execute(
            "SELECT * FROM tasks" + where + " ORDER BY created_at DESC, task_id DESC LIMIT ? OFFSET ?",
            values,
        ).fetchall()
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
        if current["status"] == "cancelling":
            status = "cancelled"
        else:
            status = "completed" if success else ("pending" if current["attempt_count"] < current["max_attempts"] else "failed")
        cursor = self.conn.execute("""
            UPDATE tasks SET status=?, output_sha256=?, output_duration_seconds=?, error_message=?,
              progress_json=?, finished_at=?, updated_at=?
            WHERE task_id=? AND assigned_worker_id=? AND status IN ('assigned','downloading','processing','uploading','cancelling')
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
        recovered = 0
        for row in rows:
            cursor = self.conn.execute("""
                UPDATE tasks SET status=CASE WHEN status='cancelling' THEN 'cancelled'
                  WHEN attempt_count < max_attempts THEN 'pending' ELSE 'failed' END,
                  assigned_worker_id=NULL,
                  progress_json=CASE WHEN status='cancelling' THEN progress_json ELSE '{}' END,
                  output_sha256=CASE WHEN status='cancelling' THEN output_sha256 ELSE NULL END,
                  output_duration_seconds=CASE WHEN status='cancelling' THEN output_duration_seconds ELSE NULL END,
                  error_message=CASE WHEN status='cancelling' THEN 'cancelled by administrator (worker heartbeat timed out)'
                    WHEN attempt_count < max_attempts THEN NULL ELSE 'worker heartbeat timeout' END,
                  started_at=CASE WHEN status='cancelling' THEN started_at WHEN attempt_count < max_attempts THEN NULL ELSE started_at END,
                  finished_at=CASE WHEN status='cancelling' THEN ? WHEN attempt_count < max_attempts THEN NULL ELSE ? END,
                  updated_at=?
                WHERE task_id=? AND status IN ('assigned','downloading','processing','uploading','cancelling')
            """, (stamp, stamp, stamp, row[1]))
            recovered += cursor.rowcount
            self.conn.execute("UPDATE workers SET status='offline', current_task_id=NULL, updated_at=? WHERE worker_id=?", (stamp, row[0]))

        # A slot can disappear or lose its lease reference outside the normal
        # finish path (for example after a host is removed).  Do not leave its
        # active task stuck forever; wait one stale period to avoid recovering
        # a transient inconsistency while an API request is in flight.
        orphaned = self.conn.execute("""
            SELECT t.task_id FROM tasks AS t
            LEFT JOIN workers AS w ON w.worker_id=t.assigned_worker_id
            WHERE t.status IN ('assigned','downloading','processing','uploading','cancelling')
              AND t.updated_at < ?
              AND (w.worker_id IS NULL OR w.current_task_id IS NULL OR w.current_task_id != t.task_id)
        """, (threshold,)).fetchall()
        for row in orphaned:
            cursor = self.conn.execute("""
                UPDATE tasks SET status=CASE WHEN status='cancelling' THEN 'cancelled'
                  WHEN attempt_count < max_attempts THEN 'pending' ELSE 'failed' END,
                  assigned_worker_id=NULL,
                  progress_json=CASE WHEN status='cancelling' THEN progress_json ELSE '{}' END,
                  output_sha256=CASE WHEN status='cancelling' THEN output_sha256 ELSE NULL END,
                  output_duration_seconds=CASE WHEN status='cancelling' THEN output_duration_seconds ELSE NULL END,
                  error_message=CASE WHEN status='cancelling' THEN 'cancelled by administrator (orphaned worker slot)'
                    WHEN attempt_count < max_attempts THEN NULL ELSE 'worker lease was orphaned' END,
                  started_at=CASE WHEN status='cancelling' THEN started_at WHEN attempt_count < max_attempts THEN NULL ELSE started_at END,
                  finished_at=CASE WHEN status='cancelling' THEN ? WHEN attempt_count < max_attempts THEN NULL ELSE ? END,
                  updated_at=?
                WHERE task_id=? AND status IN ('assigned','downloading','processing','uploading','cancelling')
            """, (stamp, stamp, stamp, row[0]))
            recovered += cursor.rowcount
        # Idle Workers also need to become offline when their process or host
        # disappears.  Previously only Workers with a leased task changed
        # state, leaving old ready records misleadingly visible forever.
        self.conn.execute("""
            UPDATE workers SET status='offline', updated_at=?
            WHERE current_task_id IS NULL AND last_seen_at < ? AND status != 'offline'
        """, (stamp, threshold))
        self.conn.commit()
        return recovered
