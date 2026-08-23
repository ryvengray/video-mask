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
            restarted_at REAL,
            finished_at REAL,
            face_annotation INTEGER,
            face_annotated_at REAL,
            content_tags_json TEXT,
            content_tagged_at REAL,
            face_review_owner TEXT,
            face_review_lease_until REAL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_tasks_worker ON tasks(assigned_worker_id, status);
        CREATE INDEX IF NOT EXISTS idx_tasks_finished ON tasks(finished_at, status);
        CREATE INDEX IF NOT EXISTS idx_tasks_started ON tasks(started_at);
        CREATE TABLE IF NOT EXISTS task_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
            worker_id TEXT NOT NULL,
            attempt_count INTEGER NOT NULL,
            created_at REAL NOT NULL,
            line TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_logs_task_id ON task_logs(task_id, id);
        CREATE TABLE IF NOT EXISTS controller_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS algorithm_defaults (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            algorithm TEXT NOT NULL DEFAULT 'video_mask_batch_fish_v1.py',
            arguments_json TEXT NOT NULL DEFAULT '[]',
            updated_at REAL NOT NULL
        );
        """)
        self.conn.execute("""
            INSERT INTO algorithm_defaults (id, algorithm, arguments_json, updated_at)
            SELECT 1, 'video_mask_batch_fish_v1.py',
                   '["--fisheye","--fisheye-device","pico4","--face-size","960",
                     "--face-int","5","--face-conf","0.4","--frame-skip","1",
                     "--face-model","yolov8+yolo11","--dual-iou","0.4"]', ?
            WHERE NOT EXISTS (SELECT 1 FROM algorithm_defaults WHERE id = 1)
        """, (now(),))
        task_columns = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(tasks)")}
        if "restarted_at" not in task_columns:
            self.conn.execute("ALTER TABLE tasks ADD COLUMN restarted_at REAL")
        for column, definition in (
            ("face_annotation", "INTEGER"),
            ("face_annotated_at", "REAL"),
            ("content_tags_json", "TEXT"),
            ("content_tagged_at", "REAL"),
            ("face_review_owner", "TEXT"),
            ("face_review_lease_until", "REAL"),
        ):
            if column not in task_columns:
                self.conn.execute(f"ALTER TABLE tasks ADD COLUMN {column} {definition}")
        # This must run after the ALTER TABLE migration above: existing
        # Controller databases do not have the annotation columns yet.
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_face_review
              ON tasks(status, face_annotation, face_review_owner, finished_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_content_review
              ON tasks(status, content_tags_json, face_review_owner, finished_at)
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
        if "content_tags_json" in value:
            value["content_tags"] = json.loads(value.pop("content_tags_json") or "[]")
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
    def get_algorithm_defaults(self) -> dict[str, Any]:
        """Return the current global default algorithm and arguments."""
        row = self.conn.execute(
            "SELECT algorithm, arguments_json FROM algorithm_defaults WHERE id=1"
        ).fetchone()
        if row is None:
            return {"algorithm": "video_mask_batch_fish_v1.py", "arguments": []}
        return {"algorithm": str(row[0]), "arguments": json.loads(row[1] or "[]")}

    @synchronized
    def set_algorithm_defaults(self, algorithm: str, arguments: list[str]) -> dict[str, Any]:
        """Update the global default algorithm and arguments."""
        stamp = now()
        self.conn.execute("""
            INSERT INTO algorithm_defaults (id, algorithm, arguments_json, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              algorithm=excluded.algorithm,
              arguments_json=excluded.arguments_json,
              updated_at=excluded.updated_at
        """, (algorithm, json.dumps(arguments), stamp))
        self.conn.commit()
        return self.get_algorithm_defaults()

    @synchronized
    def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        import uuid
        task_id = payload.get("task_id") or str(uuid.uuid4())
        defaults = self.get_algorithm_defaults()
        algorithm = payload.get("algorithm")
        if not algorithm:
            algorithm = defaults["algorithm"]
        args = payload.get("arguments")
        if args is None:
            args = defaults["arguments"]
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
                  payload.get("source_duration_seconds"), algorithm,
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
        self.conn.execute("DELETE FROM task_logs WHERE task_id=?", (task_id,))
        self.conn.execute("""
            UPDATE tasks SET status='pending', attempt_count=0, assigned_worker_id=NULL,
              progress_json='{}', error_message=NULL, started_at=NULL, restarted_at=NULL,
              finished_at=NULL, updated_at=?
            WHERE task_id=?
        """, (stamp, task_id))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def restart_task(self, task_id: str, algorithm: str | None = None,
                     arguments: list[str] | None = None) -> dict[str, Any]:
        """Queue a terminal task again, optionally with updated algorithm settings."""
        current = self.task(task_id)
        if current is None:
            raise ValueError("task does not exist")
        if current["status"] not in TERMINAL:
            raise ValueError("only completed, failed, or cancelled tasks can be restarted")
        if (algorithm is None) != (arguments is None):
            raise ValueError("algorithm and arguments must be updated together")
        stamp = now()
        self.conn.execute("DELETE FROM task_logs WHERE task_id=?", (task_id,))
        self.conn.execute("""
            UPDATE tasks SET status='pending', attempt_count=0, assigned_worker_id=NULL,
              progress_json='{}', output_sha256=NULL, output_duration_seconds=NULL,
              error_message=NULL, started_at=NULL, restarted_at=?, finished_at=NULL,
              face_review_owner=NULL, face_review_lease_until=NULL,
              algorithm=COALESCE(?, algorithm), arguments_json=COALESCE(?, arguments_json),
              updated_at=?
            WHERE task_id=?
        """, (stamp, algorithm, json.dumps(arguments) if arguments is not None else None, stamp, task_id))
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def restart_completed_tasks(self) -> dict[str, Any]:
        """Queue every completed task again and retain the bulk restart time."""
        stamp = now()
        self.conn.execute("DELETE FROM task_logs WHERE task_id IN (SELECT task_id FROM tasks WHERE status='completed')")
        cursor = self.conn.execute("""
            UPDATE tasks SET status='pending', attempt_count=0, assigned_worker_id=NULL,
              progress_json='{}', output_sha256=NULL, output_duration_seconds=NULL,
              error_message=NULL, started_at=NULL, restarted_at=?, finished_at=NULL,
              face_review_owner=NULL, face_review_lease_until=NULL, updated_at=?
            WHERE status='completed'
        """, (stamp, stamp))
        self.conn.commit()
        return {"restarted_tasks": cursor.rowcount, "restarted_at": stamp}

    def _expire_face_review_leases(self, stamp: float) -> None:
        self.conn.execute("""
            UPDATE tasks SET face_review_owner=NULL, face_review_lease_until=NULL, updated_at=?
            WHERE face_review_lease_until IS NOT NULL AND face_review_lease_until <= ?
        """, (stamp, stamp))

    @synchronized
    def claim_next_face_review(self, reviewer_id: str, lease_seconds: int) -> dict[str, Any] | None:
        """Atomically reserve a random completed video needing a manual label."""
        stamp = now()
        lease_until = stamp + lease_seconds
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self._expire_face_review_leases(stamp)
            row = self.conn.execute("""
                SELECT task_id FROM tasks
                WHERE status='completed' AND (face_annotation IS NULL OR content_tags_json IS NULL)
                  AND source_object_key IS NOT NULL
                  AND face_review_owner IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """).fetchone()
            if row is None:
                self.conn.execute("COMMIT")
                return None
            task_id = str(row[0])
            self.conn.execute("""
                UPDATE tasks SET face_review_owner=?, face_review_lease_until=?, updated_at=?
                WHERE task_id=?
            """, (reviewer_id, lease_until, stamp, task_id))
            self.conn.execute("COMMIT")
            return self.task(task_id)
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @synchronized
    def claim_face_review(self, task_id: str, reviewer_id: str, lease_seconds: int) -> dict[str, Any]:
        """Reserve one completed S3 task for editing its current face label."""
        stamp = now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self._expire_face_review_leases(stamp)
            task = self.task(task_id)
            if task is None:
                raise ValueError("task does not exist")
            if task["status"] != "completed" or not task.get("source_object_key"):
                raise ValueError("only completed S3 tasks can be manually labelled")
            owner = task.get("face_review_owner")
            if owner not in {None, reviewer_id}:
                raise ValueError("this video is currently being labelled by another browser")
            self.conn.execute("""
                UPDATE tasks SET face_review_owner=?, face_review_lease_until=?, updated_at=? WHERE task_id=?
            """, (reviewer_id, stamp + lease_seconds, stamp, task_id))
            self.conn.execute("COMMIT")
            return self.task(task_id) or {}
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @synchronized
    def renew_face_review(self, task_id: str, reviewer_id: str, lease_seconds: int) -> dict[str, Any]:
        stamp = now()
        self._expire_face_review_leases(stamp)
        cursor = self.conn.execute("""
            UPDATE tasks SET face_review_lease_until=?, updated_at=?
            WHERE task_id=? AND face_review_owner=?
        """, (stamp + lease_seconds, stamp, task_id, reviewer_id))
        if cursor.rowcount != 1:
            self.conn.commit()
            raise ValueError("face-review lease is no longer held by this browser")
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def release_face_review(self, task_id: str, reviewer_id: str) -> bool:
        stamp = now()
        cursor = self.conn.execute("""
            UPDATE tasks SET face_review_owner=NULL, face_review_lease_until=NULL, updated_at=?
            WHERE task_id=? AND face_review_owner=?
        """, (stamp, task_id, reviewer_id))
        self.conn.commit()
        return cursor.rowcount == 1

    @synchronized
    def annotate_face(self, task_id: str, reviewer_id: str, has_face: bool) -> dict[str, Any]:
        stamp = now()
        self._expire_face_review_leases(stamp)
        cursor = self.conn.execute("""
            UPDATE tasks SET face_annotation=?, face_annotated_at=?, face_review_owner=NULL,
              face_review_lease_until=NULL, updated_at=?
            WHERE task_id=? AND status='completed' AND face_review_owner=?
        """, (1 if has_face else 0, stamp, stamp, task_id, reviewer_id))
        if cursor.rowcount != 1:
            self.conn.commit()
            raise ValueError("face-review lease expired or this task is no longer available")
        self.conn.commit()
        return self.task(task_id) or {}

    @staticmethod
    def _normalise_content_tags(tags: list[str]) -> list[str]:
        """Validate and de-duplicate human-entered video-content labels."""
        if not isinstance(tags, list) or not tags:
            raise ValueError("provide at least one content tag")
        if len(tags) > 20:
            raise ValueError("at most 20 content tags may be saved")
        result: list[str] = []
        seen: set[str] = set()
        for value in tags:
            if not isinstance(value, str):
                raise ValueError("content tags must be text")
            tag = " ".join(value.split())
            if not tag or len(tag) > 64:
                raise ValueError("each content tag must contain 1 to 64 characters")
            key = tag.casefold()
            if key not in seen:
                seen.add(key)
                result.append(tag)
        if not result:
            raise ValueError("provide at least one content tag")
        return result

    @synchronized
    def annotate_content_tags(self, task_id: str, reviewer_id: str, tags: list[str]) -> dict[str, Any]:
        """Save content labels while retaining the current review lease."""
        normalised_tags = self._normalise_content_tags(tags)
        stamp = now()
        self._expire_face_review_leases(stamp)
        cursor = self.conn.execute("""
            UPDATE tasks SET content_tags_json=?, content_tagged_at=?, updated_at=?
            WHERE task_id=? AND status='completed' AND face_review_owner=?
        """, (json.dumps(normalised_tags, ensure_ascii=False), stamp, stamp, task_id, reviewer_id))
        if cursor.rowcount != 1:
            self.conn.commit()
            raise ValueError("face-review lease expired or this task is no longer available")
        self.conn.commit()
        return self.task(task_id) or {}

    @synchronized
    def content_tags(self, limit: int = 500) -> list[str]:
        """Return distinct labels already used, newest first, for the picker."""
        rows = self.conn.execute("""
            SELECT content_tags_json FROM tasks
            WHERE content_tags_json IS NOT NULL
            ORDER BY content_tagged_at DESC, updated_at DESC
            LIMIT ?
        """, (max(1, min(limit, 10_000)),)).fetchall()
        tags: list[str] = []
        seen: set[str] = set()
        for row in rows:
            try:
                values = json.loads(row[0])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str):
                    continue
                tag = " ".join(value.split())
                if tag and tag.casefold() not in seen:
                    seen.add(tag.casefold())
                    tags.append(tag)
        return tags

    @synchronized
    def face_review_status(self, task_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
        stamp = now()
        self._expire_face_review_leases(stamp)
        if task_ids:
            placeholders = ", ".join("?" for _ in task_ids)
            rows = self.conn.execute(f"""
                SELECT task_id, status, source_object_key, face_annotation, face_review_owner,
                  face_review_lease_until FROM tasks
                WHERE task_id IN ({placeholders})
            """, task_ids).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT task_id, status, source_object_key, face_annotation, face_review_owner,
                  face_review_lease_until FROM tasks
                WHERE face_annotation IS NULL AND face_review_owner IS NOT NULL
                ORDER BY face_review_lease_until
            """).fetchall()
        self.conn.commit()
        return {"reviews": [
            {
                "task_id": str(row[0]),
                "reviewable": row[1] == "completed" and bool(row[2]),
                "has_face": None if row[3] is None else bool(row[3]),
                "reviewing": bool(row[4] and float(row[5] or 0) > stamp),
                "lease_until": float(row[5] or 0),
            }
            for row in rows
        ]}

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

    @staticmethod
    def _task_filter_conditions(statuses: tuple[str, ...] | None,
                                search: str | None,
                                face_annotations: tuple[str, ...] | None = None) -> tuple[str, list[Any]]:
        """Build a WHERE clause from optional task, label, and text filters."""
        conditions: list[str] = []
        values: list[Any] = []
        if statuses:
            conditions.append("status IN (" + ", ".join("?" for _ in statuses) + ")")
            values.extend(statuses)
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like = f"%{escaped}%"
            columns = ("task_id", "source_object_key", "source_url", "output_object_key", "progress_json")
            conditions.append(
                "(" + " OR ".join(f"IFNULL({column}, '') LIKE ? ESCAPE '\\'" for column in columns) + ")")
            values.extend([like] * len(columns))
        if face_annotations:
            annotation_conditions: list[str] = []
            if "has_face" in face_annotations:
                annotation_conditions.append("face_annotation=1")
            if "no_face" in face_annotations:
                annotation_conditions.append("face_annotation=0")
            if "unlabelled" in face_annotations:
                annotation_conditions.append("face_annotation IS NULL")
            if annotation_conditions:
                conditions.append("(" + " OR ".join(annotation_conditions) + ")")
        return (" WHERE " + " AND ".join(conditions)) if conditions else "", values

    @synchronized
    def count_tasks(self, statuses: tuple[str, ...] | None = None,
                    search: str | None = None,
                    face_annotations: tuple[str, ...] | None = None) -> int:
        where, values = self._task_filter_conditions(statuses, search, face_annotations)
        return int(self.conn.execute("SELECT COUNT(*) FROM tasks" + where, values).fetchone()[0])

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
                   statuses: tuple[str, ...] | None = None,
                   search: str | None = None,
                   face_annotations: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        where, values = self._task_filter_conditions(statuses, search, face_annotations)
        values.extend((limit, max(0, offset)))
        rows = self.conn.execute(
            "SELECT * FROM tasks" + where + " ORDER BY created_at DESC, task_id DESC LIMIT ? OFFSET ?",
            values,
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    @synchronized
    def processing_statistics(self, start_at: float, end_at: float,
                              bucket_seconds: int = 3600) -> dict[str, Any]:
        """Aggregate completed output and Worker use over a selected time range.

        ``completed_*`` metrics use a task's completion time.  Concurrency is
        reconstructed from task start/end intervals, so it measures in-flight
        Worker slots (download, inference and upload).  The separate algorithm
        value uses the Worker-reported ``processing_seconds`` where available.
        """
        if end_at <= start_at:
            raise ValueError("statistics end time must be after its start time")
        if bucket_seconds < 60:
            raise ValueError("statistics bucket must be at least 60 seconds")

        bucket_start = int(start_at // bucket_seconds) * bucket_seconds
        bucket_count = max(1, int((end_at - bucket_start + bucket_seconds - 1) // bucket_seconds))
        hourly = [{
            "start_at": bucket_start + index * bucket_seconds,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "completed_video_seconds": 0.0,
            "in_flight_worker_seconds": 0.0,
            "algorithm_worker_seconds": 0.0,
            "peak_concurrency": 0,
        } for index in range(bucket_count)]

        finished_rows = self.conn.execute("""
            SELECT * FROM tasks
            WHERE finished_at >= ? AND finished_at < ?
              AND status IN ('completed', 'failed', 'cancelled')
            ORDER BY finished_at
        """, (start_at, end_at)).fetchall()
        finished_tasks = [self._row(row) or {} for row in finished_rows]

        completed = [task for task in finished_tasks if task.get("status") == "completed"]
        failed = [task for task in finished_tasks if task.get("status") == "failed"]
        total_video_seconds = sum(
            float(task.get("output_duration_seconds") or 0)
            for task in completed if float(task.get("output_duration_seconds") or 0) > 0
        )
        total_processing_seconds = sum(
            float((task.get("progress") or {}).get("processing_seconds") or 0)
            for task in completed if float((task.get("progress") or {}).get("processing_seconds") or 0) > 0
        )
        total_end_to_end_seconds = sum(
            max(0.0, float(task.get("finished_at") or 0) - float(task.get("started_at") or 0))
            for task in completed
            if task.get("started_at") is not None and task.get("finished_at") is not None
        )
        total_input_bytes = sum(
            int((task.get("progress") or {}).get("input_bytes") or task.get("source_size_bytes") or 0)
            for task in completed
        )
        total_output_bytes = sum(
            int((task.get("progress") or {}).get("output_bytes") or 0)
            for task in completed
        )

        def server_name(worker_id: Any) -> str:
            value = str(worker_id or "unassigned")
            prefix, marker, slot = value.rpartition("-slot-")
            return prefix if marker and prefix and slot.isdigit() else value

        worker_data: dict[str, dict[str, Any]] = {}
        for task in finished_tasks:
            worker = server_name(task.get("assigned_worker_id"))
            row = worker_data.setdefault(worker, {
                "worker": worker, "slots": set(), "completed_tasks": 0, "failed_tasks": 0,
                "video_seconds": 0.0, "processing_seconds": 0.0, "end_to_end_seconds": 0.0,
            })
            worker_id = str(task.get("assigned_worker_id") or "")
            if worker_id:
                row["slots"].add(worker_id)
            if task.get("status") == "failed":
                row["failed_tasks"] += 1
                continue
            if task.get("status") != "completed":
                continue
            row["completed_tasks"] += 1
            row["video_seconds"] += float(task.get("output_duration_seconds") or 0)
            progress = task.get("progress") or {}
            row["processing_seconds"] += float(progress.get("processing_seconds") or 0)
            if task.get("started_at") is not None and task.get("finished_at") is not None:
                row["end_to_end_seconds"] += max(0.0, float(task["finished_at"]) - float(task["started_at"]))

        worker_rows = []
        for row in worker_data.values():
            video_seconds = row["video_seconds"]
            processing_seconds = row["processing_seconds"]
            end_to_end_seconds = row["end_to_end_seconds"]
            worker_rows.append({
                "worker": row["worker"],
                "slot_count": len(row["slots"]),
                "completed_tasks": row["completed_tasks"],
                "failed_tasks": row["failed_tasks"],
                "video_seconds": video_seconds,
                "processing_seconds": processing_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "algorithm_realtime": video_seconds / processing_seconds if processing_seconds else None,
                "worker_hours_per_video_hour": processing_seconds / video_seconds if video_seconds else None,
                "end_to_end_realtime": video_seconds / end_to_end_seconds if end_to_end_seconds else None,
            })
        worker_rows.sort(key=lambda row: (-row["video_seconds"], row["worker"]))

        # Completed media and failures are attributed to the hour in which the
        # terminal state was recorded.  Task concurrency uses all overlapping
        # work, including tasks that started before or end after this range.
        for task in finished_tasks:
            finished_at = float(task.get("finished_at") or 0)
            bucket = int((finished_at - bucket_start) // bucket_seconds)
            if not 0 <= bucket < bucket_count:
                continue
            if task.get("status") == "completed":
                hourly[bucket]["completed_tasks"] += 1
                hourly[bucket]["completed_video_seconds"] += float(task.get("output_duration_seconds") or 0)
            elif task.get("status") == "failed":
                hourly[bucket]["failed_tasks"] += 1

        overlap_rows = self.conn.execute("""
            SELECT * FROM tasks
            WHERE started_at IS NOT NULL AND started_at < ?
              AND (finished_at IS NULL OR finished_at > ?)
        """, (end_at, start_at)).fetchall()
        events: list[list[tuple[float, int]]] = [[] for _ in hourly]
        for raw in overlap_rows:
            task = self._row(raw) or {}
            interval_start = max(start_at, float(task.get("started_at") or start_at))
            interval_end = min(end_at, float(task.get("finished_at") or end_at))
            if interval_end <= interval_start:
                continue
            first = max(0, int((interval_start - bucket_start) // bucket_seconds))
            last = min(bucket_count - 1, int((interval_end - 0.000001 - bucket_start) // bucket_seconds))
            for index in range(first, last + 1):
                period_start = max(interval_start, hourly[index]["start_at"])
                period_end = min(interval_end, hourly[index]["start_at"] + bucket_seconds)
                if period_end <= period_start:
                    continue
                hourly[index]["in_flight_worker_seconds"] += period_end - period_start
                events[index].append((period_start, 1))
                events[index].append((period_end, -1))

            progress = task.get("progress") or {}
            processing_seconds = float(progress.get("processing_seconds") or 0)
            process_start = progress.get("processing_started_at", task.get("started_at"))
            if processing_seconds > 0 and isinstance(process_start, (int, float)):
                algorithm_start = max(start_at, float(process_start))
                algorithm_end = min(end_at, float(process_start) + processing_seconds)
                if algorithm_end > algorithm_start:
                    first = max(0, int((algorithm_start - bucket_start) // bucket_seconds))
                    last = min(bucket_count - 1, int((algorithm_end - 0.000001 - bucket_start) // bucket_seconds))
                    for index in range(first, last + 1):
                        period_start = max(algorithm_start, hourly[index]["start_at"])
                        period_end = min(algorithm_end, hourly[index]["start_at"] + bucket_seconds)
                        if period_end > period_start:
                            hourly[index]["algorithm_worker_seconds"] += period_end - period_start

        for index, bucket_events in enumerate(events):
            running = peak = 0
            # End events first at a shared timestamp: intervals are [start, end).
            for _, change in sorted(bucket_events, key=lambda item: (item[0], item[1])):
                running += change
                peak = max(peak, running)
            hourly[index]["peak_concurrency"] = peak
            hourly[index]["average_concurrency"] = hourly[index]["in_flight_worker_seconds"] / bucket_seconds

        range_seconds = end_at - start_at
        return {
            "start_at": start_at,
            "end_at": end_at,
            "bucket_seconds": bucket_seconds,
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "success_rate": len(completed) / (len(completed) + len(failed)) if completed or failed else None,
            "retry_count": sum(max(0, int(task.get("attempt_count") or 0) - 1) for task in finished_tasks),
            "video_seconds": total_video_seconds,
            "input_bytes": total_input_bytes,
            "output_bytes": total_output_bytes,
            "processing_seconds": total_processing_seconds,
            "end_to_end_seconds": total_end_to_end_seconds,
            "calendar_realtime": total_video_seconds / range_seconds if range_seconds else None,
            "algorithm_realtime": total_video_seconds / total_processing_seconds if total_processing_seconds else None,
            "worker_hours_per_video_hour": total_processing_seconds / total_video_seconds if total_video_seconds else None,
            "hourly": hourly,
            "workers": worker_rows,
        }

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
    def append_task_logs(self, worker_id: str, token: str, task_id: str,
                         lines: list[str]) -> int:
        """Persist a bounded batch of Worker algorithm output for the dashboard."""
        self.authenticate_worker(worker_id, token)
        task = self.task(task_id)
        if task is None or task.get("assigned_worker_id") != worker_id or task.get("status") not in ACTIVE:
            raise ValueError("task is not actively assigned to this worker")
        normalized = [str(line).rstrip()[:2000] for line in lines if str(line).strip()]
        if not normalized:
            return 0
        stamp = now()
        self.conn.executemany("""
            INSERT INTO task_logs(task_id, worker_id, attempt_count, created_at, line)
            VALUES(?, ?, ?, ?, ?)
        """, [(task_id, worker_id, int(task.get("attempt_count") or 0), stamp, line)
              for line in normalized[:50]])
        # Keep a task's dashboard log bounded even when an algorithm is verbose.
        self.conn.execute("""
            DELETE FROM task_logs WHERE task_id=? AND id NOT IN (
              SELECT id FROM task_logs WHERE task_id=? ORDER BY id DESC LIMIT 5000
            )
        """, (task_id, task_id))
        self.conn.commit()
        return len(normalized[:50])

    @synchronized
    def task_logs(self, task_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.conn.execute("""
            SELECT id, worker_id, attempt_count, created_at, line FROM task_logs
            WHERE task_id=? ORDER BY id DESC LIMIT ?
        """, (task_id, max(1, min(limit, 5000)))).fetchall()
        return [{"id": int(row[0]), "worker_id": str(row[1]), "attempt_count": int(row[2]),
                 "created_at": float(row[3]), "line": str(row[4])} for row in reversed(rows)]

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
