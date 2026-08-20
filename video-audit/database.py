"""SQLite database layer with in-memory caching for fast queries."""

import sqlite3
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "audit.db"


class Database:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                task_id TEXT NOT NULL,
                controller_task_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                local_video_path TEXT,
                controller_info TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                review_type TEXT NOT NULL,
                note TEXT,
                screenshot_path TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reviews_task ON reviews(task_id);
        """)
        conn.commit()
        conn.close()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _now(self) -> float:
        import time
        return time.time()

    def _invalidate_cache(self, key: str) -> None:
        self._cache.pop(key, None)

    def create_project(self, name: str) -> dict[str, Any]:
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO projects (name, created_at) VALUES (?, ?)",
                (name, self._now()),
            )
            project_id = cursor.lastrowid
            conn.commit()
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        self._invalidate_cache("projects")
        return dict(row)

    def get_projects(self) -> list[dict[str, Any]]:
        cache_key = "projects"
        if cache_key in self._cache:
            return self._cache[cache_key]
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        result = [dict(row) for row in rows]
        self._cache[cache_key] = result
        return result

    def get_project(self, project_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

    def create_tasks(self, project_id: int, task_ids: list[str]) -> list[dict[str, Any]]:
        results = []
        with self._conn() as conn:
            for task_id in task_ids:
                cursor = conn.execute(
                    "INSERT INTO tasks (project_id, task_id, controller_task_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (project_id, task_id, task_id, "pending", self._now(), self._now()),
                )
                results.append(dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (cursor.lastrowid,)).fetchone()))
            conn.commit()
        self._invalidate_cache(f"tasks_{project_id}")
        return results

    def get_tasks(self, project_id: int) -> list[dict[str, Any]]:
        cache_key = f"tasks_{project_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        self._cache[cache_key] = result
        return result

    def get_task(self, task_id: int) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def update_task(self, task_id: int, **kwargs) -> None:
        allowed = {"status", "local_video_path", "controller_info"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        values.append(self._now())
        values.append(task_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE tasks SET {set_clause}, updated_at = ? WHERE id = ?", values)
            conn.commit()
        task = self.get_task(task_id)
        if task:
            self._invalidate_cache(f"tasks_{task['project_id']}")

    def create_review(self, task_id: int, frame_number: int, timestamp: float,
                      review_type: str, note: Optional[str] = None,
                      screenshot_path: Optional[str] = None) -> dict[str, Any]:
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO reviews (task_id, frame_number, timestamp, review_type, note, screenshot_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, frame_number, timestamp, review_type, note, screenshot_path, self._now()),
            )
            review_id = cursor.lastrowid
            row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
            conn.commit()
        self._invalidate_cache(f"reviews_{task_id}")
        return dict(row)

    def get_reviews(self, task_id: int) -> list[dict[str, Any]]:
        cache_key = f"reviews_{task_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE task_id = ? ORDER BY timestamp",
                (task_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        self._cache[cache_key] = result
        return result

    def get_stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE review_type = 'missed') AS missed,
                    COUNT(*) FILTER (WHERE review_type = 'false_positive') AS false_positive,
                    COUNT(*) AS total
                FROM reviews
            """).fetchone()
        return {
            "missed": row[0] or 0,
            "false_positive": row[1] or 0,
            "total": row[2] or 0,
        }
