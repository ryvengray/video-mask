import importlib.util
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "batch_scheduler.py"
SPEC = importlib.util.spec_from_file_location("batch_scheduler", MODULE_PATH)
SCHEDULER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SCHEDULER
SPEC.loader.exec_module(SCHEDULER)


def test_discover_jobs_preserves_relative_paths_and_excludes_output(tmp_path):
    source = tmp_path / "input"
    target = source / "masked"
    (source / "nested").mkdir(parents=True)
    target.mkdir()
    (source / "nested" / "clip.MOV").touch()
    (source / "same.mp4").touch()
    (source / "masked_external.mp4").touch()
    (target / "masked_same.mp4").touch()

    jobs = SCHEDULER.discover_jobs(source, target)

    assert [job.relative.as_posix() for job in jobs] == ["nested/clip.MOV", "same.mp4"]
    assert jobs[0].destination == target / "nested" / "masked_clip.mp4"
    assert jobs[1].destination == target / "masked_same.mp4"


def test_format_duration():
    assert SCHEDULER.format_duration(9) == "0:09"
    assert SCHEDULER.format_duration(3661) == "1:01:01"


def test_legacy_state_database_is_migrated_for_media_durations(tmp_path):
    state_file = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(state_file)
    connection.execute("""
        CREATE TABLE jobs (
            source TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            destination TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            duration REAL, error TEXT, updated_at REAL NOT NULL
        )
    """)
    connection.commit()
    connection.close()

    store = SCHEDULER.JobStore(state_file)
    columns = {row[1] for row in store.conn.execute("PRAGMA table_info(jobs)")}
    store.close()

    assert {"source_duration", "output_duration"} <= columns
