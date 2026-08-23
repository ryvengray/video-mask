import importlib.util
import sqlite3
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_random_task_files.py"
SPEC = importlib.util.spec_from_file_location("download_random_task_files", SCRIPT_PATH)
assert SPEC and SPEC.loader
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def make_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE tasks (
          task_id TEXT PRIMARY KEY, status TEXT NOT NULL,
          source_object_key TEXT, output_object_key TEXT,
          source_sha256 TEXT, output_sha256 TEXT
        )
    """)
    connection.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)", [
        ("completed-source", "completed", "source/a/clip.mp4", "output/a/masked.mp4", None, None),
        ("failed-source", "failed", "source/b/clip.mp4", None, None, None),
    ])
    connection.commit()
    connection.close()


def test_select_tasks_respects_file_kind_and_status(tmp_path: Path):
    database = tmp_path / "snapshot.sqlite3"
    make_snapshot(database)

    assert downloader.select_tasks(database, "input", 10, ("completed",)) == [{
        "task_id": "completed-source", "object_key": "source/a/clip.mp4", "expected_sha256": None,
    }]
    assert downloader.select_tasks(database, "output", 10, ("completed",)) == [{
        "task_id": "completed-source", "object_key": "output/a/masked.mp4", "expected_sha256": None,
    }]
    assert downloader.select_tasks(database, "input", 10, ("failed",)) == [{
        "task_id": "failed-source", "object_key": "source/b/clip.mp4", "expected_sha256": None,
    }]


def test_download_path_stays_under_destination_and_play_request_uses_controller():
    assert downloader.safe_relative_key("source/nested/video.mp4", "task-1") == Path("source/nested/video.mp4")
    assert downloader.safe_relative_key("../escape.mp4", "task-1") == Path("invalid-object-key/task-1/video.bin")
    request = downloader.build_request(
        "https://controller.example.com/", "task / 1", "input", {"Authorization": "Basic token"}, False,
    )
    assert request.full_url == "https://controller.example.com/api/tasks/task%20%2F%201/play?file=input&download=true"
    assert request.get_header("Authorization") == "Basic token"
