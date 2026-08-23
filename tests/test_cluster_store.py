from pathlib import Path
import json
import sqlite3

import pytest

from cluster.local_ingest import LocalIngestor
from cluster.store import ClusterStore


TOKEN = "worker-token-that-is-long-enough"


def new_store(tmp_path: Path) -> ClusterStore:
    return ClusterStore(tmp_path / "controller.sqlite3")


def task_payload() -> dict:
    return {
        "source_url": "https://storage.example/input.mov",
        "output_upload_url": "https://storage.example/output.mp4?signature=x",
        "source_object_key": "source/inbox/input.mov",
        "arguments": ["--no-card", "--frame-skip", "3"],
    }


def test_worker_claims_task_once_and_finishes(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {"gpu": "Tesla T4"})
    created = store.create_task(task_payload())

    claimed = store.claim("worker-01", TOKEN)
    assert claimed is not None
    assert claimed["task_id"] == created["task_id"]
    assert claimed["status"] == "assigned"
    assert store.claim("worker-01", TOKEN)["task_id"] == created["task_id"]

    store.progress("worker-01", TOKEN, created["task_id"], "processing", {"frame": 10})
    finished = store.finish("worker-01", TOKEN, created["task_id"], True, {
        "output_sha256": "abc", "output_duration_seconds": 12.5,
    })
    assert finished["status"] == "completed"
    assert store.worker("worker-01")["status"] == "ready"
    store.close()


def test_worker_task_logs_are_persisted_and_cleared_on_restart(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    assert store.append_task_logs("worker-01", TOKEN, created["task_id"], ["first line", "second line"]) == 2
    assert [entry["line"] for entry in store.task_logs(created["task_id"])] == ["first line", "second line"]

    store.finish("worker-01", TOKEN, created["task_id"], True, {})
    store.restart_task(created["task_id"])
    assert store.task_logs(created["task_id"]) == []
    store.close()


def test_task_without_upload_url_is_valid_for_worker_local_output(tmp_path: Path):
    store = new_store(tmp_path)
    payload = task_payload()
    payload.pop("output_upload_url")
    created = store.create_task(payload)

    assert created["output_upload_url"] == ""
    assert created["status"] == "pending"
    store.close()


def test_face_review_claim_lease_annotation_and_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = new_store(tmp_path)
    clock = [1_700_000_000.0]
    monkeypatch.setattr("cluster.store.now", lambda: clock[0])
    first = store.create_task(task_payload())
    second_payload = task_payload()
    second_payload["source_url"] = "https://storage.example/second.mov"
    second_payload["source_object_key"] = "source/inbox/second.mov"
    second = store.create_task(second_payload)
    store.conn.execute("UPDATE tasks SET status='completed', finished_at=?", (clock[0],))
    store.conn.commit()

    claimed_first = store.claim_next_face_review("browser-a-unique-id", 300)
    claimed_second = store.claim_next_face_review("browser-b-unique-id", 300)
    assert {claimed_first["task_id"], claimed_second["task_id"]} == {first["task_id"], second["task_id"]}
    assert store.claim_next_face_review("browser-c-unique-id", 300) is None

    labelled = store.annotate_content_tags(claimed_first["task_id"], "browser-a-unique-id", ["Laundry"])
    assert labelled["content_tags"] == ["Laundry"]
    labelled = store.annotate_face(claimed_first["task_id"], "browser-a-unique-id", True)
    assert labelled["face_annotation"] == 1
    status = store.face_review_status((claimed_first["task_id"], claimed_second["task_id"]))["reviews"]
    by_task_id = {review["task_id"]: review for review in status}
    assert by_task_id[claimed_first["task_id"]]["reviewable"] is True
    assert by_task_id[claimed_first["task_id"]]["has_face"] is True
    assert by_task_id[claimed_second["task_id"]]["reviewing"] is True

    reopened = store.claim_face_review(claimed_first["task_id"], "browser-a-unique-id", 300)
    assert reopened["face_annotation"] == 1
    changed = store.annotate_face(claimed_first["task_id"], "browser-a-unique-id", False)
    assert changed["face_annotation"] == 0

    clock[0] += 301
    reclaimed = store.claim_next_face_review("browser-c-unique-id", 300)
    assert reclaimed and reclaimed["task_id"] == claimed_second["task_id"]
    store.close()


def test_content_tags_are_saved_normalised_and_listed_for_reuse(tmp_path: Path):
    store = new_store(tmp_path)
    first = store.create_task(task_payload())
    second_payload = task_payload()
    second_payload["source_url"] = "https://storage.example/second.mov"
    second = store.create_task(second_payload)
    store.conn.execute("UPDATE tasks SET status='completed'")
    store.conn.commit()

    claimed = store.claim_next_face_review("browser-a-unique-id", 300)
    assert claimed is not None
    tagged = store.annotate_content_tags(
        claimed["task_id"], "browser-a-unique-id", ["  Washing   clothes ", "washing clothes", "Tidy room"]
    )
    assert tagged["content_tags"] == ["Washing clothes", "Tidy room"]
    assert store.content_tags() == ["Washing clothes", "Tidy room"]
    store.annotate_face(claimed["task_id"], "browser-a-unique-id", True)

    # A completed video whose face state is already known remains eligible
    # until it receives its content labels.
    other_id = second["task_id"] if claimed["task_id"] == first["task_id"] else first["task_id"]
    store.conn.execute("UPDATE tasks SET face_annotation=1 WHERE task_id=?", (other_id,))
    store.conn.commit()
    next_claim = store.claim_next_face_review("browser-b-unique-id", 300)
    assert next_claim is not None
    assert next_claim["task_id"] == other_id
    store.close()


def test_boolean_controller_setting_persists_across_store_reopens(tmp_path: Path):
    database = tmp_path / "controller.sqlite3"
    store = ClusterStore(database)
    assert store.boolean_setting("s3_ingest_enabled", True) is True
    assert store.set_boolean_setting("s3_ingest_enabled", False) is False
    store.close()

    reopened = ClusterStore(database)
    assert reopened.boolean_setting("s3_ingest_enabled", True) is False
    reopened.close()


def test_opening_pre_annotation_database_runs_columns_before_annotation_index(tmp_path: Path):
    database = tmp_path / "controller.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE tasks (
          task_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL,
          assigned_worker_id TEXT, finished_at REAL, started_at REAL, updated_at REAL NOT NULL
        );
    """)
    connection.close()

    store = ClusterStore(database)
    columns = {row[1] for row in store.conn.execute("PRAGMA table_info(tasks)")}
    indexes = {row[1] for row in store.conn.execute("PRAGMA index_list(tasks)")}
    assert {
        "face_annotation", "face_annotated_at", "content_tags_json", "content_tagged_at",
        "face_review_owner", "face_review_lease_until",
    } <= columns
    assert "idx_tasks_face_review" in indexes
    store.close()


def test_task_list_pagination_returns_total_without_overlapping_pages(tmp_path: Path):
    store = new_store(tmp_path)
    task_ids = []
    for index in range(3):
        payload = task_payload()
        payload["source_url"] = f"https://storage.example/input-{index}.mov"
        task_ids.append(store.create_task(payload)["task_id"])

    first_page = store.list_tasks(limit=2, offset=0)
    second_page = store.list_tasks(limit=2, offset=2)

    assert store.count_tasks() == 3
    assert len(first_page) == 2
    assert len(second_page) == 1
    assert {task["task_id"] for task in first_page}.isdisjoint(
        {task["task_id"] for task in second_page}
    )
    assert {task["task_id"] for task in first_page + second_page} == set(task_ids)
    store.close()


def test_task_list_can_filter_by_status(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    assigned = store.create_task(task_payload())
    pending_payload = task_payload()
    pending_payload["source_url"] = "https://storage.example/pending.mov"
    pending = store.create_task(pending_payload)
    store.claim("worker-01", TOKEN)

    assert store.count_tasks(("assigned",)) == 1
    assert [task["task_id"] for task in store.list_tasks(statuses=("assigned",))] == [assigned["task_id"]]
    assert [task["task_id"] for task in store.list_tasks(statuses=("pending",))] == [pending["task_id"]]
    store.close()


def test_task_list_can_filter_by_multiple_face_annotations(tmp_path: Path):
    store = new_store(tmp_path)
    has_face = store.create_task(task_payload())
    no_face_payload = task_payload()
    no_face_payload["source_url"] = "https://storage.example/no-face.mov"
    no_face = store.create_task(no_face_payload)
    unlabelled_payload = task_payload()
    unlabelled_payload["source_url"] = "https://storage.example/unlabelled.mov"
    unlabelled = store.create_task(unlabelled_payload)
    store.conn.execute("UPDATE tasks SET face_annotation=1 WHERE task_id=?", (has_face["task_id"],))
    store.conn.execute("UPDATE tasks SET face_annotation=0 WHERE task_id=?", (no_face["task_id"],))
    store.conn.commit()

    selected = store.list_tasks(face_annotations=("has_face", "unlabelled"))
    assert {task["task_id"] for task in selected} == {has_face["task_id"], unlabelled["task_id"]}
    assert store.count_tasks(face_annotations=("no_face",)) == 1
    store.close()


def test_task_list_can_filter_by_any_selected_content_tag(tmp_path: Path):
    store = new_store(tmp_path)
    laundry = store.create_task(task_payload())
    room_payload = task_payload()
    room_payload["source_url"] = "https://storage.example/room.mov"
    room = store.create_task(room_payload)
    unlabelled_payload = task_payload()
    unlabelled_payload["source_url"] = "https://storage.example/unlabelled.mov"
    unlabelled = store.create_task(unlabelled_payload)
    store.conn.execute("UPDATE tasks SET content_tags_json=? WHERE task_id=?", (json.dumps(["Laundry", "Home"]), laundry["task_id"]))
    store.conn.execute("UPDATE tasks SET content_tags_json=? WHERE task_id=?", (json.dumps(["Tidy room"]), room["task_id"]))
    store.conn.commit()

    selected = store.list_tasks(content_tags=("Laundry", "Tidy room"))
    assert {task["task_id"] for task in selected} == {laundry["task_id"], room["task_id"]}
    assert store.count_tasks(content_tags=("Laundry",)) == 1
    assert unlabelled["task_id"] not in {task["task_id"] for task in selected}
    store.close()


def test_processing_statistics_reports_hourly_concurrency_and_worker_efficiency(tmp_path: Path):
    store = new_store(tmp_path)
    start = 1_700_000_000.0

    first = store.create_task(task_payload())
    second_payload = task_payload()
    second_payload["source_url"] = "https://storage.example/second.mov"
    second = store.create_task(second_payload)
    failed_payload = task_payload()
    failed_payload["source_url"] = "https://storage.example/failed.mov"
    failed = store.create_task(failed_payload)
    store.conn.execute("""
        UPDATE tasks SET status='completed', assigned_worker_id='worker-01-slot-1',
          output_duration_seconds=7200, started_at=?, finished_at=?,
          progress_json=? WHERE task_id=?
    """, (start + 600, start + 1800, json.dumps({
        "processing_started_at": start + 700, "processing_seconds": 900,
        "input_bytes": 100, "output_bytes": 80,
    }), first["task_id"]))
    store.conn.execute("""
        UPDATE tasks SET status='completed', assigned_worker_id='worker-02-slot-1',
          output_duration_seconds=3600, started_at=?, finished_at=?,
          progress_json=? WHERE task_id=?
    """, (start + 1200, start + 4500, json.dumps({
        "processing_started_at": start + 1500, "processing_seconds": 900,
    }), second["task_id"]))
    store.conn.execute("""
        UPDATE tasks SET status='failed', assigned_worker_id='worker-01-slot-1',
          started_at=?, finished_at=?, progress_json=? WHERE task_id=?
    """, (start + 2000, start + 2500, json.dumps({"processing_seconds": 300}), failed["task_id"]))
    store.conn.commit()

    statistics = store.processing_statistics(start, start + 7200)

    assert statistics["completed_tasks"] == 2
    assert statistics["failed_tasks"] == 1
    assert statistics["video_seconds"] == 10_800
    assert statistics["processing_seconds"] == 1_800
    assert statistics["algorithm_realtime"] == 6
    assert statistics["worker_hours_per_video_hour"] == pytest.approx(1 / 6)
    assert statistics["hourly"][0]["completed_tasks"] == 1
    assert statistics["hourly"][1]["completed_tasks"] == 1
    assert statistics["hourly"][0]["peak_concurrency"] == 2
    assert statistics["hourly"][1]["peak_concurrency"] == 1
    assert [row["worker"] for row in statistics["workers"]] == ["worker-01", "worker-02"]
    store.close()


def test_stale_worker_requeues_task(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)
    store.progress("worker-01", TOKEN, created["task_id"], "processing", {"processing_started_at": 1})
    store.conn.execute("UPDATE workers SET last_seen_at=0 WHERE worker_id='worker-01'")
    store.conn.commit()

    assert store.requeue_stale(1) == 1
    recovered = store.task(created["task_id"])
    assert recovered["status"] == "pending"
    assert recovered["progress"] == {}
    assert recovered["error_message"] is None
    assert recovered["started_at"] is None
    assert recovered["finished_at"] is None
    store.close()


def test_stale_idle_worker_becomes_offline(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-idle", TOKEN)
    store.register_worker("worker-idle", TOKEN, {})
    store.conn.execute("UPDATE workers SET last_seen_at=0 WHERE worker_id='worker-idle'")
    store.conn.commit()

    assert store.requeue_stale(1) == 0
    assert store.worker("worker-idle")["status"] == "offline"
    store.close()


def test_stale_orphaned_cancelling_task_is_finalized(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01-slot-5", TOKEN)
    store.register_worker("worker-01-slot-5", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01-slot-5", TOKEN)
    store.cancel_task(created["task_id"])

    # Simulate a removed slot whose lease reference was cleared before stale
    # recovery had a chance to finalize its cancelling task.
    store.conn.execute("UPDATE workers SET current_task_id=NULL WHERE worker_id='worker-01-slot-5'")
    store.conn.execute("UPDATE tasks SET updated_at=0 WHERE task_id=?", (created["task_id"],))
    store.conn.commit()

    assert store.requeue_stale(1) == 1
    recovered = store.task(created["task_id"])
    assert recovered["status"] == "cancelled"
    assert recovered["assigned_worker_id"] is None
    assert recovered["error_message"] == "cancelled by administrator (orphaned worker slot)"
    store.close()


def test_stale_orphaned_active_task_returns_to_queue(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    store.conn.execute("DELETE FROM workers WHERE worker_id='worker-01'")
    store.conn.execute("UPDATE tasks SET updated_at=0 WHERE task_id=?", (created["task_id"],))
    store.conn.commit()

    assert store.requeue_stale(1) == 1
    recovered = store.task(created["task_id"])
    assert recovered["status"] == "pending"
    assert recovered["assigned_worker_id"] is None
    assert recovered["error_message"] is None
    assert recovered["progress"] == {}
    assert recovered["started_at"] is None
    store.close()


def test_failed_task_retries_until_max_attempts(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    payload = task_payload()
    payload["max_attempts"] = 2
    created = store.create_task(payload)

    store.claim("worker-01", TOKEN)
    assert store.finish("worker-01", TOKEN, created["task_id"], False, {"error_message": "network"})["status"] == "pending"
    store.claim("worker-01", TOKEN)
    assert store.finish("worker-01", TOKEN, created["task_id"], False, {"error_message": "network"})["status"] == "failed"
    store.close()


def test_failed_task_can_be_retried_by_an_admin_action(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    payload = task_payload()
    payload["max_attempts"] = 1
    created = store.create_task(payload)
    store.claim("worker-01", TOKEN)
    assert store.finish("worker-01", TOKEN, created["task_id"], False, {"error_message": "access denied"})["status"] == "failed"

    retried = store.retry_task(created["task_id"])
    assert retried["status"] == "pending"
    assert retried["attempt_count"] == 0
    assert retried["error_message"] is None
    store.close()


def test_failed_task_can_also_be_restarted(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    payload = task_payload()
    payload["max_attempts"] = 1
    created = store.create_task(payload)
    store.claim("worker-01", TOKEN)
    assert store.finish("worker-01", TOKEN, created["task_id"], False, {"error_message": "access denied"})["status"] == "failed"

    restarted = store.restart_task(created["task_id"])
    assert restarted["status"] == "pending"
    assert restarted["attempt_count"] == 0
    assert restarted["restarted_at"] is not None
    store.close()


def test_restart_task_can_update_algorithm_and_arguments(tmp_path: Path):
    store = new_store(tmp_path)
    created = store.create_task(task_payload())
    store.cancel_task(created["task_id"])

    restarted = store.restart_task(
        created["task_id"], algorithm="video_mask_batch_experiment.py",
        arguments=["--face-model", "yolo11", "--frame-skip", "2"],
    )

    assert restarted["status"] == "pending"
    assert restarted["algorithm"] == "video_mask_batch_experiment.py"
    assert restarted["arguments"] == ["--face-model", "yolo11", "--frame-skip", "2"]
    store.close()


def test_pending_task_can_be_cancelled_and_restarted(tmp_path: Path):
    store = new_store(tmp_path)
    created = store.create_task(task_payload())

    cancelled = store.cancel_task(created["task_id"])
    assert cancelled["status"] == "cancelled"
    assert "cancelled by administrator" in cancelled["error_message"]

    restarted = store.restart_task(created["task_id"])
    assert restarted["status"] == "pending"
    assert restarted["attempt_count"] == 0
    assert restarted["error_message"] is None
    assert restarted["restarted_at"] is not None
    store.close()


def test_all_completed_tasks_can_be_restarted_with_a_shared_timestamp(tmp_path: Path, monkeypatch):
    store = new_store(tmp_path)
    completed = store.create_task(task_payload())
    second_completed = store.create_task({**task_payload(), "source_url": "https://storage.example/second.mov"})
    failed = store.create_task({**task_payload(), "source_url": "https://storage.example/failed.mov"})
    store.conn.execute("""
        UPDATE tasks SET status='completed', attempt_count=2, started_at=10, finished_at=20,
          output_sha256='done', output_duration_seconds=12,
          face_annotation=CASE task_id WHEN ? THEN 1 ELSE 0 END,
          face_annotated_at=18 WHERE task_id IN (?, ?)
    """, (completed["task_id"], completed["task_id"], second_completed["task_id"]))
    store.conn.execute("UPDATE tasks SET status='failed', finished_at=20 WHERE task_id=?", (failed["task_id"],))
    store.conn.commit()
    monkeypatch.setattr("cluster.store.now", lambda: 1234.5)

    result = store.restart_completed_tasks()

    assert result == {"restarted_tasks": 2, "restarted_at": 1234.5}
    for task_id in (completed["task_id"], second_completed["task_id"]):
        restarted = store.task(task_id)
        assert restarted["status"] == "pending"
        assert restarted["attempt_count"] == 0
        assert restarted["started_at"] is None
        assert restarted["restarted_at"] == 1234.5
        assert restarted["finished_at"] is None
        assert restarted["output_sha256"] is None
        assert restarted["face_annotation"] == (1 if task_id == completed["task_id"] else 0)
        assert restarted["face_annotated_at"] == 18
    assert store.task(failed["task_id"])["status"] == "failed"
    store.close()


def test_purge_removes_all_non_active_tasks_and_clears_stale_worker_state(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-stale", TOKEN)
    store.register_worker("worker-stale", TOKEN, {})
    store.create_task(task_payload())
    failed = task_payload()
    failed["source_url"] = "https://storage.example/failed.mov"
    store.create_task(failed)
    store.conn.execute("UPDATE workers SET status='busy', current_task_id='obsolete-task' WHERE worker_id='worker-stale'")
    store.conn.commit()

    assert store.purge_tasks() == {"deleted_tasks": 2}
    assert store.count_tasks() == 0
    assert store.worker("worker-stale")["status"] == "ready"
    assert store.worker("worker-stale")["current_task_id"] is None
    store.close()


def test_purge_rejects_active_tasks(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    with pytest.raises(ValueError, match=r"task\(s\) are active"):
        store.purge_tasks()
    assert store.task(created["task_id"])["status"] == "assigned"
    store.close()


def test_active_task_cancellation_completes_as_cancelled(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    requested = store.cancel_task(created["task_id"])
    assert requested["status"] == "cancelling"
    assert store.worker("worker-01")["status"] == "cancelling"
    assert store.heartbeat("worker-01", TOKEN, "busy")["status"] == "cancelling"

    finished = store.finish("worker-01", TOKEN, created["task_id"], False, {
        "error_message": "cancelled by administrator",
        "progress": {"processing_seconds": 4.2},
    })
    assert finished["status"] == "cancelled"
    assert finished["progress"]["processing_seconds"] == 4.2
    assert store.worker("worker-01")["status"] == "ready"
    store.close()


def test_failed_task_retains_worker_reported_output_metadata(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    failed = store.finish("worker-01", TOKEN, created["task_id"], False, {
        "error_message": "uploading: output upload failed",
        "progress": {
            "input_bytes": 100,
            "output_filename": "masked_input.mp4",
            "output_bytes": 80,
            "output_duration_seconds": 12.5,
            "processing_seconds": 4.2,
        },
    })

    assert failed["progress"]["output_bytes"] == 80
    assert failed["progress"]["output_duration_seconds"] == 12.5
    assert failed["progress"]["processing_seconds"] == 4.2
    store.close()


def test_provisioning_with_the_same_token_preserves_an_active_worker(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    provisioned = store.provision_worker("worker-01", TOKEN)

    assert provisioned["status"] == "busy"
    assert provisioned["current_task_id"] == created["task_id"]
    assert store.task(created["task_id"])["status"] == "assigned"
    store.close()


def test_retire_removes_only_idle_worker_slots(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01-slot-2", TOKEN)
    store.register_worker("worker-01-slot-2", TOKEN, {})

    assert store.retire_worker("worker-01-slot-2") == {
        "worker_id": "worker-01-slot-2", "retired": True,
    }
    assert store.worker("worker-01-slot-2") is None
    assert store.retire_worker("worker-01-slot-9")["retired"] is False

    store.provision_worker("worker-01-slot-3", TOKEN)
    store.register_worker("worker-01-slot-3", TOKEN, {})
    task = store.create_task(task_payload())
    store.claim("worker-01-slot-3", TOKEN)
    with pytest.raises(ValueError, match="only an idle"):
        store.retire_worker("worker-01-slot-3")
    assert store.worker("worker-01-slot-3")["current_task_id"] == task["task_id"]
    store.close()


def test_active_task_lookup_rejects_a_different_worker_or_terminal_task(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    store.provision_worker("worker-02", TOKEN)
    store.register_worker("worker-02", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)

    assert store.active_task_for_worker("worker-01", TOKEN, created["task_id"])["task_id"] == created["task_id"]
    with pytest.raises(ValueError, match="not actively assigned"):
        store.active_task_for_worker("worker-02", TOKEN, created["task_id"])
    store.finish("worker-01", TOKEN, created["task_id"], True, {})
    with pytest.raises(ValueError, match="not actively assigned"):
        store.active_task_for_worker("worker-01", TOKEN, created["task_id"])
    store.close()


def test_local_ingestor_creates_one_file_task(tmp_path: Path):
    store = new_store(tmp_path)
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "clip.MOV"
    source.write_bytes(b"test video")
    ingestor = LocalIngestor(store, source_dir, tmp_path / "outputs")

    assert ingestor.scan() == 1
    task = store.list_tasks()[0]
    assert task["source_url"] == source.resolve().as_uri()
    assert task["output_object_key"] == "masked_clip.mp4"
    assert ingestor.scan() == 0
    store.close()


def test_duplicate_local_scan_does_not_block_worker_claim(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "clip.mp4").write_bytes(b"test video")
    ingestor = LocalIngestor(store, source_dir, tmp_path / "outputs")

    assert ingestor.scan() == 1
    # The second scan hits the task's unique ID, as it does before each claim.
    assert ingestor.scan() == 0
    claimed = store.claim("worker-01", TOKEN)

    assert claimed is not None
    assert claimed["status"] == "assigned"
    store.close()


def test_list_workers_includes_status_capabilities_and_current_task(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-ready", TOKEN)
    store.register_worker("worker-ready", TOKEN, {"gpu": "Tesla T4", "cuda_available": True})
    store.provision_worker("worker-busy", TOKEN)
    store.register_worker("worker-busy", TOKEN, {})
    task = store.create_task(task_payload())
    store.claim("worker-busy", TOKEN)

    workers = store.list_workers()

    assert [worker["worker_id"] for worker in workers] == ["worker-busy", "worker-ready"]
    assert workers[0]["status"] == "busy"
    assert workers[0]["current_task_id"] == task["task_id"]
    assert workers[1]["capabilities"]["gpu"] == "Tesla T4"
    store.close()
