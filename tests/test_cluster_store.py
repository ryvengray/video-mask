from pathlib import Path

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


def test_stale_worker_requeues_task(tmp_path: Path):
    store = new_store(tmp_path)
    store.provision_worker("worker-01", TOKEN)
    store.register_worker("worker-01", TOKEN, {})
    created = store.create_task(task_payload())
    store.claim("worker-01", TOKEN)
    store.conn.execute("UPDATE workers SET last_seen_at=0 WHERE worker_id='worker-01'")
    store.conn.commit()

    assert store.requeue_stale(1) == 1
    assert store.task(created["task_id"])["status"] == "pending"
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
