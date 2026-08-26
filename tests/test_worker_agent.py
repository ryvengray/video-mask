from types import SimpleNamespace
import json
import sys

import pytest

from cluster.worker_agent import Worker, cuda_library_paths


def test_worker_uses_controller_selected_sibling_algorithm(tmp_path):
    default_algorithm = tmp_path / "video_mask_batch_fish_v1.py"
    selected_algorithm = tmp_path / "video_mask_batch_experiment.py"
    default_algorithm.touch()
    selected_algorithm.touch()
    worker = Worker(SimpleNamespace(
        controller="http://controller.example", worker_id="worker-01", token="x" * 16,
        work_dir=tmp_path / "work", completed_output_dir=tmp_path / "completed",
        algorithm=default_algorithm, python=sys.executable, poll_seconds=1,
        extra_arg=[], allow_local_files=False,
    ))

    assert worker.algorithm_path_for_task({"algorithm": selected_algorithm.name}) == selected_algorithm.resolve()
    assert worker.algorithm_path_for_task({}) == default_algorithm.resolve()
    with pytest.raises(RuntimeError, match="filename"):
        worker.algorithm_path_for_task({"algorithm": "../outside.py"})


def test_cuda_library_paths_discovers_torch_and_nvidia_wheels(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    (site_packages / "torch" / "lib").mkdir(parents=True)
    (site_packages / "nvidia" / "cublas" / "lib").mkdir(parents=True)
    monkeypatch.setattr("cluster.worker_agent.site.getsitepackages", lambda: [str(site_packages)])

    assert cuda_library_paths() == [
        str(site_packages / "torch" / "lib"),
        str(site_packages / "nvidia" / "cublas" / "lib"),
    ]


def test_plus_algorithm_json_report_is_added_to_its_command_and_validated(tmp_path):
    algorithm = tmp_path / "video_mask_batch_fish_v1_plus.py"
    algorithm.touch()
    report_path = Worker.structured_report_path(algorithm, tmp_path / "output")
    assert report_path == tmp_path / "output" / "algorithm-report.json"
    worker = Worker(SimpleNamespace(
        controller="http://controller.example", worker_id="worker-01", token="x" * 16,
        work_dir=tmp_path / "work", completed_output_dir=tmp_path / "completed",
        algorithm=algorithm, python=sys.executable, poll_seconds=1, extra_arg=[], allow_local_files=False,
    ))
    assert worker.algorithm_command(algorithm, tmp_path / "input.mp4", tmp_path / "output", [], report_path)[-2:] == [
        "--report", str(report_path),
    ]

    report_path.parent.mkdir()
    report_path.write_text(json.dumps({"videos": [{
        "success": True, "total_frames": 3600, "frames_with_face": 123,
        "total_face_detections": 456, "duration_seconds": 98.765,
        "fps": 30.0, "width": 1920, "height": 1080,
        "input_file": "/private/work/input.mp4",  # deliberately not retained
    }]}), encoding="utf-8")
    stats = Worker.read_algorithm_report(report_path)
    assert stats == {
        "schema_version": 1, "total_frames": 3600, "frames_with_face": 123,
        "total_face_detections": 456, "duration_seconds": 98.765,
        "fps": 30.0, "width": 1920, "height": 1080,
    }
    assert "input_file" not in stats
