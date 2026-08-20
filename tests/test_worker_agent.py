from types import SimpleNamespace
import sys

import pytest

from cluster.worker_agent import Worker


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
