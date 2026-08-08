import importlib.util
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
