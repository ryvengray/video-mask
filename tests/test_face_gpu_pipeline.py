import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "video_mask_face_gpu.py"
SPEC = importlib.util.spec_from_file_location("video_mask_face_gpu", MODULE_PATH)
FACE_GPU = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = FACE_GPU
SPEC.loader.exec_module(FACE_GPU)


def test_ratio_parses_ffmpeg_fraction():
    assert abs(FACE_GPU.ratio("30000/1001") - 29.97002997) < 1e-6
    assert FACE_GPU.ratio("0/0") == 25.0


def test_collect_inputs_skips_masked_outputs(tmp_path):
    source = tmp_path / "videos"
    source.mkdir()
    original = source / "clip.MOV"
    original.touch()
    (source / "masked_clip.mp4").touch()
    (source / "notes.txt").touch()

    assert FACE_GPU.collect_inputs([str(source)]) == [original.resolve()]


def test_encoder_command_keeps_audio_and_clears_rotation(tmp_path):
    source = tmp_path / "input.mov"
    output = tmp_path / "masked.mp4"
    info = FACE_GPU.VideoInfo(1920, 1080, 30.0, 60.0, 90, True)

    command = FACE_GPU.encoder_command(source, output, info, "nvenc")

    assert "h264_nvenc" in command
    assert "1:a?" in command
    assert "rotate=0" in command
