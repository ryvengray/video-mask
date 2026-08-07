import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "video_mask_batch.py"


def test_script_is_syntactically_valid():
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_project_metadata_declares_cli_and_optional_card_dependencies():
    metadata = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'vmask = "video_mask_batch:main"' in metadata
    assert "cards = [" in metadata
    assert '"transformers>=4.40"' in metadata
    assert '"opencv-python>=4.8,<5"' in metadata
