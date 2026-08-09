"""Build only video-mask proprietary modules as native Cython extensions."""
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "video_mask_batch_fish": "video_mask_batch_fish.py",
    "cluster.controller": "cluster/controller.py",
    "cluster.worker_agent": "cluster/worker_agent.py",
    "cluster.store": "cluster/store.py",
    "cluster.local_ingest": "cluster/local_ingest.py",
}

extensions = [Extension(name, [str(ROOT / source)]) for name, source in MODULES.items()]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=False,
    ),
)
