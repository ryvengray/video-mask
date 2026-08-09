#!/usr/bin/env python3
"""Pull-based worker for a video-mask cluster."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ARGS = ["--no-card", "--face-size", "960", "--face-int", "5", "--frame-skip", "3"]


def request_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration(path: Path) -> float | None:
    process = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                             text=True, capture_output=True, check=False)
    try:
        return float(process.stdout.strip())
    except ValueError:
        return None


class Worker:
    def __init__(self, args: argparse.Namespace):
        self.controller = args.controller.rstrip("/")
        self.worker_id = args.worker_id
        self.token = args.token
        self.work_dir = args.work_dir.resolve()
        self.algorithm = args.algorithm.resolve()
        self.python = args.python
        self.poll_seconds = args.poll_seconds
        self.extra_args = args.extra_arg or DEFAULT_ARGS
        self.allow_local_files = args.allow_local_files
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def payload(self, **extra: Any) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "token": self.token, **extra}

    @staticmethod
    def capabilities() -> dict[str, Any]:
        info: dict[str, Any] = {"algorithm": "video_mask_batch_skip.py", "pid": os.getpid()}
        try:
            import torch
            info.update({"cuda_available": torch.cuda.is_available(), "torch": torch.__version__})
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["cuda_error"] = str(exc)
        return info

    def api(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        return request_json(self.controller + suffix, payload)

    def report(self, task_id: str, status: str, **progress: Any) -> None:
        self.api(f"/api/tasks/{task_id}/progress", self.payload(status=status, progress=progress))

    def busy_heartbeat(self, stop: threading.Event, capabilities: dict[str, Any]) -> None:
        """Keep long downloads/encodes leased to this Worker."""
        while not stop.wait(15):
            try:
                self.api(f"/api/workers/{self.worker_id}/heartbeat",
                         self.payload(status="busy", capabilities=capabilities))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"Worker heartbeat failed: {exc}", file=sys.stderr, flush=True)

    def download(self, url: str, destination: Path) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            if not self.allow_local_files:
                raise RuntimeError("file:// tasks require --allow-local-files")
            local_source = Path(urllib.parse.unquote(parsed.path))
            shutil.copy2(local_source, destination)
            return
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        partial.replace(destination)

    def upload(self, url: str, source: Path) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            if not self.allow_local_files:
                raise RuntimeError("file:// tasks require --allow-local-files")
            destination = Path(urllib.parse.unquote(parsed.path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return
        # curl streams the file from disk.  Do not use read_bytes() here: long
        # videos must never be loaded into a Worker's RAM for upload.
        result = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--upload-file", str(source), url],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("output upload failed: " + result.stderr[-1000:])

    def run_task(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        directory = self.work_dir / task_id
        directory.mkdir(parents=True, exist_ok=True)
        source_name = Path(urllib.parse.urlparse(task["source_url"]).path).name or "input.mp4"
        source = directory / source_name
        output_dir = directory / "output"
        output_dir.mkdir(exist_ok=True)
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(target=self.busy_heartbeat,
                                     args=(heartbeat_stop, self.capabilities()), daemon=True)
        heartbeat.start()
        try:
            self.report(task_id, "downloading", phase="downloading")
            self.download(task["source_url"], source)
            if task.get("source_sha256") and sha256(source) != task["source_sha256"]:
                raise RuntimeError("downloaded source checksum does not match task")
            self.report(task_id, "processing", phase="processing", source_bytes=source.stat().st_size)
            command = [self.python, "-u", str(self.algorithm), str(source), "--out-dir", str(output_dir),
                       *(task.get("arguments") or self.extra_args)]
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            tail: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[{task_id}] {line}", flush=True)
                    tail = (tail + [line])[-20:]
            if process.wait() != 0:
                raise RuntimeError("algorithm failed: " + " | ".join(tail)[-1500:])
            candidates = sorted(output_dir.glob("masked_*.mp4"))
            if not candidates:
                raise RuntimeError("algorithm completed without an output mp4")
            output = candidates[0]
            self.report(task_id, "uploading", phase="uploading", elapsed_seconds=round(time.monotonic() - started, 1))
            self.upload(task["output_upload_url"], output)
            self.api(f"/api/tasks/{task_id}/complete", self.payload(
                output_sha256=sha256(output), output_duration_seconds=duration(output),
                progress={"output_bytes": output.stat().st_size, "elapsed_seconds": round(time.monotonic() - started, 1)},
            ))
        except Exception as exc:
            print(f"[{task_id}] ERROR: {exc}", file=sys.stderr, flush=True)
            try:
                self.api(f"/api/tasks/{task_id}/fail", self.payload(error_message=str(exc)[:3000]))
            except Exception as report_error:
                print(f"[{task_id}] failed to report error: {report_error}", file=sys.stderr, flush=True)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
            shutil.rmtree(directory, ignore_errors=True)

    def serve(self) -> None:
        capabilities = self.capabilities()
        self.api("/api/workers/register", self.payload(capabilities=capabilities))
        print(f"Worker {self.worker_id} registered with {self.controller}", flush=True)
        while True:
            try:
                self.api(f"/api/workers/{self.worker_id}/heartbeat", self.payload(status="ready", capabilities=capabilities))
                response = self.api(f"/api/workers/{self.worker_id}/claim", self.payload(capabilities=capabilities))
                if response.get("task"):
                    self.run_task(response["task"])
                    continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"Controller unavailable: {exc}", file=sys.stderr, flush=True)
            time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-mask cluster worker agent")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/home/ubuntu/work/video-mask"))
    parser.add_argument("--algorithm", type=Path, default=Path(__file__).resolve().parents[1] / "video_mask_batch_skip.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--allow-local-files", action="store_true",
                        help="Allow file:// input/output URLs for single-machine testing only")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be positive")
    Worker(args).serve()


if __name__ == "__main__":
    main()
