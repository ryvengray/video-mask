#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Face-only streaming video mosaic pipeline.

The script is intentionally independent from the legacy face/card algorithms:

* ffmpeg streams frames through memory; no temporary JPEG directory is created.
* NVDEC is used when available and falls back to CPU decoding automatically.
* YuNet uses ONNX Runtime CUDA when available and reports the active provider.
* One detector instance is reused for every input handled by this process.
* Detection runs on key frames; LK optical flow tracks all faces in between.
* NVENC is used when available and falls back to libx264 automatically.

It keeps the same ``inputs`` / ``--out-dir`` contract as the batch scheduler.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".m2ts", ".wmv", ".3gp", ".vob", ".asf",
}
YUNET_REPO = "opencv/face_detection_yunet"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float | None
    rotation: int
    has_audio: bool


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ratio(value: str | None, default: float = 25.0) -> float:
    if not value:
        return default
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else default
        return float(value)
    except (TypeError, ValueError):
        return default


def probe_video(path: Path) -> VideoInfo:
    result = run_command([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed: {path}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream: {path}")
    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for item in video.get("side_data_list", []):
        if "rotation" in item:
            rotation = int(float(item["rotation"]))
            break
    if not rotation:
        try:
            rotation = int(float(video.get("tags", {}).get("rotate", 0)))
        except (TypeError, ValueError):
            rotation = 0
    if abs(rotation) % 180 == 90:
        width, height = height, width
    duration_raw = payload.get("format", {}).get("duration")
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None
    return VideoInfo(
        width=width,
        height=height,
        fps=ratio(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        duration=duration,
        rotation=rotation,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


class StderrCollector:
    def __init__(self, stream):
        self.lines: list[bytes] = []
        self.stream = stream
        self.thread = threading.Thread(target=self._drain, daemon=True)

    def _drain(self):
        if self.stream is None:
            return
        for line in iter(self.stream.readline, b""):
            self.lines.append(line)
            self.lines = self.lines[-80:]

    def start(self):
        self.thread.start()

    def tail(self, limit: int = 1200) -> str:
        return b"".join(self.lines).decode(errors="replace")[-limit:].strip()


def ffmpeg_has(name: str, section: str) -> bool:
    result = run_command(["ffmpeg", "-hide_banner", section])
    return result.returncode == 0 and name in result.stdout


def cuda_decode_available() -> bool:
    return shutil.which("nvidia-smi") is not None and ffmpeg_has("cuda", "-hwaccels")


def nvenc_available() -> bool:
    return shutil.which("nvidia-smi") is not None and ffmpeg_has("h264_nvenc", "-encoders")


def decoder_command(path: Path, use_cuda: bool) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda"]
    # ffmpeg autorotation is deliberately kept enabled; probe_video reports display dimensions.
    cmd += ["-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    return cmd


def encoder_command(path: Path, output: Path, info: VideoInfo,
                    encoder: str) -> list[str]:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{info.width}x{info.height}",
        "-r", f"{info.fps:.8f}", "-i", "pipe:0", "-i", str(path),
        "-map", "0:v:0", "-map", "1:a?",
    ]
    if encoder == "nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd += ["-pix_fmt", "yuv420p", "-metadata:s:v:0", "rotate=0"]
    if info.has_audio:
        cmd += ["-c:a", "copy"]
    cmd += ["-shortest", "-movflags", "+faststart", str(output)]
    return cmd


def heavy_mosaic(image: np.ndarray, box: tuple[int, int, int, int],
                 cells: int = 18, expand: float = 0.28):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - bw * expand)), max(0, int(y1 - bh * expand))
    x2, y2 = min(width, int(x2 + bw * expand)), min(height, int(y2 + bh * expand))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    region = image[y1:y2, x1:x2]
    # Blur before pixelation so facial contours cannot survive inside a large block.
    region = cv2.GaussianBlur(region, (0, 0), sigmaX=9, sigmaY=9)
    small_width = max(1, (x2 - x1) // cells)
    small_height = max(1, (y2 - y1) // cells)
    small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA)
    image[y1:y2, x1:x2] = cv2.resize(
        small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def resolve_yunet(model_dir: str | None) -> Path | None:
    candidates = []
    if model_dir:
        candidates.append(Path(model_dir).expanduser() / YUNET_FILE)
    candidates.append(Path(__file__).resolve().parent / "models" / YUNET_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    try:
        from huggingface_hub import hf_hub_download
        try:
            cached = hf_hub_download(YUNET_REPO, YUNET_FILE, local_files_only=True)
        except Exception:
            cached = hf_hub_download(YUNET_REPO, YUNET_FILE)
        return Path(cached)
    except Exception:
        return None


class YuNetOrtCuda:
    OUTPUT_NAMES = [
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32",
    ]
    STRIDES = (8, 16, 32)

    def __init__(self, model_path: Path, device_id: int = 0):
        # Import torch first so ORT can reuse the CUDA/cuDNN libraries bundled with it.
        import torch
        import onnxruntime as ort

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": device_id}),
                       "CPUExecutionProvider"],
        )
        if not self.session.get_providers() or self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider failed: {self.session.get_providers()}")
        self.providers = self.session.get_providers()
        self.input = self.session.get_inputs()[0]
        output_names = {output.name for output in self.session.get_outputs()}
        missing = set(self.OUTPUT_NAMES) - output_names
        if missing:
            raise RuntimeError(f"Incompatible YuNet outputs: {sorted(missing)}")
        shape = self.input.shape
        self.fixed_h = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else None
        self.fixed_w = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else None
        self.gpu_name = torch.cuda.get_device_name(device_id)

    def _prepare(self, image: np.ndarray):
        height, width = image.shape[:2]
        if self.fixed_w and self.fixed_h:
            scale = min(self.fixed_w / width, self.fixed_h / height)
            resized = cv2.resize(image, (max(1, round(width * scale)),
                                         max(1, round(height * scale))))
            padded = np.zeros((self.fixed_h, self.fixed_w, 3), dtype=np.uint8)
            padded[:resized.shape[0], :resized.shape[1]] = resized
        else:
            scale = 1.0
            padded_width = math.ceil(width / 32) * 32
            padded_height = math.ceil(height / 32) * 32
            padded = cv2.copyMakeBorder(
                image, 0, padded_height - height, 0, padded_width - width,
                cv2.BORDER_CONSTANT, value=(0, 0, 0))
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        return blob, scale, padded.shape[1], padded.shape[0]

    def detect(self, image: np.ndarray, confidence: float) -> list[tuple[int, int, int, int]]:
        blob, scale, padded_width, padded_height = self._prepare(image)
        values = self.session.run(self.OUTPUT_NAMES, {self.input.name: blob})
        boxes: list[list[float]] = []
        scores: list[float] = []
        for position, stride in enumerate(self.STRIDES):
            cls = np.asarray(values[position]).reshape(-1)
            obj = np.asarray(values[position + 3]).reshape(-1)
            bbox = np.asarray(values[position + 6]).reshape(-1, 4)
            rows, columns = padded_height // stride, padded_width // stride
            count = min(rows * columns, len(cls), len(obj), len(bbox))
            probabilities = np.sqrt(
                np.clip(cls[:count], 0, 1) * np.clip(obj[:count], 0, 1))
            for index in np.flatnonzero(probabilities >= confidence):
                row, column = divmod(int(index), columns)
                dx, dy, dw, dh = bbox[index]
                center_x = (column + dx) * stride
                center_y = (row + dy) * stride
                box_width = float(np.exp(dw) * stride)
                box_height = float(np.exp(dh) * stride)
                boxes.append([center_x - box_width / 2, center_y - box_height / 2,
                              box_width, box_height])
                scores.append(float(probabilities[index]))
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.3, top_k=5000)
        if indices is None or len(indices) == 0:
            return []
        height, width = image.shape[:2]
        output = []
        for index in np.asarray(indices).reshape(-1):
            x, y, box_width, box_height = boxes[int(index)]
            x1, y1 = max(0, int(x / scale)), max(0, int(y / scale))
            x2 = min(width, int((x + box_width) / scale))
            y2 = min(height, int((y + box_height) / scale))
            if x2 > x1 and y2 > y1:
                output.append((x1, y1, x2, y2))
        return output


class FaceDetector:
    def __init__(self, model_path: Path | None, face_size: int,
                 confidence: float, device_id: int = 0, force_cpu: bool = False):
        self.face_size = face_size
        self.confidence = confidence
        self.ort: YuNetOrtCuda | None = None
        self.opencv = None
        self.haar = None
        if model_path and not force_cpu:
            try:
                self.ort = YuNetOrtCuda(model_path, device_id=device_id)
                print(f"[Face model] YuNet ONNX Runtime CUDA; GPU={self.ort.gpu_name}; "
                      f"providers={self.ort.providers}", flush=True)
            except Exception as exc:
                print(f"[Face model] CUDA unavailable ({exc}); falling back to CPU", flush=True)
        if model_path and self.ort is None:
            self.opencv = cv2.FaceDetectorYN_create(
                str(model_path), "", (320, 320), score_threshold=confidence,
                nms_threshold=0.3, top_k=5000)
            print("[Face model] YuNet OpenCV CPU", flush=True)
        if self.ort is None and self.opencv is None:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("[Face model] Haar CPU fallback (YuNet model not found)", flush=True)

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.ort is not None:
            return self.ort.detect(image, self.confidence)
        height, width = image.shape[:2]
        if self.opencv is not None:
            if self.face_size and max(width, height) > self.face_size:
                scale = self.face_size / max(width, height)
                sample = cv2.resize(image, (round(width * scale), round(height * scale)))
            else:
                sample, scale = image, 1.0
            self.opencv.setInputSize((sample.shape[1], sample.shape[0]))
            self.opencv.setScoreThreshold(self.confidence)
            _, faces = self.opencv.detect(sample)
            output = []
            if faces is not None:
                for face in faces:
                    x, y, box_width, box_height = map(float, face[:4])
                    output.append((int(x / scale), int(y / scale),
                                   int((x + box_width) / scale),
                                   int((y + box_height) / scale)))
            return output
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return [(x, y, x + box_width, y + box_height)
                for x, y, box_width, box_height in faces]


class MultiFaceTracker:
    """Detect all faces on key frames and track each box between detections."""

    def __init__(self, detector: FaceDetector, interval: int = 3, grace: int = 4):
        self.detector = detector
        self.interval = max(1, interval)
        self.grace = max(0, grace)
        self.last_faces: list[tuple[int, int, int, int]] = []
        self.points: dict[int, np.ndarray] = {}
        self.previous_gray: np.ndarray | None = None
        self.missed = 0
        self.lk = dict(
            winSize=(31, 31), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    @staticmethod
    def _features(gray: np.ndarray, box: tuple[int, int, int, int]):
        x1, y1, x2, y2 = box
        height, width = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=30, qualityLevel=0.01, minDistance=5,
            mask=mask, blockSize=7)
        return None if points is None else points.reshape(-1, 1, 2)

    def update(self, image: np.ndarray, frame_index: int):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        should_detect = (frame_index - 1) % self.interval == 0 or not self.last_faces
        if should_detect:
            detected = self.detector.detect(image)
            if detected:
                self.last_faces = detected
                self.missed = 0
            elif self.missed < self.grace:
                self.missed += 1
            else:
                self.last_faces = []
            self.points = {}
            for index, box in enumerate(self.last_faces):
                points = self._features(gray, box)
                if points is not None:
                    self.points[index] = points
        elif self.previous_gray is not None and self.last_faces:
            tracked = []
            next_points = {}
            for index, box in enumerate(self.last_faces):
                points = self.points.get(index)
                if points is None or len(points) < 4:
                    tracked.append(box)
                    continue
                moved, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous_gray, gray, points, None, **self.lk)
                good = status.reshape(-1) == 1
                if good.sum() >= 4:
                    new_points = moved[good].reshape(-1, 2)
                    old_points = points[good].reshape(-1, 2)
                    dx = float(np.median(new_points[:, 0] - old_points[:, 0]))
                    dy = float(np.median(new_points[:, 1] - old_points[:, 1]))
                    x1, y1, x2, y2 = box
                    tracked.append((round(x1 + dx), round(y1 + dy),
                                    round(x2 + dx), round(y2 + dy)))
                    next_points[index] = new_points.reshape(-1, 1, 2)
                else:
                    tracked.append(box)
            self.last_faces = tracked
            self.points = next_points
        self.previous_gray = gray
        return self.last_faces


def start_decoder(path: Path, info: VideoInfo, mode: str):
    candidates = [False]
    if mode == "cuda" or (mode == "auto" and cuda_decode_available()):
        candidates = [True] if mode == "cuda" else [True, False]
    frame_size = info.width * info.height * 3
    last_error = ""
    for use_cuda in candidates:
        process = subprocess.Popen(
            decoder_command(path, use_cuda), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        collector = StderrCollector(process.stderr)
        collector.start()
        assert process.stdout is not None
        first_frame = read_exact(process.stdout, frame_size)
        if len(first_frame) == frame_size:
            print(f"[Decode] {'NVDEC/CUDA' if use_cuda else 'ffmpeg CPU'}", flush=True)
            return process, collector, first_frame
        process.kill()
        process.wait()
        last_error = collector.tail()
        if use_cuda and mode == "auto":
            print(f"[Decode] CUDA failed; falling back to CPU: {last_error}", flush=True)
            continue
        break
    raise RuntimeError(last_error or "ffmpeg decoder did not produce a complete frame")


def choose_encoder(mode: str) -> str:
    if mode == "x264":
        return "x264"
    available = nvenc_available()
    if mode == "nvenc" and not available:
        raise RuntimeError("h264_nvenc is unavailable")
    return "nvenc" if available else "x264"


def process_video(path: Path, output: Path, detector: FaceDetector,
                  face_interval: int, decode_mode: str, encoder_mode: str) -> bool:
    started = time.monotonic()
    info = probe_video(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {path}", flush=True)
    print(f"[Input] {info.width}x{info.height} {info.fps:.3f}fps "
          f"duration={info.duration if info.duration is not None else 'unknown'}s", flush=True)
    decoder, decode_errors, first_frame = start_decoder(path, info, decode_mode)
    encoder = choose_encoder(encoder_mode)
    print(f"[Encode] {'NVENC h264' if encoder == 'nvenc' else 'libx264 CPU'}", flush=True)
    encode_process = subprocess.Popen(
        encoder_command(path, output, info, encoder), stdin=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0)
    encode_errors = StderrCollector(encode_process.stderr)
    encode_errors.start()
    assert decoder.stdout is not None and encode_process.stdin is not None

    tracker = MultiFaceTracker(detector, interval=face_interval)
    frame_size = info.width * info.height * 3
    frame_index = 0
    face_instances = 0
    raw = first_frame
    try:
        while len(raw) == frame_size:
            frame_index += 1
            image = np.frombuffer(raw, dtype=np.uint8).reshape(
                info.height, info.width, 3).copy()
            faces = tracker.update(image, frame_index)
            for box in faces:
                heavy_mosaic(image, box)
            face_instances += len(faces)
            encode_process.stdin.write(image.tobytes())
            if frame_index % 100 == 0:
                elapsed = time.monotonic() - started
                effective_fps = frame_index / elapsed if elapsed else 0
                print(f"[{frame_index}] faces={len(faces)} elapsed={elapsed:.0f}s "
                      f"speed={effective_fps:.1f}fps", flush=True)
            raw = read_exact(decoder.stdout, frame_size)
    except (BrokenPipeError, KeyboardInterrupt):
        decoder.kill()
        encode_process.kill()
        raise
    finally:
        try:
            decoder.stdout.close()
        except Exception:
            pass
        decoder.wait()
        try:
            encode_process.stdin.close()
        except Exception:
            pass
        encode_process.wait()

    if decoder.returncode != 0:
        print(f"[Error] decoder failed: {decode_errors.tail()}", file=sys.stderr, flush=True)
        return False
    if encode_process.returncode != 0:
        print(f"[Error] encoder failed: {encode_errors.tail()}", file=sys.stderr, flush=True)
        return False
    elapsed = time.monotonic() - started
    source_duration = info.duration or (frame_index / info.fps if info.fps else 0)
    realtime = source_duration / elapsed if elapsed else 0
    print(f"[Done] {output} frames={frame_index} face_instances={face_instances} "
          f"elapsed={elapsed:.1f}s realtime={realtime:.2f}x", flush=True)
    return output.is_file() and output.stat().st_size > 1024


def collect_inputs(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        expanded = Path(value).expanduser()
        matches = [Path(item) for item in glob.glob(str(expanded))]
        if not matches:
            matches = [expanded]
        for match in matches:
            if match.is_dir():
                paths.extend(path for path in sorted(match.rglob("*"))
                             if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                             and not path.name.startswith("masked_"))
            elif match.is_file() and match.suffix.lower() in VIDEO_SUFFIXES:
                paths.append(match)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face-only GPU streaming mosaic (YuNet CUDA + NVDEC/NVENC)")
    parser.add_argument("inputs", nargs="+", help="Video files, directories, or glob patterns")
    parser.add_argument("--out-dir", default="masked_face_gpu", help="Output directory")
    parser.add_argument("--model-dir", default=None, help=f"Directory containing {YUNET_FILE}")
    parser.add_argument("--face-int", type=int, default=3,
                        help="Detect every N frames; intermediate frames use optical flow (default: 3)")
    parser.add_argument("--face-conf", type=float, default=0.30,
                        help="YuNet confidence threshold (default: 0.30)")
    parser.add_argument("--face-size", type=int, default=960,
                        help="Maximum OpenCV CPU detector input size (default: 960)")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--decode", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Decoder selection (default: auto)")
    parser.add_argument("--encoder", choices=("auto", "nvenc", "x264"), default="auto",
                        help="Encoder selection (default: auto)")
    parser.add_argument("--cpu", action="store_true", help="Force OpenCV CPU face inference")
    # Compatibility no-ops let the existing scheduler/menu pass legacy face-only flags safely.
    parser.add_argument("--no-card", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipe", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.face_int < 1:
        print("[Error] --face-int must be >= 1", file=sys.stderr)
        return 2
    if not 0 < args.face_conf <= 1:
        print("[Error] --face-conf must be in (0, 1]", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[Error] ffmpeg and ffprobe are required", file=sys.stderr)
        return 2
    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("[Error] no video files found", file=sys.stderr)
        return 2
    model = resolve_yunet(args.model_dir)
    detector = FaceDetector(
        model, face_size=args.face_size, confidence=args.face_conf,
        device_id=args.device_id, force_cpu=args.cpu)
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    for path in inputs:
        output = output_dir / f"masked_{path.stem}.mp4"
        try:
            if process_video(path, output, detector, args.face_int,
                             args.decode, args.encoder):
                succeeded += 1
        except KeyboardInterrupt:
            print("\n[Stopped] interrupted; incomplete output may remain", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"[Error] {path}: {exc}", file=sys.stderr, flush=True)
    print(f"\nSummary: {succeeded}/{len(inputs)} succeeded; output={output_dir}", flush=True)
    return 0 if succeeded == len(inputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
