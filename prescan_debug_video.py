#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a dual-fisheye prescan review video with face boxes and scores.

The output uses the same resize, dual-eye undistortion and detector path as
``prescan_dual_fisheye_scrfd.py``.  It is intended for human review only; it
does not mask or change the source video.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2

import prescan_dual_fisheye_scrfd as prescan


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成双鱼眼人脸预检的带框复核视频（无音轨）。")
    parser.add_argument("input", help="输入视频")
    parser.add_argument("--out-dir", required=True, help="调试视频和 JSON 的输出目录")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model", choices=["yolov8", "yolov8m", "2.5g", "10g"],
                        default="yolov8")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--sample-fps", type=float, default=30.0,
                        help="检测及输出帧率；30 fps 视频使用 30 即逐帧复核")
    parser.add_argument("--screen-eye-width", type=int, default=640)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "coreml"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--fisheye-device", choices=sorted(prescan.production.FISHEYE_PRESETS),
                        default="pico4")
    parser.add_argument("--fisheye-strength", type=float, default=1.0)
    parser.add_argument("--fisheye-downscale", type=int, default=1)
    parser.add_argument("--fisheye-crop", type=float, default=1.0)
    return parser.parse_args()


def banner(image, text, color):
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2,
                cv2.LINE_AA)


def main():
    args = parse_args()
    if not 0 < args.sample_fps or not 0 <= args.confidence <= 1:
        raise SystemExit("--sample-fps 必须大于 0，--confidence 必须在 0~1")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"找不到输入视频：{input_path}")
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"debug_{input_path.stem}.mp4"
    temporary_path = output_dir / f".debug_{input_path.stem}.mp4v.mp4"

    detector, detector_label, backend = prescan.create_detector(args)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{input_path}")
    fps = max(float(cap.get(cv2.CAP_PROP_FPS) or 0), 1.0)
    sample_every = max(1, round(fps / args.sample_fps))
    output_fps = fps / sample_every
    writer = None
    frame_index = 0
    scanned_frames = 0
    face_frames = 0
    total_detections = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % sample_every:
                continue

            scanned_frames += 1
            small_frame = prescan.resize_for_prescan(frame, args.screen_eye_width, dual=True)
            corrected = prescan.production.fisheye_undistort(
                small_frame, strength=args.fisheye_strength, device=args.fisheye_device,
                downscale=args.fisheye_downscale, dual="true", crop=args.fisheye_crop)
            detections = prescan.detect_dual_fisheye(detector, corrected, args.confidence)
            total_detections += len(detections)
            if detections:
                face_frames += 1
                prescan.draw_detections(corrected, detections, detector_label)
                status, color = f"FACE CANDIDATE: {len(detections)}", (0, 220, 0)
            else:
                status, color = "NO CANDIDATE", (60, 60, 255)
            timestamp = (frame_index - 1) / fps
            banner(corrected, f"frame {frame_index} | {timestamp:.3f}s | {status} | threshold {args.confidence:.3f}",
                   color)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(temporary_path), fourcc, output_fps,
                                         (corrected.shape[1], corrected.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("无法创建调试视频编码器")
            writer.write(corrected)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if writer is None:
        raise SystemExit("视频中没有可导出的帧")
    transcode = subprocess.run(
        ["ffmpeg", "-y", "-i", str(temporary_path), "-c:v", "libx264", "-crf", "18",
         "-preset", "medium", "-movflags", "+faststart", "-an", str(output_path)],
        text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if transcode.returncode:
        raise RuntimeError(f"调试视频转码失败：{transcode.stderr[-1000:]}")
    temporary_path.unlink(missing_ok=True)

    report = {
        "input_file": str(input_path),
        "debug_video": str(output_path),
        "model": detector_label,
        "backend": backend,
        "confidence": args.confidence,
        "source_fps": fps,
        "sample_every_frames": sample_every,
        "output_fps": output_fps,
        "decoded_frames": frame_index,
        "scanned_frames": scanned_frames,
        "candidate_frames": face_frames,
        "detections": total_detections,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "timing": detector.timing_stats(),
    }
    report_path = output_dir / f"debug_{input_path.stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
