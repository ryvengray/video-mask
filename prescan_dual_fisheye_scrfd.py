#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双鱼眼视频的人脸预检（SCRFD / YOLO）。

本脚本只做筛查，不会改写视频：完整扫描所有输入视频；某一秒首次检到
人脸时，将该帧（左右两目分别去畸变后的画面）保存为带检测框的截图。

默认输入为 tests/noface，适合先评估无脸视频的误报率：

    python prescan_dual_fisheye_scrfd.py
    python prescan_dual_fisheye_scrfd.py tests/noface --model-dir ./models
    python prescan_dual_fisheye_scrfd.py /videos --device cuda

macOS 上 --device auto 会优先使用 CoreML（可用时），否则 CPU；NVIDIA
服务器上会自动使用 CUDAExecutionProvider。正式打码前仍须使用生产脚本
复核；预检命中只表示“该视频不能跳过打码”。
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2

import video_mask_batch_fish_v1_plus as production


DEFAULT_INPUT = Path("tests/noface")
DEFAULT_OUTPUT = Path("tests/noface_prescan")


def collect_videos(inputs, output_dir):
    """递归收集视频，且不把本次截图目录当作输入。"""
    output_dir = output_dir.resolve()
    videos = []
    seen = set()
    extensions = {ext.lower() for ext in production.VIDEO_FORMATS}
    for raw in inputs:
        path = Path(raw).expanduser()
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in extensions:
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if output_dir in resolved.parents or resolved == output_dir:
                continue
            if resolved not in seen:
                videos.append(resolved)
                seen.add(resolved)
    return sorted(videos)


def resize_for_prescan(frame, eye_width, dual=True):
    """按每目最大宽度缩小，不放大，减少去畸变与预检的成本。"""
    height, width = frame.shape[:2]
    max_width = max(2, int(eye_width)) * (2 if dual else 1)
    if width <= max_width:
        return frame
    scale = max_width / width
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_detections(image, detections, label):
    """在去畸变截图上标注 SCRFD 预检框和分数。"""
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)


def detect_dual_fisheye(detector, undistorted, confidence):
    """左右眼独立推理，避免拼接画面压进方形模型后每目过小。"""
    height, width = undistorted.shape[:2]
    midpoint = width // 2
    detections = []
    for offset, eye in ((0, undistorted[:, :midpoint]),
                        (midpoint, undistorted[:, midpoint:])):
        for (x1, y1, x2, y2), score in detector.detect_with_conf(eye, confidence):
            detections.append((x1 + offset, y1, x2 + offset, y2, score))
    return detections


def scan_video(path, detector, args, output_dir):
    """顺序读取到 EOF；每个自然秒最多保存一次命中截图。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"input_file": str(path), "success": False, "error": "无法打开视频"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = production.get_fps(str(path))
    fps = max(float(fps), 1.0)
    sample_every = max(1, round(fps / args.sample_fps))
    screenshot_dir = output_dir / path.stem
    frame_index = 0
    scanned_frames = 0
    face_frames = 0
    detections_total = 0
    saved_seconds = set()
    screenshots = []
    hit_seconds = []
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

            small_frame = resize_for_prescan(frame, args.screen_eye_width, dual=True)
            corrected = production.fisheye_undistort(
                small_frame, strength=args.fisheye_strength, device=args.fisheye_device,
                downscale=args.fisheye_downscale, dual="true", crop=args.fisheye_crop)
            detections = detect_dual_fisheye(detector, corrected, args.confidence)
            if not detections:
                continue

            face_frames += 1
            detections_total += len(detections)
            second = int((frame_index - 1) / fps)
            if second in saved_seconds:
                continue
            saved_seconds.add(second)
            max_score = max(float(item[4]) for item in detections)
            hit_seconds.append({
                "second": second,
                "frame_index": frame_index,
                "detections": len(detections),
                "max_score": round(max_score, 6),
            })
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            marked = corrected.copy()
            draw_detections(marked, detections, args.model.upper())
            destination = screenshot_dir / f"second_{second:06d}_frame_{frame_index:08d}.jpg"
            if cv2.imwrite(str(destination), marked, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                screenshots.append(str(destination))
    finally:
        cap.release()

    return {
        "input_file": str(path),
        "success": True,
        "fps": round(fps, 3),
        "decoded_frames": frame_index,
        "scanned_frames": scanned_frames,
        "sample_every_frames": sample_every,
        "face_sample_frames": face_frames,
        "face_detections": detections_total,
        "face_seconds": len(saved_seconds),
        # 调参时可在不重复解码/推理的情况下，由该列表模拟更高阈值的结果。
        "hit_seconds": hit_seconds,
        "screenshots": screenshots,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="完整扫描双鱼眼视频；每个检出人脸的秒保存一张去畸变截图。")
    parser.add_argument("inputs", nargs="*", default=[str(DEFAULT_INPUT)],
                        help="视频或目录；默认 tests/noface")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT),
                        help="截图和 report.json 输出目录")
    parser.add_argument("--model-dir", default=None,
                        help="含 SCRFD/YOLO 人脸模型的目录（默认项目根目录）")
    parser.add_argument("--model", choices=["2.5g", "10g", "yolov8", "yolov8m"],
                        default="2.5g", help="预检模型；默认 SCRFD-2.5g")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "coreml"], default="auto",
                        help="auto: NVIDIA CUDA > macOS CoreML > CPU")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU 编号")
    parser.add_argument("--confidence", type=float, default=0.18,
                        help="预检阈值；低阈值提高召回，默认 0.18")
    parser.add_argument("--landmark-filter", action="store_true",
                        help="启用 SCRFD 五点关键点拓扑过滤，减少衣物/手等误报")
    parser.add_argument("--sample-fps", type=float, default=5.0,
                        help="每秒预检帧数，默认 5；设为视频 fps 可逐帧扫描")
    parser.add_argument("--screen-eye-width", type=int, default=640,
                        help="预检时每只鱼眼的最大宽度，默认 640")
    parser.add_argument("--input-size", type=int, default=320,
                        help="SCRFD 方形输入边长，默认 320")
    parser.add_argument("--fisheye-device", choices=sorted(production.FISHEYE_PRESETS),
                        default="pico4", help="必须与正式打码使用相同的鱼眼预置")
    parser.add_argument("--fisheye-strength", type=float, default=1.0)
    parser.add_argument("--fisheye-downscale", type=int, default=1,
                        help="去畸变内部降采样；预检默认 1")
    parser.add_argument("--fisheye-crop", type=float, default=1.0)
    return parser.parse_args()


def create_detector(args):
    """按命令参数创建预检模型；供单模型扫描和基准报告脚本复用。"""
    if args.model in ("2.5g", "10g"):
        detector = production.SCRFDFaceDetector(
            model_dir=args.model_dir, model=args.model, input_size=args.input_size,
            conf=args.confidence, device=args.device, gpu_id=args.gpu_id, use_gpu=True,
            landmark_filter=args.landmark_filter)
        return detector, f"SCRFD-{args.model}", detector.backend
    if args.model == "yolov8":
        detector = production.YOLOFaceDetector(
            model_dir=args.model_dir, yolo_size=args.input_size,
            use_gpu=args.device != "cpu")
        return detector, "YOLOv8-nano-face", detector.device
    detector = production.YOLOv8MFaceDetector(
        model_dir=args.model_dir, yolo_size=args.input_size,
        use_gpu=args.device != "cpu")
    return detector, "YOLOv8-medium-face", detector.device


def main():
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps 必须大于 0")
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence 必须在 0~1")
    if args.screen_eye_width < 160 or args.input_size < 160:
        raise SystemExit("--screen-eye-width 和 --input-size 均必须至少为 160")
    if not 0 < args.fisheye_crop <= 1:
        raise SystemExit("--fisheye-crop 必须在 (0, 1]")

    output_dir = Path(args.out_dir).expanduser().resolve()
    videos = collect_videos(args.inputs, output_dir)
    if not videos:
        raise SystemExit(f"未找到视频：{', '.join(args.inputs)}")

    detector, detector_label, backend = create_detector(args)
    filter_note = (f", landmark_filter={'on' if args.landmark_filter else 'off'}"
                   if args.model in ("2.5g", "10g") else "")
    print(f"[预检] {detector_label} input={detector.input_size if hasattr(detector, 'input_size') else detector.yolo_size}, backend={backend}, "
          f"conf={args.confidence:.2f}, sample={args.sample_fps:g}fps{filter_note}")
    print(f"[预检] 双鱼眼: 每目最大宽度={args.screen_eye_width}, "
          f"preset={args.fisheye_device}, crop={args.fisheye_crop}")

    results = []
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] 扫描 {video}")
        result = scan_video(video, detector, args, output_dir)
        results.append(result)
        if result["success"]:
            print(f"  完成: 扫描={result['scanned_frames']}帧, "
                  f"命中秒={result['face_seconds']}, 截图={len(result['screenshots'])}")
        else:
            print(f"  失败: {result['error']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": vars(args),
        "detector": {"model": detector_label, "backend": backend,
                     "timing": detector.timing_stats()},
        "videos": results,
        "summary": {
            "videos": len(results),
            "successful_videos": sum(item["success"] for item in results),
            "videos_with_faces": sum(item.get("face_seconds", 0) > 0 for item in results),
            "saved_screenshots": sum(len(item.get("screenshots", [])) for item in results),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[预检] 报告: {report_path}")
    return report


if __name__ == "__main__":
    main()
