#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键运行双鱼眼预检模型对照，并生成截图、JSON 与 Markdown 可行性报告。

示例（Ubuntu CPU）：

    python prescan_benchmark.py tests/prescan --device cpu \
      --out-dir prescan_benchmark

默认运行 SCRFD-2.5g/320、SCRFD-10g/320、YOLOv8-nano/320、
YOLOv8-nano/640。加 --include-yolov8m 会额外运行较慢的 YOLOv8-medium/640。

输入文件名应以 hasface 或 noface 开头，脚本据此计算视频级漏检和无脸误报。
每个模型的截图与 report.json 都保存在 --out-dir 下，最终汇总报告为
--out-dir/precheck_feasibility_report.md。
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import prescan_dual_fisheye_scrfd as prescan


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量对照双鱼眼人脸预检模型，并生成可行性报告。")
    parser.add_argument("inputs", nargs="+", help="含 hasface_/noface_ 视频的目录或文件")
    parser.add_argument("--out-dir", default="prescan_benchmark",
                        help="所有截图、JSON 和 Markdown 报告的输出目录")
    parser.add_argument("--model-dir", default=None,
                        help="SCRFD/YOLO 模型目录，默认当前项目目录")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "coreml"], default="cpu",
                        help="推理后端；Ubuntu CPU 请保持 cpu，NVIDIA 可选 cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=5.0,
                        help="每秒预检帧数，默认 5（灰度验证建议最终再提高到逐帧）")
    parser.add_argument("--screen-eye-width", type=int, default=640,
                        help="每只鱼眼预检前最大宽度，默认 640")
    parser.add_argument("--confidence", type=float, default=0.18,
                        help="基准收集候选的低阈值，默认 0.18")
    parser.add_argument("--fisheye-device", choices=sorted(prescan.production.FISHEYE_PRESETS),
                        default="pico4")
    parser.add_argument("--fisheye-strength", type=float, default=1.0)
    parser.add_argument("--fisheye-downscale", type=int, default=1)
    parser.add_argument("--fisheye-crop", type=float, default=1.0)
    parser.add_argument("--include-yolov8m", action="store_true",
                        help="额外测试 YOLOv8-medium/640；CPU 下明显更慢")
    return parser.parse_args()


def variant_specs(include_yolov8m):
    specs = [
        ("2.5g", 320, True),
        ("10g", 320, True),
        ("yolov8", 320, False),
        ("yolov8", 640, False),
    ]
    if include_yolov8m:
        specs.append(("yolov8m", 640, False))
    return specs


def classify(path):
    name = Path(path).stem.lower()
    if name.startswith("hasface"):
        return "hasface"
    if name.startswith("noface"):
        return "noface"
    return "unlabelled"


def scores(video):
    return [float(item["max_score"]) for item in video.get("hit_seconds", [])]


def next_quarter_hundredth(value):
    """取刚好越过无脸最高分的 0.025 阈值，如 0.601 -> 0.625。"""
    return min(1.0, math.ceil((float(value) + 1e-9) * 40) / 40)


def summarize_variant(report):
    videos = [item for item in report["videos"] if item.get("success")]
    positives = [item for item in videos if classify(item["input_file"]) == "hasface"]
    negatives = [item for item in videos if classify(item["input_file"]) == "noface"]
    unlabelled = [item for item in videos if classify(item["input_file"]) == "unlabelled"]
    max_negative = max((max(scores(item), default=0.0) for item in negatives), default=None)
    max_positive = max((max(scores(item), default=0.0) for item in positives), default=None)
    threshold = next_quarter_hundredth(max_negative) if max_negative is not None else None
    positive_hits = []
    negative_hits = []
    if threshold is not None:
        positive_hits = [sum(score >= threshold for score in scores(item))
                         for item in positives]
        negative_hits = [sum(score >= threshold for score in scores(item))
                         for item in negatives]
    qualified = bool(positives and negatives and threshold is not None
                     and all(count > 0 for count in positive_hits)
                     and not any(negative_hits))
    timing = report["detector"].get("timing", {})
    return {
        "model": report["detector"]["model"],
        "backend": report["detector"]["backend"],
        "input_size": report["configuration"]["input_size"],
        "max_positive": max_positive,
        "max_negative": max_negative,
        "threshold": threshold,
        "positive_videos": len(positives),
        "positive_videos_hit": sum(count > 0 for count in positive_hits),
        "positive_hit_seconds": sum(positive_hits),
        "negative_videos": len(negatives),
        "negative_videos_hit": sum(count > 0 for count in negative_hits),
        "negative_hit_seconds": sum(negative_hits),
        "unlabelled_videos": len(unlabelled),
        "average_ms": timing.get("average_ms", 0.0),
        "total_scan_seconds": sum(float(item.get("duration_seconds", 0)) for item in videos),
        "qualified": qualified,
    }


def recommendation(summaries):
    qualified = [item for item in summaries if item["qualified"]]
    if not qualified:
        return None
    # 先优先 nano / 640：它通常在双鱼眼小脸场景比 320 更稳，且比 medium 轻。
    preferred = next((item for item in qualified
                      if item["model"] == "YOLOv8-nano-face" and item["input_size"] == 640), None)
    if preferred:
        return preferred
    return min(qualified, key=lambda item: item["total_scan_seconds"])


def render_markdown(summaries, report_paths, args, elapsed):
    recommended = recommendation(summaries)
    lines = [
        "# 双鱼眼人脸预检可行性报告",
        "",
        "## 测试配置",
        "",
        f"- 低阈值候选采集：`{args.confidence:.3f}`",
        f"- 抽样率：`{args.sample_fps:g} fps`；左右眼独立去畸变、独立推理",
        f"- 鱼眼预置：`{args.fisheye_device}`，每目最大宽度：`{args.screen_eye_width}`",
        f"- 后端：`{args.device}`；基准耗时：`{elapsed:.1f} 秒`",
        "- 仅以文件名前缀 `hasface` / `noface` 进行标签统计。",
        "",
        "## 模型结果",
        "",
        "| 模型 | 输入 | 含脸最高分 | 无脸最高分 | 建议灰度阈值 | 含脸视频命中 | 无脸视频命中 | 平均推理 | 扫描耗时 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        score = lambda value: f"{value:.3f}" if value is not None else "n/a"
        threshold = score(item["threshold"])
        lines.append(
            f"| {item['model']} | {item['input_size']} | {score(item['max_positive'])} | "
            f"{score(item['max_negative'])} | {threshold} | "
            f"{item['positive_videos_hit']}/{item['positive_videos']} "
            f"({item['positive_hit_seconds']} 秒) | "
            f"{item['negative_videos_hit']}/{item['negative_videos']} "
            f"({item['negative_hit_seconds']} 秒) | "
            f"{item['average_ms']:.2f} ms | {item['total_scan_seconds']:.1f} 秒 |")

    lines += ["", "## 结论", ""]
    if recommended:
        lines += [
            f"当前测试集内的最佳灰度候选是 **{recommended['model']} / "
            f"{recommended['input_size']}**，建议阈值 **{recommended['threshold']:.3f}**。",
            "该阈值在本次命名为 `noface` 的视频中没有命中，同时每段 `hasface` 视频至少命中一次。",
        ]
    else:
        lines += [
            "没有模型能在当前样本中同时覆盖所有 `hasface` 视频并让所有 `noface` 视频零命中。",
            "不应使用单模型阈值自动跳过视频；请增加含脸样本或引入独立模型复核。",
        ]
    lines += [
        "",
        "这只代表当前样本集的分离能力，不是生产隐私安全结论。灰度期间，"
        "所有“候选跳过”视频仍必须运行现有生产打码算法反查；只有视频级漏检为 0，"
        "并覆盖短暂露脸、小脸、侧脸、遮挡、边缘和暗光后，才可启用复制/remux 跳过。",
        "",
        "## 产物",
        "",
    ]
    for path in report_paths:
        lines.append(f"- `{path}`：该模型的 JSON、按视频分目录的命中截图")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.sample_fps <= 0 or not 0 <= args.confidence <= 1:
        raise SystemExit("--sample-fps 必须大于 0，--confidence 必须在 0~1")
    output_dir = Path(args.out_dir).expanduser().resolve()
    videos = prescan.collect_videos(args.inputs, output_dir)
    if not videos:
        raise SystemExit("未找到视频输入")
    if not any(classify(path) == "hasface" for path in videos):
        raise SystemExit("未找到 hasface_ 开头的视频，无法生成可行性结论")
    if not any(classify(path) == "noface" for path in videos):
        raise SystemExit("未找到 noface_ 开头的视频，无法生成可行性结论")

    started = time.perf_counter()
    reports, report_paths = [], []
    for model, input_size, landmark_filter in variant_specs(args.include_yolov8m):
        variant_name = f"{model.replace('.', '_')}_{input_size}"
        variant_dir = output_dir / variant_name
        scan_args = SimpleNamespace(
            model=model, input_size=input_size, model_dir=args.model_dir,
            confidence=args.confidence, device=args.device, gpu_id=args.gpu_id,
            landmark_filter=landmark_filter, sample_fps=args.sample_fps,
            screen_eye_width=args.screen_eye_width,
            fisheye_device=args.fisheye_device,
            fisheye_strength=args.fisheye_strength,
            fisheye_downscale=args.fisheye_downscale,
            fisheye_crop=args.fisheye_crop,
        )
        detector, label, backend = prescan.create_detector(scan_args)
        print(f"\n[基准] {label} input={input_size}, backend={backend}")
        results = []
        for index, video in enumerate(videos, 1):
            print(f"  [{index}/{len(videos)}] {video.name}")
            results.append(prescan.scan_video(video, detector, scan_args, variant_dir))
        variant_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "configuration": vars(scan_args),
            "detector": {"model": label, "backend": backend,
                         "timing": detector.timing_stats()},
            "videos": results,
        }
        report_path = variant_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report)
        report_paths.append(report_path.relative_to(output_dir))

    summaries = [summarize_variant(report) for report in reports]
    elapsed = time.perf_counter() - started
    markdown = render_markdown(summaries, report_paths, args, elapsed)
    markdown_path = output_dir / "precheck_feasibility_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n[完成] Markdown 报告: {markdown_path}")
    print(f"[完成] 汇总 JSON: {summary_path}")
    return summaries


if __name__ == "__main__":
    main()
