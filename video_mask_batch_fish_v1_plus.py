#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_mask_batch_fish_v1_plus.py — 视频批量打码（仅人脸），SCRFD 增强版

人脸:
  YuNet(默认, 仅 OpenCV 依赖) / YOLOv8-nano(极速, 需 ultralytics)
  → LK 光流跟踪
  检测间隔+光流跟踪(每5帧检测1次, 中间帧LK跟踪)

加速:
  管道模式(ffmpeg rawvideo → Python → ffmpeg, 无磁盘I/O)
  硬件编码(macOS VideoToolbox / NVENC / QSV, 自动检测)
  关键帧检测 + 光流跟踪(人脸)
  抽帧跳帧(--frame-skip, 提速2-3倍)

用法:
  python video_mask_batch_fish_v1_plus.py video.mp4
  python video_mask_batch_fish_v1_plus.py ./videos/              # 整个目录
  python video_mask_batch_fish_v1_plus.py a.mp4 b.mov            # 多文件
  python video_mask_batch_fish_v1_plus.py video.mp4 --face-model yolov8
  python video_mask_batch_fish_v1_plus.py video.mp4 --face-model yolo11
  python video_mask_batch_fish_v1_plus.py video.mp4 --face-model scrfd --scrfd-model 10g
  python video_mask_batch_fish_v1_plus.py video.mp4 --face-model yolov8+scrfd
  python video_mask_batch_fish_v1_plus.py video.mp4 --out-dir ./out
  python video_mask_batch_fish_v1_plus.py video.mp4 --no-face            # 关闭人脸打码
  python video_mask_batch_fish_v1_plus.py video.mp4 --face-conf 0.25
  python video_mask_batch_fish_v1_plus.py video.mp4 --fisheye
  python video_mask_batch_fish_v1_plus.py video.mp4 --fisheye --fisheye-device pico4
  python video_mask_batch_fish_v1_plus.py video.mp4 --frame-skip 2

依赖:
  pip install opencv-python numpy
  # YOLOv8 人脸检测可选: pip install ultralytics
  # SCRFD 人脸检测可选: pip install onnxruntime
  # 系统需 ffmpeg(保留音轨)

代码调用:
  from video_mask_batch_fish_v1_plus import process_video
  process_video("in.mp4", "out.mp4", face_model="yunet")
"""
import argparse
import glob
import os
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

# ================= 打码/检测参数 =================
FACE_CELLS, FACE_SIGMA = 4, 45.0
FACE_INPUT = 320              # 人脸检测输入尺寸(320: 比400快约30%, 精度几乎无损失)
FACE_CONF = 0.45             # 人脸置信度阈值(0.45->0.35: 侧脸/低头/遮挡帧不漏检, 跟踪器滤误检)
FACE_EXPAND = 0.12            # 人脸打码框外扩比例(确保盖住完整脸)
FACE_YUNET_SIZE = 640         # 人脸检测输入尺寸(YuNet/YOLOv8 均自动调整为32的倍数, 640x640)
FACE_DETECT_INT = 5           # 常态每5帧检测一次；变化期临时逐帧检测
FACE_EMPTY_DETECT_INT = 5     # 无人场景每5帧扫描一次，避免空场景浪费推理
FACE_BACKFILL = 5             # 双模型确认后，最多回补此前5帧的单模型候选
FACE_BURST = 5                # 检测人数变化后，接下来5帧逐帧检测
FACE_ACTIVE_HOLD = 10         # 已出现人脸后，轨迹短暂丢失仍保持主动扫描的帧数
FACE_VISIBLE_HOLD = 2         # 已确认轨迹允许用上一可靠框保活的连续帧数
HARD_FACE_CONF = 0.20
HARD_FACE_MIN_SIZE = 100
HARD_FACE_EDGE_RATIO = 0.16
HARD_FACE_ROI_SCALE = 3.0
HARD_FACE_MAX_ROIS = 3             # 每帧最多二检的困难框数量(防止推理爆炸)
HARD_FACE_ROI_SIZE = 320           # ROI 二检推理尺寸(比主检测 640 小一倍, 提速~75%)
HARD_FACE_FULL_SCAN_CONF = 0.15    # 全帧低阈值扫描置信度(低于主检测, 独立捕获遗漏人脸)
SCRFD_ADAPTIVE_VERIFY_MARGIN = 0.15  # 新候选进入10g复核区间的分数增量
SCRFD_ADAPTIVE_VERIFY_IOU = 0.25     # 10g复核框与候选框的最低IoU
SCRFD_VERIFIER_MAX_SCORE_DROP = 0.02  # 复核时10g相对候选模型允许的最大分差
SCRFD_VERIFIER_CONF = 0.30          # 10g复核内部阈值；不改变主模型建候选阈值
SCRFD_LANDMARK_RISK_THRESHOLD = 0.35  # 五点软风险达到该值时触发10g，不直接拒绝
SCRFD_REJECT_COOLDOWN = 5           # 10g拒绝后，同位置候选暂停复核的帧数
STEREO_LOW_CONF = 0.30              # 另一目参与左右互证的最低本地候选分数
STEREO_HIGH_MARGIN = 0.05           # 已确认侧至少有一个模型高于主阈值的增量
STEREO_X_DISPARITY_GATE = 0.16      # 未标定时左右归一化横向视差容差
STEREO_Y_GATE = 0.08                # 左右归一化纵坐标容差
FRAME_SKIP = 1                # 抽帧跳过间隔(1=逐帧处理; 2=隔1帧抽1帧提速2x; 3=每3帧抽1帧提速3x)
FACE_GRACE = 3               # 单张脸允许漏掉的检测周期数
FACE_BOX_SMOOTH = 0.65       # 检测框EMA中本次检测权重(越低越稳, 越高越跟手)
# 误检几何过滤(排除手/玩具等非人脸误检, 零耗时增加)
FACE_MIN_SIZE = 15           # 人脸框最小边长(像素), 低于视为噪点丢弃
FACE_MAX_AREA_RATIO = 0.12   # 单脸最大面积占比(超过画面12%视为大物体误检, 如玩具堆)
FACE_ASPECT_MIN = 0.5        # 人脸长宽比(宽/高)下限, 低于视为细长物(手/手臂)误检
FACE_ASPECT_MAX = 2.0        # 人脸长宽比上限, 超过视为细长物误检

VIDEO_FORMATS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v",
                 ".mpg", ".mpeg", ".ts", ".m2ts", ".wmv", ".3gp", ".rmvb",
                 ".rm", ".vob", ".asf")


# ================= 打码 =================

def heavy_mosaic(img, x1, y1, x2, y2, cells=6, sigma=35.0):
    """高斯模糊 + 像素化马赛克"""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    # 限制高斯核大小: 原 sigma=45 → 核 135, 常大于人脸框; 限制到 51 提速且不影响打码效果
    k = min(int(sigma * 3) | 1, 51, max(bw, bh) | 1)
    roi = cv2.GaussianBlur(roi, (k, k), sigma)
    cw = max(1, bw // cells)
    ch = max(1, bh // cells)
    small = cv2.resize(roi, (cw, ch), interpolation=cv2.INTER_LINEAR)
    big = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = big


def _ensure_writable_frame(img):
    """ffmpeg pipe 返回 bytes，其 numpy 视图只读；需要绘制时才复制。"""
    return img if img.flags.writeable else img.copy()


def draw_face_debug_scores(img, debug_faces):
    """绘制模型分值：D=双模型、V=10g、X=左右互证、S/T/H=轨迹稳定。"""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.85, min(w, h) / 1100.0))
    thickness = max(1, int(round(scale * 2)))
    for item in debug_faces:
        x1, y1, x2, y2 = [int(round(v)) for v in item["box"]]
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        scores = item.get("scores") or ()
        text = " | ".join(
            f"{name} {score:.3f}" if score is not None else f"{name} n/a"
            for name, score in scores)
        text = f"[{item.get('source', 'T')}] {text}"
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        tx = min(x1, max(0, w - tw - 8))
        ty = y1 - 5 if y1 > th + base + 8 else y1 + th + 8
        top = max(0, ty - th - 5)
        bottom = min(h - 1, ty + base + 5)
        cv2.rectangle(img, (tx, top), (min(w - 1, tx + tw + 8), bottom), (0, 0, 0), -1)
        colors = {"D": (0, 255, 0), "V": (255, 0, 255),
                  "S": (255, 180, 0), "T": (0, 200, 255),
                  "H": (0, 165, 255), "X": (255, 255, 0)}
        color = colors.get(item.get("source"), (0, 200, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, text, (tx + 4, ty), font, scale, color, thickness, cv2.LINE_AA)


def draw_raw_face_debug(img, raw_faces):
    """绘制正式阈值一半以上的原始候选框；仅用于诊断，不参与打码。"""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.8, min(w, h) / 1200.0))
    thickness = max(1, int(round(scale * 2)))
    for item in raw_faces:
        x1, y1, x2, y2 = [int(round(v)) for v in item["box"]]
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        score = item.get("score")
        text = f"RAW {item.get('model', '?')} {score:.3f}" if score is not None else f"RAW {item.get('model', '?')} n/a"
        color = (180, 180, 180) if not item.get("formal") else (255, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, text, (x1, max(15, y1 - 5)), font, scale, color, thickness, cv2.LINE_AA)


def _interpolate_box(start_box, end_box, start_frame, end_frame, frame_idx):
    span = max(1, end_frame - start_frame)
    alpha = min(1.0, max(0.0, (frame_idx - start_frame) / span))
    return tuple(int(round((1.0 - alpha) * start_box[i] + alpha * end_box[i]))
                 for i in range(4))


def apply_face_backfill(frame_buffer, events):
    """在尚未编码的帧缓冲区中插值补码；返回实际补码帧次数。"""
    applied = 0
    for event in events:
        for item in frame_buffer:
            frame_idx = item["frame_idx"]
            if not (event["start_frame"] <= frame_idx < event["end_frame"]):
                continue
            box = _interpolate_box(
                event["start_box"], event["end_box"],
                event["start_frame"], event["end_frame"], frame_idx)
            image = _ensure_writable_frame(item["image"])
            item["image"] = image
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            heavy_mosaic(
                image, int(x1 - FACE_EXPAND * bw), int(y1 - FACE_EXPAND * bh),
                int(x2 + FACE_EXPAND * bw), int(y2 + FACE_EXPAND * bh),
                FACE_CELLS, FACE_SIGMA)
            applied += 1
    return applied


# -- 鱼眼去畸变 (无需标定, 自动估算) --

_fisheye_maps_cache = {}  # 按 (w, h, f_ratio, k1, k2, balance) 缓存 remap 映射表

# 设备预置参数: (f_ratio, k1, k2, balance)
#   f_ratio: 焦距比例, fx = max(w,h) * f_ratio, 越小=FOV越宽
#   k1/k2:  OpenCV fisheye 畸变系数, 正值=桶形畸变矫正
#   balance: 0=全矫正(最大黑边), 1=不裁剪(保留全部像素但保留畸变), 推荐 0.4~0.6
FISHEYE_PRESETS = {
    "generic": {
        "f_ratio": 0.45,   # 通用强鱼眼(≈150°+ FOV)
        "k1": 0.35,
        "k2": 0.08,
        "balance": 0.6,
    },
    "pico4": {
        # Pico 4 RGB 摄像头: H130°/V115°, 中等广角桶形畸变
        # 基于 Pico 4 官方规格: IMX471 sensor, 130° HFOV
        # 使用 equidistant 投影模型估算: f = w / (2 * tan⁻¹(HFOV/2)) ≈ w * 0.44
        "f_ratio": 0.44,
        "k1": 0.15,         # 温和桶形矫正(Pico 4 非极端鱼眼)
        "k2": 0.03,
        "balance": 0.45,    # 保留更多画面(VR 透视场景不宜裁切过多)
    },
}


def _is_dual_fisheye(width, height, dual="auto"):
    if dual == "auto":
        return width >= 2 * height
    return dual == "true" or dual is True


def fisheye_undistort(img, strength=1.0, device="generic", downscale=2, dual="auto", crop=1.0):
    """对单帧做鱼眼去畸变, 返回矫正后的图像。

    支持设备预置(device="pico4" 等)或通用模式手动 strength 调节。
    映射表按 (分辨率, 参数) 缓存, 同一视频只算一次。

    dual: 双鱼眼模式。
        "auto"=按宽高比自动检测(w>=2h 视为双鱼眼拼接, 适用于 Pico4 双目录像);
        "true"=强制切分; "false"=强制单鱼眼(兼容旧版)。
        双鱼眼会切左右两半各自独立矫正, 避免单鱼眼模型把光心设在中央黑缝导致的
        错误畸变(Pico4 3840x1456 是两个 1920x1456 鱼眼左右拼接, 必须启用此模式)。

    downscale: 降采样倍数(>=1)。>1 时在 1/downscale 分辨率上做 remap 再上采样,
        remap 面积减 downscale² 倍, 大幅提速(1920x1456 downscale=2 约 -75% 耗时)。
        画质损失可忽略(双线性插值两次, 人脸打码场景无感知)。设 1 关闭。

    crop: 裁剪比例(0~1)。矫正后裁剪边缘区域, 去除畸变最严重的外围。
        双鱼眼模式下每目独立居中裁剪, 避免从拼接缝中心裁剪导致两目内容丢失。
        默认 1.0 不裁剪。
    """
    h, w = img.shape[:2]

    # 双鱼眼模式判定: 显式指定或按宽高比自动检测
    # 宽>=2倍高 视为双鱼眼拼接(典型 Pico4 双目 3840x1456 = 2.64:1)
    is_dual = _is_dual_fisheye(w, h, dual)

    if is_dual and w >= 2:
        # 双鱼眼拼接(如 Pico4 3840x1456 = 两个 1920x1456 左右并排):
        # 切左右两半各自独立矫正, 否则单鱼眼模型把光心设在 (w/2, h/2) 正好落在
        # 中央黑缝, 黑缝被当作"鱼眼中心"展开 → 中间一大块拉成黑填充;
        # 左右两个真实鱼眼被当作"远离光心的边缘区域"反向错误畸变 → 检测率暴跌。
        # 切分后每个半边用自身维度算 fx = max(1920,1456)*0.44 = 845(正确),
        # 而非错误的 max(3840,1456)*0.44 = 1690(2x 偏大)。
        mid = w // 2
        left = fisheye_undistort(img[:, :mid], strength, device, downscale, dual="false", crop=crop)
        right = fisheye_undistort(img[:, mid:], strength, device, downscale, dual="false", crop=crop)
        return np.concatenate([left, right], axis=1)

    # === 以下为单鱼眼处理逻辑 ===

    # 获取设备参数
    preset = FISHEYE_PRESETS.get(device, FISHEYE_PRESETS["generic"])
    f_ratio = preset["f_ratio"]
    k1 = preset["k1"] * strength
    k2 = preset["k2"] * strength
    balance = preset["balance"]

    if downscale > 1:
        # 降采样路径: 在小图上 remap 再上采样回原尺寸
        sw, sh = max(2, w // downscale), max(2, h // downscale)
        ds_key = (sw, sh, f_ratio, k1, k2, balance)
        if ds_key not in _fisheye_maps_cache:
            fx = max(sw, sh) * f_ratio
            K = np.array([[fx, 0, sw / 2], [0, fx, sh / 2], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, 0, 0], dtype=np.float64)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (sw, sh), np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (sw, sh), cv2.CV_16SC2)
            _fisheye_maps_cache[ds_key] = (map1, map2)
        map1, map2 = _fisheye_maps_cache[ds_key]
        img_small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        undistorted_small = cv2.remap(img_small, map1, map2, cv2.INTER_LINEAR)
        img = cv2.resize(undistorted_small, (w, h), interpolation=cv2.INTER_LINEAR)

    else:
        # 原分辨率路径
        full_key = (w, h, f_ratio, k1, k2, balance)
        if full_key not in _fisheye_maps_cache:
            fx = max(w, h) * f_ratio
            K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, 0, 0], dtype=np.float64)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
            _fisheye_maps_cache[full_key] = (map1, map2)
        map1, map2 = _fisheye_maps_cache[full_key]
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # 裁剪边缘(单鱼眼分支, 双鱼眼已在递归调用中各自裁剪)
    if 0 < crop < 1.0:
        ch, cw = img.shape[:2]
        new_w = int(cw * crop) // 2 * 2
        new_h = int(ch * crop) // 2 * 2
        y0 = (ch - new_h) // 2
        x0 = (cw - new_w) // 2
        img = np.ascontiguousarray(img[y0:y0 + new_h, x0:x0 + new_w])
    return img


# ================= 人脸检测 =================

# YuNet 人脸检测模型 (OpenCV cv2.FaceDetectorYN, 仅依赖 opencv-python)
# OpenCV 4.8+ 内置 YuNet; 也可手动放置 face_detection_yunet_2023mar.onnx
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

# YOLOv8 nano 人脸检测模型 (ultralytics 自动下载，约 6MB)
YOLO_FACE_FILE = "yolov8n-face.pt"
YOLO_FACE_M_FILE = "yolov8m-face.pt"  # YOLOv8 medium 人脸检测模型(精度更高, 约 52MB)
YOLO11_FACE_FILE = "yolo11n-face.pt"  # YOLOv11 nano 人脸检测模型(ultralytics 最新一代, 6MB)
YOLO11_FACE_FILE_ALT = "yolov11n-face.pt"  # 社区常见命名变体(同一模型, 文件名带v)
YOLO_INPUT_SIZE = 640          # YOLOv8 检测输入尺寸(32的倍数; 降低可提速)


class YuNetFaceDetector:
    """YuNet 人脸检测器 — OpenCV 内置轻量模型, 零额外依赖。

    与 YOLOFaceDetector 保持相同接口: detect(img, conf) -> [(x1,y1,x2,y2),...]
    基于 cv2.FaceDetectorYN, 模型通过 OpenCV 内置或本地 ONNX 加载。
    适合追求最小依赖的场景(仅需 opencv-python)。
    """

    def __init__(self, model_dir=None, yunet_size=FACE_YUNET_SIZE, use_gpu=True):
        self.det_size = self._normalize_size(yunet_size)
        model_path = self._find_model(model_dir)

        try:
            if model_path:
                self._detector = cv2.FaceDetectorYN.create(
                    model_path, "",
                    (self.det_size, self.det_size),
                    score_threshold=FACE_CONF,
                )
            else:
                # 找不到本地 onnx, 让 OpenCV 自动从内置查找/下载
                self._detector = cv2.FaceDetectorYN.create(
                    YUNET_FILE, "",
                    (self.det_size, self.det_size),
                    score_threshold=FACE_CONF,
                )
        except Exception as e:
            raise RuntimeError(
                f"YuNet 模型加载失败: {e}。请下载 {YUNET_FILE} 放到 "
                f"--model-dir 指定目录或脚本当前目录。"
            ) from e

        # GPU 加速: YuNet 通过 OpenCV DNN 后端, 按优先级 CUDA > OpenCL > CPU
        # setPreferableBackend 在后端不可用时通常抛 cv2.error; 不可靠时用 warmup 耗时判断
        import time
        backend_label = "CPU"
        if use_gpu:
            for label, backend, target in [
                ("CUDA", cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA),
                ("OPENCL", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL),
            ]:
                try:
                    self._detector.setPreferableBackend(backend)
                    self._detector.setPreferableTarget(target)
                    backend_label = label
                    break
                except Exception:
                    continue

        # warmup 推理: 用户可通过耗时对比判断 GPU 是否真生效
        # (CPU 通常 >80ms, CUDA 通常 <20ms; 若声明 CUDA 但 warmup 仍 >80ms 说明静默回退)
        _dummy = np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8)
        try:
            self._detector.setInputSize((self.det_size, self.det_size))
            t0 = time.time()
            self._detector.detect(_dummy)
            warmup_ms = (time.time() - t0) * 1000
        except Exception:
            warmup_ms = -1

        print(f"[人脸] YuNet (输入{self.det_size}, 后端={backend_label}, warmup={warmup_ms:.0f}ms)")

    @staticmethod
    def _find_model(model_dir):
        """按优先级搜索 YuNet ONNX 文件。返回路径或 None(交由 OpenCV 内置查找)。

        查找顺序: --model-dir > CWD > 脚本所在目录 > OpenCV 内置目录
        (cv2.data.haarcascades 同级)。
        """
        candidates = []
        if model_dir:
            candidates.append(os.path.join(model_dir, YUNET_FILE))
        candidates.append(os.path.join(os.getcwd(), YUNET_FILE))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       YUNET_FILE))
        # OpenCV 内置目录 (cv2.data.haarcascades 同级)
        try:
            cv_data_dir = os.path.dirname(cv2.data.haarcascades)
            if cv_data_dir:
                candidates.append(os.path.join(cv_data_dir, YUNET_FILE))
        except Exception:
            pass
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _normalize_size(size):
        """调整为 32 的倍数 (与 YOLOFaceDetector 一致)。"""
        return max(32, round(size / 32) * 32)

    def detect(self, img, conf=FACE_CONF):
        """返回 bbox 列表, 格式与 YOLOFaceDetector.detect() 完全一致。"""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        # YuNet 要求每次检测前设置当前图像尺寸
        try:
            self._detector.setInputSize((w, h))
        except Exception:
            return []
        _, faces = self._detector.detect(img)
        if faces is None:
            return []
        out = []
        for row in faces:
            if len(row) < 5:
                continue
            x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(row[4])
            if score < conf:
                continue
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out


class YOLOFaceDetector:
    """YOLOv8-nano 人脸检测器 — 极速轻量级替代方案。

    与 YuNetFaceDetector 保持相同接口: detect(img, conf) -> [(x1,y1,x2,y2),...]
    比 YuNet 快, 模型仅 6MB, 适合追求速度的场景。
    使用 ultralytics 包，首次运行自动下载模型到 ~/.cache/ultralytics/。
    """

    def __init__(self, model_dir=None, yolo_size=YOLO_INPUT_SIZE, use_gpu=True):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics。请安装：pip install ultralytics"
            ) from exc
        self.yolo_size = self._normalize_size(yolo_size)
        self.device = self._select_device(use_gpu)

        # 查找模型: model_dir > CWD > 脚本目录 > ultralytics 自动下载
        model_path = self._find_model(model_dir)
        self._model = YOLO(model_path if model_path else YOLO_FACE_FILE)

        # fuse(): 融合 Conv+BN 层, 数学等价但推理快 5-15%
        try:
            self._model.fuse()
        except Exception:
            pass

        # 预热: 跑一次空推理，避免首帧卡顿
        import numpy as np
        dummy = np.zeros((self.yolo_size, self.yolo_size, 3), dtype=np.uint8)
        self._model(dummy, imgsz=self.yolo_size, device=self.device,
                    conf=0.5, verbose=False)
        self._init_timing("YOLOv8")

    def _init_timing(self, label):
        """初始化正式视频帧的模型耗时统计(不包含预热)。"""
        self.timing_label = label
        self._timing = {"calls": 0, "total": 0.0}

    def _record_timing(self, elapsed):
        self._timing["calls"] += 1
        self._timing["total"] += max(0.0, float(elapsed))

    def timing_stats(self):
        calls = self._timing["calls"]
        total = self._timing["total"]
        return {
            "calls": calls,
            "total_seconds": total,
            "average_ms": total * 1000.0 / calls if calls else 0.0,
        }

    @staticmethod
    def _find_model(model_dir):
        """按优先级搜索模型文件。返回路径或 None(触发 ultralytics 自动下载)。"""
        candidates = []
        if model_dir:
            candidates.append(os.path.join(model_dir, YOLO_FACE_FILE))
        candidates.append(os.path.join(os.getcwd(), YOLO_FACE_FILE))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       YOLO_FACE_FILE))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _normalize_size(size):
        """调整为 32 的倍数 (YOLO 要求)。"""
        return max(32, round(size / 32) * 32)

    @staticmethod
    def _select_device(use_gpu):
        """自动选择最优设备: CUDA > MPS > CPU。"""
        if not use_gpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def detect(self, img, conf=FACE_CONF):
        """返回 bbox 列表, 格式与 YuNetFaceDetector.detect() 完全一致。"""
        return [b for b, _ in self.detect_with_conf(img, conf)]

    def detect_with_conf(self, img, conf=FACE_CONF):
        """返回 [(bbox, conf), ...] 带 YOLO 置信度, 供 SCRFD 验证按 conf 分流。

        与 detect() 相同的几何过滤, 额外保留置信度供 verify() 决定是否需要二次验证。
        """
        if img is None or img.size == 0:
            return []

        started = time.perf_counter()
        # stream=True: 生成器模式, 减少结果对象包装开销
        results = self._model(img, imgsz=self.yolo_size, conf=conf,
                              device=self.device, verbose=False, stream=True)
        h, w = img.shape[:2]
        img_area = w * h
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                if cls_id != 0:  # class 0 = face (yolov8n-face 单类别模型)
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                # 几何过滤: 排除手/玩具等非人脸误检(零耗时增加)
                if bw < FACE_MIN_SIZE or bh < FACE_MIN_SIZE:
                    continue  # 太小, 噪点
                if bw * bh > img_area * FACE_MAX_AREA_RATIO:
                    continue  # 太大, 大物体误检(如玩具堆/整片区域)
                ratio = bw / bh
                if ratio < FACE_ASPECT_MIN or ratio > FACE_ASPECT_MAX:
                    continue  # 长宽比异常, 细长物(手/手臂)误检
                c = float(box.conf[0]) if box.conf is not None else 0.0
                out.append(((x1, y1, x2, y2), c))
        self._record_timing(time.perf_counter() - started)
        return out


class YOLO11FaceDetector(YOLOFaceDetector):
    """YOLOv11-nano 人脸检测器 — ultralytics 最新一代模型。

    与 YOLOFaceDetector 接口完全一致, 仅模型文件不同(yolo11n-face.pt)。
    YOLOv11 在 COCO 等数据集上 mAP 与速度均优于 YOLOv8, 作为备选方案。
    继承 YOLOFaceDetector 复用 _normalize_size / _select_device / detect 等方法。
    """

    def __init__(self, model_dir=None, yolo_size=YOLO_INPUT_SIZE, use_gpu=True):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics。请安装：pip install ultralytics"
            ) from exc
        self.yolo_size = self._normalize_size(yolo_size)
        self.device = self._select_device(use_gpu)

        # 查找模型: model_dir > CWD > 脚本目录
        # 支持 yolo11n-face.pt 和 yolov11n-face.pt 两种命名
        model_path = self._find_yolo11_model(model_dir)
        if model_path is None:
            # yolo11n-face.pt 非 ultralytics 官方模型, 不会自动下载
            # 回退到本地 yolov8n-face.pt(精度接近), 避免直接抛 FileNotFoundError
            fallback = YOLOFaceDetector._find_model(model_dir)
            if fallback:
                print(f"[人脸] [警告] 未找到 {YOLO11_FACE_FILE}/{YOLO11_FACE_FILE_ALT}, "
                      f"回退使用 {os.path.basename(fallback)} "
                      f"(yolo11n 与 yolov8n 在 WIDERFace 人脸检测上精度接近)")
                model_path = fallback
            else:
                # 本地无任何 face 模型, 交给 ultralytics 尝试下载(可能失败)
                model_path = YOLO11_FACE_FILE
        self._model = YOLO(model_path)

        # fuse(): 融合 Conv+BN 层, 数学等价但推理快 5-15%
        try:
            self._model.fuse()
        except Exception:
            pass

        # 预热: 跑一次空推理，避免首帧卡顿
        import numpy as np
        dummy = np.zeros((self.yolo_size, self.yolo_size, 3), dtype=np.uint8)
        self._model(dummy, imgsz=self.yolo_size, device=self.device,
                    conf=0.5, verbose=False)
        self._init_timing("YOLO11")

    @staticmethod
    def _find_yolo11_model(model_dir):
        """按优先级搜索 yolo11n-face.pt / yolov11n-face.pt 文件。
        返回路径或 None(交由调用方回退到 yolov8n-face.pt)。

        查找顺序: --model-dir > CWD > 脚本所在目录
        支持两种命名: yolo11n-face.pt(ultralytics 风格) / yolov11n-face.pt(社区常见)
        """
        names = [YOLO11_FACE_FILE, YOLO11_FACE_FILE_ALT]
        candidates = []
        if model_dir:
            for n in names:
                candidates.append(os.path.join(model_dir, n))
        for n in names:
            candidates.append(os.path.join(os.getcwd(), n))
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), n))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None


class YOLOv8MFaceDetector(YOLOFaceDetector):
    """YOLOv8-medium 人脸检测器 — 精度更高的中型模型。

    与 YOLOFaceDetector 接口完全一致, 仅模型文件不同(yolov8m-face.pt)。
    YOLOv8m 参数量约为 YOLOv8n 的 8 倍(52MB vs 6MB), 精度更高但速度更慢。
    适合追求检出率、对速度要求不极端的场景(如鱼眼视频、小脸场景)。
    继承 YOLOFaceDetector 复用 _normalize_size / _select_device / detect 等方法。
    """

    def __init__(self, model_dir=None, yolo_size=YOLO_INPUT_SIZE, use_gpu=True):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics。请安装：pip install ultralytics"
            ) from exc
        self.yolo_size = self._normalize_size(yolo_size)
        self.device = self._select_device(use_gpu)

        model_path = self._find_model_m(model_dir)
        if model_path is None:
            # 回退到 yolov8n-face.pt
            fallback = YOLOFaceDetector._find_model(model_dir)
            if fallback:
                print(f"[人脸] [警告] 未找到 {YOLO_FACE_M_FILE}, "
                      f"回退使用 {os.path.basename(fallback)}")
                model_path = fallback
            else:
                model_path = YOLO_FACE_M_FILE
        self._model = YOLO(model_path)

        try:
            self._model.fuse()
        except Exception:
            pass

        import numpy as np
        dummy = np.zeros((self.yolo_size, self.yolo_size, 3), dtype=np.uint8)
        self._model(dummy, imgsz=self.yolo_size, device=self.device,
                    conf=0.5, verbose=False)
        self._init_timing("YOLOv8m")

    @staticmethod
    def _find_model_m(model_dir):
        """按优先级搜索 yolov8m-face.pt 文件。"""
        candidates = []
        if model_dir:
            candidates.append(os.path.join(model_dir, YOLO_FACE_M_FILE))
        candidates.append(os.path.join(os.getcwd(), YOLO_FACE_M_FILE))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       YOLO_FACE_M_FILE))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None


class DualFaceDetector:
    """双模型共识检测器: 两个模型同时检测, IoU 匹配一致才保留。

    大幅降低单模型误检率(如 YOLO 把手/玩具误认为人脸, 但另一个模型不认;
    此时双方不一致, 不打码, 宁可漏打不误打)。
    支持任意两个检测器组合: YOLO11+YuNet, YOLOv8+YOLO11 等。
    """
    def __init__(self, det_a, det_b, name_a="A", name_b="B",
                 verifier=None, verify_margin=SCRFD_ADAPTIVE_VERIFY_MARGIN,
                 verify_iou=SCRFD_ADAPTIVE_VERIFY_IOU,
                 landmark_risk_threshold=SCRFD_LANDMARK_RISK_THRESHOLD,
                 reject_cooldown=SCRFD_REJECT_COOLDOWN):
        self._det_a = det_a
        self._det_b = det_b
        self._name_a = name_a
        self._name_b = name_b
        self._name = f"{name_a}+{name_b}"
        # 可选的高精度复核器。只用于新出现的中等置信度候选，
        # 不参与常规双模型检测，避免把10g变成逐帧主模型。
        self._verifier = verifier
        self._verify_margin = max(0.0, float(verify_margin))
        self._verify_iou = float(verify_iou)
        self._landmark_risk_threshold = float(landmark_risk_threshold)
        self._reject_cooldown = max(0, int(reject_cooldown))
        self._rejected_candidates = []
        self.last_candidates = []
        self.last_raw_candidates = []
        self.last_verified_count = 0
        self.last_score_rejected_count = 0
        self._stereo_split_enabled = False
        self._stereo_low_conf = STEREO_LOW_CONF

    def enable_stereo_split(self, enabled=True, low_conf=STEREO_LOW_CONF):
        """启用左右双鱼眼独立检测及同帧软互证。"""
        self._stereo_split_enabled = bool(enabled)
        self._stereo_low_conf = max(0.0, float(low_conf))

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def detect(self, img, conf=FACE_CONF, iou_thresh=0.2):
        return [item["box"] for item in self.detect_with_scores(
            img, conf=conf, iou_thresh=iou_thresh)]

    @staticmethod
    def _detect_with_conf(detector, img, conf):
        if hasattr(detector, "detect_with_conf"):
            return detector.detect_with_conf(img, conf=conf)
        return [(box, None) for box in detector.detect(img, conf=conf)]

    def _detect_with_details(self, detector, img, conf):
        pairs = self._detect_with_conf(detector, img, conf)
        details = getattr(detector, "last_detection_details", ())
        enriched = []
        for box, score in pairs:
            risk = 0.0
            for detail in details:
                if self._iou(box, detail["box"]) >= 0.999:
                    risk = float(detail.get("landmark_risk", 0.0))
                    break
            enriched.append({"box": box, "score": score,
                             "landmark_risk": risk})
        return enriched

    def _is_tracked(self, box, existing_boxes, iou_thresh):
        return any(self._iou(box, old_box) >= iou_thresh
                   for old_box in existing_boxes)

    def _in_reject_cooldown(self, box, frame_idx, view_key=None):
        if frame_idx is None or self._reject_cooldown <= 0:
            return False
        self._rejected_candidates = [
            item for item in self._rejected_candidates
            if item["until"] >= frame_idx]
        return any(item.get("view") == view_key
                   and self._iou(box, item["box"]) >= self._verify_iou
                   for item in self._rejected_candidates)

    def _dual_score_consistent(self, item):
        """YOLOv8+SCRFD-10g新人脸的分数一致性检查。

        仅用于两模型都检出的新人脸；已有轨迹在调用处先直接续接。
        其他模型组合和2.5g主检路径不改变原有逻辑。
        """
        if (not self._name_a.upper().startswith("YOLOV8")
                or "SCRFD-10G" not in self._name_b.upper()):
            return True
        scores = item.get("scores") or ()
        if len(scores) < 2 or scores[0][1] is None or scores[1][1] is None:
            return True
        yolo_score = float(scores[0][1])
        scrfd_score = float(scores[1][1])
        return scrfd_score + 1e-6 >= (
            yolo_score - SCRFD_VERIFIER_MAX_SCORE_DROP)

    def _verifier_score_consistent(self, item, verifier_score):
        """检查新人脸候选与SCRFD-10g复核分数是否一致。"""
        if verifier_score is None:
            return True
        scores = item.get("scores") or ()
        if len(scores) < 2:
            return True

        reference_score = None
        # 只要使用SCRFD-2.5g且它有分数，10g就必须对比2.5g；
        # 同时覆盖“2.5g单模型候选”和“YOLOv8+2.5g共识后复核”。
        if ("SCRFD-2.5G" in self._name_b.upper()
                and scores[1][1] is not None):
            reference_score = scores[1][1]
        # YOLOv8 单独候选、10g 复核：10g 对比 YOLOv8。
        elif (self._name_a.upper().startswith("YOLOV8")
              and scores[0][1] is not None and scores[1][1] is None):
            reference_score = scores[0][1]
        if reference_score is None:
            return True
        return float(verifier_score) + 1e-6 >= (
            float(reference_score) - SCRFD_VERIFIER_MAX_SCORE_DROP)

    def _has_required_primary_scores(self, item):
        """2.5g组合的新轨迹必须同时得到YOLOv8和2.5g有效分数。"""
        if "SCRFD-2.5G" not in self._name_b.upper():
            return True
        scores = item.get("scores") or ()
        return (len(scores) >= 2
                and scores[0][1] is not None
                and scores[1][1] is not None
                and float(scores[0][1]) > 0.0
                and float(scores[1][1]) > 0.0)

    @staticmethod
    def _offset_box(box, dx):
        return (box[0] + dx, box[1], box[2] + dx, box[3])

    def _offset_items(self, items, dx):
        shifted = []
        for item in items:
            copied = dict(item)
            copied["box"] = self._offset_box(item["box"], dx)
            shifted.append(copied)
        return shifted

    @staticmethod
    def _item_peak_score(item):
        values = [float(score) for _, score in (item.get("scores") or ())
                  if score is not None]
        return max(values, default=0.0)

    @staticmethod
    def _normalized_box(box, eye_w, height):
        cx = (box[0] + box[2]) * 0.5 / max(eye_w, 1)
        cy = (box[1] + box[3]) * 0.5 / max(height, 1)
        size = max(box[2] - box[0], box[3] - box[1], 1)
        return cx, cy, size

    def _stereo_pair_cost(self, left_box, right_box, eye_w, height,
                          expected_disparity=0.0, calibrated=False):
        lx, ly, ls = self._normalized_box(left_box, eye_w, height)
        rx, ry, rs = self._normalized_box(right_box, eye_w, height)
        x_gate = 0.10 if calibrated else STEREO_X_DISPARITY_GATE
        dx_error = abs((lx - rx) - expected_disparity)
        dy = abs(ly - ry)
        ratio = ls / max(rs, 1)
        if (dx_error > x_gate or dy > STEREO_Y_GATE
                or ratio < 0.45 or ratio > 2.20):
            return None
        return dx_error / x_gate + dy / STEREO_Y_GATE + abs(1.0 - ratio)

    def _match_stereo_items(self, left_items, right_items, eye_w, height,
                            expected_disparity=0.0, calibrated=False):
        pairs = []
        for li, left in enumerate(left_items):
            for ri, right in enumerate(right_items):
                cost = self._stereo_pair_cost(
                    left["box"], right["box"], eye_w, height,
                    expected_disparity, calibrated)
                if cost is not None:
                    pairs.append((cost, li, ri))
        pairs.sort()
        used_left, used_right, matched = set(), set(), []
        for _, li, ri in pairs:
            if li in used_left or ri in used_right:
                continue
            used_left.add(li)
            used_right.add(ri)
            matched.append((li, ri))
        return matched

    def _cluster_stereo_raw(self, raw_items, confirmed, iou_thresh):
        """合并同一目内YOLO/SCRFD的低分框，保留本地模型分数。"""
        clusters = []
        for raw in raw_items:
            score = raw.get("score")
            if score is None or float(score) < self._stereo_low_conf:
                continue
            if any(self._iou(raw["box"], item["box"]) >= iou_thresh
                   for item in confirmed):
                continue
            target = next((cluster for cluster in clusters
                           if self._iou(raw["box"], cluster["box"]) >= 0.30),
                          None)
            if target is None:
                target = {"box": raw["box"], "best": float(score),
                          "model_scores": {}}
                clusters.append(target)
            if float(score) > target["best"]:
                target["box"] = raw["box"]
                target["best"] = float(score)
            model = raw.get("model", "candidate")
            target["model_scores"][model] = max(
                float(score), target["model_scores"].get(model, 0.0))
        return [{
            "box": cluster["box"],
            "scores": tuple(cluster["model_scores"].items()),
            "landmark_risk": 0.0,
            "source": "C",
        } for cluster in clusters]

    def _stereo_supported_results(self, left_confirmed, right_confirmed,
                                  left_raw, right_raw, eye_w, height,
                                  conf, iou_thresh):
        # 先用两侧都已确认的人脸估计本帧平均视差；没有可靠配对时
        # 使用接近零视差的宽门限，适配Pico4相邻双目。
        confirmed_pairs = self._match_stereo_items(
            left_confirmed, right_confirmed, eye_w, height)
        disparities = []
        for li, ri in confirmed_pairs:
            lx, _, _ = self._normalized_box(
                left_confirmed[li]["box"], eye_w, height)
            rx, _, _ = self._normalized_box(
                right_confirmed[ri]["box"], eye_w, height)
            disparities.append(lx - rx)
        calibrated = bool(disparities)
        expected = float(np.median(disparities)) if disparities else 0.0
        paired_left = {li for li, _ in confirmed_pairs}
        paired_right = {ri for _, ri in confirmed_pairs}

        left_low = self._cluster_stereo_raw(
            left_raw, left_confirmed, iou_thresh)
        right_low = self._cluster_stereo_raw(
            right_raw, right_confirmed, iou_thresh)
        high_cutoff = float(conf) + STEREO_HIGH_MARGIN
        high_left = [item for i, item in enumerate(left_confirmed)
                     if i not in paired_left
                     and self._item_peak_score(item) >= high_cutoff]
        high_right = [item for i, item in enumerate(right_confirmed)
                      if i not in paired_right
                      and self._item_peak_score(item) >= high_cutoff]

        right_support = []
        for hi, ci in self._match_stereo_items(
                high_left, right_low, eye_w, height, expected, calibrated):
            candidate = dict(right_low[ci])
            candidate["source"] = "X"
            candidate["scores"] = tuple(candidate["scores"]) + (
                ("Stereo-L", self._item_peak_score(high_left[hi])),)
            right_support.append(candidate)

        left_support = []
        for ci, hi in self._match_stereo_items(
                left_low, high_right, eye_w, height, expected, calibrated):
            candidate = dict(left_low[ci])
            candidate["source"] = "X"
            candidate["scores"] = tuple(candidate["scores"]) + (
                ("Stereo-R", self._item_peak_score(high_right[hi])),)
            left_support.append(candidate)
        return left_support, right_support

    def _detect_stereo_with_scores(self, img, conf=FACE_CONF, iou_thresh=0.2,
                                   include_single=False, raw_conf=None,
                                   existing_boxes=None, frame_idx=None):
        height, width = img.shape[:2]
        mid = width // 2
        if mid < 2:
            return self._detect_single_view_with_scores(
                img, conf, iou_thresh, include_single, raw_conf,
                existing_boxes, frame_idx)
        left_img = np.ascontiguousarray(img[:, :mid])
        right_img = np.ascontiguousarray(img[:, mid:])
        left_existing, right_existing = [], []
        for box in existing_boxes or []:
            if (box[0] + box[2]) * 0.5 < mid:
                left_existing.append(box)
            else:
                right_existing.append(self._offset_box(box, -mid))
        query_raw = (self._stereo_low_conf if raw_conf is None
                     else min(float(raw_conf), self._stereo_low_conf))

        left_result = self._detect_single_view_with_scores(
            left_img, conf, iou_thresh, include_single, query_raw,
            left_existing, frame_idx, view_key="L")
        left_candidates = list(self.last_candidates)
        left_raw = list(self.last_raw_candidates)
        left_verified = self.last_verified_count
        left_rejected = self.last_score_rejected_count

        right_result = self._detect_single_view_with_scores(
            right_img, conf, iou_thresh, include_single, query_raw,
            right_existing, frame_idx, view_key="R")
        right_candidates = list(self.last_candidates)
        right_raw = list(self.last_raw_candidates)
        right_verified = self.last_verified_count
        right_rejected = self.last_score_rejected_count

        left_support, right_support = self._stereo_supported_results(
            left_result, right_result, left_raw, right_raw,
            mid, height, conf, iou_thresh)
        self.last_candidates = (left_candidates
                                + self._offset_items(right_candidates, mid))
        self.last_raw_candidates = (left_raw
                                    + self._offset_items(right_raw, mid))
        self.last_verified_count = left_verified + right_verified
        self.last_score_rejected_count = left_rejected + right_rejected
        return (left_result + left_support
                + self._offset_items(right_result + right_support, mid))

    def detect_with_scores(self, img, conf=FACE_CONF, iou_thresh=0.2,
                           include_single=False, raw_conf=None,
                           existing_boxes=None, frame_idx=None):
        if self._stereo_split_enabled and img.shape[1] >= 2 * img.shape[0]:
            return self._detect_stereo_with_scores(
                img, conf, iou_thresh, include_single, raw_conf,
                existing_boxes, frame_idx)
        return self._detect_single_view_with_scores(
            img, conf, iou_thresh, include_single, raw_conf,
            existing_boxes, frame_idx)

    def _detect_single_view_with_scores(self, img, conf=FACE_CONF,
                                        iou_thresh=0.2,
                                        include_single=False, raw_conf=None,
                                        existing_boxes=None, frame_idx=None,
                                        view_key=None):
        """返回已确认框；单模型结果只是候选，不能直接创建轨迹。"""
        query_conf = conf if raw_conf is None else min(float(conf), float(raw_conf))
        dets_a_all = self._detect_with_details(self._det_a, img, query_conf)
        dets_b_all = self._detect_with_details(self._det_b, img, query_conf)
        dets_a = [item for item in dets_a_all
                  if item["score"] is None or item["score"] >= conf]
        dets_b = [item for item in dets_b_all
                  if item["score"] is None or item["score"] >= conf]
        self.last_raw_candidates = [
            {"box": item["box"], "model": self._name_a,
             "score": item["score"],
             "formal": item["score"] is None or item["score"] >= conf}
            for item in dets_a_all
        ] + [
            {"box": item["box"], "model": self._name_b,
             "score": item["score"],
             "formal": item["score"] is None or item["score"] >= conf}
            for item in dets_b_all
        ]

        # 贪心匹配: 对每个 A 框找最佳 IoU 匹配的 B 框
        consensus = []
        used_a, used_b = set(), set()
        for ai, da in enumerate(dets_a):
            best_iou, best_j = 0, -1
            for j, db in enumerate(dets_b):
                if j in used_b:
                    continue
                score = self._iou(da["box"], db["box"])
                if score > best_iou:
                    best_iou, best_j = score, j
            if best_iou >= iou_thresh and best_j >= 0:
                db = dets_b[best_j]
                consensus.append({
                    "box": da["box"],
                    "scores": ((self._name_a, da["score"]),
                               (self._name_b, db["score"])),
                    "landmark_risk": max(da["landmark_risk"],
                                         db["landmark_risk"]),
                    "source": "D",
                })
                used_a.add(ai)
                used_b.add(best_j)

        singles = ([{
            "box": item["box"],
            "scores": ((self._name_a, item["score"]), (self._name_b, None)),
            "landmark_risk": item["landmark_risk"], "source": "C",
        } for ai, item in enumerate(dets_a) if ai not in used_a] + [{
            "box": item["box"],
            "scores": ((self._name_a, None), (self._name_b, item["score"])),
            "landmark_risk": item["landmark_risk"], "source": "C",
        } for bi, item in enumerate(dets_b) if bi not in used_b])
        self.last_candidates = list(singles) if include_single else []

        # 已确认轨迹可由双模型或任一主模型续接；只有新人脸候选才需要
        # 做分数/五点软风险判断并按需调用10g。
        self.last_verified_count = 0
        self.last_score_rejected_count = 0
        existing_boxes = existing_boxes or []
        kept = []
        verify_candidates = []
        verify_high = float(conf) + self._verify_margin
        for item in consensus:
            if self._is_tracked(item["box"], existing_boxes, iou_thresh):
                kept.append(item)
                continue
            if not self._has_required_primary_scores(item):
                self.last_score_rejected_count += 1
                continue
            if not self._dual_score_consistent(item):
                self.last_score_rejected_count += 1
                continue
            scores = [float(score) for _, score in item["scores"]
                      if score is not None]
            medium_conf = bool(scores) and min(scores) < verify_high
            risky_landmarks = (item["landmark_risk"]
                               >= self._landmark_risk_threshold)
            if self._verifier is not None and (medium_conf or risky_landmarks):
                verify_candidates.append(item)
            else:
                kept.append(item)

        if self._verifier is not None:
            # 2.5g模式下，新人脸必须先由YOLOv8+2.5g共同检出；10g
            # 只能复核双模型候选，不能替代其中任意一个创建新轨迹。
            # 已有轨迹的单模型框仍由上层续轨，避免正常人脸闪动。
            for item in singles:
                if self._is_tracked(item["box"], existing_boxes, iou_thresh):
                    continue
                if not self._has_required_primary_scores(item):
                    self.last_score_rejected_count += 1
                    continue
                verify_candidates.append(item)
        if not verify_candidates:
            return kept

        active_verify = [
            item for item in verify_candidates
            if not self._in_reject_cooldown(
                item["box"], frame_idx, view_key=view_key)]
        if not active_verify:
            return kept
        # 复核器只在已有候选框的局部位置做 IoU 匹配，因此可以使用比
        # 主模型更低的召回阈值，捞回鱼眼/小脸的 0.30~0.45 分数；
        # 主模型 conf 仍保持不变，不会因此直接创建低分新轨迹。
        verified = self._detect_with_conf(
            self._verifier, img, min(float(conf), SCRFD_VERIFIER_CONF))
        self.last_verified_count = len(active_verify)
        for item in active_verify:
            matches = [(self._iou(item["box"], box), box, score)
                       for box, score in verified]
            best = max(matches, default=(0.0, None, None), key=lambda row: row[0])
            # YOLOv8 单模型候选与 SCRFD-2.5g 单模型候选都要和10g
            # 做分数一致性检查；其他模型组合不改变原有复核逻辑。
            score_ok = self._verifier_score_consistent(item, best[2])
            if not score_ok:
                self.last_score_rejected_count += 1
            if best[0] >= self._verify_iou and score_ok:
                confirmed = dict(item)
                confirmed["source"] = "V"
                confirmed["scores"] = tuple(item["scores"]) + (
                    ("SCRFD-10g", best[2]),)
                kept.append(confirmed)
            elif frame_idx is not None and self._reject_cooldown > 0:
                self._rejected_candidates.append({
                    "box": item["box"],
                    "until": frame_idx + self._reject_cooldown,
                    "view": view_key,
                })
        return kept


class SCRFDFaceDetector:
    """SCRFD ONNX 人脸检测器，前后处理与 face-detect 项目保持一致。

    接口与 YuNetFaceDetector/YOLOFaceDetector 一致：
    detect(img, conf) -> [(x1, y1, x2, y2), ...]。
    """

    MODEL_FILES = {
        "2.5g": "scrfd_2.5g_bnkps.onnx",
        "10g": "scrfd_10g_bnkps.onnx",
    }
    STRIDES = (8, 16, 32)

    def __init__(self, model_dir=None, model="10g", input_size=640, conf=FACE_CONF,
                 nms_thresh=0.4, device="auto", gpu_id=0, use_gpu=True,
                 model_path=None, landmark_filter=True):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "缺少 onnxruntime。请安装：pip install onnxruntime"
            ) from exc

        if model not in self.MODEL_FILES:
            raise ValueError(f"不支持的 SCRFD 模型: {model}")
        model_path = model_path or self._find_model(model_dir, model)
        if model_path is None:
            name = self.MODEL_FILES[model]
            raise RuntimeError(
                f"未找到 {name}。请放到项目目录，或通过 --model-dir 指定模型目录"
            )

        self.input_size = self._normalize_size(input_size)
        self.conf = float(conf)
        self.nms_thresh = float(nms_thresh)
        self.model = model
        self.model_path = model_path
        self.landmark_filter = bool(landmark_filter)
        self.last_detection_details = []
        self._center_cache = {}

        requested = device.lower()
        if requested not in ("auto", "cpu", "cuda", "coreml"):
            raise ValueError(f"不支持的 SCRFD 推理设备: {device}")
        if not use_gpu:
            requested = "cpu"

        options = ort.SessionOptions()
        options.log_severity_level = 3
        available = ort.get_available_providers()
        provider_name = "CPUExecutionProvider"
        providers = ["CPUExecutionProvider"]
        if requested in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            provider_name = "CUDAExecutionProvider"
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": str(gpu_id),
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "1",
                }),
                "CPUExecutionProvider",
            ]
        elif requested in ("auto", "coreml") and "CoreMLExecutionProvider" in available:
            provider_name = "CoreMLExecutionProvider"
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif requested in ("cuda", "coreml"):
            print(f"[人脸] [警告] SCRFD 请求的 {requested} 后端不可用，"
                  f"可用后端={available}，回退 CPU")

        try:
            self._sess = ort.InferenceSession(model_path, options, providers=providers)
        except Exception as exc:
            if provider_name == "CPUExecutionProvider":
                raise
            print(f"[人脸] [警告] SCRFD {provider_name} 初始化失败({exc})，回退 CPU")
            provider_name = "CPUExecutionProvider"
            self._sess = ort.InferenceSession(
                model_path, options, providers=["CPUExecutionProvider"])

        self._input_name = self._sess.get_inputs()[0].name
        outputs = self._sess.get_outputs()
        if len(outputs) != 9:
            raise RuntimeError(
                f"SCRFD 模型输出数异常: {len(outputs)}，期望 9 (3尺度x[分数,框,关键点])"
            )
        self._output_names = [out.name for out in outputs]

        # 预热也能尽早发现 CUDA/CoreML 驱动或模型 shape 不兼容。
        dummy = np.zeros((1, 3, self.input_size, self.input_size), np.float32)
        try:
            self._sess.run(self._output_names, {self._input_name: dummy})
        except Exception as exc:
            if provider_name == "CPUExecutionProvider":
                raise
            print(f"[人脸] [警告] SCRFD {provider_name} 预热失败({exc})，回退 CPU")
            self._sess = ort.InferenceSession(
                model_path, options, providers=["CPUExecutionProvider"])
            self._sess.run(self._output_names, {self._input_name: dummy})
        self.backend = self._sess.get_providers()[0]
        self.timing_label = f"SCRFD-{self.model}"
        # 仅统计正式视频帧；模型加载和上面的预热不计入运行时性能数据。
        self._timing = {
            "calls": 0,
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
        }
        self._allocate_preprocess_buffers()

    def _allocate_preprocess_buffers(self):
        """为固定 SCRFD 输入尺寸分配一次可复用的预处理内存。"""
        size = self.input_size
        self._canvas = np.empty((size, size, 3), dtype=np.uint8)
        self._blob = np.empty((1, 3, size, size), dtype=np.float32)
        self._resized = None
        self._resized_shape = None

    def _prepare_input(self, img):
        """等价于原 blobFromImage 路径，但复用 canvas/blob，避免逐帧分配。"""
        h, w = img.shape[:2]
        size = self.input_size
        ratio_hw = h / w
        if ratio_hw > 1.0:
            new_h, new_w = size, int(size / ratio_hw)
        else:
            new_w, new_h = size, int(size * ratio_hw)
        scale = new_h / h

        self._canvas.fill(0)
        # canvas[:, :new_w] 通常不是连续内存；部分服务器 OpenCV 版本不接受
        # 非连续 numpy 切片作为 resize 的 dst。使用按尺寸缓存的连续缓冲区，
        # 仍然避免逐帧分配，同时兼容不同 OpenCV 构建。
        resized_shape = (new_h, new_w, 3)
        if self._resized is None or self._resized_shape != resized_shape:
            self._resized = np.empty(resized_shape, dtype=np.uint8)
            self._resized_shape = resized_shape
        cv2.resize(img, (new_w, new_h), dst=self._resized)
        self._canvas[:new_h, :new_w] = self._resized
        # BGR HWC uint8 -> RGB NCHW float32；copyto 完成转置和类型转换但不分配。
        # 少数 ONNX Runtime/CUDA 组合可能把输入标记为只读；发生时只替换一次，
        # 新 blob 仍会在后续检测中持续复用。
        if not self._blob.flags.writeable:
            self._blob = np.empty_like(self._blob)
        np.copyto(
            self._blob[0], self._canvas[:, :, ::-1].transpose(2, 0, 1),
            casting="unsafe")
        np.subtract(self._blob, 127.5, out=self._blob)
        np.multiply(self._blob, 1.0 / 128.0, out=self._blob)
        return self._blob, scale

    def _record_timing(self, preprocess, inference, postprocess):
        self._timing["calls"] += 1
        self._timing["preprocess"] += max(0.0, float(preprocess))
        self._timing["inference"] += max(0.0, float(inference))
        self._timing["postprocess"] += max(0.0, float(postprocess))

    def timing_stats(self):
        """返回 SCRFD 正式检测调用的累计与平均分阶段耗时。"""
        calls = self._timing["calls"]
        stages = ("preprocess", "inference", "postprocess")
        total = sum(self._timing[name] for name in stages)
        stats = {
            "calls": calls,
            "total_seconds": total,
            "average_ms": total * 1000.0 / calls if calls else 0.0,
        }
        for name in stages:
            seconds = self._timing[name]
            stats[f"{name}_seconds"] = seconds
            stats[f"{name}_average_ms"] = seconds * 1000.0 / calls if calls else 0.0
            stats[f"{name}_ratio"] = seconds / total if total else 0.0
        return stats

    @classmethod
    def _find_model(cls, model_dir, model):
        name = cls.MODEL_FILES[model]
        candidates = []
        if model_dir:
            candidates.append(os.path.join(model_dir, name))
        candidates.extend([
            os.path.join(os.getcwd(), name),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        ])
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _normalize_size(size):
        return max(32, round(int(size) / 32) * 32)

    def _anchors(self, stride):
        key = (self.input_size, stride)
        if key not in self._center_cache:
            side = self.input_size // stride
            centers = np.stack(np.mgrid[:side, :side][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            # SCRFD 每个网格位置有 2 个 anchor，顺序与模型输出一致。
            centers = np.stack([centers] * 2, axis=1).reshape(-1, 2)
            self._center_cache[key] = centers
        return self._center_cache[key]

    @staticmethod
    def _distance2bbox(points, distance):
        return np.stack([
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ], axis=-1)

    @staticmethod
    def _distance2kps(points, distance):
        coords = []
        for index in range(0, distance.shape[1], 2):
            coords.append(points[:, index % 2] + distance[:, index])
            coords.append(points[:, (index % 2) + 1] + distance[:, index + 1])
        return np.stack(coords, axis=-1)

    @staticmethod
    def _landmarks_plausible(box, kps):
        """用 SCRFD 五点拓扑过滤衣物褶皱等类人脸纹理。"""
        x1, y1, x2, y2 = [float(v) for v in box]
        width, height = x2 - x1, y2 - y1
        points = np.asarray(kps, dtype=np.float32)
        if width <= 0 or height <= 0 or points.shape != (5, 2):
            return False

        left_eye, right_eye, nose, left_mouth, right_mouth = points
        if left_eye[0] >= right_eye[0] or left_mouth[0] >= right_mouth[0]:
            return False

        # 强侧脸时双眼的x投影可能几乎重合；这里只排除完全坍缩的回归，
        # 可疑但仍有顺序的侧脸交给下面的软风险和10g复核。
        eye_span = float(right_eye[0] - left_eye[0]) / width
        mouth_span = float(right_mouth[0] - left_mouth[0]) / width
        if not (0.01 <= eye_span <= 0.75 and 0.03 <= mouth_span <= 0.75):
            return False

        eye_y = float(left_eye[1] + right_eye[1]) / 2.0
        mouth_y = float(left_mouth[1] + right_mouth[1]) / 2.0
        eye_to_mouth = mouth_y - eye_y
        if not (0.25 * height <= eye_to_mouth <= 0.70 * height):
            return False
        if (abs(float(right_eye[1] - left_eye[1])) > 0.35 * height
                or abs(float(right_mouth[1] - left_mouth[1])) > 0.35 * height):
            return False

        # 鼻点应处于眼线与嘴线之间。衣服褶皱误检常在这里发生点序错乱。
        nose_ratio = (float(nose[1]) - eye_y) / eye_to_mouth
        if not (0.02 <= nose_ratio <= 1.05):
            return False

        # 鼻子可越过双眼，但不能远离眼睛和嘴角共同覆盖的横向范围。
        feature_x = np.array([
            left_eye[0], right_eye[0], left_mouth[0], right_mouth[0]],
            dtype=np.float32)
        nose_margin = 0.08 * width
        if not (float(feature_x.min()) - nose_margin
                <= float(nose[0])
                <= float(feature_x.max()) + nose_margin):
            return False

        # 所有点都应落在框附近，保留 15% 余量兼容侧脸和回归误差。
        margin_x, margin_y = 0.15 * width, 0.15 * height
        if np.any(points[:, 0] < x1 - margin_x) or np.any(points[:, 0] > x2 + margin_x):
            return False
        if np.any(points[:, 1] < y1 - margin_y) or np.any(points[:, 1] > y2 + margin_y):
            return False
        return True

    @staticmethod
    def _landmark_soft_risk(box, kps):
        """返回0~1的五点软风险；只触发复核，不直接删除候选。

        基础拓扑过滤负责排除明显不可能的人脸。这里识别的是仍能通过
        基础过滤、但鼻点/五官比例比较可疑的候选。真实强侧脸也可能得到
        较高风险，因此风险只能交给10g复核，不能作为硬拒绝条件。
        """
        x1, y1, x2, y2 = [float(v) for v in box]
        width, height = x2 - x1, y2 - y1
        points = np.asarray(kps, dtype=np.float32)
        if width <= 0 or height <= 0 or points.shape != (5, 2):
            return 1.0

        left_eye, right_eye, nose, left_mouth, right_mouth = points
        eye_center_x = float(left_eye[0] + right_eye[0]) / 2.0
        mouth_center_x = float(left_mouth[0] + right_mouth[0]) / 2.0
        corridor_margin = 0.08 * width
        corridor_left = min(eye_center_x, mouth_center_x) - corridor_margin
        corridor_right = max(eye_center_x, mouth_center_x) + corridor_margin
        nose_x = float(nose[0])
        outside = max(corridor_left - nose_x, nose_x - corridor_right, 0.0) / width
        # 当前手部样例约偏出0.09个框宽；0.12以上视为满风险。
        nose_risk = min(1.0, outside / 0.12)

        eye_span = abs(float(right_eye[0] - left_eye[0])) / width
        mouth_span = abs(float(right_mouth[0] - left_mouth[0])) / width
        span_max = max(eye_span, mouth_span, 1e-6)
        span_balance = min(eye_span, mouth_span) / span_max
        span_risk = max(0.0, min(1.0, (0.35 - span_balance) / 0.35))

        eye_y = float(left_eye[1] + right_eye[1]) / 2.0
        mouth_y = float(left_mouth[1] + right_mouth[1]) / 2.0
        vertical_span = max(mouth_y - eye_y, 1e-6)
        nose_ratio = (float(nose[1]) - eye_y) / vertical_span
        vertical_risk = min(1.0, abs(nose_ratio - 0.5) / 0.55)

        return float(min(1.0, 0.70 * nose_risk
                         + 0.20 * span_risk
                         + 0.10 * vertical_risk))

    @staticmethod
    def _nms(dets, threshold):
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = dets[:, 4].argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            iw = np.maximum(0.0, xx2 - xx1 + 1)
            ih = np.maximum(0.0, yy2 - yy1 + 1)
            inter = iw * ih
            overlap = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(overlap <= threshold)[0] + 1]
        return keep

    def detect(self, img, conf=None):
        return [box for box, _ in self.detect_with_conf(img, conf)]

    def detect_with_conf(self, img, conf=None):
        """返回 [((x1,y1,x2,y2), score), ...]。"""
        self.last_detection_details = []
        if img is None or img.size == 0:
            return []
        preprocess_started = time.perf_counter()
        threshold = self.conf if conf is None else float(conf)
        h, w = img.shape[:2]
        blob, scale = self._prepare_input(img)
        preprocess_finished = time.perf_counter()
        outputs = self._sess.run(self._output_names, {self._input_name: blob})
        inference_finished = time.perf_counter()

        scores_list, boxes_list, kps_list = [], [], []
        for idx, stride in enumerate(self.STRIDES):
            scores = outputs[idx].reshape(-1)
            pos = np.where(scores >= threshold)[0]
            if pos.size == 0:
                continue
            anchors = self._anchors(stride)
            # 先筛高分 anchor，再解码框和关键点。高阈值场景下可避免为绝大多数
            # 背景 anchor 创建 boxes/kps 临时数组。
            positive_anchors = anchors[pos]
            box_distances = outputs[idx + 3].reshape(-1, 4)[pos] * stride
            boxes = self._distance2bbox(positive_anchors, box_distances)
            scores_list.append(scores[pos])
            boxes_list.append(boxes)
            kps_distances = outputs[idx + 6].reshape(-1, 10)[pos] * stride
            kps = self._distance2kps(positive_anchors, kps_distances)
            kps_list.append(kps)
        if not scores_list:
            postprocess_finished = time.perf_counter()
            self._record_timing(
                preprocess_finished - preprocess_started,
                inference_finished - preprocess_finished,
                postprocess_finished - inference_finished)
            return []

        scores = np.concatenate(scores_list)
        boxes = np.concatenate(boxes_list) / scale
        kpss = np.concatenate(kps_list).reshape(-1, 5, 2) / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep]
        kpss = kpss[keep]
        dets[:, 0::2] = np.clip(dets[:, 0::2], 0, w)
        dets[:, 1::2] = np.clip(dets[:, 1::2], 0, h)
        # SCRFD 本身没有 YOLO 那层几何过滤；补上相同约束，避免大块家具、
        # 衣物褶皱等被关键点拓扑“勉强放行”。
        widths = dets[:, 2] - dets[:, 0]
        heights = dets[:, 3] - dets[:, 1]
        areas = widths * heights
        geometric = ((widths >= FACE_MIN_SIZE) & (heights >= FACE_MIN_SIZE)
                     & (areas <= (w * h) * FACE_MAX_AREA_RATIO)
                     & ((widths / np.maximum(heights, 1e-6)) >= FACE_ASPECT_MIN)
                     & ((widths / np.maximum(heights, 1e-6)) <= FACE_ASPECT_MAX))
        dets = dets[geometric]
        kpss = kpss[geometric]
        if self.landmark_filter:
            valid = [self._landmarks_plausible(row[:4], kps)
                     for row, kps in zip(dets, kpss)]
            valid = np.asarray(valid, dtype=bool)
            dets = dets[valid]
            kpss = kpss[valid]
        self.last_detection_details = [
            {"box": tuple(row[:4]), "score": float(row[4]),
             "landmark_risk": self._landmark_soft_risk(row[:4], kps)}
            for row, kps in zip(dets, kpss)
        ]
        result = [(item["box"], item["score"])
                  for item in self.last_detection_details]
        postprocess_finished = time.perf_counter()
        self._record_timing(
            preprocess_finished - preprocess_started,
            inference_finished - preprocess_finished,
            postprocess_finished - inference_finished)
        return result


class SCRFDVerifier(SCRFDFaceDetector):
    """用 face-detect 同款 SCRFD 对 YOLO 的低置信度候选做二次验证。"""

    def __init__(self, model_dir=None, model="10g", input_size=640, conf=0.3,
                 use_gpu=False, iou_thresh=0.3, keep_conf=0.35,
                 nms_thresh=0.4, device="auto", gpu_id=0,
                 landmark_filter=True):
        super().__init__(model_dir=model_dir, model=model, input_size=input_size,
                         conf=conf, nms_thresh=nms_thresh, device=device,
                         gpu_id=gpu_id, use_gpu=use_gpu,
                         landmark_filter=landmark_filter)
        self.timing_label = f"SCRFD-{self.model}(复核)"
        self._iou_thresh = iou_thresh
        self._keep_conf = keep_conf

    def verify(self, yolo_dets, img, iou_thresh=None, keep_conf=None):
        """验证 YOLO 检测结果, 过滤误检(手/玩具)。

        yolo_dets: [(bbox, conf), ...] YOLO 带置信度的检测结果
        keep_conf: YOLO 置信度 ≥ 此值的直接保留(高置信度真脸, 不验证,
                   避免 SCRFD 漏检误伤); < 此值的用 SCRFD 验证。

        策略: 高 conf 真脸不验证(SCRFD 可能漏检), 低 conf 框才用 SCRFD 二次确认。
        手/玩具误检通常 conf 低且 SCRFD 不会检出 → 被过滤。
        """
        if iou_thresh is None:
            iou_thresh = self._iou_thresh
        if keep_conf is None:
            keep_conf = self._keep_conf
        if not yolo_dets:
            return []
        keep = [b for b, c in yolo_dets if c >= keep_conf]
        to_verify = [(b, c) for b, c in yolo_dets if c < keep_conf]
        if not to_verify:
            return keep
        scrfd_boxes = self.detect(img)
        for bbox, _ in to_verify:
            for sbox in scrfd_boxes:
                if self._iou(bbox, sbox) > iou_thresh:
                    keep.append(bbox)
                    break
        return keep

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0


def _model_timing_snapshot(detectors):
    """递归收集双模型中的各个实际模型计时快照。"""
    pending = list(detectors)
    seen = set()
    snapshot = {}
    while pending:
        detector = pending.pop(0)
        if detector is None or id(detector) in seen:
            continue
        seen.add(id(detector))
        if isinstance(detector, DualFaceDetector):
            pending.extend([detector._det_a, detector._det_b,
                            detector._verifier])
            continue
        timing = getattr(detector, "timing_stats", None)
        if timing is None:
            continue
        stats = timing()
        label = getattr(detector, "timing_label", detector.__class__.__name__)
        # 同一标签理论上不应出现两次；若出现(例如两个同规格模型)，
        # 用序号区分，避免日志覆盖统计。
        key = label
        suffix = 2
        while key in snapshot:
            key = f"{label}#{suffix}"
            suffix += 1
        snapshot[key] = {
            "calls": int(stats.get("calls", 0)),
            "total_seconds": float(stats.get("total_seconds", 0.0)),
            "average_ms": float(stats.get("average_ms", 0.0)),
        }
    return snapshot


def _format_model_timing(detectors, previous=None):
    """格式化本次日志周期和累计的模型调用次数/耗时。"""
    current = _model_timing_snapshot(detectors)
    previous = previous or {}
    parts = []
    for label, stats in current.items():
        old = previous.get(label, {})
        delta_calls = stats["calls"] - int(old.get("calls", 0))
        delta_seconds = stats["total_seconds"] - float(old.get("total_seconds", 0.0))
        parts.append(
            f"{label}: 本段{delta_calls}次/{delta_seconds * 1000.0:.1f}ms, "
            f"累计{stats['calls']}次/{stats['total_seconds']:.3f}s, "
            f"均值{stats['average_ms']:.1f}ms"
        )
    return " | ".join(parts) if parts else "模型: 无调用", current


def _log_scrfd_timing(detectors, log):
    """汇总单模型、双模型及验证器中的 SCRFD 分阶段性能。"""
    pending = list(detectors)
    seen = set()
    while pending:
        detector = pending.pop(0)
        if detector is None or id(detector) in seen:
            continue
        seen.add(id(detector))
        if isinstance(detector, DualFaceDetector):
            pending.extend([detector._det_a, detector._det_b, detector._verifier])
            continue
        if not isinstance(detector, SCRFDFaceDetector):
            continue
        stats = detector.timing_stats()
        calls = stats["calls"]
        if not calls:
            continue
        log(
            f"  [SCRFD性能] model={detector.model} backend={detector.backend} "
            f"调用={calls} 平均={stats['average_ms']:.2f}ms | "
            f"预处理={stats['preprocess_average_ms']:.2f}ms"
            f"({stats['preprocess_ratio']:.1%}) | "
            f"ONNX推理={stats['inference_average_ms']:.2f}ms"
            f"({stats['inference_ratio']:.1%}) | "
            f"后处理={stats['postprocess_average_ms']:.2f}ms"
            f"({stats['postprocess_ratio']:.1%}) | "
            f"累计={stats['total_seconds']:.2f}s"
        )


class HardFaceRecall:
    def __init__(self, conf=HARD_FACE_CONF, min_size=HARD_FACE_MIN_SIZE,
                 edge_ratio=HARD_FACE_EDGE_RATIO, roi_scale=HARD_FACE_ROI_SCALE,
                 max_rois=HARD_FACE_MAX_ROIS, roi_size=HARD_FACE_ROI_SIZE):
        self.conf = max(0.0, float(conf))
        self.min_size = max(1, int(min_size))
        self.edge_ratio = min(0.49, max(0.0, float(edge_ratio)))
        self.roi_scale = max(1.0, float(roi_scale))
        self.max_rois = max(1, int(max_rois))
        self.roi_size = max(160, int(roi_size))

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _is_difficult(self, box, width, height):
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw, bh) < self.min_size:
            return True
        margin_x, margin_y = width * self.edge_ratio, height * self.edge_ratio
        return (box[0] < margin_x or box[1] < margin_y
                or box[2] > width - margin_x or box[3] > height - margin_y)

    def _difficulty_score(self, box, width, height):
        """困难程度评分: 越小越困难, 优先处理最困难的候选。"""
        bw, bh = box[2] - box[0], box[3] - box[1]
        min_dim = min(bw, bh)
        # 小脸优先: 归一化到 [0, 2], 越小分数越低
        size_score = min_dim / max(self.min_size, 1)
        # 边缘优先: 归一化到 [0, 1], 越近边缘分数越低
        margin_x, margin_y = width * self.edge_ratio, height * self.edge_ratio
        dx = min(box[0], width - box[2]) / max(margin_x, 1)
        dy = min(box[1], height - box[3]) / max(margin_y, 1)
        edge_score = min(dx, dy)
        return size_score + edge_score

    def _roi(self, box, width, height):
        cx, cy = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
        half_w = (box[2] - box[0]) * self.roi_scale * 0.5
        half_h = (box[3] - box[1]) * self.roi_scale * 0.5
        x1, y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x2, y2 = min(width, int(cx + half_w)), min(height, int(cy + half_h))
        return x1, y1, x2, y2

    def recall(self, detector, img, raw_candidates, existing_dets,
               dual_iou, roi_detector=None):
        """低分原始候选困难脸二检。

        raw_candidates: 主检测器产出的原始候选列表, 仅使用 formal=False
                        (低置信度, 未进入共识) 的候选作为种子。
        existing_dets:  已有的共识检测结果, 用于去重。
        roi_detector:   专用轻量检测器(如小尺寸 YOLOv8)。
                        为 None 时回退到主检测器。
        """
        height, width = img.shape[:2]
        recalled = []
        # 收集低分困难候选并按困难程度排序(最困难优先)
        difficult = []
        for item in raw_candidates:
            if item.get("formal", True):      # 只处理低分候选(未进入共识的)
                continue
            box = item["box"]
            # 跳过与已有共识框重叠的(已处理过)
            if any(self._iou(box, ed) >= 0.35 for ed in existing_dets):
                continue
            if not self._is_difficult(box, width, height):
                continue
            score = self._difficulty_score(box, width, height)
            difficult.append((score, box))
        difficult.sort(key=lambda x: x[0])
        # 只处理最困难的前 max_rois 个, 防止推理爆炸
        for _, box in difficult[:self.max_rois]:
            x1, y1, x2, y2 = self._roi(box, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = img[y1:y2, x1:x2]
            if roi_detector is not None:
                boxes = roi_detector.detect(roi, conf=self.conf)
            else:
                boxes = detector.detect(roi, conf=self.conf)
            for roi_box in boxes:
                mapped = (roi_box[0] + x1, roi_box[1] + y1,
                          roi_box[2] + x1, roi_box[3] + y1)
                if any(self._iou(mapped, ed) >= 0.35 for ed in existing_dets):
                    continue
                if all(self._iou(mapped, existing) < 0.35
                       for existing in recalled):
                    recalled.append(mapped)
        return recalled


class FaceProcessor:
    """人脸检测+逐脸轨迹稳定：关键帧检测，中间帧 LK/模板跟踪。

    跟踪质量保障(防"漏一帧"):
    - 每张脸独立建轨迹；检测帧只漏掉其中一张时，该轨迹单独进入宽容期，
      不会因为同帧还有其他检测结果就立刻消失
    - F1 forward-backward LK 验证: 正向跟踪后再反向, FB 误差大的点剔除,
      median 估计更稳健(避免跟踪点跑到背景上导致框瞬移到错误位置)
    - F2 位移阈值兜底: dx/dy > 0.3×box_size 判定跟踪跑飞, 回退到旧框
      (避免框瞬移导致原人脸位置那 1 帧没被打码)
    - LK 失败时使用 face-detect 同款 NCC+PSR 模板匹配桥接快速运动
    - 已确认轨迹短时断点在检测恢复后利用帧缓存插值补码，不盲目复用旧框
    - F5 首帧强制双检: frame_idx<=2 时强制检测, 避免模型冷启动漏检
      导致视频开头几帧没打码(YuNet DNN 后端初始化可能首帧漏检)
    """
    def __init__(self, detector, detect_int=FACE_DETECT_INT,
                 empty_detect_int=FACE_EMPTY_DETECT_INT, grace=FACE_GRACE,
                 conf=FACE_CONF, scrfd_verifier=None, dual_iou=0.2,
                 box_smooth=FACE_BOX_SMOOTH, track_iou=0.2, track_dist=1.5,
                 backfill_frames=FACE_BACKFILL, burst_frames=FACE_BURST,
                 active_hold=FACE_ACTIVE_HOLD,
                 visible_hold=FACE_VISIBLE_HOLD, hard_face_recall=None,
                 roi_detector=None):
        self.detector = detector
        self.detect_int = max(1, detect_int)
        self.empty_detect_int = max(1, empty_detect_int)
        self.backfill_frames = (max(0, int(backfill_frames))
                                if isinstance(detector, DualFaceDetector) else 0)
        self.burst_frames = max(0, int(burst_frames))
        self.active_hold = max(0, int(active_hold))
        self._active_hold_remaining = 0
        self.visible_hold = max(0, int(visible_hold))
        self.conf = conf
        self.grace = max(0, int(grace))  # 单条轨迹允许漏掉的检测周期数
        self.box_smooth = min(1.0, max(0.05, float(box_smooth)))
        self.track_iou = min(1.0, max(0.0, float(track_iou)))
        self.track_dist = max(0.1, float(track_dist))
        self.tracks = []
        self._next_track_id = 1
        self.last_faces = []
        self.last_debug_faces = []
        self.last_raw_debug_faces = []
        self.last_backfill_events = []
        self._pending_single = []
        self._last_detection_count = None
        self._burst_remaining = 0
        self.prev_gray = None
        self._force_detect = False
        self.scrfd_verifier = scrfd_verifier  # SCRFD 二次验证器(可选, --scrfd-verify 开启)
        self.dual_iou = dual_iou    # 双模型共识 IoU 阈值(仅 DualFaceDetector 使用)
        self.hard_face_recall = hard_face_recall
        self.roi_detector = roi_detector  # 困难脸二检专用轻量检测器(可选)
        # 优化 LK 参数: 更小搜索窗 + 更少金字塔层数 → 每帧 tracking 提速约 30-40%
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
        # 网格采样点数: 3×3=9 点, 替代 goodFeaturesToTrack 的 corner detection 开销
        # 9 点 median 估计位移仍稳健, 比 5×5=25 点 LK 提速约 2x
        self._grid_rows, self._grid_cols = 3, 3
        # F1 forward-backward 验证: FB 误差阈值(像素), 超过此值的跟踪点被剔除
        # 2.0 像素: 对鱼眼去畸变后的非线性局部变形更宽容(鱼眼场景正向→反向 LK
        # 天然有 1-2px 误差), 普通视频也兼容(正常跟踪误差 <1px)
        self._fb_thresh = 2.0
        # F2 跟踪跑飞阈值: 位移 > 0.3×box_size 判定跟踪失败, 回退到旧框
        # 0.3 = 一帧内人脸移动不超过 30% 框宽, 超过即视为跟踪点跑到背景
        self._max_disp_ratio = 0.3
        # face-detect 同款模板桥接门槛；仅 LK 失败时运行，不增加正常帧开销。
        self._tmpl_thresh = 0.92
        self._psr_thresh = 2.0

    def _init_pts(self, gray, box):
        """用均匀网格采样替代 goodFeaturesToTrack 的角点检测。
        grid 采样 O(1) vs corner detection O(N), 对 1080P 帧约节省 2-5ms。"""
        x1, y1, x2, y2 = box
        h_img, w_img = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 12:
            return None
        # 在 bbox 内部均匀采样 3x3 网格点
        margin = 0.12
        xs = np.linspace(x1 + bw * margin, x2 - bw * margin, self._grid_cols, dtype=np.float32)
        ys = np.linspace(y1 + bh * margin, y2 - bh * margin, self._grid_rows, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        pts = np.column_stack((xv.ravel(), yv.ravel())).reshape(-1, 1, 2)
        return pts.astype(np.float32)

    def _track_box_from_flow(self, box, pts, npts, status, back_pts, back_status):
        """F1+F2+F3: 用给定特征点对 box 做 LK 跟踪。

        F1 forward-backward 验证: 正向 LK 后再反向 LK, FB 误差大的点剔除,
        median 位移估计更稳健, 避免跟踪点跑到背景导致框瞬移漏帧。
        F2 位移阈值: dx/dy > 0.3×box_size 判定跟踪跑飞, 返回 None 回退到旧框。
        F3 仿射变换: 用 estimateAffinePartial2D 拟合 4-DOF 变换(平移+旋转+均匀缩放),
           替代纯平移, 解决人脸转头(旋转)和靠近/远离(缩放)时框偏移的问题。
        """
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        box_size = max(bw, bh)
        if pts is None or len(pts) < 4:
            return None, None
        if npts is None or status is None or back_pts is None or back_status is None:
            return None, None
        fwd_mask = status.flatten() == 1
        if fwd_mask.sum() < 4:
            return None, None
        back_mask = back_status.flatten() == 1
        # FB 误差: |back_pts - 原始 pts|, 大于阈值的点视为跑飞剔除
        fb_err = np.linalg.norm(
            back_pts.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
        good_mask = fwd_mask & back_mask & (fb_err <= self._fb_thresh)
        if good_mask.sum() < 4:
            return None, None
        good = npts[good_mask].reshape(-1, 2)
        old_flat = pts[good_mask].reshape(-1, 2)
        # F2 位移阈值: 一帧内位移超过 30% 框宽判定跟踪跑飞, 回退旧框
        # (正常人脸一帧位移 < 10% 框宽; 30% 已是极端, 超过必为跟踪点跑到背景)
        dx = float(np.median(good[:, 0] - old_flat[:, 0]))
        dy = float(np.median(good[:, 1] - old_flat[:, 1]))
        if max(abs(dx), abs(dy)) > self._max_disp_ratio * box_size:
            return None, None
        # F3 仿射变换: 4-DOF (平移+旋转+均匀缩放)
        # estimateAffinePartial2D 需要 ≥2 点对, 用 RANSAC 剔除异常值
        M, inliers = cv2.estimateAffinePartial2D(
            old_flat, good, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=5000)
        if M is not None and inliers is not None and inliers.sum() >= 3:
            # 将框四角做仿射变换, 得到新框
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                               dtype=np.float64).reshape(-1, 1, 2)
            tc = cv2.transform(corners, M).reshape(-1, 2)
            nx1, ny1 = tc.min(axis=0)
            nx2, ny2 = tc.max(axis=0)
            new_box = (int(nx1), int(ny1), int(nx2), int(ny2))
        else:
            # 仿射拟合失败(点太少/退化构型), 回退纯平移
            new_box = (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))
        return new_box, good.reshape(-1, 1, 2)

    def _track_box(self, prev_gray, gray, box):
        """使用低成本 3x3 网格光流，不在失败帧追加昂贵角点检测。"""
        pts = self._init_pts(prev_gray, box)
        if pts is None:
            return None, None
        npts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None, **self.lk)
        if npts is None:
            return None, None
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, npts, None, **self.lk)
        return self._track_box_from_flow(
            box, pts, npts, status, back_pts, back_status)

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _grab_template(gray, box):
        """保存框内部灰度模板，内缩 10% 减少背景边缘干扰。"""
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
        x1, y1 = max(0, x1 + mx), max(0, y1 + my)
        x2, y2 = min(w, x2 - mx), min(h, y2 - my)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return gray[y1:y2, x1:x2].copy()

    def _appearance_score(self, gray, box, template):
        """低成本比较当前跟踪区域与上次确认的人脸外观，仅用于大位移复核。"""
        current = self._grab_template(gray, box)
        if template is None or current is None:
            return None
        side = 24
        a = cv2.resize(template, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(current, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
        a -= float(a.mean())
        b -= float(b.mean())
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.sum(a * b) / denom) if denom > 1e-6 else 0.0

    def _match_template(self, gray, track):
        """LK 失败时在上一位置附近做 NCC+PSR 模板桥接。"""
        tmpl = track.get("template")
        if tmpl is None:
            return None
        th, tw = tmpl.shape[:2]
        fh, fw = gray.shape[:2]
        box = track["box"]
        cx, cy = self._center(box)
        bw, bh = box[2] - box[0], box[3] - box[1]
        radius = int(min(max(2.5 * max(bw, bh), 32), fw, fh))
        x0 = int(max(0, cx - radius - tw / 2))
        y0 = int(max(0, cy - radius - th / 2))
        x1 = int(min(fw, cx + radius + tw / 2))
        y1 = int(min(fh, cy + radius + th / 2))
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        region = gray[y0:y1, x0:x1]
        down = 2 if tw >= 24 and region.shape[1] >= 480 else 1
        if down > 1:
            region = cv2.resize(region, (region.shape[1] // down,
                                         region.shape[0] // down),
                                interpolation=cv2.INTER_AREA)
            small_tmpl = cv2.resize(tmpl, (max(8, tw // down),
                                           max(8, th // down)),
                                    interpolation=cv2.INTER_AREA)
        else:
            small_tmpl = tmpl
        if (region.shape[0] < small_tmpl.shape[0]
                or region.shape[1] < small_tmpl.shape[1]):
            return None
        response = cv2.matchTemplate(region, small_tmpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(response)
        if peak < self._tmpl_thresh:
            return None
        psr = (peak - float(response.mean())) / (float(response.std()) + 1e-6)
        if psr < self._psr_thresh:
            return None
        return (x0 + (loc[0] + small_tmpl.shape[1] / 2) * down,
                y0 + (loc[1] + small_tmpl.shape[0] / 2) * down)

    def _track_all(self, gray, frame_idx=None):
        """先把所有存量轨迹推进到当前帧，再与本帧检测框关联。"""
        if self.prev_gray is None:
            return
        point_batches = [self._init_pts(self.prev_gray, track["box"])
                         for track in self.tracks]
        valid_batches = [(index, points) for index, points in enumerate(point_batches)
                         if points is not None and len(points) >= 4]
        flow_results = {}
        if valid_batches:
            all_points = np.concatenate([points for _, points in valid_batches], axis=0)
            all_next, all_status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, all_points, None, **self.lk)
            if all_next is not None:
                all_back, all_back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.prev_gray, all_next, None, **self.lk)
                offset = 0
                for index, points in valid_batches:
                    count = len(points)
                    flow_results[index] = (
                        points, all_next[offset:offset + count],
                        all_status[offset:offset + count] if all_status is not None else None,
                        all_back[offset:offset + count] if all_back is not None else None,
                        all_back_status[offset:offset + count]
                        if all_back_status is not None else None,
                    )
                    offset += count
        active_tracks = []
        for index, track in enumerate(self.tracks):
            old_box = track["box"]
            flow = flow_results.get(index)
            new_box, _ = (self._track_box_from_flow(old_box, *flow)
                          if flow is not None else (None, None))
            if new_box is not None:
                # 只有较大位移时才做外观复核，避免为每个普通跟踪帧增加开销。
                old_size = max(old_box[2] - old_box[0], old_box[3] - old_box[1], 1)
                old_cx, old_cy = self._center(old_box)
                new_cx, new_cy = self._center(new_box)
                motion_ratio = (((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5
                                / old_size)
                if motion_ratio >= 0.18:
                    appearance = self._appearance_score(
                        gray, new_box, track.get("template"))
                    # 大幅移动但模板几乎完全不相似，通常是 LK 跟到了背景。
                    # 严格双模型模式宁可暂时漏打，也不保留明显漂移框。
                    if (isinstance(self.detector, DualFaceDetector)
                            and appearance is not None and appearance < 0.10):
                        new_box = None
                if new_box is None:
                    # 镜头快速移动时 LK 可能只失败 1~2 帧。严格双模型
                    # 旧逻辑会在这里立即删轨迹，导致马赛克闪一下。
                    pass
                else:
                    track["box"] = new_box
                    track["visible"] = True
                    track["visibility_misses"] = 0
                    track["debug_source"] = "T"
                    if frame_idx is not None:
                        track["last_visible_frame"] = frame_idx
                        track["last_visible_box"] = track["box"]
                    active_tracks.append(track)
                    continue
            hit = self._match_template(gray, track)
            if hit is not None:
                cx, cy = self._center(old_box)
                dx, dy = hit[0] - cx, hit[1] - cy
                track["box"] = tuple(int(round(v)) for v in (
                    old_box[0] + dx, old_box[1] + dy,
                    old_box[2] + dx, old_box[3] + dy))
                track["visible"] = True
                track["visibility_misses"] = 0
                track["debug_source"] = "T"
                if frame_idx is not None:
                    track["last_visible_frame"] = frame_idx
                    track["last_visible_box"] = track["box"]
                active_tracks.append(track)
                continue
            # 无可靠位移时先不直接复用旧框；交给后面的检测关联逻辑。
            # 如果当帧检测也漏掉，_update_tracks() 会按 visible_hold
            # 给已确认轨迹短暂保活，避免只闪一帧。
            track["visible"] = False
            active_tracks.append(track)
            self._force_detect = True
        self.tracks = active_tracks

    def _update_tracks(self, dets, gray, debug_scores=None,
                       allow_new=None, debug_sources=None, frame_idx=None):
        """按 IoU/中心距离匹配；未匹配轨迹分别保活，不再整帧清空。"""
        if debug_scores is None:
            debug_scores = [None] * len(dets)
        if allow_new is None:
            allow_new = [True] * len(dets)
        if debug_sources is None:
            debug_sources = ["D"] * len(dets)
        pairs = []
        for ti, track in enumerate(self.tracks):
            current = track["box"]
            tcx, tcy = self._center(current)
            gate = self.track_dist * max(current[2] - current[0],
                                         current[3] - current[1], 1)
            for di, det in enumerate(dets):
                overlap = self._iou(current, det)
                if overlap >= self.track_iou:
                    priority = 4.0 if allow_new[di] else 0.0
                    pairs.append((priority + 2.0 + overlap, ti, di))
                    continue
                dcx, dcy = self._center(det)
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist <= gate:
                    priority = 4.0 if allow_new[di] else 0.0
                    pairs.append((priority + 2.0 - dist / gate, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            track = self.tracks[ti]
            was_visible = track.get("visible", True)
            previous_visible_frame = track.get("last_visible_frame")
            previous_visible_box = track.get("last_visible_box", track["box"])
            predicted = track["box"]
            detected = dets[di]
            alpha = self.box_smooth
            track["box"] = tuple(int(round(
                alpha * detected[j] + (1.0 - alpha) * predicted[j]))
                for j in range(4))
            track["misses"] = 0
            track["hits"] += 1
            track["visible"] = True
            track["visibility_misses"] = 0
            track["template"] = self._grab_template(gray, track["box"])
            if (frame_idx is not None and not was_visible
                    and previous_visible_frame is not None
                    and frame_idx > previous_visible_frame + 1
                    and self.backfill_frames > 0
                    and frame_idx - previous_visible_frame - 1 <= self.backfill_frames):
                # 只补“已确认轨迹”的短断点。起点从第一帧缺失帧开始，
                # 使用断点前最后一个可靠框；终点使用恢复后的平滑框。
                self.last_backfill_events.append({
                    "start_frame": previous_visible_frame + 1,
                    "start_box": previous_visible_box,
                    "end_frame": frame_idx,
                    "end_box": track["box"],
                    "kind": "track_gap",
                })
            if frame_idx is not None:
                track["last_visible_frame"] = frame_idx
                track["last_visible_box"] = track["box"]
            if debug_scores[di] is not None:
                track["debug_scores"] = debug_scores[di]
                track["debug_source"] = debug_sources[di]

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track["misses"] += 1
                if isinstance(self.detector, DualFaceDetector):
                    track["visibility_misses"] = (
                        track.get("visibility_misses", 0) + 1)
                    # 只给已确认轨迹一帧保活，降低偶发检测抖动造成的闪屏；
                    # 连续丢失超过门限后仍隐藏旧框并等待重新确认。
                    track["visible"] = (
                        track["visibility_misses"] <= self.visible_hold)
                    if track["visible"]:
                        track["debug_source"] = "H"
        # 光流已失败且当帧两个模型都无法续上的轨迹直接删除；
        # 不把没有位移证据的旧框输出为马赛克。
        self.tracks = [t for t in self.tracks if t["misses"] <= self.grace]

        for di, det in enumerate(dets):
            if di in matched_dets or not allow_new[di]:
                continue
            box = tuple(int(round(v)) for v in det)
            self.tracks.append({
                "id": self._next_track_id,
                "box": box,
                "confirmed": True,
                "misses": 0,
                "hits": 1,
                "visible": True,
                "visibility_misses": 0,
                "last_visible_frame": frame_idx,
                "last_visible_box": box,
                "template": self._grab_template(gray, box),
                "debug_scores": debug_scores[di],
                "debug_source": debug_sources[di],
            })
            self._next_track_id += 1

    def _candidate_matches_existing_track(self, box):
        """单模型框只允许续接附近的已确认轨迹，禁止创建新轨迹。"""
        dcx, dcy = self._center(box)
        for track in self.tracks:
            current = track["box"]
            if self._iou(current, box) >= self.track_iou:
                return True
            tcx, tcy = self._center(current)
            gate = self.track_dist * max(
                current[2] - current[0], current[3] - current[1], 1)
            if ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5 <= gate:
                return True
        return False

    @staticmethod
    def _backfill_match_score(old_box, new_box):
        """返回候选框中心距离/框尺寸；值越小越可能是同一张脸。"""
        ocx, ocy = FaceProcessor._center(old_box)
        ncx, ncy = FaceProcessor._center(new_box)
        size = max(old_box[2] - old_box[0], old_box[3] - old_box[1],
                   new_box[2] - new_box[0], new_box[3] - new_box[1], 1)
        return ((ocx - ncx) ** 2 + (ocy - ncy) ** 2) ** 0.5 / size

    def _update_backfill_candidates(self, frame_idx, confirmed, candidates):
        """单模型候选先暂存；后续双模型确认后产生向前补码事件。"""
        self.last_backfill_events = []
        if self.backfill_frames <= 0:
            self._pending_single = []
            return
        first_frame = frame_idx - self.backfill_frames
        self._pending_single = [
            item for item in self._pending_single
            if item["frame_idx"] >= first_frame]

        consumed = set()
        for confirmed_item in confirmed:
            end_box = confirmed_item["box"]
            matches = []
            for index, pending in enumerate(self._pending_single):
                score = self._backfill_match_score(pending["box"], end_box)
                # 最多允许约2.5个脸框的位移，覆盖短时镜头平移；仍限制空间邻近，
                # 避免把画面另一侧的单模型误检回补成马赛克。
                if score <= 2.5:
                    matches.append((pending["frame_idx"], score, index, pending))
            if not matches:
                continue
            # 优先从最早的相容候选开始回补；同帧则选择距离最近者。
            _, _, index, start = min(matches, key=lambda item: (item[0], item[1]))
            if start["frame_idx"] < frame_idx:
                self.last_backfill_events.append({
                    "start_frame": start["frame_idx"],
                    "start_box": start["box"],
                    "end_frame": frame_idx,
                    "end_box": end_box,
                })
            for other_index, pending in enumerate(self._pending_single):
                if (pending["frame_idx"] >= start["frame_idx"]
                        and self._backfill_match_score(pending["box"], end_box) <= 2.5):
                    consumed.add(other_index)
        if consumed:
            self._pending_single = [
                item for index, item in enumerate(self._pending_single)
                if index not in consumed]

        added = False
        for item in candidates:
            box = item["box"]
            if any(self._iou(box, confirmed_item["box"]) >= self.dual_iou
                   for confirmed_item in confirmed):
                continue
            # 已有轨迹可直接用 S 续轨的候选不需要回补，避免重复事件。
            if self._candidate_matches_existing_track(box):
                continue
            self._pending_single.append({
                "frame_idx": frame_idx,
                "box": tuple(box),
                "scores": item.get("scores"),
            })
            added = True
        # 单模型已经看到疑似人脸时，下一帧立即复检，而不是继续等待空场景间隔。
        if added:
            self._force_detect = True

    def process(self, img, frame_idx, raw_debug=False):
        gray = None
        had_confirmed_tracks = any(
            track.get("confirmed", True) for track in self.tracks)
        if had_confirmed_tracks:
            # 轨迹在本帧检测前即使暂时不可见，也说明场景不是空场景；
            # 保持主动扫描，避免丢轨后退回每5帧一次的空场景节奏。
            self._active_hold_remaining = self.active_hold
        if self.tracks:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self._track_all(gray, frame_idx=frame_idx)
        self.last_raw_debug_faces = []
        self.last_backfill_events = []
        # F5 首帧强制双检: frame_idx<=2 时强制检测, 避免模型冷启动漏检
        # (YuNet 首帧可能因 DNN 后端初始化开销漏检, 普通帧间隔兜底无效)
        cold_start = frame_idx <= 2
        # 有轨迹时按 --face-int 周期检测；中间帧由光流/模板跟踪。
        # 快速运动、跟踪失败或人数变化会设置 _force_detect/_burst_remaining，
        # 临时切换为逐帧检测，不会误入无人场景等待周期。
        active_scene = bool(self.tracks) or self._active_hold_remaining > 0
        interval = self.detect_int if active_scene else self.empty_detect_int
        need_detect = (((frame_idx - 1) % interval == 0)
                       or cold_start or self._force_detect
                       or self._burst_remaining > 0)
        if need_detect:
            if gray is None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self._force_detect = False
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
            # 双模型共识: 两模型同时检测, IoU 匹配一致才保留
            if isinstance(self.detector, DualFaceDetector):
                dual_dets = self.detector.detect_with_scores(
                    img, conf=self.conf, iou_thresh=self.dual_iou,
                    include_single=(bool(self.tracks) or raw_debug
                                    or self.backfill_frames > 0),
                    raw_conf=self.conf / 2.0 if raw_debug else None,
                    existing_boxes=[track["box"] for track in self.tracks
                                    if track.get("confirmed", True)],
                    frame_idx=frame_idx)
                single_candidates = list(self.detector.last_candidates)
                self.last_raw_debug_faces = list(self.detector.last_raw_candidates) if raw_debug else []
                dets = [item["box"] for item in dual_dets]
                debug_scores = [item["scores"] for item in dual_dets]
                allow_new = [True] * len(dets)
                debug_sources = [item.get("source", "D") for item in dual_dets]
                if self.hard_face_recall is not None:
                    raw_cands = list(self.detector.last_raw_candidates)
                    hard_boxes = self.hard_face_recall.recall(
                        self.detector, img, raw_cands, dets, self.dual_iou,
                        roi_detector=self.roi_detector)
                    self.detector.last_candidates = single_candidates
                    for box in hard_boxes:
                        if any(self._iou(box, existing) >= 0.35 for existing in dets):
                            continue
                        dets.append(box)
                        debug_scores.append((("HardROI", self.hard_face_recall.conf),))
                        allow_new.append(True)
                        debug_sources.append("R")
                self._update_backfill_candidates(
                    frame_idx, dual_dets, single_candidates)
                detection_count = len(dets)
                # 新人脸必须双模型共识；已确认人脸遇到运动模糊时，
                # 允许 YOLO/SCRFD 任一模型续轨。这些框 allow_new=False，
                # 因此不会把单模型误检建成新马赛克。
                for item in single_candidates:
                    box = item["box"]
                    if any(self._iou(box, confirmed["box"]) >= self.dual_iou
                           for confirmed in dual_dets):
                        continue
                    if not self._candidate_matches_existing_track(box):
                        continue
                    dets.append(box)
                    debug_scores.append(item["scores"])
                    allow_new.append(False)
                    debug_sources.append("S")
            # SCRFD 二次验证: 高conf框直接保留, 低conf框用SCRFD确认(过滤手/玩具误检)
            elif self.scrfd_verifier is not None and hasattr(self.detector, 'detect_with_conf'):
                dets_with_conf = self.detector.detect_with_conf(img, conf=self.conf)
                dets = self.scrfd_verifier.verify(dets_with_conf, img)
                detection_count = len(dets)
                debug_scores = None
                allow_new = debug_sources = None
                self.last_raw_debug_faces = []
            else:
                dets = self.detector.detect(img, conf=self.conf)
                detection_count = len(dets)
                debug_scores = None
                allow_new = debug_sources = None
                self.last_raw_debug_faces = []
            self._update_tracks(
                dets, gray, debug_scores,
                allow_new=allow_new, debug_sources=debug_sources,
                frame_idx=frame_idx)
            if (self._last_detection_count is not None
                    and detection_count != self._last_detection_count
                    and self._burst_remaining == 0):
                self._burst_remaining = self.burst_frames
            self._last_detection_count = detection_count
            if detection_count or getattr(self.detector, "last_candidates", None):
                self._active_hold_remaining = self.active_hold
            elif not self.tracks and self._active_hold_remaining > 0:
                self._active_hold_remaining -= 1
        elif not self.tracks and self._active_hold_remaining > 0:
            self._active_hold_remaining -= 1
        self.last_faces = [
            track["box"] for track in self.tracks
            if track.get("confirmed", True) and track.get("visible", True)]
        self.last_debug_faces = [
            {"box": track["box"], "scores": track.get("debug_scores"),
             "source": track.get("debug_source", "T")}
            for track in self.tracks
            if (track.get("confirmed", True) and track.get("visible", True)
                and track.get("debug_scores") is not None)
        ]
        self.prev_gray = gray
        return self.last_faces


# ================= 视频处理 =================

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def has_audio(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", path])
    return bool(r.stdout.strip())


def get_fps(path):
    """获取视频真实帧率(优先 avg_frame_rate, 无效时回退 duration/帧数推算)"""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames",
              "-of", "default=noprint_wrappers=1:nokey=1", path])
    lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
    # 优先 avg_frame_rate(最可靠); r_frame_rate 可能是错误元数据(如 600fps)
    for line in lines:
        if "/" in line:
            num, den = line.split("/")
            try:
                fps = float(num) / float(den)
                if 1 <= fps <= 240:  # 合理范围
                    return fps
            except (ValueError, ZeroDivisionError):
                pass
    # 都不合理时用 duration/nb_frames 推算
    dur = get_duration(path)
    for line in lines:
        try:
            nb = int(line)
            if nb > 0 and dur and dur > 0:
                return nb / dur
        except ValueError:
            pass
    return 25.0


def get_duration(path):
    """获取视频时长(秒)"""
    r = _run(["ffprobe", "-v", "error", "-show_entries",
              "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path])
    try:
        return float(r.stdout.strip())
    except (ValueError, TypeError):
        return None


def get_video_info(path):
    """获取视频宽、高、帧率(已考虑旋转)"""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height",
              "-of", "csv=p=0", path])
    parts = r.stdout.strip().split(",")
    if len(parts) < 2:
        return 1920, 1080, 25.0, 0
    w, h = int(parts[0]), int(parts[1])
    fps = get_fps(path)  # 使用修复后的帧率获取
    # 检测旋转(iPhone等设备拍摄的竖屏视频有旋转元数据)
    rot = get_rotation(path)
    if abs(rot) == 90:
        w, h = h, w  # 旋转90°时宽高互换
    return w, h, fps, rot


def get_rotation(path):
    """获取视频旋转角度(0/90/180/270), 只读1帧避免输出过多"""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "side_data_list",
              "-read_intervals", "%+#1",
              "-of", "default=noprint_wrappers=1", path])
    for line in r.stdout.split("\n"):
        line = line.strip()
        if line.startswith("rotation="):
            try:
                return int(float(line.split("=")[1]))
            except (ValueError, IndexError):
                pass
    return 0


def get_transpose_filter(rotation):
    """获取ffmpeg transpose滤镜参数"""
    if rotation == -90 or rotation == 270:
        return "transpose=1"      # 90°顺时针
    elif rotation == 90:
        return "transpose=0"      # 90°逆时针
    elif rotation == 180 or rotation == -180:
        return "transpose=2,transpose=2"  # 180°
    return None


_HW_ENCODER_USABILITY = {}
_CUDA_DECODE_AVAILABLE = None
_NVENC_AVAILABLE = None


def cuda_decode_available():
    """仅在 CUDA 解码器确实可用时启用 ffmpeg 硬件解码。"""
    global _CUDA_DECODE_AVAILABLE
    if _CUDA_DECODE_AVAILABLE is not None:
        return _CUDA_DECODE_AVAILABLE

    if shutil.which("nvidia-smi") is None:
        _CUDA_DECODE_AVAILABLE = False
        return False

    result = _run(["ffmpeg", "-hide_banner", "-hwaccels"])
    _CUDA_DECODE_AVAILABLE = result.returncode == 0 and "cuda" in result.stdout.split()
    return _CUDA_DECODE_AVAILABLE


def nvenc_available():
    """确认 NVIDIA 编码器可用，避免无 CUDA 设备时管道模式报错。"""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    if shutil.which("nvidia-smi") is None:
        _NVENC_AVAILABLE = False
        return False

    encoders = _run(["ffmpeg", "-hide_banner", "-encoders"])
    _NVENC_AVAILABLE = (
        encoders.returncode == 0
        and "h264_nvenc" in encoders.stdout
        and _hw_encoder_is_usable("h264_nvenc")
    )
    return _NVENC_AVAILABLE


def _hw_encoder_is_usable(encoder):
    """确认硬件编码器不只是被 ffmpeg 编译进去了，而且当前机器真的可用。

    有些云镜像会带有 nvenc 编码器，但没有把 NVIDIA 设备/驱动暴露给
    ffmpeg。这种情况下 ``ffmpeg -encoders`` 仍会列出 ``*_nvenc``，而真正
    合成到最后一步才报 CUDA_ERROR_NO_DEVICE。用一个极小的空帧编码探测
    可以在开始处理前安全地回退到 libx264。
    """
    if encoder in _HW_ENCODER_USABILITY:
        return _HW_ENCODER_USABILITY[encoder]

    probe = _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        # NVENC rejects very small frames; 256x256 is safely above its minimum
        # supported dimension while remaining a negligible one-frame probe.
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30",
        "-frames:v", "1", "-an", "-c:v", encoder,
        "-f", "null", "-",
    ])
    usable = probe.returncode == 0
    _HW_ENCODER_USABILITY[encoder] = usable
    return usable


def find_hw_encoder(family="hevc"):
    """检测可用的硬件编码器。

    family="hevc"(默认): 优先 HEVC(与原片同格式, 画质更优)
    family="h264": 强制 H.264(兼容性无敌, 适合微信/Android/旧播放器)
    其次回退到对方族, 都没有返回 None(由调用方回退 libx264)
    """
    r = _run(["ffmpeg", "-hide_banner", "-encoders"])
    if family == "h264":
        order = ["h264_videotoolbox", "h264_nvenc", "h264_qsv",
                 "hevc_videotoolbox", "hevc_nvenc", "hevc_qsv"]
    else:
        order = ["hevc_videotoolbox", "h265_videotoolbox", "hevc_nvenc", "h265_nvenc",
                 "hevc_qsv", "h265_qsv", "h264_videotoolbox", "h264_nvenc", "h264_qsv"]
    for enc in order:
        if enc in r.stdout:
            if _hw_encoder_is_usable(enc):
                return enc
    return None


def get_bitrate(path):
    """获取视频总码率(bps), 用于编码时不超过原片码率"""
    r = _run(["ffprobe", "-v", "error", "-show_entries",
              "format=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", path])
    try:
        return int(r.stdout.strip())
    except (ValueError, TypeError):
        return None


def hw_bitrate(w, h, fps, encoder, src_bitrate=None):
    """根据编码器和分辨率算目标码率(接近原片画质)。

    HEVC 效率高, 0.10 系数即可接近原片; H.264 效率低需更高码率补偿。
    下限 1Mbps(保证极低分辨率视频也有基本画质)。
    如果原片码率更低则不超过原片码率(避免小视频无故膨胀)。
    """
    factor = 0.10 if (encoder or "").startswith(("hevc", "h265")) else 0.15
    target = max(int(w * h * fps * factor), 1000000)
    if src_bitrate and src_bitrate > 0:
        target = min(target, int(src_bitrate))
    return target


def _create_face_detector(face_model, model_dir, face_size, face_conf, use_gpu,
                          scrfd_model, scrfd_device, scrfd_nms, scrfd_gpu_id,
                          scrfd_landmark_filter, log):
    """按 --face-model 创建单模型或双模型共识检测器。"""
    def make_scrfd(model_name=scrfd_model):
        return SCRFDFaceDetector(
            model_dir=model_dir, model=model_name, input_size=face_size,
            conf=face_conf, nms_thresh=scrfd_nms, device=scrfd_device,
            gpu_id=scrfd_gpu_id, use_gpu=use_gpu,
            landmark_filter=scrfd_landmark_filter)

    def make_adaptive_verifier():
        # 2.5g主检时，10g只在中等置信度新人脸上按需调用。
        if scrfd_model != "2.5g":
            return None
        try:
            verifier = make_scrfd("10g")
            log(f"  [人脸] SCRFD-10g 按需复核已启用 "
                f"(分数区间={face_conf:.2f}~{face_conf + SCRFD_ADAPTIVE_VERIFY_MARGIN:.2f}, "
                f"复核conf={min(face_conf, SCRFD_VERIFIER_CONF):.2f}, "
                f"IoU={SCRFD_ADAPTIVE_VERIFY_IOU:.2f}, "
                f"新轨迹要求YOLOv8+2.5g均有效, "
                f"2.5g→10g最大分差={SCRFD_VERIFIER_MAX_SCORE_DROP:.2f})")
            return verifier
        except Exception as exc:
            log(f"  [警告] SCRFD-10g 按需复核初始化失败: {exc}; 继续使用主模型")
            return None

    if face_model == "yolo11+yunet":
        detector = DualFaceDetector(
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu),
            name_a="YOLO11", name_b="YuNet")
        log(f"  [人脸] YOLO11+YuNet 双模型共识 (输入{face_size})")
    elif face_model == "yolov8+yolo11":
        detector = DualFaceDetector(
            YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            name_a="YOLOv8", name_b="YOLO11")
        log(f"  [人脸] YOLOv8+YOLO11 双模型共识 (输入{face_size})")
    elif face_model == "yolov8+scrfd":
        scrfd = make_scrfd()
        verifier = make_adaptive_verifier()
        detector = DualFaceDetector(
            YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            scrfd, name_a="YOLOv8", name_b=f"SCRFD-{scrfd_model}",
            verifier=verifier)
        log(f"  [人脸] YOLOv8+SCRFD-{scrfd_model} 严格双模型共识 "
            f"(输入{face_size}, SCRFD五点拓扑={scrfd_landmark_filter}, "
            f"SCRFD后端={scrfd.backend}"
            + (f", YOLOv8→10g最大分差={SCRFD_VERIFIER_MAX_SCORE_DROP:.2f}"
               if scrfd_model == "10g" else "")
            + ")")
    elif face_model == "yolo11+scrfd":
        scrfd = make_scrfd()
        verifier = make_adaptive_verifier()
        detector = DualFaceDetector(
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            scrfd, name_a="YOLO11", name_b=f"SCRFD-{scrfd_model}",
            verifier=verifier)
        log(f"  [人脸] YOLO11+SCRFD-{scrfd_model} 严格双模型共识 "
            f"(输入{face_size}, SCRFD五点拓扑={scrfd_landmark_filter}, "
            f"SCRFD后端={scrfd.backend})")
    elif face_model == "scrfd":
        detector = make_scrfd()
        log(f"  [人脸] SCRFD-{scrfd_model} (输入{face_size}, "
            f"conf={face_conf}, nms={scrfd_nms}, 后端={detector.backend})")
    elif face_model == "yolov8m":
        detector = YOLOv8MFaceDetector(
            model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-medium (输入{face_size})")
    elif face_model == "yolov8":
        detector = YOLOFaceDetector(
            model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-nano (输入{face_size})")
    elif face_model == "yolo11":
        detector = YOLO11FaceDetector(
            model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv11-nano (输入{face_size})")
    else:
        detector = YuNetFaceDetector(
            model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YuNet (输入{face_size})")
    return detector


def _create_scrfd_verifier(enabled, face_model, model_dir, face_size, use_gpu,
                           scrfd_model, scrfd_device, scrfd_nms, scrfd_gpu_id,
                           scrfd_conf, scrfd_iou, scrfd_keep_conf,
                           scrfd_landmark_filter, log):
    if not enabled:
        return None
    if "scrfd" in face_model:
        log("  [警告] --face-model 已包含 SCRFD，忽略重复的 --scrfd-verify")
        return None
    try:
        verifier = SCRFDVerifier(
            model_dir=model_dir, model=scrfd_model, input_size=face_size,
            conf=scrfd_conf, use_gpu=use_gpu, iou_thresh=scrfd_iou,
            keep_conf=scrfd_keep_conf, nms_thresh=scrfd_nms,
            device=scrfd_device, gpu_id=scrfd_gpu_id,
            landmark_filter=scrfd_landmark_filter)
        log(f"  [人脸] SCRFD-{scrfd_model} 二次验证 "
            f"(conf={scrfd_conf}, iou={scrfd_iou}, keep_conf={scrfd_keep_conf}, "
            f"后端={verifier.backend})")
        return verifier
    except Exception as exc:
        log(f"  [警告] SCRFD 初始化失败: {exc}, 跳过二次验证")
        return None


def _process_pipe(src, dst, face_on, model_dir, face_size,
                  face_int, face_empty_int, face_conf, face_model, keep_tmp, force_h264, use_gpu,
                  frame_skip, fisheye, fisheye_strength, fisheye_device,
                  fisheye_downscale, fisheye_dual, fisheye_crop, log,
                  scrfd_verify=False, scrfd_conf=0.3, scrfd_iou=0.3, scrfd_keep_conf=0.35,
                  dual_iou=0.2, scrfd_model="10g", scrfd_device="auto",
                  scrfd_nms=0.4, scrfd_gpu_id=0, dual_mirror=False,
                  face_grace=FACE_GRACE, face_smooth=FACE_BOX_SMOOTH,
                  scrfd_landmark_filter=True, debug=False, raw_debug=False,
                  face_backfill=FACE_BACKFILL, face_burst=FACE_BURST,
                  hard_face_recall=False, hard_face_conf=HARD_FACE_CONF,
                  hard_face_min_size=HARD_FACE_MIN_SIZE,
                  hard_face_roi_scale=HARD_FACE_ROI_SCALE,
                  hard_face_max_rois=HARD_FACE_MAX_ROIS,
                  hard_face_roi_size=HARD_FACE_ROI_SIZE,
                  hard_face_full_scan=False,
                  hard_face_full_scan_conf=HARD_FACE_FULL_SCAN_CONF):
    """流式管道: ffmpeg解码(rawvideo pipe) → Python处理 → ffmpeg编码, 全程0磁盘IO。

    三级流水线(读帧线程 → 主线程处理 → 写帧线程), 解码/编码与 Python 处理并行,
    相比串行模式提速 20-40%。音频直接从源文件内联输入, 无需临时抽取。
    frame_skip: 跳帧间隔, 管道模式通过跳过读取来实现。
    """
    t0 = time.time()
    w, h, fps, rot = get_video_info(src)
    has_aud = has_audio(src)

    # 鱼眼裁剪: 计算输出尺寸
    if fisheye and 0 < fisheye_crop < 1.0:
        # 用 dummy 帧探测 fisheye_undistort 的实际输出尺寸,
        # 避免双鱼眼 per-eye 裁剪与整图计算不一致导致编码器帧数据错位 → 死锁
        _probe = fisheye_undistort(np.zeros((h, w, 3), dtype=np.uint8),
                                   fisheye_strength, fisheye_device,
                                   downscale=fisheye_downscale,
                                   dual=fisheye_dual,
                                   crop=fisheye_crop)
        out_h, out_w = _probe.shape[:2]
        # 对齐偶数(编码器要求)
        out_w = out_w // 2 * 2
        out_h = out_h // 2 * 2
        log(f"  [鱼眼] 裁剪边缘: crop={fisheye_crop}, 输出 {out_w}x{out_h} (原 {w}x{h})")
    elif fisheye_crop != 1.0 and not fisheye:
        log(f"  [警告] --fisheye-crop 需要 --fisheye 同时启用, 已忽略")
        out_w, out_h = w, h
    else:
        out_w, out_h = w, h

    log(f"  [管道模式] {w}x{h} {fps:.1f}fps 音轨={'有' if has_aud else '无'}"
        + (f" 旋转{rot}°" if rot else ""))

    # 检测硬件编码器
    hw = find_hw_encoder(family="h264" if force_h264 else "hevc")

    # 初始化检测器 — 根据 face_model 参数选择
    fd = _create_face_detector(
        face_model, model_dir, face_size, face_conf, use_gpu,
        scrfd_model, scrfd_device, scrfd_nms, scrfd_gpu_id,
        scrfd_landmark_filter, log)
    stereo_split = (_is_dual_fisheye(w, h, fisheye_dual)
                    and isinstance(fd, DualFaceDetector))
    if stereo_split:
        fd.enable_stereo_split(True)
        log(f"  [双鱼眼] 左右独立检测 + 低分互证 "
            f"(每目输入={face_size}, 低分候选>={fd._stereo_low_conf:.2f}, "
            f"高分增量={STEREO_HIGH_MARGIN:.2f})")
    scrfd_verifier = _create_scrfd_verifier(
        scrfd_verify and face_on, face_model, model_dir, face_size, use_gpu,
        scrfd_model, scrfd_device, scrfd_nms, scrfd_gpu_id,
        scrfd_conf, scrfd_iou, scrfd_keep_conf,
        scrfd_landmark_filter, log)
    # 困难召唤: 创建专用轻量 ROI 检测器(小尺寸 YOLOv8, 跳过双模型共识)
    roi_detector = None
    if hard_face_recall and face_on:
        try:
            roi_detector = YOLOFaceDetector(
                model_dir=model_dir, yolo_size=hard_face_roi_size,
                use_gpu=use_gpu)
            log(f"  [困难召唤] ROI二检: YOLOv8@{roi_detector.yolo_size}px "
                f"(上限{hard_face_max_rois}框/帧, conf={hard_face_conf})")
        except Exception as exc:
            log(f"  [警告] ROI二检检测器初始化失败: {exc}, 回退主检测器")
            roi_detector = None
    face_proc = FaceProcessor(fd, detect_int=face_int,
                              empty_detect_int=face_empty_int, conf=face_conf,
                              scrfd_verifier=scrfd_verifier, dual_iou=dual_iou,
                              grace=face_grace, box_smooth=face_smooth,
                              backfill_frames=face_backfill,
                              burst_frames=face_burst,
                              hard_face_recall=(HardFaceRecall(
                                  conf=hard_face_conf,
                                  min_size=hard_face_min_size,
                                  roi_scale=hard_face_roi_scale,
                                  max_rois=hard_face_max_rois,
                                  roi_size=hard_face_roi_size)
                                  if hard_face_recall else None),
                              roi_detector=roi_detector) if face_on else None
    if face_proc is not None:
        log(f"  [稳定] 逐脸轨迹 + LK/模板桥接 "
            f"(有人检测间隔={face_proc.detect_int}, "
            f"无人扫描间隔={face_proc.empty_detect_int}, "
            f"向前补码={face_proc.backfill_frames}帧, "
            f"变化突检={face_proc.burst_frames}帧, "
            f"丢轨主动扫描保持={face_proc.active_hold}帧, "
            f"单帧保活={face_proc.visible_hold}帧, "
            f"grace={face_proc.grace}检测周期, smooth={face_proc.box_smooth})")
    backfill_buffer_size = face_proc.backfill_frames if face_proc is not None else 0

    # 启动抽帧进程(raw BGR → stdout pipe)
    # NVDEC 硬件解码(可用时): -hwaccel cuda 让 ffmpeg 用 GPU 解码
    # rawvideo 不自动旋转, 需手动加transpose滤镜(iPhone等竖屏视频)
    extract_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if cuda_decode_available():
        extract_cmd += ["-hwaccel", "cuda"]
        log("  [解码] NVDEC GPU 硬件解码")
    extract_cmd += ["-i", src]
    filters = []
    transpose = get_transpose_filter(rot)
    if transpose:
        filters.append(transpose)
    if frame_skip > 1:
        filters.extend([
            f"select=not(mod(n\\,{frame_skip}))",
            "setpts=N/FRAME_RATE/TB",
        ])
    if filters:
        extract_cmd += ["-vf", ",".join(filters)]
    extract_cmd += ["-map", "0:v:0", "-an", "-sn", "-dn",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    extract = subprocess.Popen(
        extract_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # 构建编码命令(stdin ← raw BGR, 第二输入 ← 原视频音频流)
    # 音频内联: 直接 -i src 作为第二输入, ffmpeg 内部并行复用音频流,
    # 省去原先 ffmpeg -vn -acodec copy audio.aac 的临时抽取步骤。
    enc_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s:v", f"{out_w}x{out_h}", "-r", str(fps / frame_skip), "-i", "pipe:0"]
    if has_aud:
        enc_cmd += ["-i", src]  # 第二输入: 内联读取音频, 省去临时抽取
    # 输出选项(必须在所有 -i 之后)
    if nvenc_available():
        # NVENC 调优: 优先 hevc_nvenc(与源 HEVC 同格式, 省 ~40% 码率, 避免输出膨胀);
        # force_h264 或 hevc_nvenc 不可用时回退 h264_nvenc(兼容性无敌).
        # -cq 20 + -maxrate 双重限速: -b:v 0 单独使用无码率上限, 复杂场景码率会失控膨胀;
        #   加 maxrate 按 hw_bitrate 算(不超过源片码率), 防止输出比源片大.
        # preset p6: 比 p4 画质更好, NVENC 吞吐影响小(硬件编码 preset 间差异远小于 CPU).
        use_hevc = (not force_h264) and _hw_encoder_is_usable("hevc_nvenc")
        nvenc_enc = "hevc_nvenc" if use_hevc else "h264_nvenc"
        bitrate = hw_bitrate(out_w, out_h, fps, nvenc_enc, get_bitrate(src))
        enc_cmd += ["-c:v", nvenc_enc, "-preset", "p6", "-tune", "hq",
                    "-rc", "vbr", "-cq", "20", "-b:v", "0",
                    "-maxrate", f"{bitrate // 1000}k",
                    "-bufsize", f"{bitrate // 2000}k"]
        if use_hevc:
            enc_cmd += ["-tag:v", "hvc1"]  # HEVC必须hvc1标签才能被QuickTime/iOS/多数播放器播放
        log(f"  [编码] NVENC GPU 硬件编码 ({nvenc_enc}, preset=p6, maxrate={bitrate // 1000}kbps)")
    elif hw:
        bitrate = hw_bitrate(out_w, out_h, fps, hw, get_bitrate(src))
        enc_cmd += ["-c:v", hw, "-b:v", str(bitrate)]
        if hw.startswith(("hevc", "h265")):
            enc_cmd += ["-tag:v", "hvc1"]  # HEVC必须hvc1标签才能被QuickTime/iOS/多数播放器播放
        log(f"  [编码] 硬件加速: {hw} ({bitrate // 1000}kbps)")
    else:
        enc_cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-threads", "0"]
        log("  [编码] libx264 veryfast (CPU多核)")
    enc_cmd += ["-pix_fmt", "yuv420p"]
    if has_aud:
        enc_cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    enc_cmd += ["-shortest", "-movflags", "+faststart", dst]
    encode = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # 用线程排空 encode stderr(避免管道死锁)
    enc_err_buf = []
    def _drain_stderr():
        while True:
            line = encode.stderr.readline()
            if not line:
                break
            enc_err_buf.append(line)
    _drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
    _drain_thread.start()

    def _read_frame(stream, size):
        """高效帧读取: 一次性 read 大部分 pipe buffer 可以满足, 不满足时再补读。
        对比逐字节/小块循环拼接, 1080P 帧(≈6MB)只需 1~2 次 read 调用。"""
        data = stream.read(size)
        if data is None:
            return b""
        # 偶尔 pipe buffer 不够大, 补齐剩余部分
        remaining = size - len(data)
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        return data

    # 三级流水线: 读帧线程 → 主线程处理 → 写帧线程
    # 解码/编码与 Python 处理并行, 消除串行等待。
    # None 作为哨兵通知下游结束。队列容量小(3)以控制内存(1080P≈18MB/队列)。
    read_q = queue.Queue(maxsize=3)
    write_q = queue.Queue(maxsize=3)
    frame_size = w * h * 3
    pipeline_error = []  # 跨线程传递异常
    writer_done = threading.Event()  # writer 退出信号(检测编码器提前结束)

    def _reader():
        """生产者: 从 ffmpeg stdout 读 raw 帧, 入队。"""
        try:
            while True:
                raw = _read_frame(extract.stdout, frame_size)
                if len(raw) < frame_size:
                    break
                read_q.put(raw)
        except Exception as e:
            pipeline_error.append(("reader", e))
        finally:
            read_q.put(None)  # 哨兵: 通知主线程读取结束

    def _writer():
        """消费者: 从写队列取帧, 写入 ffmpeg stdin。
        用 select 检测可写性, 避免编码器崩溃时 write() 永久阻塞。
        """
        try:
            while True:
                item = write_q.get()
                if item is None:  # 哨兵
                    break
                # select 等待 stdin 可写, 超时检查编码器是否存活
                fd = encode.stdin.fileno()
                try:
                    mv = memoryview(item).cast('B')  # 零拷贝字节视图
                except TypeError:
                    # 非 C-contiguous (不应发生, fisheye crop 已做 ascontiguousarray)
                    mv = memoryview(bytes(item))
                offset = 0
                while offset < len(mv):
                    _, w_ready, _ = select.select([], [fd], [], 3.0)
                    if not w_ready:
                        # 超时: 检查编码器是否已退出
                        if encode.poll() is not None:
                            writer_done.set()
                            return
                        continue
                    n = os.write(fd, mv[offset:])
                    if n == 0:
                        raise BrokenPipeError("编码器 stdin 已关闭")
                    offset += n
        except BrokenPipeError:
            pass
        except Exception as e:
            pipeline_error.append(("writer", e))
        finally:
            writer_done.set()  # 通知主循环: writer 已退出(含编码器提前结束场景)
            try:
                encode.stdin.close()
            except BrokenPipeError:
                pass

    reader_thread = threading.Thread(target=_reader, daemon=True)
    writer_thread = threading.Thread(target=_writer, daemon=True)
    reader_thread.start()
    writer_thread.start()

    # 主循环: 取帧 → 处理 → 入写队列
    # 队列操作带超时, 防止编码器提前退出(-shortest, 音频短于视频)导致死锁:
    # 超时后检查 writer_done, 若 writer 已退出则视为"编码器提前结束",
    # 记警告并正常 break, 走后续完整性检查(输出文件已生成)。
    frame_idx = 0
    total_face = 0
    encoder_finished_early = False
    _PIPE_TIMEOUT = 5  # 秒, 队列超时阈值
    frame_buffer = []
    timing_snapshot = _model_timing_snapshot((fd, scrfd_verifier))

    def _enqueue_image(image):
        nonlocal encoder_finished_early
        try:
            write_q.put(memoryview(image), timeout=_PIPE_TIMEOUT)
            return True
        except queue.Full:
            if writer_done.is_set():
                log("  [警告] 编码器提前结束(音频短于视频, -shortest 触发), "
                    "输出已生成, 走完整性检查")
                encoder_finished_early = True
                return False
            raise

    try:
        while True:
            try:
                raw = read_q.get(timeout=_PIPE_TIMEOUT)
            except queue.Empty:
                if writer_done.is_set():
                    log("  [警告] 编码器提前结束(音频短于视频, -shortest 触发), "
                        "输出已生成, 走完整性检查")
                    encoder_finished_early = True
                    break
                continue
            if raw is None:
                break
            frame_idx += 1
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

            # 鱼眼去畸变: 矫正后再检测人脸, 大幅提升识别率
            if fisheye:
                img = fisheye_undistort(img, fisheye_strength, fisheye_device,
                                        downscale=fisheye_downscale,
                                        dual=fisheye_dual,
                                        crop=fisheye_crop)

            faces = []
            if face_on:
                faces = face_proc.process(img, frame_idx, raw_debug=raw_debug)
                # 双鱼眼镜像打码: 一侧检出人脸 → 另一侧同步打码
                if dual_mirror and fisheye and fisheye_dual != "false":
                    out_h, out_w = img.shape[:2]
                    mid = out_w // 2
                    if mid >= 2 and out_w >= 2 * out_h:
                        mirrored = list(faces)
                        for (x1, y1, x2, y2) in faces:
                            cx = (x1 + x2) / 2
                            if cx < mid:
                                mirrored.append((x1 + mid, y1, x2 + mid, y2))
                            else:
                                mirrored.append((x1 - mid, y1, x2 - mid, y2))
                        faces = mirrored
                if faces:
                    # np.frombuffer(raw_bytes) 是只读视图；只在人脸帧需要打码时
                    # 才复制为可写数组，空场景仍保持零拷贝。
                    img = _ensure_writable_frame(img)
                for (x1, y1, x2, y2) in faces:
                    bw, bh = x2 - x1, y2 - y1
                    heavy_mosaic(img, int(x1 - FACE_EXPAND * bw),
                                 int(y1 - FACE_EXPAND * bh),
                                 int(x2 + FACE_EXPAND * bw),
                                 int(y2 + FACE_EXPAND * bh),
                                 FACE_CELLS, FACE_SIGMA)
                if debug:
                    if raw_debug and face_proc.last_raw_debug_faces:
                        # 原始候选可能没有进入打码，此时 pipe 帧仍是只读视图。
                        img = _ensure_writable_frame(img)
                        draw_raw_face_debug(img, face_proc.last_raw_debug_faces)
                    draw_face_debug_scores(img, face_proc.last_debug_faces)
                total_face += len(faces)

            # 延迟若干帧再编码，使后续双模型确认能回改此前单模型候选帧。
            # face_backfill=0 时仍然即时零拷贝写入。
            frame_buffer.append({"frame_idx": frame_idx, "image": img})
            if face_on and face_proc.last_backfill_events:
                total_face += apply_face_backfill(
                    frame_buffer, face_proc.last_backfill_events)
            if len(frame_buffer) > backfill_buffer_size:
                oldest = frame_buffer.pop(0)
                if not _enqueue_image(oldest["image"]):
                    break

            if frame_idx % 50 == 0:
                model_timing, timing_snapshot = _format_model_timing(
                    (fd, scrfd_verifier), timing_snapshot)
                log(f"    [{frame_idx}] 人脸={len(faces)} "
                    f"elapsed={time.time()-t0:.0f}s | {model_timing}")
        if not encoder_finished_early:
            for item in frame_buffer:
                if not _enqueue_image(item["image"]):
                    break
            frame_buffer.clear()
    except BrokenPipeError:
        log("  [警告] 管道中断")
    finally:
        # 通知 writer 结束, 等待两个线程退出
        if not writer_done.is_set():
            # writer 仍在运行, 投递哨兵触发退出
            try:
                write_q.put(None, timeout=_PIPE_TIMEOUT)
            except queue.Full:
                pass  # writer 可能已提前退出
        writer_thread.join(timeout=30)
        # 提前结束时 reader 可能阻塞在 read_q.put(队列满), 终止解码器并排空队列,
        # 使 reader 读到 EOF 自行退出, 避免 join 长时间空等
        if encoder_finished_early:
            try:
                extract.terminate()
            except Exception:
                pass
            try:
                while True:
                    read_q.get_nowait()
            except queue.Empty:
                pass
        reader_thread.join(timeout=10)
        try:
            extract.stdout.close()
        except Exception:
            pass
        extract.wait()
        encode.wait()

    # 错误诊断
    if encode.returncode != 0 and enc_err_buf:
        err_tail = b"".join(enc_err_buf).decode()[-500:]
        if err_tail.strip():
            log(f"  [ffmpeg错误] {err_tail}")
    if pipeline_error:
        for who, e in pipeline_error:
            log(f"  [流水线错误] {who}: {e}")

    final_model_timing, _ = _format_model_timing((fd, scrfd_verifier))
    log(f"  [模型性能汇总] {final_model_timing}")
    _log_scrfd_timing((fd, scrfd_verifier), log)
    elapsed = time.time() - t0
    # 完整性检查: 超高帧率视频管道可能跟不上下游, 若实际处理帧远低于预期则自动回退文件模式
    # 编码器提前结束时输出已完整生成, 跳过帧数比例检查(避免误判回退文件模式)
    duration = get_duration(src)
    if not encoder_finished_early and duration and duration > 5:
        total_expected = int(fps * duration)  # 视频总帧数(含跳帧前)
        actual_ratio = frame_idx / max(1, total_expected)
        if actual_ratio < 0.3:
            # 流式管道提前结束(<30%帧处理完成), 自动回退文件模式
            raise RuntimeError(
                f"管道提前结束(仅处理{frame_idx}/{total_expected}帧, {actual_ratio:.0%}), 回退文件模式")
    log(f"  [完成] {dst}  耗时 {elapsed:.0f}s  人脸帧次={total_face}")
    return encode.returncode == 0


def _process_files(src, dst, face_on, model_dir, face_size,
                   face_int, face_empty_int, face_conf, face_model, keep_tmp, force_h264, use_gpu,
                   frame_skip, fisheye, fisheye_strength, fisheye_device,
                   fisheye_downscale, fisheye_dual, fisheye_crop, log,
                   scrfd_verify=False, scrfd_conf=0.3, scrfd_iou=0.3, scrfd_keep_conf=0.35,
                   dual_iou=0.2, scrfd_model="10g", scrfd_device="auto",
                   scrfd_nms=0.4, scrfd_gpu_id=0, dual_mirror=False,
                   face_grace=FACE_GRACE, face_smooth=FACE_BOX_SMOOTH,
                   scrfd_landmark_filter=True, debug=False, raw_debug=False,
                   face_backfill=FACE_BACKFILL, face_burst=FACE_BURST,
                   hard_face_recall=False, hard_face_conf=HARD_FACE_CONF,
                   hard_face_min_size=HARD_FACE_MIN_SIZE,
                   hard_face_roi_scale=HARD_FACE_ROI_SCALE,
                   hard_face_max_rois=HARD_FACE_MAX_ROIS,
                   hard_face_roi_size=HARD_FACE_ROI_SIZE,
                   hard_face_full_scan=False,
                   hard_face_full_scan_conf=HARD_FACE_FULL_SCAN_CONF):
    """文件模式: ffmpeg抽帧JPEG → 打码 → 重新编码。

    frame_skip 控制"每隔多少帧抽一帧"(1=逐帧, 2=隔1抽1提速2x)。
    跳帧时输出帧率同步降低(fps/frame_skip), 保持视频时长不变。
    """
    tmp = tempfile.mkdtemp(prefix="vmb_")
    fin, fout = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    os.makedirs(fin)
    os.makedirs(fout)
    t0 = time.time()
    fps = get_fps(src)
    w, h, _, _ = get_video_info(src)

    # 鱼眼裁剪: 计算输出尺寸
    if fisheye and 0 < fisheye_crop < 1.0:
        # 用 dummy 帧探测 fisheye_undistort 的实际输出尺寸
        _probe = fisheye_undistort(np.zeros((h, w, 3), dtype=np.uint8),
                                   fisheye_strength, fisheye_device,
                                   downscale=fisheye_downscale,
                                   dual=fisheye_dual,
                                   crop=fisheye_crop)
        out_h, out_w = _probe.shape[:2]
        out_w = out_w // 2 * 2
        out_h = out_h // 2 * 2
        log(f"  [鱼眼] 裁剪边缘: crop={fisheye_crop}, 输出 {out_w}x{out_h} (原 {w}x{h})")
    elif fisheye_crop != 1.0 and not fisheye:
        log(f"  [警告] --fisheye-crop 需要 --fisheye 同时启用, 已忽略")
        out_w, out_h = w, h
    else:
        out_w, out_h = w, h

    log(f"  [文件模式] 抽帧: {src}" + (f" (每{frame_skip}帧抽1帧,提速约{frame_skip}x)" if frame_skip > 1 else ""))
    extract_cmd = ["ffmpeg", "-y", "-i", src, "-q:v", "2"]
    if frame_skip > 1:
        extract_cmd += ["-vf", f"select=not(mod(n\\,{frame_skip})),setpts=N/FRAME_RATE/TB"]
    extract_cmd += [os.path.join(fin, "frame_%05d.jpg")]
    r = _run(extract_cmd)
    if r.returncode != 0:
        log("  [错误] ffmpeg 抽帧失败: " + r.stderr[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    frames = sorted(f for f in os.listdir(fin) if f.endswith(".jpg"))
    log(f"  共 {len(frames)} 帧")

    fd = _create_face_detector(
        face_model, model_dir, face_size, face_conf, use_gpu,
        scrfd_model, scrfd_device, scrfd_nms, scrfd_gpu_id,
        scrfd_landmark_filter, log)
    stereo_split = (_is_dual_fisheye(w, h, fisheye_dual)
                    and isinstance(fd, DualFaceDetector))
    if stereo_split:
        fd.enable_stereo_split(True)
        log(f"  [双鱼眼] 左右独立检测 + 低分互证 "
            f"(每目输入={face_size}, 低分候选>={fd._stereo_low_conf:.2f}, "
            f"高分增量={STEREO_HIGH_MARGIN:.2f})")
    scrfd_verifier = _create_scrfd_verifier(
        scrfd_verify and face_on, face_model, model_dir, face_size, use_gpu,
        scrfd_model, scrfd_device, scrfd_nms, scrfd_gpu_id,
        scrfd_conf, scrfd_iou, scrfd_keep_conf,
        scrfd_landmark_filter, log)
    roi_detector = None
    if hard_face_recall and face_on:
        try:
            roi_detector = YOLOFaceDetector(
                model_dir=model_dir, yolo_size=hard_face_roi_size,
                use_gpu=use_gpu)
            log(f"  [困难召唤] ROI二检: YOLOv8@{roi_detector.yolo_size}px "
                f"(上限{hard_face_max_rois}框/帧, conf={hard_face_conf})")
        except Exception as exc:
            log(f"  [警告] ROI二检检测器初始化失败: {exc}, 回退主检测器")
            roi_detector = None
    face_proc = FaceProcessor(fd, detect_int=face_int,
                              empty_detect_int=face_empty_int, conf=face_conf,
                              scrfd_verifier=scrfd_verifier, dual_iou=dual_iou,
                              grace=face_grace, box_smooth=face_smooth,
                              backfill_frames=face_backfill,
                              burst_frames=face_burst,
                              hard_face_recall=(HardFaceRecall(
                                  conf=hard_face_conf,
                                  min_size=hard_face_min_size,
                                  roi_scale=hard_face_roi_scale,
                                  max_rois=hard_face_max_rois,
                                  roi_size=hard_face_roi_size)
                                  if hard_face_recall else None),
                              roi_detector=roi_detector) if face_on else None
    if face_proc is not None:
        log(f"  [稳定] 逐脸轨迹 + LK/模板桥接 "
            f"(有人检测间隔={face_proc.detect_int}, "
            f"无人扫描间隔={face_proc.empty_detect_int}, "
            f"向前补码={face_proc.backfill_frames}帧, "
            f"变化突检={face_proc.burst_frames}帧, "
            f"丢轨主动扫描保持={face_proc.active_hold}帧, "
            f"单帧保活={face_proc.visible_hold}帧, "
            f"grace={face_proc.grace}检测周期, smooth={face_proc.box_smooth})")
    backfill_buffer_size = face_proc.backfill_frames if face_proc is not None else 0
    total_face = 0
    frame_buffer = []
    timing_snapshot = _model_timing_snapshot((fd, scrfd_verifier))

    for i, fn in enumerate(frames, 1):
        img = cv2.imread(os.path.join(fin, fn))
        if img is None:
            continue
        # 鱼眼去畸变
        if fisheye:
            img = fisheye_undistort(img, fisheye_strength, fisheye_device,
                                    downscale=fisheye_downscale,
                                    dual=fisheye_dual,
                                    crop=fisheye_crop)
        faces = []
        if face_on:
            faces = face_proc.process(img, i, raw_debug=raw_debug)
            # 双鱼眼镜像打码: 一侧检出人脸 → 另一侧同步打码
            if dual_mirror and fisheye and fisheye_dual != "false":
                fh, fw = img.shape[:2]
                mid = fw // 2
                if mid >= 2 and fw >= 2 * fh:
                    mirrored = list(faces)
                    for (x1, y1, x2, y2) in faces:
                        cx = (x1 + x2) / 2
                        if cx < mid:
                            mirrored.append((x1 + mid, y1, x2 + mid, y2))
                        else:
                            mirrored.append((x1 - mid, y1, x2 - mid, y2))
                    faces = mirrored
            for (x1, y1, x2, y2) in faces:
                bw, bh = x2 - x1, y2 - y1
                ex1, ey1 = int(x1 - FACE_EXPAND * bw), int(y1 - FACE_EXPAND * bh)
                ex2, ey2 = int(x2 + FACE_EXPAND * bw), int(y2 + FACE_EXPAND * bh)
                heavy_mosaic(img, ex1, ey1, ex2, ey2, FACE_CELLS, FACE_SIGMA)
            if debug:
                if raw_debug:
                    draw_raw_face_debug(img, face_proc.last_raw_debug_faces)
                draw_face_debug_scores(img, face_proc.last_debug_faces)
            total_face += len(faces)
        frame_buffer.append({"frame_idx": i, "image": img, "filename": fn})
        if face_on and face_proc.last_backfill_events:
            total_face += apply_face_backfill(
                frame_buffer, face_proc.last_backfill_events)
        if len(frame_buffer) > backfill_buffer_size:
            oldest = frame_buffer.pop(0)
            cv2.imwrite(
                os.path.join(fout, oldest["filename"]), oldest["image"],
                [cv2.IMWRITE_JPEG_QUALITY, 100])
        if i % 50 == 0 or i == len(frames):
            model_timing, timing_snapshot = _format_model_timing(
                (fd, scrfd_verifier), timing_snapshot)
            log(f"    [{i}/{len(frames)}] 人脸帧={len(faces)} "
                f"elapsed={time.time()-t0:.0f}s | {model_timing}")

    for item in frame_buffer:
        cv2.imwrite(
            os.path.join(fout, item["filename"]), item["image"],
            [cv2.IMWRITE_JPEG_QUALITY, 100])

    log("  合成视频...")
    hw = find_hw_encoder(family="h264" if force_h264 else "hevc")
    out_fps = fps / frame_skip  # 跳帧时帧率同步降低
    vcmd = ["ffmpeg", "-y", "-framerate", str(out_fps), "-i", os.path.join(fout, "frame_%05d.jpg")]
    if has_audio(src):
        vcmd += ["-i", src, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    if hw:
        vcmd += ["-c:v", hw, "-b:v", str(hw_bitrate(out_w, out_h, fps, hw, get_bitrate(src)))]
        if hw.startswith(("hevc", "h265")):
            vcmd += ["-tag:v", "hvc1"]
        log(f"  [编码] 硬件: {hw}")
    else:
        # CPU x264: preset veryfast(原slow太慢) + threads 0(全核) + crf 20(原18偏慢)
        vcmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-threads", "0"]
        log("  [编码] libx264 veryfast(CPU多核)")
    vcmd += ["-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart", dst]
    r = _run(vcmd)
    if r.returncode != 0:
        log("  [错误] ffmpeg 合成失败: " + r.stderr[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    final_model_timing, _ = _format_model_timing((fd, scrfd_verifier))
    log(f"  [模型性能汇总] {final_model_timing}")
    _log_scrfd_timing((fd, scrfd_verifier), log)
    log(f"  [完成] {dst}  耗时 {time.time()-t0:.0f}s  人脸帧次={total_face}")
    return True


def process_video(src, dst, face_on=True, model_dir=None,
                  face_size=FACE_YUNET_SIZE, face_int=FACE_DETECT_INT,
                  face_conf=FACE_CONF, face_model="yunet",
                  use_pipe=True, keep_tmp=False,
                  force_h264=False, use_gpu=True, frame_skip=FRAME_SKIP,
                  fisheye=False, fisheye_strength=1.0, fisheye_device="pico4",
                  fisheye_downscale=2, fisheye_dual="auto", fisheye_crop=1.0, log=print,
                  scrfd_verify=False, scrfd_conf=0.3, scrfd_iou=0.3, scrfd_keep_conf=0.35,
                  dual_iou=0.2, scrfd_model="10g", scrfd_device="auto",
                  scrfd_nms=0.4, scrfd_gpu_id=0, dual_mirror=False,
                  face_grace=FACE_GRACE, face_smooth=FACE_BOX_SMOOTH,
                  scrfd_landmark_filter=True, debug=False, raw_debug=False,
                  face_empty_int=FACE_EMPTY_DETECT_INT,
                  face_backfill=FACE_BACKFILL, face_burst=FACE_BURST,
                  hard_face_recall=False, hard_face_conf=HARD_FACE_CONF,
                  hard_face_min_size=HARD_FACE_MIN_SIZE,
                  hard_face_roi_scale=HARD_FACE_ROI_SCALE,
                  hard_face_max_rois=HARD_FACE_MAX_ROIS,
                  hard_face_roi_size=HARD_FACE_ROI_SIZE,
                  hard_face_full_scan=False,
                  hard_face_full_scan_conf=HARD_FACE_FULL_SCAN_CONF):
    """处理单个视频: 人脸打码, 保留音轨。返回是否成功。"""
    if use_pipe:
        try:
            return _process_pipe(src, dst, face_on, model_dir, face_size,
                                 face_int, face_empty_int, face_conf, face_model,
                                 keep_tmp, force_h264, use_gpu,
                                 frame_skip, fisheye, fisheye_strength,
                                 fisheye_device, fisheye_downscale,
                                 fisheye_dual, fisheye_crop, log,
                                 scrfd_verify, scrfd_conf, scrfd_iou, scrfd_keep_conf,
                                 dual_iou, scrfd_model, scrfd_device,
                                 scrfd_nms, scrfd_gpu_id, dual_mirror,
                                 face_grace, face_smooth,
                                 scrfd_landmark_filter, debug, raw_debug,
                                 face_backfill, face_burst,
                                 hard_face_recall, hard_face_conf,
                                 hard_face_min_size, hard_face_roi_scale,
                                 hard_face_max_rois, hard_face_roi_size,
                                 hard_face_full_scan, hard_face_full_scan_conf)
        except Exception as e:
            log(f"  [警告] 管道模式失败({e}), 回退文件模式")
    return _process_files(src, dst, face_on, model_dir, face_size,
                          face_int, face_empty_int, face_conf, face_model,
                          keep_tmp, force_h264, use_gpu,
                          frame_skip, fisheye, fisheye_strength,
                          fisheye_device, fisheye_downscale,
                          fisheye_dual, fisheye_crop, log,
                          scrfd_verify, scrfd_conf, scrfd_iou, scrfd_keep_conf,
                          dual_iou, scrfd_model, scrfd_device,
                          scrfd_nms, scrfd_gpu_id, dual_mirror,
                          face_grace, face_smooth,
                          scrfd_landmark_filter, debug, raw_debug,
                          face_backfill, face_burst,
                          hard_face_recall, hard_face_conf,
                          hard_face_min_size, hard_face_roi_scale,
                          hard_face_max_rois, hard_face_roi_size,
                          hard_face_full_scan, hard_face_full_scan_conf)


def expand_inputs(inputs):
    files = []
    for it in inputs:
        if os.path.isdir(it):
            for ext in VIDEO_FORMATS:
                files += glob.glob(os.path.join(it, "*" + ext))
                files += glob.glob(os.path.join(it, "*" + ext.upper()))
        else:
            files += glob.glob(it)
    return [f for f in files if os.path.isfile(f)]


def main():
    ap = argparse.ArgumentParser(description="视频批量打码(仅人脸, YuNet/YOLO/SCRFD)")
    ap.add_argument("inputs", nargs="+", help="视频文件/目录/通配符")
    ap.add_argument("--out-dir", default="masked_out", help="输出目录(默认 masked_out)")
    ap.add_argument("--face-size", type=int, default=FACE_YUNET_SIZE,
                    help="人脸检测输入尺寸(默认640; YuNet/YOLO/SCRFD 均调整为32的倍数)")
    ap.add_argument("--face-model", default="yunet",
                    choices=["yunet", "yolov8", "yolov8m", "yolo11", "scrfd",
                             "yolo11+yunet", "yolov8+yolo11",
                             "yolov8+scrfd", "yolo11+scrfd"],
                    help="人脸检测模型: yunet(默认,轻量,仅opencv) / yolov8(极速) / yolov8m(高精度) / "
                         "yolo11(最新) / scrfd(face-detect同款) / yolo11+yunet / yolov8+yolo11 / "
                         "yolov8+scrfd / yolo11+scrfd(含+号的为严格双模型共识)")
    ap.add_argument("--dual-iou", type=float, default=0.2, metavar="IOU",
                    help="双模型共识 IoU 阈值(默认0.2; 仅--face-model 含+号时生效; "
                         "越高越严格,越低越宽容)")
    ap.add_argument("--face-int", type=int, default=FACE_DETECT_INT,
                    help=f"常态人脸检测间隔(默认{FACE_DETECT_INT}; "
                         "变化期由 --face-burst 临时逐帧检测)")
    ap.add_argument("--face-empty-int", type=int, default=FACE_EMPTY_DETECT_INT,
                    help=f"无人脸轨迹时的扫描间隔(默认{FACE_EMPTY_DETECT_INT}; "
                         "只影响空场景，新人脸最迟等待该帧数被发现)")
    ap.add_argument("--face-backfill", type=int, default=FACE_BACKFILL,
                    help=f"双模型确认后向前补码的最大帧数(默认{FACE_BACKFILL}; "
                         "单模型候选不会直接打码，0=关闭)")
    ap.add_argument("--face-burst", type=int, default=FACE_BURST,
                    help=f"检测到人脸数量变化后逐帧复检的帧数(默认{FACE_BURST}; 0=关闭)")
    ap.add_argument("--face-grace", type=int, default=FACE_GRACE,
                    help=f"单张脸漏检保活周期数(默认{FACE_GRACE}; 每周期=--face-int帧; "
                         "增大更不易闪, 但人脸离场后框保留更久)")
    ap.add_argument("--face-smooth", type=float, default=FACE_BOX_SMOOTH,
                    help=f"检测框平滑权重(默认{FACE_BOX_SMOOTH}; 0.05~1, "
                         "越低越稳, 1=不平滑)")
    ap.add_argument("--fisheye", action="store_true",
                    help="鱼眼视频去畸变(提升鱼眼镜头下人脸识别率, 输出也为去畸变后视频)")
    ap.add_argument("--fisheye-strength", type=float, default=1.0,
                    help="鱼眼畸变强度(默认1.0; 桶形畸变越严重调越大, 1.5~2.0)")
    ap.add_argument("--fisheye-device", default="generic",
                    choices=["generic", "pico4"],
                    help="鱼眼设备预置: generic(通用强鱼眼,默认) / pico4(Pico 4 RGB摄像头, 130°FOV)")
    ap.add_argument("--fisheye-downscale", type=int, default=1, metavar="N",
                    help="鱼眼remap降采样倍数(默认1=原分辨率remap, 画质最好且误检最少(实测downscale=2大框误检+30%%, 且提速仅+5%%因瓶颈在检测); "
                         "2=1/2分辨率remap提速但bilinear上采样让YuNet更易误检; 3=1/3分辨率更快但可能影响小脸检出)")
    ap.add_argument("--fisheye-dual", default="auto",
                    choices=["auto", "true", "false"],
                    help="双鱼眼拼接模式(默认auto=按宽高比自动检测,w>=2h视为双鱼眼); "
                         "true=强制切左右两半各自矫正; false=强制单鱼眼(兼容旧版). "
                         "Pico4双目录像(3840x1456=两个1920x1456左右拼接)必须启用(auto或true), "
                         "否则光心落在中央黑缝导致错误畸变,人脸检测率暴跌")
    ap.add_argument("--fisheye-crop", type=float, default=1.0, metavar="RATIO",
                    help="鱼眼矫正后裁剪比例(默认1.0=不裁剪; 0.8=保留中心80%%, 四周各裁10%%; "
                         "仅--fisheye启用时生效, 输出分辨率同步缩小)")
    ap.add_argument("--dual-mirror", action="store_true",
                    help="双鱼眼镜像打码: 一侧检出人脸后另一侧同步打码"
                         "(仅双鱼眼视频生效, 避免单侧漏检导致另一侧未打码)")
    ap.add_argument("--scrfd-verify", action="store_true",
                    help="SCRFD二次验证: YOLO检出后用SCRFD关键点模型确认, 过滤手/玩具误检 "
                         "(高conf框不验证, 低conf框才验证; 要求全部框均由SCRFD确认请使用"
                         "--face-model yolov8+scrfd)")
    ap.add_argument("--scrfd-model", choices=["2.5g", "10g"], default="2.5g",
                    help="SCRFD模型规格(默认2.5g速度快; 10g精度更高, 模型从--model-dir/当前目录查找)")
    ap.add_argument("--scrfd-device", choices=["auto", "cpu", "cuda", "coreml"],
                    default="auto", help="SCRFD推理后端(默认auto: CUDA > CoreML > CPU)")
    ap.add_argument("--scrfd-nms", type=float, default=0.4,
                    help="SCRFD NMS IoU阈值(默认0.4)")
    ap.add_argument("--scrfd-gpu-id", type=int, default=0,
                    help="SCRFD使用的CUDA GPU编号(默认0)")
    ap.add_argument("--scrfd-conf", type=float, default=0.3,
                    help="SCRFD二次验证阈值(默认0.3; 仅--scrfd-verify生效; "
                         "SCRFD主模型/共识模式使用--face-conf)")
    ap.add_argument("--no-scrfd-landmark-filter", action="store_true",
                    help="关闭SCRFD五点拓扑过滤(默认开启；仅在极端侧脸被误删时使用)")
    ap.add_argument("--scrfd-iou", type=float, default=0.3,
                    help="YOLO框与SCRFD框匹配的IoU阈值(默认0.3; 降如0.2匹配更宽松)")
    ap.add_argument("--scrfd-keep-conf", type=float, default=0.35,
                    help="YOLO conf≥此值直接保留不验证(默认0.35; 避免SCRFD漏检误伤高置信度真脸)")
    ap.add_argument("--face-conf", type=float, default=FACE_CONF,
                    help=f"人脸置信度阈值(默认{FACE_CONF}; 降底如0.25可检出更多侧脸/遮挡, 可能增误检)")
    ap.add_argument("--hard-face-recall", action="store_true",
                    help="启用困难脸ROI二检：对小脸/边缘候选放大区域后低阈值复检")
    ap.add_argument("--hard-face-conf", type=float, default=HARD_FACE_CONF,
                    help=f"困难脸ROI二检阈值(默认{HARD_FACE_CONF})")
    ap.add_argument("--hard-face-min-size", type=int, default=HARD_FACE_MIN_SIZE,
                    help=f"候选最小边小于该值时进入困难ROI二检(默认{HARD_FACE_MIN_SIZE}px)")
    ap.add_argument("--hard-face-roi-scale", type=float, default=HARD_FACE_ROI_SCALE,
                    help=f"困难候选ROI外扩倍数(默认{HARD_FACE_ROI_SCALE})")
    ap.add_argument("--hard-face-max-rois", type=int, default=HARD_FACE_MAX_ROIS,
                    help=f"每帧最多二检的困难框数(默认{HARD_FACE_MAX_ROIS}; 防止推理爆炸)")
    ap.add_argument("--hard-face-roi-size", type=int, default=HARD_FACE_ROI_SIZE,
                    help=f"ROI二检推理尺寸(默认{HARD_FACE_ROI_SIZE}px; 比主检测小一倍提速~75%%)")
    ap.add_argument("--hard-face-full-scan", action="store_true",
                    help="全帧低阈值扫描: ROI检测器独立扫描全帧, 主动发现主检测遗漏的人脸(需要--hard-face-recall同时开启)")
    ap.add_argument("--hard-face-full-scan-conf", type=float, default=HARD_FACE_FULL_SCAN_CONF,
                    help=f"全帧扫描阈值(默认{HARD_FACE_FULL_SCAN_CONF}; 低于主检测阈值, 独立捕获漏检人脸)")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP,
                    help="【抽帧频率】跳过间隔(默认1=逐帧; 2=隔1抽1提速2x; 3=每3帧抽1提速3x; 配合--face-int效果叠加)")
    ap.add_argument("--no-pipe", action="store_true",
                    help="禁用管道模式, 强制文件模式(调试或特殊场景; 默认已启用流式管道)")
    ap.add_argument("--debug", action="store_true",
                    help="显示双模型分值及来源: D=双模型确认, "
                         "S=单模型续轨, T=光流/模板跟踪")
    ap.add_argument("--debug-raw", action="store_true",
                    help="原始调试：绘制置信度达到 face-conf/2 的 YOLO/SCRFD 候选框（不参与打码）")
    ap.add_argument("--force-h264", action="store_true",
                    help="强制H.264输出(兼容性无敌: 微信/Android/旧播放器都能播; 画质仍近无损高码率)")
    ap.add_argument("--no-face", action="store_true", help="关闭人脸打码")
    ap.add_argument("--no-gpu", action="store_true",
                    help="关闭 GPU 加速(CUDA/MPS 均禁用, 强制 CPU)")
    ap.add_argument("--model-dir", default=None, help="人脸模型目录")
    ap.add_argument("--keep-tmp", action="store_true", help="保留中间帧")
    args = ap.parse_args()

    if args.face_int < 1 or args.face_empty_int < 1:
        ap.error("--face-int 和 --face-empty-int 必须 >= 1")
    if args.face_backfill < 0:
        ap.error("--face-backfill 必须 >= 0")
    if args.face_burst < 0:
        ap.error("--face-burst 必须 >= 0")
    if not 0 <= args.hard_face_conf <= 1:
        ap.error("--hard-face-conf 必须在 0~1 之间")
    if args.hard_face_min_size < 1:
        ap.error("--hard-face-min-size 必须 >= 1")
    if args.hard_face_roi_scale < 1:
        ap.error("--hard-face-roi-scale 必须 >= 1")
    if args.hard_face_max_rois < 1:
        ap.error("--hard-face-max-rois 必须 >= 1")
    if not 160 <= args.hard_face_roi_size <= 640:
        ap.error("--hard-face-roi-size 必须在 160~640 之间")
    if not 0 <= args.hard_face_full_scan_conf <= 1:
        ap.error("--hard-face-full-scan-conf 必须在 0~1 之间")
    if args.hard_face_full_scan and not args.hard_face_recall:
        print("  [警告] --hard-face-full-scan 需要 --hard-face-recall 同时开启, 已忽略")

    files = expand_inputs(args.inputs)
    if not files:
        print("[错误] 未找到任何视频文件")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"待处理 {len(files)} 个视频 -> {args.out_dir}")
    ok = 0
    for src in files:
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(args.out_dir, "masked_" + name + ".mp4")
        print(f"\n>>> {src}")
        if process_video(src, dst, face_on=not args.no_face,
                         model_dir=args.model_dir,
                         face_size=args.face_size, face_int=args.face_int,
                         face_empty_int=args.face_empty_int,
                         face_backfill=args.face_backfill,
                         face_burst=args.face_burst,
                         hard_face_recall=args.hard_face_recall,
                         hard_face_conf=args.hard_face_conf,
                         hard_face_min_size=args.hard_face_min_size,
                         hard_face_roi_scale=args.hard_face_roi_scale,
                         hard_face_max_rois=args.hard_face_max_rois,
                         hard_face_roi_size=args.hard_face_roi_size,
                         hard_face_full_scan=args.hard_face_full_scan,
                         hard_face_full_scan_conf=args.hard_face_full_scan_conf,
                         face_conf=args.face_conf, face_model=args.face_model,
                         use_pipe=not args.no_pipe, keep_tmp=args.keep_tmp,
                         force_h264=args.force_h264, use_gpu=not args.no_gpu,
                         frame_skip=args.frame_skip,
                         fisheye=args.fisheye, fisheye_strength=args.fisheye_strength,
                         fisheye_device=args.fisheye_device,
                         fisheye_downscale=args.fisheye_downscale,
                         fisheye_dual=args.fisheye_dual,
                         fisheye_crop=args.fisheye_crop,
                         scrfd_verify=args.scrfd_verify,
                         scrfd_conf=args.scrfd_conf,
                         scrfd_iou=args.scrfd_iou,
                         scrfd_keep_conf=args.scrfd_keep_conf,
                         dual_iou=args.dual_iou,
                         dual_mirror=args.dual_mirror,
                         scrfd_model=args.scrfd_model,
                         scrfd_device=args.scrfd_device,
                         scrfd_nms=args.scrfd_nms,
                         scrfd_gpu_id=args.scrfd_gpu_id,
                         face_grace=args.face_grace,
                         face_smooth=args.face_smooth,
                         scrfd_landmark_filter=not args.no_scrfd_landmark_filter,
                         debug=args.debug or args.debug_raw,
                         raw_debug=args.debug_raw):
            ok += 1
    print(f"\n全部完成: {ok}/{len(files)} 成功, 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
