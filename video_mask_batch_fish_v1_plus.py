#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_mask_batch_fish_v1.py — 视频批量打码（仅人脸），精简版

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
  python video_mask_batch_fish_v1.py video.mp4
  python video_mask_batch_fish_v1.py ./videos/              # 整个目录
  python video_mask_batch_fish_v1.py a.mp4 b.mov            # 多文件
  python video_mask_batch_fish_v1.py video.mp4 --face-model yolov8
  python video_mask_batch_fish_v1.py video.mp4 --face-model yolo11
  python video_mask_batch_fish_v1.py video.mp4 --out-dir ./out
  python video_mask_batch_fish_v1.py video.mp4 --no-face            # 关闭人脸打码
  python video_mask_batch_fish_v1.py video.mp4 --face-conf 0.25
  python video_mask_batch_fish_v1.py video.mp4 --fisheye
  python video_mask_batch_fish_v1.py video.mp4 --fisheye --fisheye-device pico4
  python video_mask_batch_fish_v1.py video.mp4 --frame-skip 2

依赖:
  pip install opencv-python numpy
  # YOLOv8 人脸检测可选: pip install ultralytics
  # 系统需 ffmpeg(保留音轨)

代码调用:
  from video_mask_batch_fish_v1 import process_video
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
FACE_DETECT_INT = 5           # 人脸检测间隔: 每5帧检测1次, 中间帧光流跟踪(默认5; 运动剧烈可改2~3)
FRAME_SKIP = 1                # 抽帧跳过间隔(1=逐帧处理; 2=隔1帧抽1帧提速2x; 3=每3帧抽1帧提速3x)
FACE_GRACE = 8               # 人脸漏检沿用旧框帧数(运动时收紧, 防打码框滞后)
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
    if dual == "auto":
        # 宽>=2倍高 视为双鱼眼拼接(典型 Pico4 双目 3840x1456 = 2.64:1)
        is_dual = (w >= 2 * h)
    else:
        is_dual = (dual == "true" or dual is True)

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
    def __init__(self, det_a, det_b, name_a="A", name_b="B"):
        self._det_a = det_a
        self._det_b = det_b
        self._name = f"{name_a}+{name_b}"

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
        dets_a = self._det_a.detect(img, conf=conf)
        if not dets_a:
            return []  # A 没检出 → 共识必为空, 跳过 B 的推理
        dets_b = self._det_b.detect(img, conf=conf)
        if not dets_b:
            return []
        # 贪心匹配: 对每个 A 框找最佳 IoU 匹配的 B 框
        consensus = []
        used = set()
        for da in dets_a:
            best_iou, best_j = 0, -1
            for j, db in enumerate(dets_b):
                if j in used:
                    continue
                score = self._iou(da, db)
                if score > best_iou:
                    best_iou, best_j = score, j
            if best_iou >= iou_thresh and best_j >= 0:
                consensus.append(da)
                used.add(best_j)
        return consensus


class SCRFDVerifier:
    """SCRFD 人脸验证器 — 用关键点模型验证 YOLO 检出的候选框是否为真脸。

    项目已有 scrfd_10g_bnkps.onnx (带5关键点: 双眼/鼻/双嘴角)。
    YOLO 检出候选框后, SCRFD 全图检测, 两者 IoU 匹配:
    - 匹配上(SCRFD 也认为是脸) → 保留 YOLO 框打码
    - 未匹配(SCRFD 不认为是脸, 如手/玩具) → 丢弃误检

    仅在 --scrfd-verify 开启时使用, 避免双模型推理的耗时翻倍。
    """

    SCRFD_FILE = "scrfd_10g_bnkps.onnx"
    SCRFD_INPUT = 640
    SCRFD_STRIDES = [8, 16, 32]
    SCRFD_NUM_ANCHORS = 2

    def __init__(self, model_path=None, input_size=640, conf=0.3, use_gpu=False,
                 iou_thresh=0.3, keep_conf=0.35):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("缺少 onnxruntime。请安装：pip install onnxruntime") from exc

        if model_path is None:
            for p in [os.path.join(os.getcwd(), self.SCRFD_FILE),
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), self.SCRFD_FILE)]:
                if os.path.isfile(p):
                    model_path = p
                    break
        if model_path is None:
            raise RuntimeError(f"未找到 {self.SCRFD_FILE}, 请放到项目目录")

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            avail = ort.get_available_providers()
            if "CoreMLExecutionProvider" in avail:
                providers.insert(0, "CoreMLExecutionProvider")
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._input_size = input_size
        self._conf = conf
        self._iou_thresh = iou_thresh    # YOLO框与SCRFD框匹配的IoU阈值
        self._keep_conf = keep_conf      # YOLO conf≥此值直接保留不验证(避免SCRFD漏检误伤)

    def detect(self, img):
        """全图检测, 返回 [(x1,y1,x2,y2), ...] 真脸列表(SCRFD 检出即带关键点)。"""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        sz = self._input_size
        scale = min(sz / h, sz / w)
        nh, nw = int(h * scale), int(w * scale)
        canvas = np.zeros((sz, sz, 3), dtype=np.float32)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        # BGR→RGB, HWC→CHW, 归一化 (x-127.5)/128
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)
        blob = (blob - 127.5) / 128.0
        blob = blob[np.newaxis].astype(np.float32)

        outs = self._sess.run(None, {self._input_name: blob})

        # 9 个输出: 3 score + 3 bbox + 3 kps (stride 8/16/32)
        scores_all, boxes_all = [], []
        for idx, stride in enumerate(self.SCRFD_STRIDES):
            scores = outs[idx]        # [N, 1]
            bboxes = outs[idx + 3]    # [N, 4] left/top/right/bottom 偏移
            h_grid = sz // stride
            n = h_grid * h_grid * self.SCRFD_NUM_ANCHORS
            if scores.shape[0] != n:
                continue
            # anchor 中心点 (每个 grid 点 num_anchors 个 anchor)
            centers = np.zeros((n, 2), dtype=np.float32)
            i = 0
            for y in range(h_grid):
                for x in range(h_grid):
                    cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
                    for _ in range(self.SCRFD_NUM_ANCHORS):
                        centers[i] = (cx, cy)
                        i += 1
            mask = scores[:, 0] > self._conf
            if mask.sum() == 0:
                continue
            s = scores[mask, 0]
            b = bboxes[mask]
            c = centers[mask]
            # anchor-free 解码: bbox = center ± offset*stride
            x1 = (c[:, 0] - b[:, 0] * stride) / scale
            y1 = (c[:, 1] - b[:, 1] * stride) / scale
            x2 = (c[:, 0] + b[:, 2] * stride) / scale
            y2 = (c[:, 1] + b[:, 3] * stride) / scale
            for j in range(len(s)):
                scores_all.append(float(s[j]))
                boxes_all.append((max(0.0, x1[j]), max(0.0, y1[j]),
                                  min(float(w), x2[j]), min(float(h), y2[j])))

        if not boxes_all:
            return []
        boxes_np = np.array(boxes_all, dtype=np.float32)
        scores_np = np.array(scores_all, dtype=np.float32)
        xywh = boxes_np.copy()
        xywh[:, 2] -= xywh[:, 0]
        xywh[:, 3] -= xywh[:, 1]
        idxs = cv2.dnn.NMSBoxes(xywh.tolist(), scores_np.tolist(), self._conf, 0.4)
        if len(idxs) == 0:
            return []
        return [tuple(boxes_np[i]) for i in idxs.flatten()]

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


class FaceProcessor:
    """人脸检测+光流跟踪：关键帧检测，中间帧LK光流跟踪。支持多人。

    跟踪质量保障(防"漏一帧"):
    - F1 forward-backward LK 验证: 正向跟踪后再反向, FB 误差大的点剔除,
      median 估计更稳健(避免跟踪点跑到背景上导致框瞬移到错误位置)
    - F2 位移阈值兜底: dx/dy > 0.3×box_size 判定跟踪跑飞, 回退到旧框
      (避免框瞬移导致原人脸位置那 1 帧没被打码)
    - F5 首帧强制双检: frame_idx<=2 时强制检测, 避免模型冷启动漏检
      导致视频开头几帧没打码(YuNet DNN 后端初始化可能首帧漏检)
    """
    def __init__(self, detector, detect_int=FACE_DETECT_INT, grace=FACE_GRACE, conf=FACE_CONF, scrfd_verifier=None, dual_iou=0.2):
        self.detector = detector
        self.detect_int = max(1, detect_int)
        self.conf = conf
        self.grace = grace          # 检测失败时沿用旧框的帧数
        self.miss_count = 0         # 连续检测失败的周期数
        self.last_faces = []
        self.prev_gray = None
        self.face_pts = {}
        self.scrfd_verifier = scrfd_verifier  # SCRFD 二次验证器(可选, --scrfd-verify 开启)
        self.dual_iou = dual_iou    # 双模型共识 IoU 阈值(仅 DualFaceDetector 使用)
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

    def _track_box(self, prev_gray, gray, box):
        """F1+F2: 对一个 box 做 LK 跟踪, 返回 (new_box, tracked_pts) 或 (None, None)。

        F1 forward-backward 验证: 正向 LK 后再反向 LK, FB 误差大的点剔除,
        median 位移估计更稳健, 避免跟踪点跑到背景导致框瞬移漏帧。
        F2 位移阈值: dx/dy > 0.3×box_size 判定跟踪跑飞, 返回 None 回退到旧框。
        """
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        box_size = max(bw, bh)
        pts = self._init_pts(prev_gray, box)
        if pts is None or len(pts) < 4:
            return None, None
        # 正向 LK: prev_gray → gray
        npts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None, **self.lk)
        fwd_mask = status.flatten() == 1
        if fwd_mask.sum() < 4:
            return None, None
        # F1 反向 LK: gray → prev_gray, 验证点是否真的跟踪正确
        # 原理: 真实跟踪的点反向应该回到原位, 跑到背景的点反向会跑飞
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, npts, None, **self.lk)
        back_mask = back_status.flatten() == 1
        # FB 误差: |back_pts - 原始 pts|, 大于阈值的点视为跑飞剔除
        fb_err = np.linalg.norm(
            back_pts.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
        good_mask = fwd_mask & back_mask & (fb_err <= self._fb_thresh)
        if good_mask.sum() < 4:
            return None, None
        good = npts[good_mask].reshape(-1, 2)
        old_flat = pts[good_mask].reshape(-1, 2)
        dx = float(np.median(good[:, 0] - old_flat[:, 0]))
        dy = float(np.median(good[:, 1] - old_flat[:, 1]))
        # F2 位移阈值: 一帧内位移超过 30% 框宽判定跟踪跑飞, 回退旧框
        # (正常人脸一帧位移 < 10% 框宽; 30% 已是极端, 超过必为跟踪点跑到背景)
        if max(abs(dx), abs(dy)) > self._max_disp_ratio * box_size:
            return None, None
        new_box = (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))
        return new_box, good.reshape(-1, 1, 2)

    def process(self, img, frame_idx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # F5 首帧强制双检: frame_idx<=2 时强制检测, 避免模型冷启动漏检
        # (YuNet 首帧可能因 DNN 后端初始化开销漏检, 普通帧间隔兜底无效)
        cold_start = frame_idx <= 2
        need_detect = ((frame_idx - 1) % self.detect_int == 0) or not self.last_faces or cold_start
        if need_detect:
            # 双模型共识: 两模型同时检测, IoU 匹配一致才保留
            if isinstance(self.detector, DualFaceDetector):
                dets = self.detector.detect(img, conf=self.conf, iou_thresh=self.dual_iou)
            # SCRFD 二次验证: 高conf框直接保留, 低conf框用SCRFD确认(过滤手/玩具误检)
            elif self.scrfd_verifier is not None and hasattr(self.detector, 'detect_with_conf'):
                dets_with_conf = self.detector.detect_with_conf(img, conf=self.conf)
                dets = self.scrfd_verifier.verify(dets_with_conf, img)
            else:
                dets = self.detector.detect(img, conf=self.conf)
            if dets:
                faces = dets
                self.miss_count = 0
            elif self.last_faces and self.miss_count < self.grace:
                # 检测失败: 宽容期内沿用上一帧框(LK跟踪会继续修正)
                faces = self.last_faces
                self.miss_count += 1
            else:
                faces = []
                self.miss_count = 0
            self.face_pts = {}
            for i, box in enumerate(faces):
                pts = self._init_pts(gray, box)
                if pts is not None:
                    self.face_pts[i] = pts
            self.last_faces = faces
        elif self.prev_gray is not None and self.last_faces and self.face_pts:
            tracked, new_pts = [], {}
            for i, box in enumerate(self.last_faces):
                if i not in self.face_pts or len(self.face_pts[i]) < 4:
                    tracked.append(box)
                    # 跟踪点缺失/不足: 在当前框上重新初始化, 确保下一帧仍有 LK 点可用
                    pts = self._init_pts(gray, box)
                    if pts is not None:
                        new_pts[i] = pts
                    continue
                # F1+F2 跟踪: forward-backward 验证 + 位移阈值兜底
                new_box, good_pts = self._track_box(self.prev_gray, gray, box)
                if new_box is not None:
                    tracked.append(new_box)
                    new_pts[i] = good_pts
                else:
                    # 跟踪跑飞或点不足: 沿用旧框, 重新初始化跟踪点
                    # (避免跟踪点完全丢失导致后续帧无法 LK 跟踪 → 闪烁)
                    tracked.append(box)
                    pts = self._init_pts(gray, box)
                    if pts is not None:
                        new_pts[i] = pts
            self.face_pts = new_pts
            self.last_faces = tracked
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


def _process_pipe(src, dst, face_on, model_dir, face_size,
                  face_int, face_conf, face_model, keep_tmp, force_h264, use_gpu,
                  frame_skip, fisheye, fisheye_strength, fisheye_device,
                  fisheye_downscale, fisheye_dual, fisheye_crop, log,
                  scrfd_verify=False, scrfd_conf=0.3, scrfd_iou=0.3, scrfd_keep_conf=0.35,
                  dual_iou=0.2):
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
    if face_model == "yolo11+yunet":
        fd = DualFaceDetector(
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu),
            name_a="YOLO11", name_b="YuNet")
        log(f"  [人脸] YOLO11+YuNet 双模型共识 (输入{face_size})")
    elif face_model == "yolov8+yolo11":
        fd = DualFaceDetector(
            YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            name_a="YOLOv8", name_b="YOLO11")
        log(f"  [人脸] YOLOv8+YOLO11 双模型共识 (输入{face_size})")
    elif face_model == "yolov8m":
        fd = YOLOv8MFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-medium (输入{face_size})")
    elif face_model == "yolov8":
        fd = YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-nano (输入{face_size})")
    elif face_model == "yolo11":
        fd = YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv11-nano (输入{face_size})")
    else:
        fd = YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YuNet (输入{face_size})")
    scrfd_verifier = None
    if scrfd_verify and face_on:
        try:
            scrfd_verifier = SCRFDVerifier(conf=scrfd_conf, use_gpu=use_gpu,
                                           iou_thresh=scrfd_iou, keep_conf=scrfd_keep_conf)
            log(f"  [人脸] SCRFD 二次验证 (conf={scrfd_conf}, iou={scrfd_iou}, keep_conf={scrfd_keep_conf})")
        except Exception as e:
            log(f"  [警告] SCRFD 初始化失败: {e}, 跳过二次验证")
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf,
                              scrfd_verifier=scrfd_verifier, dual_iou=dual_iou) if face_on else None

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
                faces = face_proc.process(img, frame_idx)
                for (x1, y1, x2, y2) in faces:
                    bw, bh = x2 - x1, y2 - y1
                    heavy_mosaic(img, int(x1 - FACE_EXPAND * bw),
                                 int(y1 - FACE_EXPAND * bh),
                                 int(x2 + FACE_EXPAND * bw),
                                 int(y2 + FACE_EXPAND * bh),
                                 FACE_CELLS, FACE_SIGMA)
                total_face += len(faces)

            # 零拷贝写入: memoryview 直接引用 numpy 数组内部缓冲区,
            # 替代 img.tobytes() 的每帧 6MB 内存拷贝。
            try:
                write_q.put(memoryview(img), timeout=_PIPE_TIMEOUT)
            except queue.Full:
                if writer_done.is_set():
                    log("  [警告] 编码器提前结束(音频短于视频, -shortest 触发), "
                        "输出已生成, 走完整性检查")
                    encoder_finished_early = True
                    break
                raise  # 队列阻塞且 writer 未退出, 疑似死锁, 抛出便于诊断

            if frame_idx % 50 == 0:
                log(f"    [{frame_idx}] 人脸={len(faces)} "
                    f"elapsed={time.time()-t0:.0f}s")
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
                   face_int, face_conf, face_model, keep_tmp, force_h264, use_gpu,
                   frame_skip, fisheye, fisheye_strength, fisheye_device,
                   fisheye_downscale, fisheye_dual, fisheye_crop, log,
                   scrfd_verify=False, scrfd_conf=0.3, scrfd_iou=0.3, scrfd_keep_conf=0.35,
                   dual_iou=0.2):
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

    if face_model == "yolo11+yunet":
        fd = DualFaceDetector(
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu),
            name_a="YOLO11", name_b="YuNet")
        log(f"  [人脸] YOLO11+YuNet 双模型共识 (输入{face_size})")
    elif face_model == "yolov8+yolo11":
        fd = DualFaceDetector(
            YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu),
            name_a="YOLOv8", name_b="YOLO11")
        log(f"  [人脸] YOLOv8+YOLO11 双模型共识 (输入{face_size})")
    elif face_model == "yolov8m":
        fd = YOLOv8MFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-medium (输入{face_size})")
    elif face_model == "yolov8":
        fd = YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-nano (输入{face_size})")
    elif face_model == "yolo11":
        fd = YOLO11FaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv11-nano (输入{face_size})")
    else:
        fd = YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YuNet (输入{face_size})")
    scrfd_verifier = None
    if scrfd_verify and face_on:
        try:
            scrfd_verifier = SCRFDVerifier(conf=scrfd_conf, use_gpu=use_gpu,
                                           iou_thresh=scrfd_iou, keep_conf=scrfd_keep_conf)
            log(f"  [人脸] SCRFD 二次验证 (conf={scrfd_conf}, iou={scrfd_iou}, keep_conf={scrfd_keep_conf})")
        except Exception as e:
            log(f"  [警告] SCRFD 初始化失败: {e}, 跳过二次验证")
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf,
                              scrfd_verifier=scrfd_verifier, dual_iou=dual_iou) if face_on else None
    total_face = 0

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
            faces = face_proc.process(img, i)
            for (x1, y1, x2, y2) in faces:
                bw, bh = x2 - x1, y2 - y1
                ex1, ey1 = int(x1 - FACE_EXPAND * bw), int(y1 - FACE_EXPAND * bh)
                ex2, ey2 = int(x2 + FACE_EXPAND * bw), int(y2 + FACE_EXPAND * bh)
                heavy_mosaic(img, ex1, ey1, ex2, ey2, FACE_CELLS, FACE_SIGMA)
            total_face += len(faces)
        cv2.imwrite(os.path.join(fout, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if i % 50 == 0 or i == len(frames):
            log(f"    [{i}/{len(frames)}] 人脸帧={len(faces)} elapsed={time.time()-t0:.0f}s")

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
                  dual_iou=0.2):
    """处理单个视频: 人脸打码, 保留音轨。返回是否成功。"""
    if use_pipe:
        try:
            return _process_pipe(src, dst, face_on, model_dir, face_size,
                                 face_int, face_conf, face_model,
                                 keep_tmp, force_h264, use_gpu,
                                 frame_skip, fisheye, fisheye_strength,
                                 fisheye_device, fisheye_downscale,
                                 fisheye_dual, fisheye_crop, log,
                                 scrfd_verify, scrfd_conf, scrfd_iou, scrfd_keep_conf,
                                 dual_iou)
        except Exception as e:
            log(f"  [警告] 管道模式失败({e}), 回退文件模式")
    return _process_files(src, dst, face_on, model_dir, face_size,
                          face_int, face_conf, face_model,
                          keep_tmp, force_h264, use_gpu,
                          frame_skip, fisheye, fisheye_strength,
                          fisheye_device, fisheye_downscale,
                          fisheye_dual, fisheye_crop, log,
                          scrfd_verify, scrfd_conf, scrfd_iou, scrfd_keep_conf,
                          dual_iou)


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
    ap = argparse.ArgumentParser(description="视频批量打码(仅人脸, YuNet/YOLOv8)")
    ap.add_argument("inputs", nargs="+", help="视频文件/目录/通配符")
    ap.add_argument("--out-dir", default="masked_out", help="输出目录(默认 masked_out)")
    ap.add_argument("--face-size", type=int, default=FACE_YUNET_SIZE,
                    help="人脸检测输入尺寸(默认640; YuNet/YOLOv8 均调整为32的倍数)")
    ap.add_argument("--face-model", default="yunet",
                    choices=["yunet", "yolov8", "yolov8m", "yolo11", "yolo11+yunet", "yolov8+yolo11"],
                    help="人脸检测模型: yunet(默认,轻量,仅opencv) / yolov8(极速) / yolov8m(高精度) / "
                         "yolo11(最新) / yolo11+yunet(双模型共识) / yolov8+yolo11(双模型共识)")
    ap.add_argument("--dual-iou", type=float, default=0.2, metavar="IOU",
                    help="双模型共识 IoU 阈值(默认0.2; 仅--face-model 含+号时生效; "
                         "越高越严格,越低越宽容)")
    ap.add_argument("--face-int", type=int, default=FACE_DETECT_INT,
                    help="人脸检测间隔帧数(默认5: 每5帧检测1次,中间帧光流跟踪; 1=每帧检测最准)")
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
    ap.add_argument("--scrfd-verify", action="store_true",
                    help="SCRFD二次验证: YOLO检出后用SCRFD关键点模型确认, 过滤手/玩具误检 "
                         "(需scrfd_10g_bnkps.onnx+onnxruntime; 高conf框不验证, 低conf框才验证)")
    ap.add_argument("--scrfd-conf", type=float, default=0.3,
                    help="SCRFD检测置信度阈值(默认0.3; 降如0.15检出更多但可能增误检)")
    ap.add_argument("--scrfd-iou", type=float, default=0.3,
                    help="YOLO框与SCRFD框匹配的IoU阈值(默认0.3; 降如0.2匹配更宽松)")
    ap.add_argument("--scrfd-keep-conf", type=float, default=0.35,
                    help="YOLO conf≥此值直接保留不验证(默认0.35; 避免SCRFD漏检误伤高置信度真脸)")
    ap.add_argument("--face-conf", type=float, default=FACE_CONF,
                    help=f"人脸置信度阈值(默认{FACE_CONF}; 降底如0.25可检出更多侧脸/遮挡, 可能增误检)")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP,
                    help="【抽帧频率】跳过间隔(默认1=逐帧; 2=隔1抽1提速2x; 3=每3帧抽1提速3x; 配合--face-int效果叠加)")
    ap.add_argument("--no-pipe", action="store_true",
                    help="禁用管道模式, 强制文件模式(调试或特殊场景; 默认已启用流式管道)")
    ap.add_argument("--force-h264", action="store_true",
                    help="强制H.264输出(兼容性无敌: 微信/Android/旧播放器都能播; 画质仍近无损高码率)")
    ap.add_argument("--no-face", action="store_true", help="关闭人脸打码")
    ap.add_argument("--no-gpu", action="store_true",
                    help="关闭 GPU 加速(CUDA/MPS 均禁用, 强制 CPU)")
    ap.add_argument("--model-dir", default=None, help="人脸模型目录")
    ap.add_argument("--keep-tmp", action="store_true", help="保留中间帧")
    args = ap.parse_args()

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
                         dual_iou=args.dual_iou):
            ok += 1
    print(f"\n全部完成: {ok}/{len(files)} 成功, 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
