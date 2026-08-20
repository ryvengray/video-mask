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
import concurrent.futures
import glob
import math
import os
import queue
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
FACE_CONF = 0.25             # 人脸置信度阈值(0.45->0.35: 侧脸/低头/遮挡帧不漏检, 跟踪器滤误检)
FACE_EXPAND = 0.12            # 人脸打码框外扩比例(确保盖住完整脸)
FACE_YUNET_SIZE = 640         # 人脸检测输入尺寸(YuNet/YOLOv8 均自动调整为32的倍数, 640x640)
FACE_DETECT_INT = 5           # 人脸检测间隔: 每5帧检测1次, 中间帧光流跟踪(默认5; 运动剧烈可改2~3)
FRAME_SKIP = 1                # 抽帧跳过间隔(1=逐帧处理; 2=隔1帧抽1帧提速2x; 3=每3帧抽1帧提速3x)
FACE_GRACE = 4               # 人脸漏检沿用旧框帧数(运动时收紧, 防打码框滞后)

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


def fisheye_undistort(img, strength=1.0, device="generic", downscale=2):
    """对单帧做鱼眼去畸变, 返回矫正后的图像。

    支持设备预置(device="pico4" 等)或通用模式手动 strength 调节。
    映射表按 (分辨率, 参数) 缓存, 同一视频只算一次。

    downscale: 降采样倍数(>=1)。>1 时在 1/downscale 分辨率上做 remap 再上采样,
        remap 面积减 downscale² 倍, 大幅提速(1920x1456 downscale=2 约 -75% 耗时)。
        画质损失可忽略(双线性插值两次, 人脸打码场景无感知)。设 1 关闭。
    """
    h, w = img.shape[:2]

    # 获取设备参数
    preset = FISHEYE_PRESETS.get(device, FISHEYE_PRESETS["generic"])
    f_ratio = preset["f_ratio"]
    k1 = preset["k1"] * strength
    k2 = preset["k2"] * strength
    balance = preset["balance"]

    if downscale > 1:
        # 降采样路径: 在小图上 remap 再上采样回原尺寸
        sw, sh = max(2, w // downscale), max(2, h // downscale)
        cache_key = (sw, sh, f_ratio, k1, k2, balance)
        if cache_key not in _fisheye_maps_cache:
            fx = max(sw, sh) * f_ratio
            K = np.array([[fx, 0, sw / 2], [0, fx, sh / 2], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, 0, 0], dtype=np.float64)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (sw, sh), np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (sw, sh), cv2.CV_16SC2)
            _fisheye_maps_cache[cache_key] = (map1, map2)
        map1, map2 = _fisheye_maps_cache[cache_key]
        img_small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        undistorted_small = cv2.remap(img_small, map1, map2, cv2.INTER_LINEAR)
        return cv2.resize(undistorted_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 原分辨率路径
    cache_key = (w, h, f_ratio, k1, k2, balance)
    if cache_key in _fisheye_maps_cache:
        map1, map2 = _fisheye_maps_cache[cache_key]
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # 估算内参矩阵 K
    fx = max(w, h) * f_ratio
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
    # 畸变系数: k1 主导桶形畸变, k2 高阶修正
    D = np.array([k1, k2, 0, 0], dtype=np.float64)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    _fisheye_maps_cache[cache_key] = (map1, map2)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


# ================= 人脸检测 =================

# YuNet 人脸检测模型 (OpenCV cv2.FaceDetectorYN, 仅依赖 opencv-python)
# OpenCV 4.8+ 内置 YuNet; 也可手动放置 face_detection_yunet_2023mar.onnx
YUNET_FILE = "face_detection_yunet_2023mar.onnx"

# YOLOv8 nano 人脸检测模型 (ultralytics 自动下载，约 6MB)
YOLO_FACE_FILE = "yolov8n-face.pt"
YOLO_INPUT_SIZE = 640          # YOLOv8 检测输入尺寸(32的倍数; 降低可提速)


def _select_accel_backend(accel, face_model, use_gpu):
    """选择检测器加速后端, 返回实际后端名(可能因依赖缺失回退 'none')。

    accel: "auto"|"onnx"|"coreml"|"tensorrt"|"none"
    face_model: "yunet"|"yolov8"
    use_gpu: 是否允许使用 GPU(影响 auto 选择 EP)
    返回: "coreml"|"tensorrt"|"onnx"|"none"
    """
    if accel == "none":
        return "none"
    if not use_gpu and accel == "auto":
        # GPU 关闭时, 即使 auto 也走 none(纯 CPU 用原检测器更省心)
        return "none"

    is_macos = sys.platform == "darwin"
    is_linux_cuda = False
    if sys.platform.startswith("linux") and use_gpu:
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():
                is_linux_cuda = True
        except Exception:
            pass

    def _try_auto():
        # auto 路径: 按平台+模型优先级
        if is_macos and face_model == "yolov8":
            try:
                import coremltools  # noqa: F401
                return "coreml"
            except Exception as e:
                print(f"[加速] coreml 不可用: {e}, 回退 none")
                return "none"
        if is_macos and face_model == "yunet":
            try:
                import onnxruntime as ort
                eps = ort.get_available_providers()
                if "CoreMLExecutionProvider" in eps:
                    return "onnx"
                print(f"[加速] onnxruntime 无 CoreMLExecutionProvider ({eps}), 回退 none")
                return "none"
            except Exception as e:
                print(f"[加速] onnxruntime 不可用: {e}, 回退 none")
                return "none"
        if is_linux_cuda and face_model == "yolov8":
            try:
                import torch  # noqa: F401
                return "tensorrt"
            except Exception as e:
                print(f"[加速] tensorrt 不可用: {e}, 回退 none")
                return "none"
        if is_linux_cuda and face_model == "yunet":
            try:
                import onnxruntime as ort
                eps = ort.get_available_providers()
                if "CUDAExecutionProvider" in eps:
                    return "onnx"
                print(f"[加速] onnxruntime 无 CUDAExecutionProvider ({eps}), 回退 none")
                return "none"
            except Exception as e:
                print(f"[加速] onnxruntime 不可用: {e}, 回退 none")
                return "none"
        return "none"

    if accel == "auto":
        return _try_auto()

    # 显式指定后端: 检查依赖是否齐全
    if accel == "coreml":
        try:
            import coremltools  # noqa: F401
            return "coreml"
        except Exception as e:
            print(f"[加速] 显式 coreml 不可用: {e}, 回退 none")
            return "none"
    if accel == "tensorrt":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[加速] 显式 tensorrt 不可用: CUDA 不可用, 回退 none")
                return "none"
            return "tensorrt"
        except Exception as e:
            print(f"[加速] 显式 tensorrt 不可用: {e}, 回退 none")
            return "none"
    if accel == "onnx":
        try:
            import onnxruntime as ort  # noqa: F401
            return "onnx"
        except Exception as e:
            print(f"[加速] 显式 onnx 不可用: {e}, 回退 none")
            return "none"
    # 未知值视为 none
    print(f"[加速] 未知后端 {accel}, 回退 none")
    return "none"


def _export_yolo_model(model_path, fmt, imgsz, half, model_dir):
    """调用 ultralytics YOLO.export 导出加速后端模型, 返回导出文件绝对路径。

    fmt: "coreml"|"engine"|"onnx" 等
    imgsz: 输入尺寸(整数)
    half: 是否 fp16
    model_dir: 缓存目录优先级 model_dir > CWD > 脚本目录

    导出文件名规则: {stem}_{imgsz}_{'fp16' if half else 'fp32'}.{ext}
      coreml -> .mlpackage (目录)
      engine -> .engine
      其他 -> 按 ultralytics 默认后缀
    检测文件已存在则直接返回路径, 不重复导出。
    """
    from ultralytics import YOLO

    ext_map = {"coreml": ".mlpackage", "engine": ".engine"}
    ext = ext_map.get(fmt, "")  # 其他格式由 ultralytics 自行决定后缀
    stem = os.path.splitext(os.path.basename(model_path))[0]
    precision = "fp16" if half else "fp32"

    # 缓存目录优先级
    cache_dirs = []
    if model_dir:
        cache_dirs.append(model_dir)
    cache_dirs.append(os.getcwd())
    cache_dirs.append(os.path.dirname(os.path.abspath(__file__)))

    if ext:
        target_name = f"{stem}_{imgsz}_{precision}{ext}"
        for d in cache_dirs:
            target = os.path.join(d, target_name)
            # coreml .mlpackage 是目录, engine 是文件, 两者都用 exists 检查
            if os.path.exists(target):
                return os.path.abspath(target)

    # 调用 ultralytics 导出
    # 新版 ultralytics(8.4+) half 已废弃改用 quantize; 旧版仍用 half
    # 先尝试新 API, 失败回退旧 API
    model = YOLO(model_path)
    export_kwargs = {"format": fmt, "imgsz": imgsz}
    if half:
        export_kwargs["half"] = True  # 旧版参数(新版会 warning 但仍可用)
    try:
        exported_path = model.export(**export_kwargs)
    except (TypeError, Exception) as e:
        # half 参数不被接受时回退到 quantize
        if "half" in str(e) or "quantize" in str(e):
            export_kwargs.pop("half", None)
            export_kwargs["quantize"] = "16" if half else None
            export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None}
            exported_path = model.export(**export_kwargs)
        else:
            raise

    # exported_path 是 ultralytics 返回的实际导出路径(字符串)
    if not exported_path or not os.path.exists(exported_path):
        raise RuntimeError(f"YOLO 导出失败: fmt={fmt}, imgsz={imgsz}, half={half}")

    # 重命名为带 imgsz+precision 后缀的缓存名, 避免不同尺寸互相覆盖
    target_dir = cache_dirs[0] if cache_dirs else os.getcwd()
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    if ext:
        target_path = os.path.join(target_dir, f"{stem}_{imgsz}_{precision}{ext}")
    else:
        # 其他格式沿用 ultralytics 默认文件名
        target_path = os.path.join(target_dir, os.path.basename(exported_path))

    if os.path.abspath(exported_path) != os.path.abspath(target_path):
        try:
            import shutil as _sh
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    _sh.rmtree(target_path)
                else:
                    os.remove(target_path)
            _sh.move(exported_path, target_path)
        except Exception:
            # 移动失败, 用 ultralytics 默认路径
            target_path = exported_path
    return os.path.abspath(target_path)


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

        # GPU 加速: YuNet 通过 OpenCV DNN 后端, 可尝试 CUDA; 失败则默认 CPU
        if use_gpu:
            try:
                self._detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                pass

        print(f"[人脸] YuNet (输入{self.det_size})")

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
        if img is None or img.size == 0:
            return []

        # stream=True: 生成器模式, 减少结果对象包装开销
        results = self._model(img, imgsz=self.yolo_size, conf=conf,
                              device=self.device, verbose=False, stream=True)
        bboxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                if cls_id != 0:  # class 0 = face (yolov8n-face 单类别模型)
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h, w = img.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    bboxes.append((x1, y1, x2, y2))
        return bboxes


class ONNXYuNetDetector:
    """YuNet 人脸检测器 — ONNX Runtime / OpenCL 加速后端版。

    简化方案: 复用 cv2.FaceDetectorYN 的完整后处理, 仅切换推理后端:
      - macOS: DNN_BACKEND_INFERENCE_ENGINE + DNN_TARGET_OPENCL_FP16 (OpenVINO/OpenCL)
      - Linux+CUDA: DNN_BACKEND_CUDA + DNN_TARGET_CUDA (与 YuNetFaceDetector 一致)
      - 其他: 回退默认(cv2 默认后端)

    detect() 行为与 YuNetFaceDetector 完全一致, 返回 [(x1,y1,x2,y2), ...]。
    """

    def __init__(self, model_dir=None, yunet_size=FACE_YUNET_SIZE, use_gpu=True):
        # 延迟 import: 无 onnxruntime 时本类不会被实例化(由 _create_detector 工厂保证)
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "缺少 onnxruntime。请安装：pip install onnxruntime"
            ) from exc

        self.det_size = YuNetFaceDetector._normalize_size(yunet_size)
        model_path = YuNetFaceDetector._find_model(model_dir)

        try:
            if model_path:
                self._detector = cv2.FaceDetectorYN.create(
                    model_path, "",
                    (self.det_size, self.det_size),
                    score_threshold=FACE_CONF,
                )
            else:
                self._detector = cv2.FaceDetectorYN.create(
                    YUNET_FILE, "",
                    (self.det_size, self.det_size),
                    score_threshold=FACE_CONF,
                )
        except Exception as e:
            raise RuntimeError(
                f"YuNet-ONNX 模型加载失败: {e}。请下载 {YUNET_FILE} 放到 "
                f"--model-dir 指定目录或脚本当前目录。"
            ) from e

        # 切换推理后端
        providers_used = "default"
        if sys.platform == "darwin":
            # macOS: 优先 OpenVINO/OpenCL FP16
            try:
                self._detector.setPreferableBackend(
                    cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                self._detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
                providers_used = "OpenCL_FP16"
            except Exception:
                try:
                    self._detector.setPreferableBackend(
                        cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
                    self._detector.setPreferableTarget(
                        cv2.dnn.DNN_TARGET_OPENCL)
                    providers_used = "OpenCL"
                except Exception:
                    providers_used = "default(opencl 不可用)"
        elif sys.platform.startswith("linux") and use_gpu:
            # Linux + CUDA: 同 YuNetFaceDetector
            try:
                self._detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                providers_used = "CUDA"
            except Exception:
                providers_used = "default(cuda 不可用)"
        else:
            providers_used = "default"

        print(f"[人脸] YuNet-ONNX (输入{self.det_size}, EP={providers_used})")

    def detect(self, img, conf=FACE_CONF):
        """返回 bbox 列表, 格式与 YuNetFaceDetector.detect() 完全一致。"""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
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


class CoreMLYOLODetector:
    """YOLOv8 人脸检测器 — Apple CoreML 后端版(macOS 专用)。

    与 YOLOFaceDetector 同接口: detect(img, conf) -> [(x1,y1,x2,y2),...]
    通过 ultralytics 导出 .mlpackage, 用 coremltools 加载并推理。
    """

    def __init__(self, model_dir=None, yolo_size=YOLO_INPUT_SIZE, use_gpu=True):
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError(
                "缺少 coremltools。请安装：pip install coremltools"
            ) from exc
        self._ct = ct

        self.yolo_size = YOLOFaceDetector._normalize_size(yolo_size)
        pt_path = YOLOFaceDetector._find_model(model_dir)
        if not pt_path:
            # ultralytics 自动下载路径无法直接 export, 给出明确错误
            raise RuntimeError(
                f"未找到 {YOLO_FACE_FILE}, 无法导出 CoreML 模型。"
                f"请将 .pt 文件放到 --model-dir 或 CWD 后重试。"
            )

        # 导出/加载 .mlpackage (fp16, macOS 推荐配置)
        mlpackage_path = _export_yolo_model(
            pt_path, "coreml", self.yolo_size, half=True, model_dir=model_dir)

        try:
            self._model = ct.models.MLModel(mlpackage_path)
        except Exception as e:
            raise RuntimeError(f"CoreML 模型加载失败: {e}") from e

        # 预热: 跑一次空推理避免首帧卡顿
        dummy = np.zeros((self.yolo_size, self.yolo_size, 3), dtype=np.uint8)
        try:
            self._predict(dummy)
        except Exception:
            pass  # 预热失败可忽略, 后续推理报错才真正失败

        print(f"[人脸] YOLOv8-CoreML (输入{self.yolo_size})")

    @staticmethod
    def _letterbox(img, new_shape):
        """ultralytics 风格的 letterbox resize。返回 (out, (r, (left, top)))。"""
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        nh, nw = int(h * r), int(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_h = new_shape[0] - nh
        pad_w = new_shape[1] - nw
        top, left = pad_h // 2, pad_w // 2
        out = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return out, (r, (left, top))

    def _predict(self, img_letterboxed_rgb_chw_f32):
        """调用 coremltools predict; 处理可能的多种输入键名。"""
        arr = img_letterboxed_rgb_chw_f32
        # 试探常见输入键名
        last_err = None
        for key in ("image", "images", "input"):
            try:
                return self._model.predict({key: arr})
            except Exception as e:
                last_err = e
        # 找 spec 里的真正输入名
        try:
            spec = self._model.get_spec()
            inputs = spec.description.input
            for inp in inputs:
                name = inp.name
                try:
                    return self._model.predict({name: arr})
                except Exception as e:
                    last_err = e
        except Exception as e:
            last_err = e
        raise RuntimeError(f"CoreML predict 失败: {last_err}")

    def detect(self, img, conf=FACE_CONF):
        """返回 bbox 列表, 格式与 YOLOFaceDetector.detect() 完全一致。"""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]

        # 预处理: letterbox → BGR2RGB → /255 → HWC2CHW → float32
        lb, (r, (left, top)) = self._letterbox(img, (self.yolo_size, self.yolo_size))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))            # HWC -> CHW
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        # 加 batch 维
        arr = arr[np.newaxis, ...]                      # (1, 3, H, W)

        try:
            out_dict = self._predict(arr)
        except Exception:
            return []

        if not isinstance(out_dict, dict):
            return []

        # 后处理: coreml 导出的 YOLOv8 输出可能是已解码的 boxes + confs, 也可能是原始 tensor
        # 通用策略: 找形状最接近 (1, N, 4) 的当 boxes, (1, N, num_classes) 的当 confs
        boxes_arr = None
        confs_arr = None
        for key, val in out_dict.items():
            v = np.asarray(val)
            if v.ndim == 3 and v.shape[2] == 4 and boxes_arr is None:
                boxes_arr = v
            elif v.ndim == 3 and v.shape[2] >= 1 and confs_arr is None and v.shape[2] != 4:
                confs_arr = v
            elif v.ndim == 2 and v.shape[1] == 4 and boxes_arr is None:
                boxes_arr = v[np.newaxis, ...]
            elif v.ndim == 2 and v.shape[1] >= 1 and confs_arr is None and v.shape[1] != 4:
                confs_arr = v[np.newaxis, ...]
            elif v.ndim == 3 and v.shape[2] == 6 and boxes_arr is None:
                # [x1,y1,x2,y2,conf,cls] 合并输出格式
                boxes_arr = v[..., :4]
                confs_arr = v[..., 4:5]

        if boxes_arr is None or confs_arr is None:
            # 找不到匹配形状, 兜底: 单输出已是 [x1,y1,x2,y2,conf,cls]
            for key, val in out_dict.items():
                v = np.asarray(val)
                if v.ndim >= 2 and v.shape[-1] >= 6:
                    boxes_arr = v[..., :4]
                    confs_arr = v[..., 4:5]
                    break
            if boxes_arr is None:
                return []

        # boxes_arr: (1, N, 4) or (N, 4); confs_arr: (1, N, num_classes) or (N, num_classes)
        b = boxes_arr.reshape(-1, 4)
        c = confs_arr.reshape(confs_arr.shape[1] if confs_arr.ndim == 3 else confs_arr.shape[0],
                               -1) if confs_arr.ndim >= 2 else confs_arr.reshape(-1, 1)
        # 确保对齐
        n = min(b.shape[0], c.shape[0])
        b = b[:n]
        c = c[:n]
        scores = c.max(axis=1) if c.shape[1] > 1 else c[:, 0]

        # 过滤 conf, 仅取 cls==0(人脸); 若 confs 单类(=人脸)则只按分数过滤
        mask = scores >= conf
        if c.shape[1] > 1:
            cls_ids = c.argmax(axis=1)
            mask &= (cls_ids == 0)
        b = b[mask]
        scores = scores[mask]
        if len(b) == 0:
            return []

        # NMS 简化版(置信度排序 + IoU 抑制)
        idxs = np.argsort(-scores)
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            iou = self._iou(b[i], b[idxs[1:]])
            idxs = idxs[1:][iou < 0.45]

        # 坐标反映射回原图
        bboxes = []
        for i in keep:
            x1, y1, x2, y2 = b[i].tolist()
            x1 = (x1 - left) / r
            y1 = (y1 - top) / r
            x2 = (x2 - left) / r
            y2 = (y2 - top) / r
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))
        return bboxes

    @staticmethod
    def _iou(box, others):
        """numpy 向量化 IoU。box=(4,), others=(N,4)。"""
        x1 = np.maximum(box[0], others[:, 0])
        y1 = np.maximum(box[1], others[:, 1])
        x2 = np.minimum(box[2], others[:, 2])
        y2 = np.minimum(box[3], others[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        union = area1 + area2 - inter
        return np.where(union > 0, inter / union, 0.0)


class TensorRTYOLODetector:
    """YOLOv8 人脸检测器 — NVIDIA TensorRT 后端版(Linux+CUDA 专用)。

    与 YOLOFaceDetector 同接口: detect(img, conf) -> [(x1,y1,x2,y2),...]
    通过 ultralytics 导出 .engine, 仍用 ultralytics 推理 API 解码。
    """

    def __init__(self, model_dir=None, yolo_size=YOLO_INPUT_SIZE, use_gpu=True):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用")
        except ImportError as exc:
            raise RuntimeError(
                "缺少 torch。请安装：pip install torch --index-url "
                "https://download.pytorch.org/whl/cu118"
            ) from exc
        except RuntimeError as e:
            raise RuntimeError(f"TensorRT 后端不可用: {e}") from e

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ultralytics。请安装：pip install ultralytics"
            ) from exc

        self.yolo_size = YOLOFaceDetector._normalize_size(yolo_size)
        pt_path = YOLOFaceDetector._find_model(model_dir)
        if not pt_path:
            raise RuntimeError(
                f"未找到 {YOLO_FACE_FILE}, 无法导出 TensorRT 模型。"
                f"请将 .pt 文件放到 --model-dir 或 CWD 后重试。"
            )

        # 导出/加载 .engine (fp16)
        engine_path = _export_yolo_model(
            pt_path, "engine", self.yolo_size, half=True, model_dir=model_dir)

        try:
            self._model = YOLO(engine_path, task="detect")
        except Exception as e:
            raise RuntimeError(f"TensorRT 模型加载失败: {e}") from e

        # 预热: 跑一次空推理避免首帧卡顿
        try:
            dummy = np.zeros((self.yolo_size, self.yolo_size, 3), dtype=np.uint8)
            self._model(dummy, imgsz=self.yolo_size, conf=0.5,
                        device="cuda:0", verbose=False, stream=True)
        except Exception:
            pass

        print(f"[人脸] YOLOv8-TensorRT (输入{self.yolo_size})")

    def detect(self, img, conf=FACE_CONF):
        """返回 bbox 列表, 格式与 YOLOFaceDetector.detect() 完全一致。"""
        if img is None or img.size == 0:
            return []

        # 复用 ultralytics 统一接口
        results = self._model(img, imgsz=self.yolo_size, conf=conf,
                              device="cuda:0", verbose=False, stream=True)
        bboxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                if cls_id != 0:  # class 0 = face (yolov8n-face 单类别模型)
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h, w = img.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    bboxes.append((x1, y1, x2, y2))
        return bboxes


class FaceProcessor:
    """人脸检测+光流跟踪：关键帧检测，中间帧LK光流跟踪。支持多人。"""
    def __init__(self, detector, detect_int=FACE_DETECT_INT, grace=FACE_GRACE, conf=FACE_CONF):
        self.detector = detector
        self.detect_int = max(1, detect_int)
        self.conf = conf
        self.grace = grace          # 检测失败时沿用旧框的帧数
        self.miss_count = 0         # 连续检测失败的周期数
        self.last_faces = []
        self.prev_gray = None
        self.face_pts = {}
        # 优化 LK 参数: 更小搜索窗 + 更少金字塔层数 → 每帧 tracking 提速约 30-40%
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
        # 网格采样点数: 3×3=9 点, 替代 goodFeaturesToTrack 的 corner detection 开销
        # 9 点 median 估计位移仍稳健, 比 5×5=25 点 LK 提速约 2x
        self._grid_rows, self._grid_cols = 3, 3

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

    def process(self, img, frame_idx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        need_detect = ((frame_idx - 1) % self.detect_int == 0) or not self.last_faces
        if need_detect:
            dets = self.detector.detect(img, conf=self.conf)
            # 多人场景: 保留全部检测结果, 不再用单人 Tracker 过滤
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
            for i, (x1, y1, x2, y2) in enumerate(self.last_faces):
                if i not in self.face_pts or len(self.face_pts[i]) < 4:
                    tracked.append((x1, y1, x2, y2))
                    continue
                pts = self.face_pts[i]
                npts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts, None, **self.lk)
                good_mask = status.flatten() == 1
                if good_mask.sum() >= 4:
                    good = npts[good_mask].reshape(-1, 2)
                    old_flat = pts[good_mask].reshape(-1, 2)
                    dx = float(np.median(good[:, 0] - old_flat[:, 0]))
                    dy = float(np.median(good[:, 1] - old_flat[:, 1]))
                    tracked.append((int(x1 + dx), int(y1 + dy),
                                    int(x2 + dx), int(y2 + dy)))
                    new_pts[i] = good.reshape(-1, 1, 2)
                else:
                    tracked.append((x1, y1, x2, y2))
            self.face_pts = new_pts
            self.last_faces = tracked
        self.prev_gray = gray
        return self.last_faces


# ================= 视频处理 =================

def _create_detector(face_model, accel="auto", model_dir=None,
                     face_size=FACE_YUNET_SIZE, use_gpu=True, log=print):
    """检测器工厂: 按 (face_model, accel) 路由选择后端。

    若加速后端不可用则回退到 (face_model, none) 对应的原检测器。
    log: 日志回调, 默认 print
    """
    backend = _select_accel_backend(accel, face_model, use_gpu)

    def _fallback_default(reason):
        if reason:
            log(f"  [加速] {reason}; 回退到原检测器")
        if face_model == "yolov8":
            return YOLOFaceDetector(
                model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        return YuNetFaceDetector(
            model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)

    try:
        if face_model == "yunet":
            if backend == "onnx":
                return ONNXYuNetDetector(
                    model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
            # backend in {"none", "coreml", "tensorrt"}: yunet 仅支持 none/onnx
            return YuNetFaceDetector(
                model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
        elif face_model == "yolov8":
            if backend == "coreml":
                return CoreMLYOLODetector(
                    model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
            if backend == "tensorrt":
                return TensorRTYOLODetector(
                    model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
            # backend in {"none", "onnx"}: yolov8 默认走 PyTorch 原检测器
            return YOLOFaceDetector(
                model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        else:
            return _fallback_default(f"未知 face_model={face_model}")
    except Exception as e:
        log(f"  [加速] 后端 {backend} 初始化失败: {e}")
        return _fallback_default(None)
    finally:
        log(f"  [加速] 后端={backend}")


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


def _build_encode_cmd(w, h, fps, frame_skip, src, has_aud, dst, force_h264, log=None):
    """构建 ffmpeg 编码命令(stdin ← raw BGR, 可选第二输入 ← 源音频)。

    供 _process_pipe 和 _process_segment 共用, 保证编码参数一致。
    has_aud=False 时不含音频流(段处理用, 音频在合并时从源复用)。
    """
    if log is None:
        log = lambda *a, **kw: None

    hw = find_hw_encoder(family="h264" if force_h264 else "hevc")

    enc_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s:v", f"{w}x{h}", "-r", str(fps / frame_skip), "-i", "pipe:0"]
    if has_aud:
        enc_cmd += ["-i", src]  # 第二输入: 内联读取音频, 省去临时抽取
    # 输出选项(必须在所有 -i 之后)
    if nvenc_available():
        # NVENC 调优参数: p4 速度优先, hq 画质微调, vbr+cq20 近无损
        enc_cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                    "-rc", "vbr", "-cq", "20", "-b:v", "0"]
        log("  [编码] NVENC GPU 硬件编码 (h264_nvenc, preset=p4)")
    elif hw:
        bitrate = hw_bitrate(w, h, fps, hw, get_bitrate(src))
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
    return enc_cmd


def _process_pipe(src, dst, face_on, model_dir, face_size,
                  face_int, face_conf, face_model, keep_tmp, force_h264, use_gpu,
                  frame_skip, fisheye, fisheye_strength, fisheye_device,
                  fisheye_downscale, log, accel="auto"):
    """流式管道: ffmpeg解码(rawvideo pipe) → Python处理 → ffmpeg编码, 全程0磁盘IO。

    三级流水线(读帧线程 → 主线程处理 → 写帧线程), 解码/编码与 Python 处理并行,
    相比串行模式提速 20-40%。音频直接从源文件内联输入, 无需临时抽取。
    frame_skip: 跳帧间隔, 管道模式通过跳过读取来实现。
    accel: 检测器加速后端(auto/onnx/coreml/tensorrt/none), 由 _create_detector 解析。
    """
    t0 = time.time()
    w, h, fps, rot = get_video_info(src)
    has_aud = has_audio(src)

    log(f"  [管道模式] {w}x{h} {fps:.1f}fps 音轨={'有' if has_aud else '无'}"
        + (f" 旋转{rot}°" if rot else ""))

    # 初始化检测器 — 由 _create_detector 工厂按 (face_model, accel) 路由选择后端
    fd = _create_detector(face_model, accel=accel, model_dir=model_dir,
                          face_size=face_size, use_gpu=use_gpu, log=log)
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf) if face_on else None

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
    enc_cmd = _build_encode_cmd(w, h, fps, frame_skip, src, has_aud, dst,
                                force_h264, log=log)
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
        """消费者: 从写队列取帧, 写入 ffmpeg stdin。"""
        try:
            while True:
                item = write_q.get()
                if item is None:  # 哨兵
                    break
                encode.stdin.write(item)
        except BrokenPipeError:
            pass
        except Exception as e:
            pipeline_error.append(("writer", e))
        finally:
            try:
                encode.stdin.close()
            except BrokenPipeError:
                pass

    reader_thread = threading.Thread(target=_reader, daemon=True)
    writer_thread = threading.Thread(target=_writer, daemon=True)
    reader_thread.start()
    writer_thread.start()

    # 主循环: 取帧 → 处理 → 入写队列
    frame_idx = 0
    total_face = 0
    try:
        while True:
            raw = read_q.get()
            if raw is None:
                break
            frame_idx += 1
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

            # 鱼眼去畸变: 矫正后再检测人脸, 大幅提升识别率
            if fisheye:
                img = fisheye_undistort(img, fisheye_strength, fisheye_device,
                                        downscale=fisheye_downscale)

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
            write_q.put(memoryview(img))

            if frame_idx % 50 == 0:
                log(f"    [{frame_idx}] 人脸={len(faces)} "
                    f"elapsed={time.time()-t0:.0f}s")
    except BrokenPipeError:
        log("  [警告] 管道中断")
    finally:
        # 通知 writer 结束, 等待两个线程退出
        write_q.put(None)
        writer_thread.join(timeout=30)
        reader_thread.join(timeout=10)
        extract.stdout.close()
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
    duration = get_duration(src)
    if duration and duration > 5:
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
                   fisheye_downscale, log, accel="auto"):
    """文件模式: ffmpeg抽帧JPEG → 打码 → 重新编码。

    frame_skip 控制"每隔多少帧抽一帧"(1=逐帧, 2=隔1抽1提速2x)。
    跳帧时输出帧率同步降低(fps/frame_skip), 保持视频时长不变。
    accel: 检测器加速后端, 由 _create_detector 解析。
    """
    tmp = tempfile.mkdtemp(prefix="vmb_")
    fin, fout = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    os.makedirs(fin)
    os.makedirs(fout)
    t0 = time.time()
    fps = get_fps(src)
    w, h, _, _ = get_video_info(src)

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

    # 初始化检测器 — 由 _create_detector 工厂按 (face_model, accel) 路由选择后端
    fd = _create_detector(face_model, accel=accel, model_dir=model_dir,
                          face_size=face_size, use_gpu=use_gpu, log=log)
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf) if face_on else None
    total_face = 0

    for i, fn in enumerate(frames, 1):
        img = cv2.imread(os.path.join(fin, fn))
        if img is None:
            continue
        # 鱼眼去畸变
        if fisheye:
            img = fisheye_undistort(img, fisheye_strength, fisheye_device,
                                    downscale=fisheye_downscale)
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
        vcmd += ["-c:v", hw, "-b:v", str(hw_bitrate(w, h, fps, hw, get_bitrate(src)))]
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


# ================= 切片并行处理 =================

# worker 进程全局变量(每个 worker 进程独立, 避免重复加载检测器)
_WORKER_DETECTOR = None
_WORKER_FACE_PROC = None


def _worker_init(face_model, model_dir, face_size, use_gpu, face_int, face_conf,
                 face_on, accel="auto"):
    """ProcessPoolExecutor initializer: 每个 worker 进程启动时加载一次检测器。

    后续 _process_segment 调用直接复用 _WORKER_DETECTOR / _WORKER_FACE_PROC,
    避免每段重复加载模型(模型加载耗时 1-3s)。
    accel: 由 _create_detector 工厂解析的加速后端(auto/onnx/coreml/tensorrt/none)。
    """
    global _WORKER_DETECTOR, _WORKER_FACE_PROC
    if face_on:
        _WORKER_DETECTOR = _create_detector(
            face_model, accel=accel, model_dir=model_dir,
            face_size=face_size, use_gpu=use_gpu, log=print)
        _WORKER_FACE_PROC = FaceProcessor(
            _WORKER_DETECTOR, detect_int=face_int, conf=face_conf)
    print("[worker] 检测器已加载")


def _slice_video(src, duration, segment_duration, fps, frame_skip, detect_int, grace):
    """按 segment_duration 切片, 段间重叠 overlap_frames 帧保证跟踪连续性。

    返回 [(start_time, end_time, overlap_frames), ...]
    - seg 0: start=0, overlap=0
    - seg N (N>=1): start=N*segment_duration - overlap_time, overlap=overlap_frames
    - overlap = ceil((detect_int + grace) / frame_skip) * frame_skip, 最少 10
    - overlap_time = overlap / fps
    """
    if not duration or duration <= 0:
        return [(0.0, 0.0, 0)]
    if duration <= segment_duration:
        return [(0.0, duration, 0)]

    overlap = max(10, math.ceil((detect_int + grace) / frame_skip) * frame_skip)
    overlap_time = overlap / fps if fps > 0 else 0

    segments = []
    n = 0
    while n * segment_duration < duration:
        if n == 0:
            start = 0.0
            ov = 0
        else:
            start = max(0.0, n * segment_duration - overlap_time)
            ov = overlap
        end = min((n + 1) * segment_duration, duration)
        segments.append((start, end, ov))
        n += 1
    return segments


def _merge_segments(segment_files, dst, src, log):
    """用 ffmpeg concat demuxer 合并各段, 音轨从源文件复用。

    无损合并(-c copy)后校验时长与源一致(误差 <1 帧), 失败回退重编码合并。
    返回是否成功。
    """
    if not segment_files:
        return False

    # 写 concat 列表文件(用绝对路径避免 concat demuxer 路径解析问题)
    list_path = dst + ".concat.txt"
    with open(list_path, "w") as f:
        for seg in segment_files:
            f.write(f"file '{os.path.abspath(seg)}'\n")

    src_duration = get_duration(src)
    fps = get_fps(src)
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30

    # 无损合并: concat demuxer + 复用源音轨
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", list_path,
           "-i", src,
           "-map", "0:v", "-map", "1:a?",
           "-c", "copy", "-c:a", "copy",
           "-movflags", "+faststart", dst]
    r = _run(cmd)

    if r.returncode == 0:
        merged_duration = get_duration(dst)
        if merged_duration and src_duration:
            # 容忍 2 秒差异: 精确 seek + 重叠帧丢弃后仍可能有编码器 padding 误差
            # 严格 1 帧阈值会导致大量误报回退重编码, 浪费时间
            if abs(merged_duration - src_duration) < 2.0:
                os.unlink(list_path)
                log(f"  [合并] 无损合并成功 ({merged_duration:.2f}s vs 源 {src_duration:.2f}s)")
                return True
            else:
                log(f"  [合并] 时长不匹配 ({merged_duration:.2f}s vs {src_duration:.2f}s), 回退重编码")
        else:
            os.unlink(list_path)
            log("  [合并] 无损合并成功(无法校验时长)")
            return True
    else:
        log(f"  [合并] 无损合并失败: {r.stderr[-300:]}")

    # 回退重编码合并
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", list_path,
           "-i", src,
           "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "copy",
           "-movflags", "+faststart", dst]
    r = _run(cmd)
    os.unlink(list_path)
    if r.returncode != 0:
        log(f"  [合并] 重编码合并失败: {r.stderr[-300:]}")
        return False
    log("  [合并] 重编码合并成功")
    return True


def _process_segment(args):
    """worker 入口: 处理单个视频段, 输出到临时 mp4。

    args: dict, 包含 src, seg_idx, start_time, end_time, overlap_frames,
          fps, w, h, rot, has_aud, face_on, fisheye, fisheye_strength,
          fisheye_device, fisheye_downscale, force_h264, frame_skip, tmp_dir
    返回: (seg_idx, output_path, frame_count, encoder_params)
    """
    global _WORKER_DETECTOR, _WORKER_FACE_PROC

    src = args["src"]
    seg_idx = args["seg_idx"]
    start_time = args["start_time"]
    end_time = args["end_time"]
    overlap_frames = args["overlap_frames"]
    fps = args["fps"]
    w = args["w"]
    h = args["h"]
    rot = args["rot"]
    face_on = args["face_on"]
    fisheye = args["fisheye"]
    fisheye_strength = args["fisheye_strength"]
    fisheye_device = args["fisheye_device"]
    fisheye_downscale = args["fisheye_downscale"]
    force_h264 = args["force_h264"]
    frame_skip = args["frame_skip"]
    tmp_dir = args["tmp_dir"]

    out_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}.mp4")
    t0 = time.time()
    duration = end_time - start_time

    print(f"[worker] 段 {seg_idx}: [{start_time:.2f}s, {end_time:.2f}s] "
          f"overlap={overlap_frames}帧 时长={duration:.2f}s")

    # 构造解码命令: 精确 seek(-ss 在 -i 之后, 帧级精确) + -t 限定段时长
    # 注意: -ss 在 -i 之后是精确 seek(慢但准), 在 -i 之前是快速 seek(快但只到关键帧)
    # 切片并行场景必须精确 seek, 否则各段起点偏移累积导致合并时长不匹配
    extract_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if cuda_decode_available():
        extract_cmd += ["-hwaccel", "cuda"]
    extract_cmd += ["-i", src]
    if seg_idx == 0:
        extract_cmd += ["-ss", "0"]
    else:
        # 从段开始前 overlap_time 处开始读, 读完整个段(含重叠)
        # overlap 帧由后续 select 滤镜丢弃
        extract_cmd += ["-ss", str(start_time)]
    extract_cmd += ["-t", str(duration)]

    filters = []
    transpose = get_transpose_filter(rot)
    if transpose:
        filters.append(transpose)
    # 跳过段开头重叠帧(seg_idx>=1) + 跳帧, 合并为单个 select 表达式
    if overlap_frames > 0 and frame_skip > 1:
        filters.append(
            f"select=gte(n\\,{overlap_frames})*not(mod(n\\,{frame_skip}))")
    elif overlap_frames > 0:
        filters.append(f"select=gte(n\\,{overlap_frames})")
    elif frame_skip > 1:
        filters.append(f"select=not(mod(n\\,{frame_skip}))")
    if filters:
        filters.append("setpts=N/FRAME_RATE/TB")
        extract_cmd += ["-vf", ",".join(filters)]
    extract_cmd += ["-map", "0:v:0", "-an", "-sn", "-dn",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    extract = subprocess.Popen(
        extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # 构造编码命令(段不含音频, 音频在合并时从源复用)
    enc_cmd = _build_encode_cmd(w, h, fps, frame_skip, src, False, out_path,
                                force_h264, log=print)
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
        data = stream.read(size)
        if data is None:
            return b""
        remaining = size - len(data)
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        return data

    # 三级流水线: 读帧线程 → 主线程处理 → 写帧线程
    read_q = queue.Queue(maxsize=3)
    write_q = queue.Queue(maxsize=3)
    frame_size = w * h * 3
    pipeline_error = []

    def _reader():
        try:
            while True:
                raw = _read_frame(extract.stdout, frame_size)
                if len(raw) < frame_size:
                    break
                read_q.put(raw)
        except Exception as e:
            pipeline_error.append(("reader", e))
        finally:
            read_q.put(None)

    def _writer():
        try:
            while True:
                item = write_q.get()
                if item is None:
                    break
                encode.stdin.write(item)
        except BrokenPipeError:
            pass
        except Exception as e:
            pipeline_error.append(("writer", e))
        finally:
            try:
                encode.stdin.close()
            except BrokenPipeError:
                pass

    reader_thread = threading.Thread(target=_reader, daemon=True)
    writer_thread = threading.Thread(target=_writer, daemon=True)
    reader_thread.start()
    writer_thread.start()

    # 主循环: 取帧 → 处理 → 入写队列
    face_proc = _WORKER_FACE_PROC
    frame_idx = 0
    total_face = 0
    try:
        while True:
            raw = read_q.get()
            if raw is None:
                break
            frame_idx += 1
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

            # 鱼眼去畸变: 矫正后再检测人脸
            if fisheye:
                img = fisheye_undistort(img, fisheye_strength, fisheye_device,
                                        downscale=fisheye_downscale)

            faces = []
            if face_on and face_proc is not None:
                faces = face_proc.process(img, frame_idx)
                for (x1, y1, x2, y2) in faces:
                    bw, bh = x2 - x1, y2 - y1
                    heavy_mosaic(img, int(x1 - FACE_EXPAND * bw),
                                 int(y1 - FACE_EXPAND * bh),
                                 int(x2 + FACE_EXPAND * bw),
                                 int(y2 + FACE_EXPAND * bh),
                                 FACE_CELLS, FACE_SIGMA)
                total_face += len(faces)

            write_q.put(memoryview(img))
    except BrokenPipeError:
        print(f"[worker] 段 {seg_idx} 管道中断")
    finally:
        write_q.put(None)
        writer_thread.join(timeout=30)
        reader_thread.join(timeout=10)
        extract.stdout.close()
        extract.wait()
        encode.wait()

    # 错误诊断
    if encode.returncode != 0 and enc_err_buf:
        err_tail = b"".join(enc_err_buf).decode()[-500:]
        if err_tail.strip():
            print(f"[worker] 段 {seg_idx} ffmpeg错误: {err_tail}")
    if pipeline_error:
        for who, e in pipeline_error:
            print(f"[worker] 段 {seg_idx} 流水线错误: {who}: {e}")

    elapsed = time.time() - t0
    print(f"[worker] 段 {seg_idx} 完成: {frame_idx} 帧, 人脸帧次={total_face}, 耗时 {elapsed:.0f}s")

    if encode.returncode != 0:
        raise RuntimeError(f"段 {seg_idx} 编码失败 (returncode={encode.returncode})")

    encoder_params = {"w": w, "h": h, "fps": fps / frame_skip, "pix_fmt": "yuv420p"}
    return (seg_idx, out_path, frame_idx, encoder_params)


def _parallel_process_video(src, dst, face_on, model_dir, face_size,
                            face_int, face_conf, face_model, keep_tmp,
                            force_h264, use_gpu, frame_skip, fisheye,
                            fisheye_strength, fisheye_device, fisheye_downscale,
                            parallel, segment_duration, log, use_pipe=True,
                            accel="auto", parallel_max_gpu=2):
    """切片并行处理编排: 切片 → 多 worker 并行处理 → 无损合并。

    GPU + PyTorch 原路径(accel==none)下 parallel 无收益, 降级为单进程;
    GPU + 加速后端(accel!=none)下解除并行限制, 由 parallel_max_gpu 控制上限。
    """
    # 解析实际后端(auto 可能因依赖缺失回退 none)
    actual_accel = _select_accel_backend(accel, face_model, use_gpu)
    if actual_accel != accel:
        log(f"  [加速] auto 解析为 {actual_accel}")

    # GPU 路径降级: 直接调用单进程路径, 避免递归调 process_video
    if use_gpu and actual_accel == "none" and parallel > 1:
        log("[并行] PyTorch GPU 路径下并行无收益, 强制 --parallel 1")
        if use_pipe:
            try:
                return _process_pipe(src, dst, face_on, model_dir, face_size,
                                     face_int, face_conf, face_model,
                                     keep_tmp, force_h264, use_gpu,
                                     frame_skip, fisheye, fisheye_strength,
                                     fisheye_device, fisheye_downscale, log,
                                     accel=actual_accel)
            except Exception as e:
                log(f"  [警告] 管道模式失败({e}), 回退文件模式")
        return _process_files(src, dst, face_on, model_dir, face_size,
                              face_int, face_conf, face_model,
                              keep_tmp, force_h264, use_gpu,
                              frame_skip, fisheye, fisheye_strength,
                              fisheye_device, fisheye_downscale, log,
                              accel=actual_accel)
    elif use_gpu and actual_accel != "none" and parallel > 1:
        actual_parallel = min(parallel, parallel_max_gpu)
        log(f"[并行] 加速后端={actual_accel}, GPU worker 上限={parallel_max_gpu}, 实际并行={actual_parallel}")

    # 获取视频信息
    w, h, fps, rot = get_video_info(src)
    duration = get_duration(src)
    has_aud = has_audio(src)

    # 计算实际并行 worker 数: 加速后端启用时受 parallel_max_gpu 限制
    if use_gpu and actual_accel != "none" and parallel > 1:
        actual_parallel = min(parallel, parallel_max_gpu)
    else:
        actual_parallel = parallel

    # 短视频不切片, 走单进程
    if not duration or duration <= segment_duration:
        log(f"  [并行] 视频时长 {duration}s <= {segment_duration}s, 不切片, 走单进程")
        if use_pipe:
            try:
                return _process_pipe(src, dst, face_on, model_dir, face_size,
                                     face_int, face_conf, face_model,
                                     keep_tmp, force_h264, use_gpu,
                                     frame_skip, fisheye, fisheye_strength,
                                     fisheye_device, fisheye_downscale, log,
                                     accel=actual_accel)
            except Exception as e:
                log(f"  [警告] 管道模式失败({e}), 回退文件模式")
        return _process_files(src, dst, face_on, model_dir, face_size,
                              face_int, face_conf, face_model,
                              keep_tmp, force_h264, use_gpu,
                              frame_skip, fisheye, fisheye_strength,
                              fisheye_device, fisheye_downscale, log,
                              accel=actual_accel)

    # 切片
    segments = _slice_video(src, duration, segment_duration, fps, frame_skip,
                            face_int, FACE_GRACE)
    overlap_info = segments[1][2] if len(segments) > 1 else 0
    log(f"  [并行] 切成 {len(segments)} 段, 每段 {segment_duration}s, 重叠 {overlap_info} 帧")

    # 创建临时目录
    tmp_dir = tempfile.mkdtemp(prefix="vmb_par_")

    # 构造 worker 参数
    segment_args = []
    for seg_idx, (start_time, end_time, overlap_frames) in enumerate(segments):
        segment_args.append({
            "src": src, "seg_idx": seg_idx, "start_time": start_time,
            "end_time": end_time, "overlap_frames": overlap_frames,
            "fps": fps, "w": w, "h": h, "rot": rot, "has_aud": has_aud,
            "face_on": face_on, "fisheye": fisheye,
            "fisheye_strength": fisheye_strength, "fisheye_device": fisheye_device,
            "fisheye_downscale": fisheye_downscale, "force_h264": force_h264,
            "frame_skip": frame_skip, "tmp_dir": tmp_dir,
        })

    # 并行处理
    t0 = time.time()
    results = []
    failed_seg = None
    error_msg = None
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=actual_parallel,
            initializer=_worker_init,
            initargs=(face_model, model_dir, face_size, use_gpu, face_int,
                      face_conf, face_on, actual_accel)
        ) as executor:
            future_to_idx = {
                executor.submit(_process_segment, arg): arg["seg_idx"]
                for arg in segment_args
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                seg_idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    log(f"  [并行] 段 {seg_idx} 完成: {result[2]} 帧")
                except Exception as e:
                    failed_seg = seg_idx
                    error_msg = str(e)
                    log(f"  [并行] 段 {seg_idx} 失败: {e}")
                    for f in future_to_idx:
                        f.cancel()
                    break
    except Exception as e:
        failed_seg = -1
        error_msg = str(e)
        log(f"  [并行] 进程池异常: {e}")

    # 任一段失败则整体失败
    if failed_seg is not None:
        log(f"  [并行] 段 {failed_seg} 失败, 整体失败: {error_msg}")
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # 按 seg_idx 排序
    results.sort(key=lambda x: x[0])
    segment_files = [r[1] for r in results]

    # 合并
    log(f"  [并行] 合并 {len(segment_files)} 段...")
    ok = _merge_segments(segment_files, dst, src, log)

    # 清理临时段文件
    if not keep_tmp:
        for f in segment_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    total_time = time.time() - t0
    log(f"  [并行] 总耗时 {total_time:.0f}s, 各段帧数: {[r[2] for r in results]}")
    return ok


def process_video(src, dst, face_on=True, model_dir=None,
                  face_size=FACE_YUNET_SIZE, face_int=FACE_DETECT_INT,
                  face_conf=FACE_CONF, face_model="yunet",
                  use_pipe=True, keep_tmp=False,
                  force_h264=False, use_gpu=True, frame_skip=FRAME_SKIP,
                  fisheye=False, fisheye_strength=1.0, fisheye_device="pico4",
                  fisheye_downscale=1, log=print,
                  parallel=0, segment_duration=60,
                  accel="auto", parallel_max_gpu=2):
    """处理单个视频: 人脸打码, 保留音轨。返回是否成功。

    parallel>1 时走切片并行路径:
      - GPU + PyTorch 原路径(accel=none): 自动降级为单进程;
      - GPU + 加速后端(accel!=none): 解除并行限制, 上限 parallel_max_gpu;
      - 纯 CPU: 直接走多进程。
    accel: 检测器加速后端(auto/onnx/coreml/tensorrt/none)。
    """
    if parallel > 1:
        return _parallel_process_video(src, dst, face_on, model_dir, face_size,
                                       face_int, face_conf, face_model,
                                       keep_tmp, force_h264, use_gpu,
                                       frame_skip, fisheye, fisheye_strength,
                                       fisheye_device, fisheye_downscale,
                                       parallel, segment_duration, log,
                                       use_pipe=use_pipe,
                                       accel=accel,
                                       parallel_max_gpu=parallel_max_gpu)
    if use_pipe:
        try:
            return _process_pipe(src, dst, face_on, model_dir, face_size,
                                 face_int, face_conf, face_model,
                                 keep_tmp, force_h264, use_gpu,
                                 frame_skip, fisheye, fisheye_strength,
                                 fisheye_device, fisheye_downscale, log,
                                 accel=accel)
        except Exception as e:
            log(f"  [警告] 管道模式失败({e}), 回退文件模式")
    return _process_files(src, dst, face_on, model_dir, face_size,
                          face_int, face_conf, face_model,
                          keep_tmp, force_h264, use_gpu,
                          frame_skip, fisheye, fisheye_strength,
                          fisheye_device, fisheye_downscale, log,
                          accel=accel)


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
    ap.add_argument("--face-model", default="yunet", choices=["yunet", "yolov8"],
                    help="人脸检测模型: yunet(默认,轻量,仅opencv) / yolov8(极速,需ultralytics)")
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
                    help="鱼眼remap降采样倍数(默认1=关闭; 2=在1/2分辨率remap再上采样提速但会降低人脸检出率,不推荐)")
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
    ap.add_argument("--parallel", type=int, default=0,
                    help="并行 worker 数(仅 CPU 路径有效, 0=关闭; GPU 路径自动降级为单进程)")
    ap.add_argument("--segment-duration", type=int, default=60,
                    help="切片时长(秒, 默认60; 仅 --parallel>1 时生效)")
    ap.add_argument("--accel", default="auto",
                    choices=["auto", "onnx", "coreml", "tensorrt", "none"],
                    help="检测器加速后端(auto=自动选择/none=禁用; macOS+yolov8→coreml,"
                         " macOS+yunet→onnx, Linux+CUDA+yolov8→tensorrt, Linux+CUDA+yunet→onnx;"
                         " auto 失败静默回退 none, 行为与原检测器一致)")
    ap.add_argument("--parallel-max-gpu", type=int, default=2,
                    help="加速后端启用(--accel!=none)时 GPU worker 并发上限, 默认2;"
                         " 仅在 --parallel>1 且 --accel 解析出非 none 后端时生效")
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        print("[错误] 未找到任何视频文件")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"待处理 {len(files)} 个视频 -> {args.out_dir}")
    ok = 0
    total_t0 = time.time()
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
                         parallel=args.parallel,
                         segment_duration=args.segment_duration,
                         accel=args.accel,
                         parallel_max_gpu=args.parallel_max_gpu):
            ok += 1
    total_elapsed = time.time() - total_t0
    print(f"\n全部完成: {ok}/{len(files)} 成功, 输出目录: {args.out_dir}, 总耗时: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
