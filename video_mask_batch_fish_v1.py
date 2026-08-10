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
import glob
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
    k = int(sigma * 3) | 1
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


def fisheye_undistort(img, strength=1.0, device="generic"):
    """对单帧做鱼眼去畸变, 返回矫正后的图像。

    支持设备预置(device="pico4" 等)或通用模式手动 strength 调节。
    映射表按 (分辨率, 参数) 缓存, 同一视频只算一次。
    """
    h, w = img.shape[:2]

    # 获取设备参数
    preset = FISHEYE_PRESETS.get(device, FISHEYE_PRESETS["generic"])
    f_ratio = preset["f_ratio"]
    k1 = preset["k1"] * strength
    k2 = preset["k2"] * strength
    balance = preset["balance"]

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

        results = self._model(img, imgsz=self.yolo_size, conf=conf,
                              device=self.device, verbose=False)
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
        # 网格采样点数: 5×5=25 点, 替代 goodFeaturesToTrack 的 corner detection 开销
        self._grid_rows, self._grid_cols = 5, 5

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
        # 在 bbox 内部均匀采样 5x5 网格点
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
                  frame_skip, fisheye, fisheye_strength, fisheye_device, log):
    """流式管道: ffmpeg解码(rawvideo pipe) → Python处理 → ffmpeg编码, 全程0磁盘IO。

    三级流水线(读帧线程 → 主线程处理 → 写帧线程), 解码/编码与 Python 处理并行,
    相比串行模式提速 20-40%。音频直接从源文件内联输入, 无需临时抽取。
    frame_skip: 跳帧间隔, 管道模式通过跳过读取来实现。
    """
    t0 = time.time()
    w, h, fps, rot = get_video_info(src)
    has_aud = has_audio(src)

    log(f"  [管道模式] {w}x{h} {fps:.1f}fps 音轨={'有' if has_aud else '无'}"
        + (f" 旋转{rot}°" if rot else ""))

    # 检测硬件编码器
    hw = find_hw_encoder(family="h264" if force_h264 else "hevc")

    # 初始化检测器 — 根据 face_model 参数选择
    if face_model == "yolov8":
        fd = YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-nano (输入{face_size})")
    else:
        fd = YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YuNet (输入{face_size})")
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
    enc_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s:v", f"{w}x{h}", "-r", str(fps / frame_skip), "-i", "pipe:0"]
    if has_aud:
        enc_cmd += ["-i", src]  # 第二输入: 内联读取音频, 省去临时抽取
    # 输出选项(必须在所有 -i 之后)
    if nvenc_available():
        # NVENC 调优参数(face_gpu.py 移植): p4 速度优先, hq 画质微调, vbr+cq20 近无损
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
                img = fisheye_undistort(img, fisheye_strength, fisheye_device)

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
                   frame_skip, fisheye, fisheye_strength, fisheye_device, log):
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

    if face_model == "yolov8":
        fd = YOLOFaceDetector(model_dir=model_dir, yolo_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YOLOv8-nano (输入{face_size})")
    else:
        fd = YuNetFaceDetector(model_dir=model_dir, yunet_size=face_size, use_gpu=use_gpu)
        log(f"  [人脸] YuNet (输入{face_size})")
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf) if face_on else None
    total_face = 0

    for i, fn in enumerate(frames, 1):
        img = cv2.imread(os.path.join(fin, fn))
        if img is None:
            continue
        # 鱼眼去畸变
        if fisheye:
            img = fisheye_undistort(img, fisheye_strength)
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


def process_video(src, dst, face_on=True, model_dir=None,
                  face_size=FACE_YUNET_SIZE, face_int=FACE_DETECT_INT,
                  face_conf=FACE_CONF, face_model="yunet",
                  use_pipe=True, keep_tmp=False,
                  force_h264=False, use_gpu=True, frame_skip=FRAME_SKIP,
                  fisheye=False, fisheye_strength=1.0, fisheye_device="pico4",
                  log=print):
    """处理单个视频: 人脸打码, 保留音轨。返回是否成功。"""
    if use_pipe:
        try:
            return _process_pipe(src, dst, face_on, model_dir, face_size,
                                 face_int, face_conf, face_model,
                                 keep_tmp, force_h264, use_gpu,
                                 frame_skip, fisheye, fisheye_strength,
                                 fisheye_device, log)
        except Exception as e:
            log(f"  [警告] 管道模式失败({e}), 回退文件模式")
    return _process_files(src, dst, face_on, model_dir, face_size,
                          face_int, face_conf, face_model,
                          keep_tmp, force_h264, use_gpu,
                          frame_skip, fisheye, fisheye_strength,
                          fisheye_device, log)


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
                         fisheye_device=args.fisheye_device):
            ok += 1
    print(f"\n全部完成: {ok}/{len(files)} 成功, 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
