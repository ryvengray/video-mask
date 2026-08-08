#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_mask_batch.py — 视频批量打码（人脸 + 信用卡/银行卡），通用完整版

人脸:
  YuNet 深度模型(优先, 强: 侧脸/小脸/戴帽/低光照) → res10 SSD → Haar 级联(兜底, 零依赖)
  检测间隔+光流跟踪(每3帧检测1次, 中间帧LK跟踪)
信用卡/银行卡:
  OWLv2 深度学习零样本检测(默认, 高准确率, 任意颜色/遮挡) + LK 光流跟踪
  可选几何法(--card-detector geo, 快但准确率一般, 适合边缘完整的卡)

加速:
  管道模式(ffmpeg rawvideo → Python → ffmpeg, 无磁盘I/O)
  硬件编码(macOS VideoToolbox / NVENC / QSV, 自动检测)
  MPS GPU 加速(Apple Silicon OWLv2)
  关键帧检测 + 光流跟踪(人脸+卡)

用法:
  # 批量处理(默认 OWLv2 高精度卡检测)
  python video_mask_batch.py video.mp4
  python video_mask_batch.py ./videos/                     # 整个目录
  python video_mask_batch.py a.mp4 b.mov                    # 多文件

  # 快速模式(几何法卡检测, 适合边缘完整的卡)
  python video_mask_batch.py video.mp4 --card-detector geo

  # 几何法 + 颜色提示(卡被遮挡且知道颜色)
  python video_mask_batch.py video.mp4 --card-detector geo --card-color yellow

  # 其他参数
  python video_mask_batch.py video.mp4 --out-dir ./out      # 输出目录
  python video_mask_batch.py video.mp4 --no-card            # 只打码人脸
  python video_mask_batch.py video.mp4 --no-face            # 只打码卡
  python video_mask_batch.py video.mp4 --card-conf 0.3      # OWLv2置信度阈值

依赖:
  pip install opencv-python numpy
  # OWLv2 卡检测需: pip install torch transformers pillow (模型首次自动下载约600MB)
  # 系统需 ffmpeg(保留音轨)

代码调用:
  from video_mask_batch import process_video
  process_video("in.mp4", "out.mp4", card_detector="owlv2")
"""
import argparse
import glob
import os
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
FACE_INPUT = 400              # 人脸检测输入尺寸(300->400: 小脸/侧脸召回更高)
FACE_CONF = 0.35              # 人脸置信度阈值(0.45->0.35: 侧脸/低头/遮挡帧不漏检, 跟踪器滤误检)
FACE_EXPAND = 0.12            # 人脸打码框外扩比例(确保盖住完整脸)
FACE_YUNET_SIZE = 1280        # YuNet 输入最长边像素(0=原图最准最慢; 1280快且准)
FACE_DETECT_INT = 3           # 人脸检测间隔: 每3帧检测1次, 中间帧光流跟踪(1=每帧检测)
CARD_CELLS, CARD_SIGMA = 6, 35.0
FACE_GRACE = 4               # 人脸漏检沿用旧框帧数(运动时收紧, 防打码框滞后)
CARD_KEY_INT = 10            # OWLv2 每10帧检测一次(运动场景更频繁; 跟踪质量差时自动提前重检)
CARD_OWL_CONF = 0.25          # OWLv2 置信度阈值
CARD_OWL_SIZE = 1280          # OWLv2 输入最长边像素(0=原图; 1280≈提速3-8倍, 精度略降)
GEO_SCORE = 0.8               # 几何法得分阈值
GEO_COLOR_SCORE = 0.4         # 几何法+颜色提示阈值
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
CARD_TEXTS = ["a credit card", "a bank card", "a debit card"]

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


class Tracker:
    """短时漏检沿用上一帧框(几何法/人脸用)"""
    def __init__(self, grace):
        self.grace = grace
        self.last_box = None
        self.miss = 0

    def update(self, dets):
        if dets:
            if self.last_box is not None:
                best, best_iou = dets[0], -1.0
                lx1, ly1, lx2, ly2 = self.last_box
                la = max(1, (lx2 - lx1) * (ly2 - ly1))
                for b in dets:
                    x1, y1, x2, y2 = b
                    ix1, iy1 = max(lx1, x1), max(ly1, y1)
                    ix2, iy2 = min(lx2, x2), min(ly2, y2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    a = max(1, (x2 - x1) * (y2 - y1))
                    u = la + a - inter
                    iou = inter / u if u > 0 else 0
                    if iou > best_iou:
                        best_iou, best = iou, b
            else:
                best = max(dets, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            self.last_box, self.miss = best, 0
            return [best]
        if self.last_box is not None and self.miss < self.grace:
            self.miss += 1
            return [self.last_box]
        if self.last_box is not None:
            self.last_box, self.miss = None, 0
        return []


# ================= 人脸检测 =================

class FaceDetector:
    """人脸检测: YuNet(强: 侧脸/小脸/戴帽/低光照) → res10 SSD → Haar 兜底"""
    def __init__(self, model_dir=None, yunet_size=FACE_YUNET_SIZE):
        self.yunet = None
        self.ssd = None
        self.yunet_size = yunet_size
        # 1. 优先 YuNet(从 model_dir 或 HuggingFace 缓存找)
        yunet_path = None
        if model_dir:
            cand = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
            if os.path.exists(cand):
                yunet_path = cand
        if not yunet_path:
            try:
                # 绕过本机代理(HF 直连可用, 走代理会 502) + 离线模式用缓存
                for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                          "https_proxy", "ALL_PROXY", "all_proxy"):
                    os.environ.pop(k, None)
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from huggingface_hub import hf_hub_download
                yunet_path = hf_hub_download(
                    repo_id="opencv/face_detection_yunet",
                    filename="face_detection_yunet_2023mar.onnx")
            except Exception:
                pass
        if yunet_path and os.path.exists(yunet_path):
            self.yunet = cv2.FaceDetectorYN_create(
                yunet_path, "", (320, 320),
                score_threshold=FACE_CONF, nms_threshold=0.3, top_k=5000)
            print(f"[人脸] YuNet 模型(强: 侧脸/小脸/戴帽/低光照, 输入{yunet_size})")
        # 2. res10 SSD 兜底
        if self.yunet is None and model_dir:
            proto = os.path.join(model_dir, "deploy.prototxt")
            model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(proto) and os.path.exists(model):
                self.ssd = cv2.dnn.readNetFromCaffe(proto, model)
                print("[人脸] res10 SSD(YuNet 未找到)")
        # 3. Haar 最后兜底
        if self.yunet is None and self.ssd is None:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("[人脸] Haar 级联(零依赖)")

    def detect(self, img, conf=FACE_CONF):
        h0, w0 = img.shape[:2]
        # YuNet: 缩放输入提速, 坐标映射回原图
        if self.yunet is not None:
            if self.yunet_size and max(w0, h0) > self.yunet_size:
                scale = self.yunet_size / max(w0, h0)
                small = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
            else:
                small, scale = img, 1.0
            self.yunet.setInputSize((small.shape[1], small.shape[0]))
            self.yunet.setScoreThreshold(conf)
            _, faces = self.yunet.detect(small)
            out = []
            if faces is not None:
                for f in faces:
                    x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                    if scale != 1.0:
                        x, y, w, h = x / scale, y / scale, w / scale, h / scale
                    out.append((int(x), int(y), int(x + w), int(y + h)))
            return out
        # res10 SSD
        if self.ssd is not None:
            s = FACE_INPUT
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (s, s)), 1.0,
                                         (s, s), (104.0, 177.0, 123.0))
            self.ssd.setInput(blob)
            dets = self.ssd.forward()
            out = []
            for i in range(dets.shape[2]):
                if dets[0, 0, i, 2] < conf:
                    continue
                out.append((int(dets[0, 0, i, 3] * w0), int(dets[0, 0, i, 4] * h0),
                            int(dets[0, 0, i, 5] * w0), int(dets[0, 0, i, 6] * h0)))
            return out
        # Haar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [(x, y, x + w, y + h) for (x, y, w, h) in rects]


class FaceProcessor:
    """人脸检测+光流跟踪: 每N帧检测1次, 中间帧LK光流跟随, 减少检测调用"""
    def __init__(self, detector, detect_int=FACE_DETECT_INT, grace=FACE_GRACE, conf=FACE_CONF):
        self.detector = detector
        self.detect_int = max(1, detect_int)
        self.conf = conf
        self.tracker = Tracker(grace)
        self.last_faces = []
        self.prev_gray = None
        self.face_pts = {}
        self.lk = dict(winSize=(31, 31), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def _init_pts(self, gray, box):
        x1, y1, x2, y2 = box
        h_img, w_img = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.01,
                                      minDistance=5, mask=mask, blockSize=7)
        return None if pts is None else pts.reshape(-1, 1, 2)

    def process(self, img, frame_idx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        need_detect = ((frame_idx - 1) % self.detect_int == 0) or not self.last_faces
        if need_detect:
            dets = self.detector.detect(img, conf=self.conf)
            faces = self.tracker.update(dets)
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


# ================= 信用卡检测 =================
# 通道1: OWLv2 深度学习(默认, 高准确率)

class OwlCardDetector:
    """OWLv2 零样本检测, 懒加载(首次自动下载模型约600MB)。
    支持 CUDA/MPS(GPU) 加速 + 降分辨率输入提速, 检测框自动映射回原图坐标。"""
    _inst = None
    _inst_size = None
    _inst_gpu = None

    def __init__(self, owl_size=CARD_OWL_SIZE, use_gpu=True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from PIL import Image
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.torch = torch
        self.Image = Image
        self.owl_size = owl_size
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL)
        # GPU 检测: CUDA(优先) > MPS(Apple Silicon) > CPU(兜底)
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.to("cuda")
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA"
                print(f"[信用卡] OWLv2 模型已加载 (CUDA GPU: {gpu_name})")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.model.to("mps")
                print("[信用卡] OWLv2 模型已加载 (MPS GPU 加速)")
            else:
                self.device = "cpu"
                print("[信用卡] OWLv2 模型已加载 (CPU, 未检测到 GPU)")
        else:
            self.device = "cpu"
            print("[信用卡] OWLv2 模型已加载 (CPU, GPU 已关闭)")

    @classmethod
    def get(cls, owl_size=CARD_OWL_SIZE, use_gpu=True):
        if cls._inst is None or cls._inst_size != owl_size or cls._inst_gpu != use_gpu:
            cls._inst = cls(owl_size, use_gpu)
            cls._inst_size = owl_size
            cls._inst_gpu = use_gpu
        return cls._inst

    def detect(self, img, thr=CARD_OWL_CONF):
        h0, w0 = img.shape[:2]
        image = self.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 降分辨率提速: 最长边缩到 owl_size, 框坐标映射回原图
        if self.owl_size and max(w0, h0) > self.owl_size:
            scale = self.owl_size / max(w0, h0)
            image = image.resize((int(w0 * scale), int(h0 * scale)))
            sw, sh = image.size
        else:
            sw, sh, scale = w0, h0, 1.0
        inputs = self.processor(text=CARD_TEXTS, images=image, return_tensors="pt")
        if self.device in ("cuda", "mps"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        ts = self.torch.tensor([[sh, sw]])
        res = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=thr, target_sizes=ts,
            text_labels=[CARD_TEXTS])[0]
        out = []
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in b.tolist()]
            if scale != 1.0:    # 映射回原图坐标
                x1, y1 = int(x1 / scale), int(y1 / scale)
                x2, y2 = int(x2 / scale), int(y2 / scale)
            out.append((x1, y1, x2, y2, float(s)))
        return out


# 通道2: 几何法(快速备选, 边缘轮廓 + 可选颜色)

CARD_ASPECT = 1.586
GEO_COLORS = {"yellow": (14, 45, 50, 80), "gold": (14, 45, 50, 80),
              "blue": (95, 130, 50, 50), "green": (45, 85, 50, 50),
              "orange": (10, 24, 60, 100)}


def _tex_std(gray, x1, y1, x2, y2):
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 8:
        return 0.0
    roi = gray[int(y1 + 0.15 * h):int(y2 - 0.15 * h), int(x1 + 0.15 * w):int(x2 - 0.15 * w)]
    return float(roi.std()) if roi.size else 0.0


def detect_cards_geo(img, color_hint=None):
    """几何法卡检测, 返回 [(x1,y1,x2,y2,score), ...]"""
    H, W = img.shape[:2]
    min_area, max_area = 0.0015 * H * W, 0.06 * H * W
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    found = []
    for lo, hi in [(25, 100), (40, 150), (60, 200)]:
        edges = cv2.Canny(gray, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            hull = cv2.convexHull(c)
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(hull)
            if min(rw, rh) < 1:
                continue
            ra = max(rw, rh) / min(rw, rh)
            if not (1.25 <= ra <= 2.0):
                continue
            fill = cv2.contourArea(hull) / (rw * rh)
            if fill < 0.55 or max(rw, rh) < 90:
                continue
            x1, y1 = max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2))
            x2, y2 = min(W, int(cx + rw / 2)), min(H, int(cy + rh / 2))
            std = _tex_std(gray, x1, y1, x2, y2)
            if std > 46:
                continue
            score = (1.0 - abs(ra - CARD_ASPECT) / 0.75) * 0.6 + fill * 0.3 + \
                    (1.0 - min(std, 46) / 46) * 0.1
            found.append((x1, y1, x2, y2, round(float(score), 3)))
    # 颜色提示补充(边缘被遮挡时)
    if color_hint:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color_hint in GEO_COLORS:
            hmin, hmax, smin, vmin = GEO_COLORS[color_hint]
            mask = cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, 255, 255]))
        elif color_hint == "red":
            mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255])) | \
                   cv2.inRange(hsv, np.array([170, 60, 60]), np.array([179, 255, 255]))
        elif color_hint == "black":
            mask = (hsv[:, :, 2] < 60).astype(np.uint8) * 255
        elif color_hint == "white":
            mask = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 195)).astype(np.uint8) * 255
        else:
            mask = None
        if mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area or area > min(max_area, 0.035 * H * W):
                    continue
                (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
                if min(rw, rh) < 1:
                    continue
                ra = max(rw, rh) / min(rw, rh)
                if not (1.15 <= ra <= 2.4) or max(rw, rh) < 90:
                    continue
                x1, y1 = max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2))
                x2, y2 = min(W, int(cx + rw / 2)), min(H, int(cy + rh / 2))
                std = _tex_std(gray, x1, y1, x2, y2)
                if std > 46:
                    continue
                score = (1.0 - abs(ra - CARD_ASPECT) / 0.9) * 0.6 + 0.35
                found.append((x1, y1, x2, y2, round(float(score), 3)))
    if not found:
        return []
    boxes = [f[:4] for f in found]
    scores = [f[4] for f in found]
    rects = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in boxes]
    idxs = cv2.dnn.NMSBoxes(rects, scores, 0.0, 0.5)
    if idxs is None or len(idxs) == 0:
        return []
    idxs = np.asarray(idxs).flatten()
    res = [boxes[int(i)] + (scores[int(i)],) for i in idxs]
    res.sort(key=lambda r: -r[4])
    return res


# ================= 卡的 LK 跟踪器(OWLv2 关键帧 + 光流) =================

class CardTracker:
    def __init__(self, detector, key_int=CARD_KEY_INT, conf=CARD_OWL_CONF):
        self.detector = detector          # OwlCardDetector 或 None(几何法)
        self.color_hint = None
        self.key_int = key_int
        self.conf = conf
        self.box = None
        self.pts = None
        self.prev_gray = None
        self.next_detect = 1          # 下一帧需要 OWLv2 重检的帧号
        self.lk = dict(winSize=(31, 31), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def _init_pts(self, gray, box):
        x1, y1, x2, y2 = box
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01,
                                      minDistance=10, mask=mask, blockSize=9)
        return None if pts is None else pts.reshape(-1, 1, 2)

    def update(self, img, frame_idx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            need = (self.box is None) or (frame_idx >= self.next_detect)
            if need:
                dets = self.detector.detect(img, self.conf)
                if dets:
                    best = max(dets, key=lambda b: b[4])
                    self.box = best[:4]
                    self.pts = self._init_pts(gray, self.box)
                    self.next_detect = frame_idx + self.key_int
                else:
                    self.next_detect = frame_idx + 1   # 未检出, 下帧重试
            elif self.box is not None and self.pts is not None and len(self.pts) >= 8 \
                    and self.prev_gray is not None:
                npts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.pts, None, **self.lk)
                gn = npts[status.flatten() == 1].reshape(-1, 2)
                go = self.pts[status.flatten() == 1].reshape(-1, 2)
                if len(gn) >= 8:
                    dx = float(np.median(gn[:, 0] - go[:, 0]))
                    dy = float(np.median(gn[:, 1] - go[:, 1]))
                    self.box = (int(self.box[0] + dx), int(self.box[1] + dy),
                                int(self.box[2] + dx), int(self.box[3] + dy))
                    self.pts = gn.reshape(-1, 1, 2)
                    # 跟踪质量差(位移过大/点数太少) → 提前重检, 防止打码框漂移
                    box_w = max(1, self.box[2] - self.box[0])
                    if len(gn) < 15 or abs(dx) > box_w * 0.5 or abs(dy) > box_w * 0.5:
                        self.next_detect = frame_idx + 1
                else:
                    self.pts = self._init_pts(gray, self.box)
                    self.next_detect = frame_idx + 1
        # 结果
        self.prev_gray = gray
        return [self.box] if self.box is not None else []


# ================= 视频处理 =================

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def has_audio(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", path])
    return bool(r.stdout.strip())


def get_fps(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=r_frame_rate",
              "-of", "default=noprint_wrappers=1:nokey=1", path])
    s = r.stdout.strip()
    if "/" in s:
        num, den = s.split("/")
        try:
            return float(num) / float(den)
        except ZeroDivisionError:
            pass
    return 25.0


def get_video_info(path):
    """获取视频宽、高、帧率(已考虑旋转)"""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height,r_frame_rate",
              "-of", "csv=p=0", path])
    parts = r.stdout.strip().split(",")
    if len(parts) < 3:
        return 1920, 1080, 25.0, 0
    w, h = int(parts[0]), int(parts[1])
    fps_str = parts[2]
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 25.0
    else:
        fps = float(fps_str) if fps_str else 25.0
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
            return enc
    return None


def hw_bitrate(w, h, fps, encoder):
    """根据编码器和分辨率算目标码率(接近原片画质)。

    HEVC 效率高, 0.10 系数即可接近原片; H.264 效率低需更高码率补偿。
    下限 4Mbps, 保证小视频也有基本画质。
    """
    factor = 0.10 if (encoder or "").startswith(("hevc", "h265")) else 0.15
    return max(int(w * h * fps * factor), 4000000)


def _process_pipe(src, dst, face_on, card_on, card_detector, card_conf,
                  card_color, model_dir, card_key_int, owl_size, face_size,
                  face_int, face_conf, keep_tmp, force_h264, use_gpu, log):
    """管道模式: ffmpeg抽帧(rawvideo) → Python处理 → ffmpeg编码, 无磁盘I/O"""
    t0 = time.time()
    w, h, fps, rot = get_video_info(src)
    has_aud = has_audio(src)
    tmp = tempfile.mkdtemp(prefix="vmb_")

    log(f"  [管道模式] {w}x{h} {fps:.1f}fps 音轨={'有' if has_aud else '无'}"
        + (f" 旋转{rot}°" if rot else ""))

    # 抽取音轨到临时文件(管道模式需单独处理音频)
    audio_tmp = None
    if has_aud:
        audio_tmp = os.path.join(tmp, "audio.aac")
        r = _run(["ffmpeg", "-y", "-i", src, "-vn", "-acodec", "copy", audio_tmp])
        if r.returncode != 0:
            audio_tmp = None

    # 检测硬件编码器
    hw = find_hw_encoder(family="h264" if force_h264 else "hevc")

    # 初始化检测器
    fd = FaceDetector(model_dir, yunet_size=face_size)
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf) if face_on else None
    owl = None
    geo_thr = GEO_SCORE
    if card_on:
        if card_detector == "owlv2":
            try:
                owl = OwlCardDetector.get(owl_size=owl_size, use_gpu=use_gpu)
                log(f"  [信用卡] OWLv2 (每{card_key_int}帧检测, 输入{owl_size if owl_size else '原图'})")
            except Exception as e:
                log(f"  [警告] OWLv2 加载失败({e}), 降级几何法")
                card_detector = "geo"
        if card_detector == "geo":
            geo_thr = GEO_COLOR_SCORE if card_color else GEO_SCORE
            log(f"  [信用卡] 几何法" + (f" + 颜色:{card_color}" if card_color else ""))
    card_tr = CardTracker(owl, key_int=card_key_int, conf=card_conf) if owl else None
    geo_tr = Tracker(10) if (card_on and not owl) else None

    # 启动抽帧进程(raw BGR → stdout pipe, stderr → DEVNULL 避免管道死锁)
    # 注意: rawvideo不自动旋转, 需手动加transpose滤镜(iPhone等竖屏视频)
    extract_cmd = ["ffmpeg", "-i", src]
    vf = get_transpose_filter(rot)
    if vf:
        extract_cmd += ["-vf", vf]
    extract_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    extract = subprocess.Popen(
        extract_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # 构建编码命令(stdin ← raw BGR)
    # 注意: ffmpeg参数顺序必须是 [输入选项 -i 输入] [输出选项 输出文件]
    enc_cmd = ["ffmpeg", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_tmp:
        enc_cmd += ["-i", audio_tmp]
    # 输出选项(必须在所有 -i 之后)
    if hw:
        bitrate = hw_bitrate(w, h, fps, hw)
        enc_cmd += ["-c:v", hw, "-b:v", str(bitrate), "-real_time", "0"]
        if hw.startswith(("hevc", "h265")):
            enc_cmd += ["-tag:v", "hvc1"]  # HEVC必须hvc1标签才能被QuickTime/iOS/多数播放器播放
        log(f"  [编码] 硬件加速: {hw} ({bitrate // 1000}kbps)")
    else:
        enc_cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
        log(f"  [编码] libx264 fast (CPU)")
    enc_cmd += ["-pix_fmt", "yuv420p"]
    if audio_tmp:
        enc_cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "copy"]
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

    # 主循环: 读帧 → 处理 → 写帧
    frame_size = w * h * 3
    frame_idx = 0
    total_face = total_card = 0

    def _read_frame(stream, size):
        """逐块读取完整一帧(pipe buffer 远小于一帧, 需循环读取)"""
        data = b""
        while len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                break
            data += chunk
        return data

    try:
        while True:
            raw = _read_frame(extract.stdout, frame_size)
            if len(raw) < frame_size:
                break
            frame_idx += 1
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

            faces, cards = [], []
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
            if card_on:
                if owl is not None:
                    cards = card_tr.update(img, frame_idx)
                else:
                    dets = [b[:4] for b in detect_cards_geo(img, card_color)
                            if b[4] >= geo_thr]
                    cards = geo_tr.update(dets)
                for (x1, y1, x2, y2) in cards:
                    bw, bh = x2 - x1, y2 - y1
                    heavy_mosaic(img, int(x1 - 0.1 * bw), int(y1 - 0.1 * bh),
                                 int(x2 + 0.1 * bw), int(y2 + 0.1 * bh),
                                 CARD_CELLS, CARD_SIGMA)
                total_card += len(cards)

            encode.stdin.write(img.tobytes())

            if frame_idx % 50 == 0:
                log(f"    [{frame_idx}] 人脸={len(faces)} 卡={len(cards)} "
                    f"elapsed={time.time()-t0:.0f}s")
    except BrokenPipeError:
        log("  [警告] 管道中断")
    finally:
        extract.stdout.close()
        extract.wait()
        try:
            encode.stdin.close()
        except BrokenPipeError:
            pass
        encode.wait()

    # 错误诊断
    if encode.returncode != 0 and enc_err_buf:
        err_tail = b"".join(enc_err_buf).decode()[-500:]
        if err_tail.strip():
            log(f"  [ffmpeg错误] {err_tail}")

    if not keep_tmp and encode.returncode == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    elapsed = time.time() - t0
    log(f"  [完成] {dst}  耗时 {elapsed:.0f}s  人脸帧次={total_face} 卡帧次={total_card}")
    return encode.returncode == 0


def _process_files(src, dst, face_on, card_on, card_detector, card_conf,
                   card_color, model_dir, card_key_int, owl_size, face_size,
                   face_int, face_conf, keep_tmp, force_h264, use_gpu, log):
    """文件模式(后备): ffmpeg抽帧JPEG → 处理 → ffmpeg编码JPEG"""
    tmp = tempfile.mkdtemp(prefix="vmb_")
    fin, fout = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    os.makedirs(fin)
    os.makedirs(fout)
    t0 = time.time()
    fps = get_fps(src)
    w, h, _, _ = get_video_info(src)

    log(f"  [文件模式] 抽帧: {src}")
    r = _run(["ffmpeg", "-y", "-i", src, "-q:v", "2", os.path.join(fin, "frame_%05d.jpg")])
    if r.returncode != 0:
        log("  [错误] ffmpeg 抽帧失败: " + r.stderr[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    frames = sorted(f for f in os.listdir(fin) if f.endswith(".jpg"))
    log(f"  共 {len(frames)} 帧")

    owl = None
    geo_thr = GEO_SCORE
    if card_on:
        if card_detector == "owlv2":
            try:
                owl = OwlCardDetector.get(owl_size=owl_size, use_gpu=use_gpu)
                log(f"  [信用卡] OWLv2 (每{card_key_int}帧检测, 输入{owl_size if owl_size else '原图'})")
            except Exception as e:
                log(f"  [警告] OWLv2 加载失败({e}), 降级几何法")
                card_detector = "geo"
        if card_detector == "geo":
            geo_thr = GEO_COLOR_SCORE if card_color else GEO_SCORE
            log(f"  [信用卡] 几何法" + (f" + 颜色:{card_color}" if card_color else ""))

    fd = FaceDetector(model_dir, yunet_size=face_size)
    face_proc = FaceProcessor(fd, detect_int=face_int, conf=face_conf) if face_on else None
    card_tr = CardTracker(owl, key_int=card_key_int, conf=card_conf) if owl else None
    geo_tr = Tracker(10)
    total_face = total_card = 0

    for i, fn in enumerate(frames, 1):
        img = cv2.imread(os.path.join(fin, fn))
        if img is None:
            continue
        faces, cards = [], []
        if face_on:
            faces = face_proc.process(img, i)
            for (x1, y1, x2, y2) in faces:
                bw, bh = x2 - x1, y2 - y1
                ex1, ey1 = int(x1 - FACE_EXPAND * bw), int(y1 - FACE_EXPAND * bh)
                ex2, ey2 = int(x2 + FACE_EXPAND * bw), int(y2 + FACE_EXPAND * bh)
                heavy_mosaic(img, ex1, ey1, ex2, ey2, FACE_CELLS, FACE_SIGMA)
            total_face += len(faces)
        if card_on:
            if owl is not None:
                cards = card_tr.update(img, i)
            else:
                dets = [b[:4] for b in detect_cards_geo(img, card_color) if b[4] >= geo_thr]
                cards = geo_tr.update(dets)
            for (x1, y1, x2, y2) in cards:
                bw, bh = x2 - x1, y2 - y1
                heavy_mosaic(img, int(x1 - 0.1 * bw), int(y1 - 0.1 * bh),
                             int(x2 + 0.1 * bw), int(y2 + 0.1 * bh),
                             CARD_CELLS, CARD_SIGMA)
            total_card += len(cards)
        cv2.imwrite(os.path.join(fout, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if i % 50 == 0 or i == len(frames):
            log(f"    [{i}/{len(frames)}] 人脸帧={len(faces)} 卡帧={len(cards)} elapsed={time.time()-t0:.0f}s")

    log("  合成视频...")
    hw = find_hw_encoder(family="h264" if force_h264 else "hevc")
    vcmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fout, "frame_%05d.jpg")]
    if has_audio(src):
        vcmd += ["-i", src, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    if hw:
        vcmd += ["-c:v", hw, "-b:v", str(hw_bitrate(w, h, fps, hw)), "-real_time", "0"]
        if hw.startswith(("hevc", "h265")):
            vcmd += ["-tag:v", "hvc1"]  # HEVC必须hvc1标签才能被QuickTime/iOS/多数播放器播放
    else:
        vcmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18"]
    vcmd += ["-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart", dst]
    r = _run(vcmd)
    if r.returncode != 0:
        log("  [错误] ffmpeg 合成失败: " + r.stderr[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"  [完成] {dst}  耗时 {time.time()-t0:.0f}s  人脸帧次={total_face} 卡帧次={total_card}")
    return True


def process_video(src, dst, face_on=True, card_on=True, card_detector="owlv2",
                  card_conf=CARD_OWL_CONF, card_color=None, model_dir=None,
                  card_key_int=CARD_KEY_INT, owl_size=CARD_OWL_SIZE,
                  face_size=FACE_YUNET_SIZE, face_int=FACE_DETECT_INT,
                  face_conf=FACE_CONF, use_pipe=False, keep_tmp=False,
                  force_h264=False, use_gpu=True, log=print):
    """处理单个视频: 人脸+信用卡打码, 保留音轨。返回是否成功。"""
    if use_pipe:
        try:
            return _process_pipe(src, dst, face_on, card_on, card_detector,
                                card_conf, card_color, model_dir,
                                card_key_int, owl_size, face_size,
                                face_int, face_conf, keep_tmp, force_h264, use_gpu, log)
        except Exception as e:
            log(f"  [警告] 管道模式失败({e}), 回退文件模式")
    return _process_files(src, dst, face_on, card_on, card_detector,
                         card_conf, card_color, model_dir,
                         card_key_int, owl_size, face_size,
                         face_int, face_conf, keep_tmp, force_h264, use_gpu, log)


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
    ap = argparse.ArgumentParser(description="视频批量打码(人脸+信用卡, OWLv2高精度)")
    ap.add_argument("inputs", nargs="+", help="视频文件/目录/通配符")
    ap.add_argument("--out-dir", default="masked_out", help="输出目录(默认 masked_out)")
    ap.add_argument("--card-detector", default="owlv2", choices=["owlv2", "geo"],
                    help="信用卡检测方式: owlv2(深度学习,默认,高准确率) / geo(几何法,快)")
    ap.add_argument("--card-conf", type=float, default=CARD_OWL_CONF, help="OWLv2置信度阈值(默认0.25)")
    ap.add_argument("--card-key-int", type=int, default=CARD_KEY_INT,
                    help="OWLv2检测间隔帧数(默认25; 调大提速但快速移动可能漏, 如40/50)")
    ap.add_argument("--owl-size", type=int, default=CARD_OWL_SIZE,
                    help="OWLv2输入最长边像素(默认1280; 0=原图最准最慢, 1280快3-8倍)")
    ap.add_argument("--face-size", type=int, default=FACE_YUNET_SIZE,
                    help="YuNet人脸检测输入最长边(默认1280快且准; 0=原图最准最慢; 远景多人脸可调低)")
    ap.add_argument("--face-int", type=int, default=FACE_DETECT_INT,
                    help="人脸检测间隔帧数(默认3: 每3帧检测1次,中间帧光流跟踪; 1=每帧检测最准)")
    ap.add_argument("--face-conf", type=float, default=FACE_CONF,
                    help=f"人脸置信度阈值(默认{FACE_CONF}; 降底如0.25可检出更多侧脸/遮挡, 可能增误检)")
    ap.add_argument("--pipe", action="store_true",
                    help="启用管道模式(无磁盘I/O, 更快但对HDR/HEVC+旋转视频有兼容问题)")
    ap.add_argument("--force-h264", action="store_true",
                    help="强制H.264输出(兼容性无敌: 微信/Android/旧播放器都能播; 画质仍近无损高码率)")
    ap.add_argument("--card-color", default=None,
                    help="几何法颜色提示: yellow/gold/blue/green/red/orange/black/white")
    ap.add_argument("--no-face", action="store_true", help="关闭人脸打码")
    ap.add_argument("--no-card", action="store_true", help="关闭信用卡打码")
    ap.add_argument("--no-gpu", action="store_true",
                    help="关闭 GPU 加速(CUDA/MPS 均禁用, 强制 CPU; OWLv2 加载慢且检测慢约 3-8 倍)")
    ap.add_argument("--model-dir", default=None, help="SSD人脸模型目录")
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
        if process_video(src, dst, face_on=not args.no_face, card_on=not args.no_card,
                         card_detector=args.card_detector, card_conf=args.card_conf,
                         card_color=args.card_color, model_dir=args.model_dir,
                         card_key_int=args.card_key_int, owl_size=args.owl_size,
                         face_size=args.face_size, face_int=args.face_int,
                         face_conf=args.face_conf,
                         use_pipe=args.pipe, keep_tmp=args.keep_tmp,
                         force_h264=args.force_h264, use_gpu=not args.no_gpu):
            ok += 1
    print(f"\n全部完成: {ok}/{len(files)} 成功, 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
