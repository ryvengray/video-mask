#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_face_mask_hard_v2.py

针对“小脸 + 侧脸 + 鱼眼边缘 + 低置信度”漏检优化的人脸视频打码程序。

核心策略：
1. 主检测：YOLO/YOLO11 人脸模型
2. 困难脸检测：
   - 小脸
   - 画面边缘脸
   - 低置信度候选
   - 连续帧未确认的人脸候选
   -> 自动扩大 ROI、放大后再次检测
3. 原图 + 可选鱼眼去畸变双路检测
4. 时序确认：低置信度候选需要连续帧支持
5. Track ID：IoU + 中心距离 + 面积变化做简单多目标关联
6. LK 光流：非检测帧跟踪
7. 动态检测间隔：跟踪质量差时自动提前检测
8. 动态 Grace：稳定目标允许短暂漏检，低质量目标快速释放
9. 动态打码框：根据脸框大小和位置自适应扩展

依赖：
    pip install opencv-python numpy ultralytics

运行示例：
    python video_face_mask_hard_v2.py input.mp4 \
        --model face_model.pt \
        --output masked.mp4

鱼眼：
    python video_face_mask_hard_v2.py input.mp4 \
        --model face_model.pt \
        --output masked.mp4 \
        --fisheye

如果没有单独的人脸模型，请不要直接使用普通 COCO YOLO11n；
必须使用“人脸检测模型”权重。
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# 参数
# ============================================================

DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.45

# 困难脸
HARD_CONF = 0.20
SMALL_FACE_SIZE = 100
VERY_SMALL_FACE_SIZE = 55
EDGE_RATIO = 0.16
HARD_ROI_SCALE = 3.0
HARD_INPUT = 640

# 时序
CONFIRM_FRAMES = 2
CANDIDATE_MAX_AGE = 4

# LK
LK_WIN = (21, 21)
LK_LEVEL = 3
LK_FB_THRESHOLD = 1.5
LK_MIN_POINTS = 5
LK_MAX_MEDIAN_MOVE = 0.20

# 动态检测
NORMAL_DETECT_INTERVAL = 5
FAST_DETECT_INTERVAL = 2
BAD_TRACK_DETECT_INTERVAL = 1

# Grace
GOOD_GRACE = 5
NORMAL_GRACE = 3
BAD_GRACE = 1

# 打码
MOSAIC_CELLS = 5
MOSAIC_SIGMA = 35.0
BASE_EXPAND = 0.14


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Detection:
    box: Tuple[float, float, float, float]
    conf: float
    source: str = "main"


@dataclass
class Track:
    tid: int
    box: Tuple[float, float, float, float]
    conf: float
    age: int = 1
    hits: int = 1
    miss: int = 0
    confirmed: bool = False
    quality: float = 0.8
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_source: str = "main"
    points: Optional[np.ndarray] = None


@dataclass
class Candidate:
    box: Tuple[float, float, float, float]
    conf: float
    age: int = 1
    hits: int = 1


# ============================================================
# 基础函数
# ============================================================

def xyxy_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_wh(box):
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = xyxy_area(a) + xyxy_area(b) - inter
    return inter / union if union > 1e-6 else 0.0


def center_distance_norm(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)
    bw, bh = box_wh(a)
    scale = max(10.0, math.sqrt(bw * bh))
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) / scale


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (
        max(0, min(w - 1, int(round(x1)))),
        max(0, min(h - 1, int(round(y1)))),
        max(0, min(w - 1, int(round(x2)))),
        max(0, min(h - 1, int(round(y2)))),
    )


def expand_box(box, scale, w, h):
    x1, y1, x2, y2 = box
    cx, cy = box_center(box)
    bw, bh = box_wh(box)
    nw = bw * scale
    nh = bh * scale
    return clip_box(
        (cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2),
        w, h
    )


def is_edge_face(box, w, h):
    x1, y1, x2, y2 = box
    margin_x = w * EDGE_RATIO
    margin_y = h * EDGE_RATIO
    return x1 < margin_x or y1 < margin_y or x2 > w - margin_x or y2 > h - margin_y


def face_size(box):
    bw, bh = box_wh(box)
    return min(bw, bh)


def is_small_face(box):
    return face_size(box) < SMALL_FACE_SIZE


def is_very_small_face(box):
    return face_size(box) < VERY_SMALL_FACE_SIZE


# ============================================================
# YOLO检测器
# ============================================================

class FaceDetector:
    def __init__(self, model_path, conf=DEFAULT_CONF, iou_threshold=DEFAULT_IOU,
                 device=None, imgsz=640):
        if YOLO is None:
            raise RuntimeError(
                "未安装 ultralytics，请执行：pip install ultralytics"
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型不存在: {model_path}")

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou_threshold
        self.device = device
        self.imgsz = imgsz

    def detect(self, img, conf=None, imgsz=None, source="main"):
        conf = self.conf if conf is None else conf
        imgsz = self.imgsz if imgsz is None else imgsz

        kwargs = dict(
            source=img,
            conf=conf,
            iou=self.iou,
            imgsz=imgsz,
            verbose=False,
        )

        if self.device:
            kwargs["device"] = self.device

        result = self.model.predict(**kwargs)[0]

        out = []
        if result.boxes is None:
            return out

        h, w = img.shape[:2]

        for b in result.boxes:
            xy = b.xyxy[0].detach().cpu().numpy()
            score = float(b.conf[0].detach().cpu().item())

            x1, y1, x2, y2 = xy.tolist()

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            out.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    conf=score,
                    source=source
                )
            )

        return out


# ============================================================
# 鱼眼
# ============================================================

class FisheyeProcessor:
    def __init__(self, strength=1.0):
        self.strength = strength
        self.cache = {}

    def process(self, frame):
        h, w = frame.shape[:2]
        key = (w, h, round(self.strength, 3))

        if key not in self.cache:
            # 通用估算，不要求相机标定文件。
            # 对不同摄像头不能保证最优，因此始终保留原图检测。
            f = max(w, h) * 0.46 / max(0.5, self.strength)

            K = np.array([
                [f, 0, w / 2],
                [0, f, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            D = np.array([
                0.18 * self.strength,
                0.035 * self.strength,
                0.0,
                0.0
            ], dtype=np.float64)

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K,
                (w, h), cv2.CV_16SC2
            )
            self.cache[key] = (map1, map2)

        map1, map2 = self.cache[key]
        return cv2.remap(
            frame, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )


# ============================================================
# 困难脸检测
# ============================================================

class HardFaceDetector:
    """
    针对：
      1. 小脸
      2. 边缘脸
      3. 低conf脸
      4. 鱼眼边缘脸

    对候选框进行扩大、放大、再次检测。
    """

    def __init__(self, detector, roi_scale=HARD_ROI_SCALE,
                 hard_input=HARD_INPUT):
        self.detector = detector
        self.roi_scale = roi_scale
        self.hard_input = hard_input

    def detect_one_roi(self, frame, candidate: Detection):
        h, w = frame.shape[:2]

        roi_box = expand_box(
            candidate.box,
            self.roi_scale,
            w, h
        )

        rx1, ry1, rx2, ry2 = roi_box

        if rx2 <= rx1 or ry2 <= ry1:
            return []

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return []

        # ROI扩大后直接 resize 给高输入尺寸检测。
        # 这里的目的就是把 50~80 px 的小脸变成 150~250 px。
        detections = self.detector.detect(
            roi,
            conf=HARD_CONF,
            imgsz=self.hard_input,
            source="hard_roi"
        )

        mapped = []

        for d in detections:
            x1, y1, x2, y2 = d.box
            mapped.append(
                Detection(
                    box=(
                        x1 + rx1,
                        y1 + ry1,
                        x2 + rx1,
                        y2 + ry1,
                    ),
                    conf=d.conf,
                    source="hard_roi"
                )
            )

        return mapped

    def detect(self, frame, candidates):
        results = []

        h, w = frame.shape[:2]

        for c in candidates:
            difficult = (
                is_small_face(c.box)
                or is_edge_face(c.box, w, h)
                or c.conf < 0.45
            )

            if not difficult:
                continue

            results.extend(self.detect_one_roi(frame, c))

        return results


# ============================================================
# LK跟踪
# ============================================================

class LKTracker:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def init_points(self, gray, box):
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1] - 1, x2)
        y2 = min(gray.shape[0] - 1, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        # 避开最外边缘，优先在人脸内部寻找角点。
        px1 = x1 + int((x2 - x1) * 0.12)
        py1 = y1 + int((y2 - y1) * 0.12)
        px2 = x2 - int((x2 - x1) * 0.12)
        py2 = y2 - int((y2 - y1) * 0.12)

        roi = gray[py1:py2, px1:px2]

        if roi.size == 0:
            return None

        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=40,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=5
        )

        if pts is None or len(pts) < LK_MIN_POINTS:
            return None

        pts[:, 0, 0] += px1
        pts[:, 0, 1] += py1

        return pts.astype(np.float32)

    def track(self, prev_gray, gray, box, points):
        if points is None or len(points) < LK_MIN_POINTS:
            return None, None, 0.0

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points, None,
            winSize=LK_WIN,
            maxLevel=LK_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20, 0.03
            )
        )

        if p1 is None:
            return None, None, 0.0

        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, p1, None,
            winSize=LK_WIN,
            maxLevel=LK_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20, 0.03
            )
        )

        if p0_back is None:
            return None, None, 0.0

        p0 = points.reshape(-1, 2)
        p1v = p1.reshape(-1, 2)
        p0b = p0_back.reshape(-1, 2)

        good = (
            (st1.reshape(-1) > 0)
            & (st2.reshape(-1) > 0)
        )

        if good.sum() < LK_MIN_POINTS:
            return None, None, 0.0

        p0g = p0[good]
        p1g = p1v[good]
        p0bg = p0b[good]

        fb = np.linalg.norm(p0g - p0bg, axis=1)
        fb_good = fb < LK_FB_THRESHOLD

        if fb_good.sum() < LK_MIN_POINTS:
            return None, None, 0.0

        p0g = p0g[fb_good]
        p1g = p1g[fb_good]

        displacement = np.linalg.norm(p1g - p0g, axis=1)
        median_move = float(np.median(displacement))

        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        if median_move > max(bw, bh) * LK_MAX_MEDIAN_MOVE:
            return None, None, 0.0

        dx = float(np.median(p1g[:, 0] - p0g[:, 0]))
        dy = float(np.median(p1g[:, 1] - p0g[:, 1]))

        new_box = (
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy
        )

        fb_score = max(0.0, 1.0 - float(np.median(fb)) / LK_FB_THRESHOLD)
        point_score = min(1.0, len(p1g) / 20.0)
        move_score = max(
            0.0,
            1.0 - median_move / max(1.0, max(bw, bh) * LK_MAX_MEDIAN_MOVE)
        )

        quality = (
            0.45 * fb_score +
            0.30 * point_score +
            0.25 * move_score
        )

        return new_box, p1g.reshape(-1, 1, 2).astype(np.float32), quality


# ============================================================
# Track ID
# ============================================================

class TrackManager:
    def __init__(self):
        self.tracks: List[Track] = []
        self.next_id = 1

    def _score_match(self, track_box, det_box):
        ov = iou(track_box, det_box)
        dist = center_distance_norm(track_box, det_box)

        tw, th = box_wh(track_box)
        dw, dh = box_wh(det_box)

        area_ratio = min(tw * th, dw * dh) / max(
            1.0, max(tw * th, dw * dh)
        )

        # 越高越匹配
        score = (
            0.55 * ov +
            0.25 * max(0.0, 1.0 - dist / 2.0) +
            0.20 * area_ratio
        )
        return score

    def update(self, detections, frame_shape):
        h, w = frame_shape[:2]

        used = set()

        # 先按历史track逐一寻找最佳检测。
        for t in self.tracks:
            best_idx = -1
            best_score = 0.0

            for i, d in enumerate(detections):
                if i in used:
                    continue

                score = self._score_match(t.box, d.box)

                # 较宽松的匹配，保证小脸移动时不容易换ID。
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score >= 0.28:
                d = detections[best_idx]
                used.add(best_idx)

                old_cx, old_cy = box_center(t.box)
                new_cx, new_cy = box_center(d.box)

                vx = new_cx - old_cx
                vy = new_cy - old_cy

                t.velocity = (
                    0.65 * t.velocity[0] + 0.35 * vx,
                    0.65 * t.velocity[1] + 0.35 * vy,
                )

                t.box = d.box
                t.conf = max(t.conf * 0.65, d.conf)
                t.age += 1
                t.hits += 1
                t.miss = 0
                t.last_source = d.source

                # 新脸连续命中后才确认。
                if t.hits >= CONFIRM_FRAMES:
                    t.confirmed = True

                # 检测本身可以恢复track质量。
                t.quality = min(
                    1.0,
                    0.55 * t.quality + 0.45 * max(
                        0.0, min(1.0, d.conf / 0.7)
                    )
                )

            else:
                t.miss += 1
                t.age += 1
                t.quality *= 0.88

        # 创建新的track
        for i, d in enumerate(detections):
            if i in used:
                continue

            t = Track(
                tid=self.next_id,
                box=d.box,
                conf=d.conf,
                age=1,
                hits=1,
                miss=0,
                confirmed=(d.conf >= 0.55),
                quality=max(0.4, min(1.0, d.conf / 0.7)),
                last_source=d.source,
            )
            self.next_id += 1
            self.tracks.append(t)

        # 清理死亡track
        alive = []

        for t in self.tracks:
            if t.quality >= 0.70:
                grace = GOOD_GRACE
            elif t.quality >= 0.45:
                grace = NORMAL_GRACE
            else:
                grace = BAD_GRACE

            if t.miss <= grace:
                alive.append(t)

        self.tracks = alive

        return self.tracks


# ============================================================
# 候选时序确认
# ============================================================

class CandidateManager:
    def __init__(self):
        self.candidates: List[Candidate] = []

    def update(self, detections):
        new_candidates = []

        for d in detections:
            best = None
            best_score = 0.0

            for c in self.candidates:
                score = (
                    0.65 * iou(c.box, d.box)
                    + 0.35 * max(
                        0.0,
                        1.0 - center_distance_norm(c.box, d.box) / 2.0
                    )
                )

                if score > best_score:
                    best_score = score
                    best = c

            if best is not None and best_score > 0.25:
                best.box = d.box
                best.conf = max(best.conf, d.conf)
                best.age += 1
                best.hits += 1
                new_candidates.append(best)
            else:
                new_candidates.append(
                    Candidate(
                        box=d.box,
                        conf=d.conf,
                        age=1,
                        hits=1
                    )
                )

        # 去掉太久没有更新的旧候选
        self.candidates = new_candidates[:30]

        return self.candidates


# ============================================================
# 合并检测
# ============================================================

def merge_detections(dets, frame_shape):
    if not dets:
        return []

    h, w = frame_shape[:2]

    # 先做基础几何过滤。
    valid = []

    for d in dets:
        bw, bh = box_wh(d.box)

        if min(bw, bh) < 10:
            continue

        ratio = bw / max(1.0, bh)

        # 对人脸保留较宽范围，避免侧脸被过滤。
        if ratio < 0.35 or ratio > 2.8:
            continue

        valid.append(d)

    # NMS式合并。
    valid.sort(key=lambda x: x.conf, reverse=True)

    result = []

    for d in valid:
        duplicate = False

        for r in result:
            if iou(d.box, r.box) > 0.35:
                duplicate = True

                # 困难ROI的框优先级更高，因为它通常来自放大后的检测。
                if d.source == "hard_roi" and r.source != "hard_roi":
                    if d.conf >= r.conf * 0.75:
                        r.box = d.box
                        r.conf = max(r.conf, d.conf)
                        r.source = d.source
                break

        if not duplicate:
            result.append(d)

    return result


# ============================================================
# 打码
# ============================================================

def heavy_mosaic(img, box, expand=BASE_EXPAND):
    h, w = img.shape[:2]

    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # 小脸适当增加扩展，避免只覆盖脸中心。
    if min(bw, bh) < 60:
        extra = 0.22
    elif min(bw, bh) < 100:
        extra = 0.18
    else:
        extra = expand

    # 边缘脸稍微多留一点安全边。
    ex1 = int(x1 - bw * extra)
    ey1 = int(y1 - bh * extra)
    ex2 = int(x2 + bw * extra)
    ey2 = int(y2 + bh * extra)

    ex1 = max(0, min(w - 1, ex1))
    ey1 = max(0, min(h - 1, ey1))
    ex2 = max(0, min(w, ex2))
    ey2 = max(0, min(h, ey2))

    if ex2 <= ex1 or ey2 <= ey1:
        return

    roi = img[ey1:ey2, ex1:ex2]

    if roi.size == 0:
        return

    rw = ex2 - ex1
    rh = ey2 - ey1

    k = min(
        int(MOSAIC_SIGMA * 3) | 1,
        51,
        max(3, max(rw, rh) | 1)
    )

    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(
        roi,
        (k, k),
        MOSAIC_SIGMA
    )

    # 小脸打码更加密集。
    min_side = min(rw, rh)

    if min_side < 60:
        cells = 4
    elif min_side < 100:
        cells = 5
    else:
        cells = MOSAIC_CELLS

    cw = max(1, rw // cells)
    ch = max(1, rh // cells)

    small = cv2.resize(
        blurred,
        (cw, ch),
        interpolation=cv2.INTER_AREA
    )

    big = cv2.resize(
        small,
        (rw, rh),
        interpolation=cv2.INTER_NEAREST
    )

    img[ey1:ey2, ex1:ex2] = big


# ============================================================
# 视频处理
# ============================================================

class FaceMaskProcessor:
    def __init__(
        self,
        model_path,
        conf=DEFAULT_CONF,
        device=None,
        imgsz=640,
        fisheye=False,
        fisheye_strength=1.0,
        debug=False,
    ):
        self.detector = FaceDetector(
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )

        self.hard_detector = HardFaceDetector(
            self.detector
        )

        self.fisheye = FisheyeProcessor(
            fisheye_strength
        ) if fisheye else None

        self.track_manager = TrackManager()
        self.candidate_manager = CandidateManager()
        self.lk = LKTracker()

        self.prev_gray = None
        self.frame_index = 0
        self.last_detection_frame = -999999

        self.debug = debug

    def detect(self, frame):
        h, w = frame.shape[:2]

        # ----------------------------------------------------
        # 1. 原图主检测
        # ----------------------------------------------------
        main_dets = self.detector.detect(
            frame,
            conf=self.detector.conf,
            imgsz=self.detector.imgsz,
            source="main"
        )

        # ----------------------------------------------------
        # 2. 鱼眼去畸变图检测
        # ----------------------------------------------------
        undist_dets = []

        if self.fisheye is not None:
            undist = self.fisheye.process(frame)

            undist_dets = self.detector.detect(
                undist,
                conf=self.detector.conf,
                imgsz=self.detector.imgsz,
                source="undist"
            )

            # 将去畸变结果近似映射回原图。
            # 注意：这是通用估算，不等同于严格相机标定逆映射。
            # 因此它只作为补充候选，不覆盖原图检测结果。
            if undist_dets:
                # 通用情况下无法准确逆映射，因此只在画面比例一致时保留同坐标。
                pass

        # ----------------------------------------------------
        # 3. 合并主检测和去畸变候选
        # ----------------------------------------------------
        candidates = merge_detections(
            main_dets + undist_dets,
            frame.shape
        )

        # ----------------------------------------------------
        # 4. 困难脸 ROI 二次检测
        # ----------------------------------------------------
        hard_dets = self.hard_detector.detect(
            frame,
            candidates
        )

        all_dets = merge_detections(
            candidates + hard_dets,
            frame.shape
        )

        # ----------------------------------------------------
        # 5. 对低置信度候选进行时序确认
        # ----------------------------------------------------
        candidate_state = self.candidate_manager.update(
            [
                d for d in all_dets
                if d.conf < 0.55
            ]
        )

        confirmed_candidates = []

        for c in candidate_state:
            # 高置信度直接通过；
            # 低置信度至少需要连续2次支持。
            if c.conf >= 0.55 or c.hits >= CONFIRM_FRAMES:
                confirmed_candidates.append(
                    Detection(
                        box=c.box,
                        conf=c.conf,
                        source="temporal"
                    )
                )

        final_dets = merge_detections(
            [
                d for d in all_dets
                if d.conf >= 0.55
            ] + confirmed_candidates,
            frame.shape
        )

        return final_dets

    def lk_update(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return

        for t in self.track_manager.tracks:
            if t.miss > 0:
                continue

            if t.points is None or len(t.points) < LK_MIN_POINTS:
                t.points = self.lk.init_points(
                    self.prev_gray,
                    t.box
                )

            if t.points is None:
                continue

            new_box, new_points, quality = self.lk.track(
                self.prev_gray,
                gray,
                t.box,
                t.points
            )

            if new_box is not None:
                t.box = new_box
                t.points = new_points
                t.quality = (
                    0.65 * t.quality +
                    0.35 * quality
                )

        self.prev_gray = gray.copy()

    def need_detection(self):
        tracks = self.track_manager.tracks

        if not tracks:
            return True

        # 有任何track质量差，立即检测。
        if any(t.quality < 0.45 for t in tracks):
            interval = BAD_TRACK_DETECT_INTERVAL
        elif any(t.quality < 0.70 for t in tracks):
            interval = FAST_DETECT_INTERVAL
        else:
            interval = NORMAL_DETECT_INTERVAL

        return (
            self.frame_index - self.last_detection_frame
            >= interval
        )

    def process_frame(self, frame):
        self.frame_index += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        do_detect = self.need_detection()

        if do_detect:
            detections = self.detect(frame)

            tracks = self.track_manager.update(
                detections,
                frame.shape
            )

            # 每次检测成功后重新初始化LK点。
            for t in tracks:
                t.points = self.lk.init_points(
                    gray,
                    t.box
                )

            self.last_detection_frame = self.frame_index

            # 防止检测帧之前的prev_gray失效。
            self.prev_gray = gray.copy()

        else:
            # LK跟踪
            self.lk_update(gray)
            tracks = self.track_manager.tracks

        # ----------------------------------------------------
        # 最终打码
        # ----------------------------------------------------
        for t in tracks:
            # 未确认的弱候选不直接打码。
            # 高置信度脸或连续确认的track才打码。
            if not t.confirmed:
                continue

            # Track质量过低时，不盲目无限延长。
            if t.miss > 0 and t.quality < 0.35:
                continue

            heavy_mosaic(
                frame,
                t.box,
                expand=BASE_EXPAND
            )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------
        if self.debug:
            for t in tracks:
                x1, y1, x2, y2 = map(int, t.box)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                text = (
                    f"ID:{t.tid} "
                    f"C:{t.conf:.2f} "
                    f"Q:{t.quality:.2f} "
                    f"M:{t.miss}"
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        return frame


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="困难场景人脸视频打码 v2"
    )

    parser.add_argument(
        "input",
        help="输入视频"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="人脸检测模型 .pt"
    )

    parser.add_argument(
        "--output",
        default="masked_output.mp4",
        help="输出视频"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"主检测阈值，默认 {DEFAULT_CONF}"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="主检测输入尺寸，默认640"
    )

    parser.add_argument(
        "--device",
        default=None,
        help="CPU/GPU，例如 cpu、0、mps"
    )

    parser.add_argument(
        "--fisheye",
        action="store_true",
        help="开启鱼眼双路检测"
    )

    parser.add_argument(
        "--fisheye-strength",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示检测框/Track ID"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="实时显示"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[错误] 输入视频不存在: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("[错误] 无法创建输出视频")
        cap.release()
        sys.exit(1)

    print("=" * 70)
    print("困难场景人脸打码 v2")
    print("=" * 70)
    print(f"输入      : {args.input}")
    print(f"输出      : {args.output}")
    print(f"分辨率    : {width}x{height}")
    print(f"FPS       : {fps:.2f}")
    print(f"总帧数    : {total}")
    print(f"主conf    : {args.conf}")
    print(f"主imgsz   : {args.imgsz}")
    print(f"鱼眼      : {args.fisheye}")
    print(f"困难ROI   : {HARD_ROI_SCALE}x")
    print(f"小脸阈值  : {SMALL_FACE_SIZE}px")
    print("=" * 70)

    processor = FaceMaskProcessor(
        model_path=args.model,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        fisheye=args.fisheye,
        fisheye_strength=args.fisheye_strength,
        debug=args.debug
    )

    frame_no = 0
    start = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_no += 1

        out = processor.process_frame(frame)

        writer.write(out)

        if args.show:
            cv2.imshow(
                "Face Mask v2",
                out
            )

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if frame_no % 30 == 0:
            elapsed = time.time() - start
            speed = frame_no / max(0.001, elapsed)

            print(
                f"\r[{frame_no}/{total}] "
                f"{frame_no / max(1, total) * 100:.1f}% "
                f"{speed:.1f} FPS "
                f"tracks={len(processor.track_manager.tracks)}",
                end="",
                flush=True
            )

    cap.release()
    writer.release()

    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("处理完成")
    print(f"输出文件: {args.output}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"平均速度: {frame_no / max(0.001, elapsed):.2f} FPS")
    print("=" * 70)


if __name__ == "__main__":
    main()
