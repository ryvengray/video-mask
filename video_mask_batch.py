#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_mask_batch.py — 视频批量打码（人脸 + 信用卡/银行卡），通用完整版

人脸:
  res10 SSD 深度模型(优先) → OpenCV Haar 级联(兜底, 零依赖)
信用卡/银行卡:
  OWLv2 深度学习零样本检测(默认, 高准确率, 任意颜色/遮挡) + LK 光流跟踪
  可选几何法(--card-detector geo, 快但准确率一般, 适合边缘完整的卡)

性能: OWLv2 每 25 帧检测一次, 中间帧光流跟踪(4K 视频约 4-6 分钟/10秒)

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
import time

import cv2
import numpy as np

# ================= 打码/检测参数 =================
FACE_CELLS, FACE_SIGMA = 4, 45.0
CARD_CELLS, CARD_SIGMA = 6, 35.0
FACE_GRACE = 8
CARD_KEY_INT = 25             # OWLv2 每25帧检测一次
CARD_OWL_CONF = 0.25          # OWLv2 置信度阈值
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
    def __init__(self, model_dir=None):
        self.ssd = None
        if model_dir:
            proto = os.path.join(model_dir, "deploy.prototxt")
            model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(proto) and os.path.exists(model):
                self.ssd = cv2.dnn.readNetFromCaffe(proto, model)
                print("[人脸] SSD 深度模型")
        if self.ssd is None:
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("[人脸] Haar 级联(零依赖)")

    def detect(self, img, conf=0.5):
        h, w = img.shape[:2]
        if self.ssd is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            self.ssd.setInput(blob)
            dets = self.ssd.forward()
            out = []
            for i in range(dets.shape[2]):
                if dets[0, 0, i, 2] < conf:
                    continue
                out.append((int(dets[0, 0, i, 3] * w), int(dets[0, 0, i, 4] * h),
                            int(dets[0, 0, i, 5] * w), int(dets[0, 0, i, 6] * h)))
            return out
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return [(x, y, x + w, y + h) for (x, y, w, h) in rects]


# ================= 信用卡检测 =================
# 通道1: OWLv2 深度学习(默认, 高准确率)

class OwlCardDetector:
    """OWLv2 零样本检测, 懒加载(首次自动下载模型约600MB)"""
    _inst = None

    def __init__(self):
        import torch
        from PIL import Image
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.torch = torch
        self.Image = Image
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL)
        print("[信用卡] OWLv2 深度学习模型已加载")

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def detect(self, img, thr=CARD_OWL_CONF):
        image = self.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=CARD_TEXTS, images=image, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        ts = self.torch.tensor([image.size[::-1]])
        res = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=thr, target_sizes=ts,
            text_labels=[CARD_TEXTS])[0]
        out = []
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in b.tolist()]
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
            # OWLv2: 关键帧检测 + 中间 LK 跟踪
            need = (self.box is None) or (frame_idx % self.key_int == 1)
            if need:
                dets = self.detector.detect(img, self.conf)
                if dets:
                    best = max(dets, key=lambda b: b[4])
                    self.box = best[:4]
                    self.pts = self._init_pts(gray, self.box)
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
                else:
                    self.pts = self._init_pts(gray, self.box)
        # 结果
        self.prev_gray = gray
        return [self.box] if self.box is not None else []


# ================= 视频处理 =================

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def require_media_tools():
    """在开始耗时处理前，给出明确的 ffmpeg 环境错误。"""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("未找到 " + ", ".join(missing) + "，请安装 ffmpeg 并确保它在 PATH 中")


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


def process_video(src, dst, face_on=True, card_on=True, card_detector="owlv2",
                  card_conf=CARD_OWL_CONF, card_color=None, model_dir=None,
                  keep_tmp=False, log=print):
    """处理单个视频: 人脸+信用卡打码, 保留音轨。返回是否成功。"""
    require_media_tools()
    if not face_on and not card_on:
        raise ValueError("人脸和信用卡检测不能同时关闭")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="vmb_")
    fin, fout = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    os.makedirs(fin)
    os.makedirs(fout)
    t0 = time.time()
    fps = get_fps(src)

    log(f"  抽帧: {src}")
    r = _run(["ffmpeg", "-y", "-i", src, "-q:v", "2", os.path.join(fin, "frame_%05d.jpg")])
    if r.returncode != 0:
        log("  [错误] ffmpeg 抽帧失败: " + r.stderr[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    frames = sorted(f for f in os.listdir(fin) if f.endswith(".jpg"))
    log(f"  共 {len(frames)} 帧")

    # 卡检测器
    owl = None
    geo_thr = GEO_SCORE
    if card_on:
        if card_detector == "owlv2":
            try:
                owl = OwlCardDetector.get()
            except Exception as e:
                log(f"  [警告] OWLv2 加载失败({e}), 降级几何法")
                card_detector = "geo"
        if card_detector == "geo":
            geo_thr = GEO_COLOR_SCORE if card_color else GEO_SCORE
            log(f"  [信用卡] 几何法" + (f" + 颜色:{card_color}" if card_color else ""))

    fd = FaceDetector(model_dir) if face_on else None
    face_tr = Tracker(FACE_GRACE)
    card_tr = CardTracker(owl, conf=card_conf) if owl else None
    geo_tr = Tracker(10)
    total_face = total_card = 0

    for i, fn in enumerate(frames, 1):
        img = cv2.imread(os.path.join(fin, fn))
        if img is None:
            continue
        faces, cards = [], []
        if face_on:
            faces = face_tr.update(fd.detect(img))
            for (x1, y1, x2, y2) in faces:
                heavy_mosaic(img, x1, y1, x2, y2, FACE_CELLS, FACE_SIGMA)
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
        cv2.imwrite(os.path.join(fout, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 50 == 0 or i == len(frames):
            log(f"    [{i}/{len(frames)}] 人脸帧={len(faces)} 卡帧={len(cards)} elapsed={time.time()-t0:.0f}s")

    log("  合成视频...")
    vcmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(fout, "frame_%05d.jpg")]
    if has_audio(src):
        vcmd += ["-i", src, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    vcmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart", dst]
    r = _run(vcmd)
    if r.returncode != 0:
        log("  [错误] ffmpeg 合成失败: " + r.stderr[-300:])
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"  [完成] {dst}  耗时 {time.time()-t0:.0f}s  人脸帧次={total_face} 卡帧次={total_card}")
    return True


def expand_inputs(inputs):
    files = []
    for it in inputs:
        if os.path.isdir(it):
            for ext in VIDEO_FORMATS:
                files += glob.glob(os.path.join(it, "*" + ext))
                files += glob.glob(os.path.join(it, "*" + ext.upper()))
        else:
            files += glob.glob(it)
    # 输入可能同时来自目录和通配符；去重并保持稳定顺序。
    return list(dict.fromkeys(f for f in files if os.path.isfile(f)))


def main():
    ap = argparse.ArgumentParser(description="视频批量打码(人脸+信用卡, OWLv2高精度)")
    ap.add_argument("inputs", nargs="+", help="视频文件/目录/通配符")
    ap.add_argument("--out-dir", default="masked_out", help="输出目录(默认 masked_out)")
    ap.add_argument("--card-detector", default="owlv2", choices=["owlv2", "geo"],
                    help="信用卡检测方式: owlv2(深度学习,默认,高准确率) / geo(几何法,快)")
    ap.add_argument("--card-conf", type=float, default=CARD_OWL_CONF, help="OWLv2置信度阈值(默认0.25)")
    ap.add_argument("--card-color", default=None,
                    help="几何法颜色提示: yellow/gold/blue/green/red/orange/black/white")
    ap.add_argument("--no-face", action="store_true", help="关闭人脸打码")
    ap.add_argument("--no-card", action="store_true", help="关闭信用卡打码")
    ap.add_argument("--model-dir", default=None, help="SSD人脸模型目录")
    ap.add_argument("--keep-tmp", action="store_true", help="保留中间帧")
    ap.add_argument("--offline", action="store_true",
                    help="禁止 OWLv2 下载模型，仅使用本机 Hugging Face 缓存")
    args = ap.parse_args()

    if args.no_face and args.no_card:
        ap.error("--no-face 和 --no-card 不能同时使用")
    if not 0.0 <= args.card_conf <= 1.0:
        ap.error("--card-conf 必须在 0 到 1 之间")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    files = expand_inputs(args.inputs)
    if not files:
        print("[错误] 未找到任何视频文件")
        sys.exit(1)
    try:
        require_media_tools()
    except RuntimeError as exc:
        ap.error(str(exc))
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
                         keep_tmp=args.keep_tmp):
            ok += 1
    print(f"\n全部完成: {ok}/{len(files)} 成功, 输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
