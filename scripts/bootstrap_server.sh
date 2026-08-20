#!/usr/bin/env bash
# 在全新 Ubuntu/Debian 服务器初始化 video-mask 运行环境。
# 用法：bash scripts/bootstrap_server.sh [--cuda|--cpu] [--face-only] [--skip-models] [--torch-index-url URL]
set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="auto"                 # auto | cuda | cpu
DOWNLOAD_MODELS=1
FACE_ONLY=0
# 如官方 PyTorch 安装页给出不同 CUDA wheel，请通过 --torch-index-url 覆盖。
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

usage() {
  cat <<'EOF'
用法：bash scripts/bootstrap_server.sh [选项]

  --cuda                    强制使用 CUDA；缺少驱动时自动安装 Ubuntu 推荐驱动
  --cpu                     强制安装 CPU 版 PyTorch
  --face-only               部署仅人脸流水线；默认要求 CUDA，配合 --cpu 才使用 CPU
  --skip-models             不预下载 OWLv2 和 YuNet 模型
  --torch-index-url URL     指定 PyTorch wheel 索引（默认 cu128）
  -h, --help                显示帮助

示例：
  bash scripts/bootstrap_server.sh --cuda --face-only
  bash scripts/bootstrap_server.sh --torch-index-url https://download.pytorch.org/whl/cu126
  bash scripts/bootstrap_server.sh --cpu --skip-models
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) MODE="cuda" ;;
    --cpu) MODE="cpu" ;;
    --face-only) FACE_ONLY=1 ;;
    --skip-models) DOWNLOAD_MODELS=0 ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || { echo "缺少 --torch-index-url 的值" >&2; exit 2; }
      TORCH_INDEX_URL="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知选项：$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# This project is deployed to GPU workers.  A face-only installation must not
# silently become a CPU deployment merely because GPU auto-detection failed.
# Users who intentionally need CPU inference can opt in with --face-only --cpu.
if [[ "$FACE_ONLY" -eq 1 && "$MODE" == "auto" ]]; then
  MODE="cuda"
fi

if [[ ! -f "$APP_DIR/pyproject.toml" || ! -f "$APP_DIR/video_mask_batch_flast.py" ]]; then
  echo "错误：请在 video-mask 项目目录中运行此脚本。" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "错误：此引导脚本仅面向 Ubuntu/Debian Linux。" >&2
  exit 1
fi

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
    echo "警告：检测到 ${PRETTY_NAME:-未知 Linux}，将尝试使用 apt-get。"
  fi
fi

if [[ $EUID -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "错误：需要 root 权限或 sudo 来安装系统依赖。" >&2
  exit 1
fi

echo "==> 安装系统依赖"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg curl pciutils python3 python3-venv python3-pip ca-certificates \
  ubuntu-drivers-common util-linux

if [[ "$MODE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    MODE="cuda"
  else
    MODE="cpu"
  fi
fi

if [[ "$MODE" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    if ! lspci -nn 2>/dev/null | grep -qi 'NVIDIA'; then
      echo "错误：请求 CUDA 模式，但此机器未检测到 NVIDIA GPU。请确认 Worker 使用 GPU 实例。" >&2
      exit 1
    fi
    echo "==> 未检测到可用 NVIDIA 驱动，安装 Ubuntu 推荐驱动"
    "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive ubuntu-drivers install
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
      cat >&2 <<'EOF'
NVIDIA 驱动已安装，但内核模块尚未加载。请重启系统后重新运行本脚本。
Ansible Worker 部署会识别此状态、重启 Worker 一次并自动继续。
EOF
      exit 75
    fi
  fi
  echo "==> 检测到 NVIDIA GPU"
  nvidia-smi -L
else
  echo "==> 未使用 NVIDIA GPU，将安装 CPU 版 PyTorch"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi

echo "==> 创建虚拟环境"
cd "$APP_DIR"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
PYTHON="$APP_DIR/.venv/bin/python"
"$PYTHON" -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo "==> 安装 PyTorch（${MODE}）"
"$PYTHON" -m pip install --no-cache-dir --upgrade torch torchvision --index-url "$TORCH_INDEX_URL"

echo "==> 安装项目依赖"
"$PYTHON" -m pip install --no-cache-dir -e '.[cards]'

if [[ "$MODE" == "cuda" ]]; then
  echo "==> 安装 YuNet ONNX Runtime CUDA 依赖"
  "$PYTHON" -m pip install --upgrade --no-cache-dir -r "$APP_DIR/requirements-cuda.txt"
  "$PYTHON" -m pip show onnxruntime-gpu >/dev/null || {
    echo "错误：onnxruntime-gpu 安装后未找到。" >&2
    exit 1
  }
fi

echo "==> 创建默认任务目录"
mkdir -p /home/ubuntu/sources /home/ubuntu/outputs

echo "==> 验证 ffmpeg 与 PyTorch"
ffmpeg -version | head -n 1
"$PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
PY

if [[ "$MODE" == "cuda" ]]; then
  if ! "$PYTHON" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    cat >&2 <<EOF
错误：NVIDIA 驱动已检测到，但当前 PyTorch 无法使用 CUDA。
请在 https://pytorch.org/get-started/locally/ 选择与该服务器驱动兼容的 CUDA wheel，
然后用 --torch-index-url 重新运行本脚本。
EOF
    exit 1
  fi
fi

if [[ "$MODE" == "cuda" ]]; then
  echo "==> 验证 ONNX Runtime CUDA provider"
  VIDEO_MASK_FACE_ONLY="$FACE_ONLY" "$PYTHON" - <<'PY'
import torch
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime CUDAExecutionProvider 不可用")
PY
  echo "==> 检查 NVIDIA 视频编解码能力"
  ffmpeg -hide_banner -hwaccels | grep -q cuda && echo "NVDEC/CUDA: available" || echo "NVDEC/CUDA: not listed"
  ffmpeg -hide_banner -encoders | grep -q h264_nvenc && echo "NVENC H.264: available" || echo "NVENC H.264: not listed"
fi

if [[ "$DOWNLOAD_MODELS" -eq 1 ]]; then
  if [[ "$FACE_ONLY" -eq 1 ]]; then
    echo "==> 预下载 YuNet 人脸模型"
  else
    echo "==> 预下载 OWLv2 与 YuNet 模型（首次约 600MB）"
  fi
  # 确保此前 shell 中遗留的离线配置不会阻止首次下载。
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  VIDEO_MASK_FACE_ONLY="$FACE_ONLY" "$PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
import os

if os.environ.get("VIDEO_MASK_FACE_ONLY") != "1":
    snapshot_download("google/owlv2-base-patch16-ensemble")
hf_hub_download(
    repo_id="opencv/face_detection_yunet",
    filename="face_detection_yunet_2023mar.onnx",
)
print("模型缓存完成")
PY
else
  echo "==> 已跳过模型下载；首次任务前请自行准备 OWLv2 与 YuNet 缓存。"
fi

if [[ "$MODE" == "cuda" && "$DOWNLOAD_MODELS" -eq 1 ]]; then
  echo "==> 验证 YuNet ONNX Runtime CUDA Session"
  "$PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download
import torch
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
model = hf_hub_download(
    repo_id="opencv/face_detection_yunet",
    filename="face_detection_yunet_2023mar.onnx",
    local_files_only=True,
)
session = ort.InferenceSession(
    model,
    providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
)
providers = session.get_providers()
print("YuNet session providers:", providers)
if not torch.cuda.is_available() or not providers or providers[0] != "CUDAExecutionProvider":
    raise SystemExit("YuNet ONNX Runtime CUDA Session 不可用")
print("YuNet CUDA Session: OK; GPU:", torch.cuda.get_device_name(0))
PY
fi

echo
echo "部署完成。建议先执行："
echo "  cd $APP_DIR"
if [[ "$FACE_ONLY" -eq 1 ]]; then
  echo "  .venv/bin/python video_mask_face_gpu.py /home/ubuntu/sources/test.mp4 --out-dir /home/ubuntu/outputs"
  echo "日志应显示 '[Face model] YuNet ONNX Runtime CUDA'、'[Decode] NVDEC/CUDA' 与 '[Encode] NVENC h264'。"
else
  echo "  .venv/bin/python video_mask_batch_flast.py /path/to/test.mp4 --out-dir /path/to/output"
  echo "日志应显示 '[信用卡] OWLv2 模型已加载 (CUDA GPU: ...)'。"
fi
