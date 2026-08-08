# 人脸 GPU 流水线使用指南

`video_mask_face_gpu.py` 是仅做人脸马赛克的独立算法，不加载信用卡模型，也不修改历史算法文件。

## 设计

- FFmpeg 内存管道，不生成临时 JPEG 帧。
- 自动尝试 NVDEC，失败时回退 CPU 解码。
- YuNet 优先使用 ONNX Runtime CUDA，失败时回退 OpenCV CPU。
- 每隔若干帧检测一次，其余帧使用多目标 LK 光流跟踪。
- 自动尝试 NVENC H.264，失败时可显式使用 CPU `libx264`。
- 同一进程处理多个输入时只加载一次人脸模型。

## 服务器依赖

```bash
cd /home/ubuntu/video-mask
.venv/bin/pip install -r requirements-cuda.txt
```

验证 CUDA：

```bash
.venv/bin/python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), torch.version.cuda); print(ort.get_available_providers())"
```

输出应包含 `True` 和 `CUDAExecutionProvider`。

## 单文件运行

```bash
.venv/bin/python video_mask_face_gpu.py \
  /home/ubuntu/sources/example.MOV \
  --out-dir /home/ubuntu/outputs
```

推荐的 T4 参数：

```bash
.venv/bin/python video_mask_face_gpu.py \
  /home/ubuntu/sources/example.MOV \
  --out-dir /home/ubuntu/outputs \
  --face-int 5 \
  --decode auto \
  --encoder auto
```

## 使用批量调度器

```bash
.venv/bin/python scripts/batch_scheduler.py \
  /home/ubuntu/sources \
  /home/ubuntu/outputs \
  --algorithm video_mask_face_gpu.py \
  --workers 2
```

在 `scripts/task_manager.sh` 中直接选择 `video_mask_face_gpu.py`。该算法只处理人脸，不需要再输入 `--no-card`；如需调整检测间隔，可输入：

```text
--face-int 5
```

任务管理器会把它转换成调度器可识别的透传参数。

## 性能调节

- `--workers 2`：T4 + 8 核 CPU 的建议起点；整体吞吐提升，但单个视频不一定更快。
- `--face-int 3`：检测更密集，适合快速运动，速度较慢。
- `--face-int 5`：默认平衡值。
- `--face-int 6`：更快，但快速转头或突然入镜时可能短暂漏检。
- `--cpu`：用于与 GPU 做同条件基准测试。
- `--decode cpu` / `--encoder x264`：用于排查 NVIDIA 视频编解码问题。

启动日志会明确记录实际设备，例如：

```text
[Face model] YuNet ONNX Runtime CUDA; GPU=Tesla T4; providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[Decode] NVDEC/CUDA
[Encode] NVENC h264
```

任一 GPU 环节不可用时会记录原因并回退或终止，不会把 CPU 路径误报为 GPU。
