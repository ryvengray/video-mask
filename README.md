# video-mask

批量为视频中的人脸和银行卡添加马赛克。视频编码、帧率保持不变；安装 `ffmpeg` 后会保留原音轨。

## 安装

前置条件：Python 3.9+ 与 `ffmpeg`（命令行中应能执行 `ffmpeg -version`）。建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

以上安装基础功能（人脸 + 几何法银行卡检测）。如需默认的 OWLv2 高精度银行卡检测：

```bash
python -m pip install -e '.[cards]'
```

OWLv2 首次使用会下载模型到 Hugging Face 缓存；如需严格离线运行，先确保模型已经缓存，再传入 `--offline`。

## 使用

```bash
# 高精度：人脸 + 银行卡（首次会下载 OWLv2 模型）
vmask input.mp4

# 快速：人脸 + 几何法银行卡检测
vmask input.mp4 --card-detector geo

# 仅人脸
vmask input.mp4 --no-card

# 整个目录、指定输出目录
vmask ./videos --out-dir ./masked
```

未安装为命令时，也可用：`python video_mask_batch.py input.mp4`。

输出默认位于 `masked_out/masked_<原文件名>.mp4`。完整选项请执行 `vmask --help`，详细中文说明见 [视频打码工具使用文档.md](./视频打码工具使用文档.md)。

## 开发检查

```bash
python -m compileall -q video_mask_batch.py
python -m pytest
```
