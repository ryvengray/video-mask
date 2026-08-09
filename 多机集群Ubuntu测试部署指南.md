# 多机视频打码集群：Ubuntu 单机测试部署指南

本指南用于当前测试：**一台 Ubuntu GPU 服务器同时运行 Controller 和一个 Worker**。

```text
Controller（127.0.0.1:8080）
        │
        └── Worker Agent（同一台服务器）
                │
                ├── /home/ubuntu/cluster_test_sources/
                └── /home/ubuntu/cluster_test_outputs/
```

这次测试不使用 S3，不会扫描或改动现有的 `/home/ubuntu/sources/`、`/home/ubuntu/outputs/`。先只处理一个短视频，验证 Controller、Worker、GPU、任务领取和输出上传（本地复制）完整链路。

## 0. 前提：确认服务器已拿到集群代码

当前集群代码必须已经提交并推送到仓库后，服务器才能拉取。进入项目后检查：

```bash
cd /home/ubuntu/video-mask
git pull

test -f cluster/controller.py && \
test -f cluster/worker_agent.py && \
test -f ansible/site.yml && \
echo "Cluster code is ready"
```

如果最后一条没有输出 `Cluster code is ready`，不要继续部署；说明服务器仓库尚未包含集群代码。

检查服务器已有修改，避免误覆盖：

```bash
git status --short
```

## 1. 安装 Ansible

```bash
sudo apt-get update
sudo apt-get install -y ansible

ansible-playbook --version
```

## 2. 创建独立测试目录

```bash
mkdir -p /home/ubuntu/cluster_test_sources
mkdir -p /home/ubuntu/cluster_test_outputs

# 仅复制一个短视频用于首次验证；-n 表示目标存在时不覆盖。
cp -n /home/ubuntu/sources/video_41s.mp4 \
  /home/ubuntu/cluster_test_sources/

ls -lh /home/ubuntu/cluster_test_sources
```

不要把 `video_25m.mp4` 放入测试目录，先验证短视频链路成功。

## 3. 创建 Ansible inventory

进入 Ansible 目录：

```bash
cd /home/ubuntu/video-mask/ansible
cp -n inventory.yml.example inventory.yml
mkdir -p group_vars
cp -n group_vars/all.yml.example group_vars/all.yml
```

编辑 inventory：

```bash
nano inventory.yml
```

将内容改为：

```yaml
controller:
  hosts:
    controller-01:
      ansible_connection: local

gpu_workers:
  hosts:
    worker-01:
      ansible_connection: local
      video_mask_worker_id: worker-01
```

保存 `nano`：按 `Ctrl+O`、回车，再按 `Ctrl+X`。

## 4. 配置本地测试参数和 Token

先生成两个随机 Token：

```bash
openssl rand -hex 32
openssl rand -hex 32
```

编辑变量文件：

```bash
nano group_vars/all.yml
```

保留 `video_mask_repo` 等已有配置，并至少设置以下内容；把两处 Token 替换成刚刚生成的不同随机值：

```yaml
video_mask_repo: https://gitee.com/ryven/video-mask.git
video_mask_ref: main
video_mask_app_dir: /home/ubuntu/video-mask

# Controller 与 Worker 同机，不需要域名或 HTTPS。
video_mask_controller_url: http://127.0.0.1:8080
video_mask_admin_token: REPLACE_WITH_FIRST_RANDOM_TOKEN
video_mask_worker_token: REPLACE_WITH_SECOND_RANDOM_TOKEN

video_mask_storage_mode: local
video_mask_local_source_dir: /home/ubuntu/cluster_test_sources
video_mask_local_output_dir: /home/ubuntu/cluster_test_outputs
```

S3 Bucket、Region、跨账号 Role ARN 和输出 Bucket 的配置先保持为空。不要把真实 Token 提交到 Git。

## 5. 部署 Controller 和 Worker

先做语法检查：

```bash
cd /home/ubuntu/video-mask
ansible-playbook -i ansible/inventory.yml ansible/site.yml --syntax-check
```

开始部署：

```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml -K
```

`-K` 会询问当前 Ubuntu 用户的 sudo 密码。部署过程会：

```text
安装/检查系统依赖
→ 校验 CUDA、ONNX Runtime CUDA 与 YuNet Session
→ 启动 video-mask-controller.service
→ 在控制器预注册 worker-01
→ 启动 video-mask-worker.service
```

## 6. 验证服务

```bash
sudo systemctl status video-mask-controller --no-pager
sudo systemctl status video-mask-worker --no-pager

curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/
```

健康检查应返回包含 `"status":"ok"` 的 JSON。

查看实时日志：

```bash
journalctl -u video-mask-controller -f
```

另开一个 SSH 窗口查看 Worker：

```bash
journalctl -u video-mask-worker -f
```

Worker 应先显示已注册，然后领取测试目录中的 `video_41s.mp4`。

## 7. 查看测试结果

```bash
ls -lh /home/ubuntu/cluster_test_outputs

ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height \
  -of default=noprint_wrappers=1 \
  /home/ubuntu/cluster_test_outputs/masked_video_41s.mp4
```

Controller 页面在服务器本机：

```text
http://127.0.0.1:8080/
```

由于当前没有公网 HTTPS 域名，暂时通过 SSH 隧道在本地浏览器查看：

```bash
# 在你的本地电脑执行
ssh -L 8080:127.0.0.1:8080 ubuntu@43.166.162.143
```

然后在本地浏览器打开：

```text
http://127.0.0.1:8080/
```

## 8. 停止或重新测试

停止两项服务：

```bash
sudo systemctl stop video-mask-worker
sudo systemctl stop video-mask-controller
```

重新测试同一视频前，停止 Worker 后删除测试输出和 Controller 状态库：

```bash
rm -f /home/ubuntu/cluster_test_outputs/masked_video_41s.mp4
sudo rm -f /var/lib/video-mask-controller/controller.sqlite3
```

然后重新启动：

```bash
sudo systemctl start video-mask-controller
sudo systemctl start video-mask-worker
```

仅删除上述测试目录和测试数据库；不要删除 `/home/ubuntu/sources/`、`/home/ubuntu/outputs/`、`/home/ubuntu/video-mask/`。

## 常见问题

### Worker 显示未注册或 Token 无效

确认 `group_vars/all.yml` 中的 `video_mask_worker_token` 未修改，并重新执行：

```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml -K
```

### Controller 端口 8080 已被占用

```bash
sudo ss -ltnp | grep :8080
```

停止冲突服务，或修改 `video-mask-controller.service.j2` 中的端口后重新部署。

### CUDA / ONNX Runtime 验证失败

```bash
nvidia-smi
/home/ubuntu/video-mask/.venv/bin/python -c \
  "import torch, onnxruntime as ort; print(torch.cuda.is_available()); print(ort.get_available_providers())"
```

必须看到 `True` 和 `CUDAExecutionProvider`。如果缺失，重新执行 GPU 引导：

```bash
cd /home/ubuntu/video-mask
bash scripts/bootstrap_server.sh --cuda --face-only
```

## 通过测试后的下一步

确认短视频完整处理成功后，再决定是否把 `video_25m.mp4` 放入测试目录。S3 权限到位后，将本地存储模式切换为 S3；正式多机时每台 Worker 使用独立本地工作目录，源视频和输出通过 S3 传递。
