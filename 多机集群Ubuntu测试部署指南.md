# 多机视频打码集群：Ubuntu 单机测试部署指南

本指南先完成：**一台 Ubuntu GPU 服务器同时运行 Controller 和一个 Worker** 的测试；随后可按第 5.1 节用 Ansible 和 `terraform-host.pem` 添加独立 Worker。

```text
Controller（127.0.0.1:8080）
        │
        └── Worker Agent（同一台服务器）
                │
                ├── /home/ubuntu/cluster_test_sources/
                └── /home/ubuntu/cluster_test_outputs/
```

这次测试不使用 S3，不会扫描或改动现有的 `/home/ubuntu/sources/`、`/home/ubuntu/outputs/`。先只处理一个短视频，验证 Controller、Worker、GPU、任务领取和输出上传（本地复制）完整链路。

## 0. 新 Ubuntu Controller（同时作为 Ansible 控制机）一键启动

以下命令在新的 Ubuntu Controller 上、以 `ubuntu` 用户执行。请把 `CONTROLLER_PRIVATE_IP` 替换为此机器可被 Worker 访问的 VPC 私网 IP；不要填写 `127.0.0.1`，否则远程 Worker 无法领取任务。

```bash
git clone https://github.com/ryvengray/video-mask.git /home/ubuntu/video-mask
cd /home/ubuntu/video-mask
git pull --ff-only

bash scripts/bootstrap_cluster_controller.sh \
  --controller-url http://CONTROLLER_PRIVATE_IP:8080
```

脚本会安装 Ansible 和 Controller 依赖、生成随机 Controller/Worker Token、创建本机 Controller inventory，并启动 `video-mask-controller.service`。它不会部署本机 Worker；后续远程 Worker 使用第 5.1 节的 `terraform-host.pem` 流程添加。

首次启动完成后验证：

```bash
curl http://127.0.0.1:8080/healthz
sudo systemctl status video-mask-controller --no-pager
```

脚本依赖 Ubuntu/Debian 的 `apt`，并固定以 `ubuntu` 用户运行服务；请使用 Ubuntu AMI，不支持 Amazon Linux 的 `ec2-user` 运行环境。

### 手动部署前提：确认服务器已拿到集群代码

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

仅启动 Controller、暂时没有 Worker 时，可运行
`bash scripts/bootstrap_cluster_controller.sh` 自动生成本机配置并部署。
多机 AWS 环境请指定 Controller 私网地址，例如：

```bash
bash scripts/bootstrap_cluster_controller.sh \
  --controller-url http://10.0.1.25:8080
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
mkdir -p group_vars/all
cp -n group_vars/all/settings.yml.example group_vars/all/settings.yml
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

这是 Controller 与首个 Worker 同机的测试 inventory。新增独立机器时，不要保留
`ansible_connection: local`；请使用第 5.1 节的远程 Worker 配置。

## 4. 配置本地测试参数和 Token

`terraform-host.pem` 仅用于 Ansible SSH 登录。Worker 不再拉取私有仓库：运行代码由执行 Ansible 的 Controller 通过 rsync 推送，因此 Worker 不需要 GitHub Deploy Key。若需要让 Controller 自己从私有仓库更新代码，才为 Controller 保存只读 Deploy Key，并可用 Ansible Vault 管理其私钥：

```bash
cd /home/ubuntu/video-mask/ansible
ansible-vault create group_vars/all/vault.yml
```

在 Vault 文件中填写（保留 Deploy Key 的完整多行内容）：

```yaml
vault_video_mask_source_deploy_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  REPLACE_WITH_READ_ONLY_REPOSITORY_DEPLOY_KEY
  -----END OPENSSH PRIVATE KEY-----
```

下面的 `settings.yml` 保持对该 Vault 变量的引用，并将仓库地址设为 SSH 地址：

```yaml
video_mask_repo: git@github.com:ryvengray/video-mask.git
video_mask_source_deploy_key: "{{ vault_video_mask_source_deploy_key }}"
```

创建 Vault 后，所有 Ansible 命令均需提供 Vault 密码；下文命令以 `--ask-vault-pass` 为例。

先生成两个随机 Token：

```bash
openssl rand -hex 32
openssl rand -hex 32
```

编辑变量文件：

```bash
nano group_vars/all/settings.yml
```

保留 `video_mask_repo` 等已有配置，并至少设置以下内容；把两处 Token 替换成刚刚生成的不同随机值：

```yaml
video_mask_repo: git@github.com:ryvengray/video-mask.git
video_mask_ref: main
video_mask_app_dir: /home/ubuntu/video-mask
video_mask_source_deploy_key: "{{ vault_video_mask_source_deploy_key }}"

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
ansible-playbook -i ansible/inventory.yml ansible/site.yml --syntax-check --ask-vault-pass
```

开始部署：

```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml -K --ask-vault-pass
```

`-K` 会询问当前 Ubuntu 用户的 sudo 密码。部署过程会：

```text
安装/检查系统依赖
→ 校验 CUDA、ONNX Runtime CUDA 与 YuNet Session
→ 启动 video-mask-controller.service
→ 在控制器预注册 worker-01
→ 启动 video-mask-worker.service
```

### 5.1 使用 `terraform-host.pem` 新增远程 Worker

以下操作在**运行 Ansible 的控制机**上执行；私钥必须位于这台机器，不能只存在于新 Worker 上。先限制私钥权限并验证 SSH 与免密 sudo：

```bash
chmod 600 /absolute/path/to/terraform-host.pem
ssh -i /absolute/path/to/terraform-host.pem ubuntu@WORKER_PUBLIC_OR_PRIVATE_IP \
  'sudo -n true && echo "SSH and passwordless sudo are ready"'
```

将 `inventory.yml` 的 `gpu_workers` 改为（Controller 仍保留为 `ansible_connection: local`）：

```yaml
gpu_workers:
  vars:
    ansible_user: ubuntu
    # 此路径属于运行 Ansible 的控制机。
    ansible_ssh_private_key_file: /absolute/path/to/terraform-host.pem
  hosts:
    worker-02:
      ansible_host: WORKER_PUBLIC_OR_PRIVATE_IP
      video_mask_worker_id: worker-02
      video_mask_worker_slots: 1
```

先检查连通性和 playbook，再只部署这台新 Worker：

```bash
cd /home/ubuntu/video-mask
ansible -i ansible/inventory.yml gpu_workers -m ping --limit worker-02 --ask-vault-pass
ansible-playbook -i ansible/inventory.yml ansible/site.yml --syntax-check --ask-vault-pass
ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit worker-02 --ask-vault-pass
```

若 `ubuntu` 用户没有免密 sudo，最后一条加 `-K`。也可以不在 inventory 保存私钥路径，而在两条 Ansible 命令末尾追加：

```bash
--private-key /absolute/path/to/terraform-host.pem
```

新 Worker 的安全组必须允许控制机到 Worker 的 TCP 22；同时 Worker 必须能访问 `video_mask_controller_url` 的 TCP 8080。私钥仅用于部署，Worker 运行时仍通过 HTTP 主动向 Controller 领取任务。

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
ssh -i /absolute/path/to/terraform-host.pem \
  -L 8080:127.0.0.1:8080 ubuntu@43.166.162.143
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

确认 `group_vars/all/settings.yml` 中的 `video_mask_worker_token` 未修改，并重新执行：

```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml -K --ask-vault-pass
```

远程 Worker 则将命令限制到对应主机，并使用 inventory 中的私钥配置（或追加 `--private-key`）：

```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit worker-02 --ask-vault-pass
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
