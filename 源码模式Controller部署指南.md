# 源码模式 Controller 部署指南

本文用于在客户的 Controller 机器上部署视频打码集群。Controller 与 Worker 都从私有 `video-mask` 源码仓库运行；Controller 同时作为 Ansible 控制端，通过 SSH 初始化和升级 Worker。

## 1. 网络与机器要求

- Controller：Ubuntu 24.04，建议 2 核 CPU、4 GB 内存起。
- Worker：Ubuntu 24.04 x86_64，NVIDIA GPU Worker 建议 T4 及以上，驱动需预先由云镜像或云厂商安装。
- Controller 和 Worker 使用私网通信；Worker 可访问 Controller 的 TCP `8080`。
- Controller 需要能访问 GitHub，Worker 需要能访问 GitHub、PyPI、PyTorch 下载源和 Hugging Face（首次下载模型时）。

Controller 私网 IP 可通过下面命令查看：

```bash
hostname -I
```

## 2. 创建共享源码只读 Deploy Key（一次性）

这把 Key 用于 Controller 和所有 Worker 读取私有源码仓库。它只授权 `ryvengray/video-mask`，不要勾选 GitHub 写权限。

在受信任机器生成：

```bash
ssh-keygen -t ed25519 \
  -f ~/.ssh/video-mask-source \
  -C "video-mask-cluster-source-readonly" \
  -N ""

cat ~/.ssh/video-mask-source.pub
```

在 GitHub 的 `ryvengray/video-mask` 仓库中进入 `Settings` → `Deploy keys` → `Add deploy key`，粘贴公钥，**不要**勾选 `Allow write access`。

将私钥安全传到 Controller，保存为：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# 将私钥内容保存到 ~/.ssh/video-mask-source 后执行：
chmod 600 ~/.ssh/video-mask-source

ssh -i ~/.ssh/video-mask-source -o IdentitiesOnly=yes -T git@github.com
```

看到 GitHub 认证成功但不提供 shell 即为正常。

## 3. 在 Controller 安装工具并 clone 源码

```bash
sudo apt-get update
sudo apt-get install -y ansible git

GIT_SSH_COMMAND="ssh -i ~/.ssh/video-mask-source -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  git clone git@github.com:ryvengray/video-mask.git ~/video-mask

cd ~/video-mask/ansible
```

## 4. 配置 Ansible

创建客户本机配置，这些文件已经被 `.gitignore` 忽略，不能提交：

```bash
cp inventory.yml.example inventory.yml
mkdir -p group_vars/all
cp group_vars/all/settings.yml.example group_vars/all/settings.yml
ansible-vault create group_vars/all/vault.yml
```

Vault 密码是本地加密密码，请保存到密码管理器。创建 Vault 后填写：

```yaml
vault_video_mask_source_deploy_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  粘贴 ~/.ssh/video-mask-source 的完整私钥内容
  -----END OPENSSH PRIVATE KEY-----

vault_video_mask_admin_token: 替换为至少32位随机Token
vault_video_mask_worker_token: 替换为至少32位随机Token
```

Token 可生成：

```bash
openssl rand -hex 32
openssl rand -hex 32
```

编辑 `inventory.yml`。Controller 与 Ansible 在同一台机器：

```yaml
controller:
  hosts:
    controller-01:
      ansible_connection: local
      ansible_user: ubuntu

gpu_workers:
  hosts: {}
```

编辑 `group_vars/all/settings.yml`，至少替换 Controller 私网 IP：

```yaml
video_mask_repo: git@github.com:ryvengray/video-mask.git
video_mask_ref: main
video_mask_app_dir: /home/ubuntu/video-mask
video_mask_source_deploy_key_path: /home/ubuntu/.ssh/video-mask-source
video_mask_source_deploy_key: "{{ vault_video_mask_source_deploy_key }}"

video_mask_controller_url: http://Controller私网IP:8080
video_mask_admin_token: "{{ vault_video_mask_admin_token }}"
video_mask_worker_token: "{{ vault_video_mask_worker_token }}"

video_mask_storage_mode: local
video_mask_local_source_dir: /home/ubuntu/cluster_test_sources
video_mask_local_output_dir: /home/ubuntu/cluster_test_outputs
```

## 5. 启动 Controller

先检查变量与本机连接：

```bash
ansible -i inventory.yml controller -m ping --ask-vault-pass
```

预期得到 `ping: pong`。

初始化并启动：

```bash
ansible-playbook -i inventory.yml site.yml \
  --limit controller \
  -K --ask-vault-pass
```

其中：

- `-K`：输入 Ubuntu 的 sudo 密码；若用户已配置免密 sudo，可省略。
- `--ask-vault-pass`：输入第 4 步创建 Vault 时的密码。

Ansible 会拉取源码、创建 `/home/ubuntu/video-mask/.venv`、安装 Controller 依赖，并创建 `video-mask-controller.service`。

## 6. 验证 Controller

```bash
sudo systemctl status video-mask-controller --no-pager
curl -fsS http://127.0.0.1:8080/healthz
sudo journalctl -u video-mask-controller -n 100 --no-pager
```

浏览器访问 Controller 页面时，建议仅通过私网 VPN、堡垒机或严格限制的安全组访问 `http://Controller私网IP:8080/`，不要对公网开放。

## 7. 添加 Worker

### 7.1 选择 Controller 到 Worker 的 SSH 认证方式

Ansible 需要 Controller 无交互登录每台 Worker。支持两种方式：统一密码 SSH（部署方便）或 Controller 专用 SSH Key（更推荐）。两种方式只能选一种，不要在 inventory 同时配置 `ansible_password` 与 `ansible_ssh_private_key_file`。

#### 方式 A：所有 Worker 使用同一账号密码

Controller 先安装密码 SSH 所需组件：

```bash
sudo apt-get update
sudo apt-get install -y sshpass
```

Worker 会动态新增时，在 inventory 中使用 `StrictHostKeyChecking=accept-new`。它会自动记录**首次出现**的 Worker 主机指纹；若同一个 IP 的指纹之后发生变化，仍会拒绝连接，避免无条件信任未知主机。

将 Worker 的 SSH 登录密码和 sudo 密码加密保存到 Vault。若二者相同，可以填写相同内容：

```bash
cd ~/video-mask/ansible
ansible-vault edit group_vars/all/vault.yml
```

添加：

```yaml
vault_video_mask_worker_ssh_password: "所有Worker共用的SSH登录密码"
vault_video_mask_worker_sudo_password: "所有Worker共用的sudo密码"
```

密码只能存在 `vault.yml` 中，不能写进 `inventory.yml`、Shell 历史或 Git 仓库。

#### 方式 B：Controller 专用 SSH Key（推荐）

若当前只能输入 `ubuntu` 密码 SSH，使用该密码一次性安装 Controller 的 SSH 公钥；之后不再需要保存 Worker 登录密码。

在 Controller 生成管理 Key（若文件已存在，不要覆盖）：

```bash
test -f ~/.ssh/video-mask-ansible || \
  ssh-keygen -t ed25519 -f ~/.ssh/video-mask-ansible -C "video-mask-ansible-controller" -N ""
chmod 600 ~/.ssh/video-mask-ansible
```

把公钥安装到新 Worker。该命令会询问 **一次** Worker 的 `ubuntu` 密码：

```bash
ssh-copy-id -i ~/.ssh/video-mask-ansible.pub ubuntu@10.200.0.7
```

测试无密码 SSH：

```bash
ssh -i ~/.ssh/video-mask-ansible -o IdentitiesOnly=yes ubuntu@10.200.0.7 'hostname && nvidia-smi -L'
```

若 Worker 的 sudo 需要密码，部署时使用 Playbook 的 `-K` 输入 sudo 密码；更推荐使用云镜像默认的受控免密 sudo。

### 7.2 添加 inventory

在 `inventory.yml` 增加 Worker 私网 IP，并按上一节选择一种认证配置。

**统一密码 SSH 示例：**

```yaml
gpu_workers:
  vars:
    ansible_user: ubuntu
    ansible_password: "{{ vault_video_mask_worker_ssh_password }}"
    ansible_become_password: "{{ vault_video_mask_worker_sudo_password }}"
    ansible_ssh_common_args: "-o StrictHostKeyChecking=accept-new"
  hosts:
    worker-01:
      ansible_host: 10.0.2.20
      video_mask_worker_id: worker-01
      video_mask_worker_slots: 2
```

没有配置 `accept-new` 时，需要在 Controller 上先手动连接每台新 Worker 并确认一次主机指纹：

```bash
ssh ubuntu@Worker私网IP
# 首次提示时输入 yes，再输入 Worker 密码，登录后 exit
```

**SSH Key 示例：**

```yaml
gpu_workers:
  vars:
    ansible_user: ubuntu
    ansible_ssh_private_key_file: /home/ubuntu/.ssh/video-mask-ansible
  hosts:
    worker-01:
      ansible_host: 10.0.2.20
      video_mask_worker_id: worker-01
      video_mask_worker_slots: 2
```

确认安全组允许 Controller SSH 到 Worker 的 TCP `22`，并允许 Worker 访问 Controller TCP `8080`。然后在 Controller 上执行：

```bash
# 密码 SSH：密码已在 Vault 中，不需要 -K
ansible -i inventory.yml gpu_workers -m ping --ask-vault-pass
ansible-playbook -i inventory.yml site.yml \
  --limit gpu_workers --ask-vault-pass

# SSH Key：若 Worker sudo 需要密码，使用 -K
ansible-playbook -i inventory.yml site.yml \
  --limit gpu_workers \
  -K --ask-vault-pass
```

`video_mask_worker_slots` 表示一台物理 Worker 同时处理的任务数，默认值为 `1`。每个槽位会成为一个独立的 Controller Worker，例如上例会注册并启动 `worker-01-slot-1` 与 `worker-01-slot-2`；每个槽位一次领取一个任务。

建议先使用 `2` 个槽位，观察 GPU SM、显存、CPU 与单视频速度后再增至 `4`。同一台 T4 上盲目设置过多槽位可能降低单任务速度，应该以总吞吐量判断。部署会停用旧的单实例 `video-mask-worker.service`，改为 `video-mask-worker@slot-1.service` 等模板实例；请在该 Worker 没有集群任务执行时升级。

Worker 会自动获得源码只读 Key、clone 源码、安装 CUDA/Python 依赖，并启动每个槽位服务。启动后 Worker 会自行轮询 Controller 领取任务；不需要让 Ansible 常驻运行。

## 8. 添加视频打码任务

Worker 安装完成后会自动注册到 Controller 并持续轮询任务。Ansible 只负责部署，不负责添加视频任务。

### 本机目录模式仅限单机测试

`video_mask_storage_mode: local` 会扫描 Controller 的 `video_mask_local_source_dir`，生成 `file://` 任务。该方式仅适用于 Controller 与 Worker 在同一台机器，或两者挂载了同一个共享文件系统且路径完全一致。

**当前远程 Worker 不能直接处理 Controller 本机 `/home/ubuntu/cluster_test_sources` 中的文件。**

### 远程 Worker 使用任务 API

给远程 Worker 添加任务时，调用 Controller 的 `POST /api/tasks`。任务必须提供：

- `source_url`：Worker 能下载源视频的 URL；生产环境使用 S3 预签名 GET URL。
- `output_upload_url`：可选。生产环境使用 S3 预签名 PUT URL；省略时，Worker 将结果保存到自身的 `/home/ubuntu/outputs`。

在 Controller 上执行（将 Token 和两个 URL 替换为真实值）：

```bash
curl -fsS -X POST "http://127.0.0.1:8080/api/tasks" \
  -H "Authorization: Bearer 你的管理员Token" \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://S3预签名下载URL",
    "output_upload_url": "https://S3预签名上传URL",
    "source_object_key": "incoming/video_001.mp4",
    "output_object_key": "masked/video_001.mp4"
  }'
```

Controller 会立即将任务标记为 `pending`；空闲 Worker 会领取、下载、处理并上传结果。当前默认算法与参数为：

```text
video_mask_batch_fish.py
--fisheye --fisheye-device pico4 --no-card --face-size 960 --face-int 5 --frame-skip 3 --face-model yolov8
```

查看任务与 Worker 状态：

```bash
curl -fsS "http://127.0.0.1:8080/api/tasks" \
  -H "Authorization: Bearer 你的管理员Token"

curl -fsS "http://127.0.0.1:8080/api/workers" \
  -H "Authorization: Bearer 你的管理员Token"
```

浏览器访问 `http://Controller私网IP:8080/` 也可以查看最近任务与 Worker 状态。Worker 表会显示 Worker ID、主机名、Controller 实际看到的私网 IP、槽位编号、GPU、CUDA 状态、Worker PID、当前任务和最后心跳时间。

> 后续可在 Controller 增加 S3 任务导入服务：扫描指定 Prefix、生成预签名 URL、调用本 API。这样业务人员只需上传视频到 S3，无需手动执行 `curl`。

### S3 自动分发与结果上传

S3 模式下，Controller 使用自身的 AWS Profile 扫描源桶，将视频自动加入任务队列。Worker 领取任务的瞬间，Controller 才生成新的预签名下载 URL 和上传 URL：Worker 不需要 AWS AK/SK，也不会因为任务排队导致 URL 在处理前过期。

在 **Controller** 上以运行服务的 `ubuntu` 用户配置 AWS Profile（不要用 `root` 配置，否则 systemd 服务无法读取）：

```bash
sudo -u ubuntu -H aws configure --profile s3-test
sudo -u ubuntu -H aws sts get-caller-identity --profile s3-test
```

所用 IAM 身份至少需要：源桶 `s3:ListBucket`、`s3:GetObject`；结果桶 `s3:ListBucket`、`s3:GetObject`（用于跳过已有结果）、`s3:PutObject`。AK/SK 只保存在 Controller 的 `/home/ubuntu/.aws/`，不要写入 Git、Ansible inventory 或 Vault 以外的明文文件。

编辑 `ansible/group_vars/all/settings.yml`：

```yaml
video_mask_storage_mode: s3
video_mask_s3_source_bucket: dataai-mp4-685538570851-us-east-2-an
video_mask_s3_source_prefix: ''
video_mask_s3_source_region: us-east-2
video_mask_s3_output_bucket: processed-video-685538570851-us-east-2-an
video_mask_s3_output_prefix: outputs/
video_mask_s3_profile: s3-test
video_mask_s3_poll_seconds: 60
video_mask_s3_presign_seconds: 86400
```

部署 Controller：

```bash
cd ~/video-mask/ansible
ansible-playbook -i inventory.yml site.yml \
  --limit controller -K --ask-vault-pass
```

首次健康检查会扫描源桶；之后 Controller 每 60 秒扫描一次，或有 Worker 领取任务时按周期扫描一次：

```bash
curl -fsS http://127.0.0.1:8080/healthz
```

返回中的 `s3_ingested` 是本次新创建任务数。结果对象会上传至 `s3://processed-video-685538570851-us-east-2-an/outputs/`，保留源视频的子目录，并加上 `masked_` 前缀。数据库已记录的同一对象版本、或结果桶已存在的输出，会自动跳过。

预签名 URL 默认在任务领取后有效 24 小时，可设置到最多 7 天。当前上传方式是单次 S3 `PutObject`，单个结果文件应小于 5 GiB；更大的文件需要后续增加 multipart upload。

若修复 IAM 权限、网络或算法配置后需要重新执行一个已失败的任务，使用管理员 Token 将其重新放回队列（重置该任务的重试次数）：

```bash
curl -fsS -X POST "http://127.0.0.1:8080/api/tasks/任务ID/retry" \
  -H "Authorization: Bearer 你的管理员Token"
```

### 无 S3 时的远程 Worker 测试

可以只提供一个 Worker 能访问的 HTTP/HTTPS 下载 URL，并省略 `output_upload_url`：

```bash
curl -fsS -X POST "http://127.0.0.1:8080/api/tasks" \
  -H "Authorization: Bearer 你的管理员Token" \
  -H "Content-Type: application/json" \
  -d '{"source_url":"https://可下载的视频URL"}'
```

Worker 下载、处理后会将视频保存至该 Worker 的 `/home/ubuntu/outputs/`，并向 Controller 标记 `completed`。任务详情的 `progress.output_location` 会记录实际本地路径。

## 9. 更新源码版本

所有更新命令都在 **Controller** 的 `~/video-mask/ansible` 目录执行。先更新 Controller，再更新 Worker；这样新任务 API（例如省略 `output_upload_url` 时保存到 Worker 本地）会先在 Controller 生效。

先更新 Controller 本机的源码与 Ansible 文件：

```bash
cd ~/video-mask
git pull --ff-only
cd ansible
```

Controller 或 Worker 升级时，先在 `group_vars/all/settings.yml` 确认：

```yaml
video_mask_ref: main
```

也可以填写某个固定 Git tag 或 commit SHA，以保证生产版本稳定。随后重新执行对应范围的 Playbook：

```bash
# 更新 Controller
ansible-playbook -i inventory.yml site.yml --limit controller -K --ask-vault-pass

# 更新所有 Worker
ansible-playbook -i inventory.yml site.yml --limit gpu_workers -K --ask-vault-pass
```

更新 Worker 后，可确认新服务参数已生效：

```bash
ssh ubuntu@Worker私网IP 'sudo systemctl status "video-mask-worker@slot-*" --no-pager && ls -ld ~/outputs'
```

## 10. 常见问题

### `Attempting to decrypt but no vault secrets found`

命令缺少 `--ask-vault-pass`。所有 Ansible 命令均应带上该参数。

### `Permission denied (publickey)`

检查 `~/.ssh/video-mask-source`、GitHub `video-mask` 仓库 Deploy Key 与 Vault 中的 `vault_video_mask_source_deploy_key` 是否为同一把私钥。

### `nvidia-smi` 不可用

Worker 的 NVIDIA 驱动尚未安装或没有使用 GPU 云镜像。先按云厂商文档安装驱动并确认：

```bash
nvidia-smi
```

然后重新运行 Worker Playbook。
