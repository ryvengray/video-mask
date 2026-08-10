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

在 `inventory.yml` 增加 Worker 私网 IP：

```yaml
gpu_workers:
  vars:
    ansible_user: ubuntu
    ansible_ssh_private_key_file: /secure/path/worker.pem
  hosts:
    worker-01:
      ansible_host: 10.0.2.20
      video_mask_worker_id: worker-01
```

确认安全组允许 Controller SSH 到 Worker，并允许 Worker 访问 Controller TCP `8080`。然后在 Controller 上执行：

```bash
ansible-playbook -i inventory.yml site.yml \
  --limit gpu_workers \
  -K --ask-vault-pass
```

Worker 会自动获得源码只读 Key、clone 源码、安装 CUDA/Python 依赖，并启动 `video-mask-worker.service`。启动后 Worker 会自行轮询 Controller 领取任务；不需要让 Ansible 常驻运行。

## 8. 更新源码版本

Controller 或 Worker 升级时，先在 `group_vars/all/settings.yml` 修改：

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

## 9. 常见问题

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
