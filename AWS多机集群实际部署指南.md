# AWS 多机视频打码集群实际部署指南

本指南用于正式 AWS 环境：一台 Ubuntu Controller 同时作为 Ansible 控制机，多个 Ubuntu GPU Worker 通过私网领取任务；源视频和结果使用 S3 传递。它不替代 [多机集群Ubuntu测试部署指南.md](多机集群Ubuntu测试部署指南.md)，后者仅用于 Controller 与 Worker 同机的本地测试。

```text
管理员 SSH / Ansible
        │
Ubuntu Controller（Ansible + Controller API + SQLite + IAM Role）
        │ TCP 8080（仅 Worker 安全组）
        ├──────── Ubuntu GPU Worker 01
        ├──────── Ubuntu GPU Worker 02
        └──────── Ubuntu GPU Worker N

S3 source/inbox/ ──Controller 扫描──> Worker 以预签名 URL 下载
S3 outputs/      <──Worker 以预签名 URL 上传── Controller 记录状态
```

## 1. 部署前确认

- 所有节点必须使用 **Ubuntu 22.04 或 24.04**，SSH 登录用户为 `ubuntu`。当前 Ansible role 使用 `apt`、`/home/ubuntu` 和 `ubuntu` systemd 服务用户，不支持 Amazon Linux 的 `ec2-user`。
- Controller 可以是无 GPU 的 EC2；Worker 必须是带 NVIDIA GPU 的实例。Worker 初始化会在缺少 `nvidia-smi` 时安装 Ubuntu 推荐 NVIDIA 驱动，并在需要时自动重启一次后继续；非 GPU 实例仍会明确失败。
- 建议 Controller 和 Worker 都在私有子网。Controller 需要出站访问 GitHub、Ubuntu 软件源、AWS S3；Worker 首次部署还需访问 GitHub、Ubuntu 软件源、PyTorch/Hugging Face，运行时需访问 S3 预签名 URL。
- 准备 Terraform 生成的 SSH 私钥 `terraform-host.pem`。它只保存在 Controller（运行 Ansible 的机器）或你的受控运维主机，绝不提交到仓库。
- 准备 GitHub 仓库的**只读 Deploy Key**。它与 `terraform-host.pem` 不同，用于每台远程 Worker 拉取代码。
- 准备两个 S3 Bucket，或同一 Bucket 的两个不重叠前缀：输入 `source/inbox/`、输出 `outputs/`。

## 2. AWS 网络与 IAM

### 安全组

建议创建 `video-mask-controller-sg` 与 `video-mask-worker-sg`：

| 目标 | 入站规则 | 用途 |
| --- | --- | --- |
| Controller | TCP 22，仅运维出口 IP | 管理 Controller |
| Controller | TCP 8080，仅 `video-mask-worker-sg` | Worker 注册、心跳和领取任务 |
| Worker | TCP 22，仅 `video-mask-controller-sg` | Controller 上 Ansible 部署 Worker |
| Worker | 不开放公网入站 | Worker 不接收任务推送 |

不要将 TCP 8080 暴露到 `0.0.0.0/0`。Controller 页面需要查看时，通过 SSH 隧道访问：

```bash
ssh -i /absolute/path/to/terraform-host.pem \
  -L 8080:127.0.0.1:8080 ubuntu@CONTROLLER_PUBLIC_IP
```

### Controller IAM Role

只给 Controller EC2 Instance Profile 授权；Worker 不保存 AWS AK/SK，也不需要 Instance Profile。Controller 生成短时效的 S3 GET/PUT 预签名 URL 给 Worker。

将下列策略中的 Bucket 和前缀替换为实际值；输入与输出 Bucket 相同时，合并对应 ARN：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListInputPrefix",
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": "arn:aws:s3:::INPUT_BUCKET",
      "Condition": {"StringLike": {"s3:prefix": ["source/inbox/*"]}}
    },
    {
      "Sid": "ReadInput",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::INPUT_BUCKET/source/inbox/*"
    },
    {
      "Sid": "ReadAndWriteOutput",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::OUTPUT_BUCKET/outputs/*"
    },
    {
      "Sid": "IdentifyOutputBucketRegion",
      "Effect": "Allow",
      "Action": "s3:GetBucketLocation",
      "Resource": "arn:aws:s3:::OUTPUT_BUCKET"
    }
  ]
}
```

S3 Bucket Policy 还必须允许该 Controller IAM Role 访问。不要先启用要求额外上传请求头的 Bucket Policy（例如强制 SSE-KMS header）；当前 Worker 上传的预签名 PUT 不携带这些额外 header。

上传输入视频时先传到临时 key，例如 `uploading/<uuid>.partial`；校验完成后复制/移动到 `source/inbox/`。Controller 会周期扫描 `source/inbox/`，不要将未上传完成的对象直接放入此目录。

## 3. 用启动脚本部署同机的 Ansible + Controller

Controller 本机就是 Ansible 控制机：无需在 inventory 为 Controller 配置 SSH、`terraform-host.pem` 或 `ansible_user`。启动脚本会生成以下本地 Controller inventory：

```yaml
controller:
  hosts:
    controller-01:
      ansible_connection: local
      video_mask_manage_source: false
```

以下在新的 Controller 上以 `ubuntu` 用户执行。`CONTROLLER_PRIVATE_IP` 必须是 Worker 可访问的 VPC 私网 IP：

```bash
git clone https://github.com/ryvengray/video-mask.git /home/ubuntu/video-mask
cd /home/ubuntu/video-mask
git pull --ff-only

bash scripts/bootstrap_cluster_controller.sh \
  --controller-url http://CONTROLLER_PRIVATE_IP:8080 \
  --no-deploy
```

`--no-deploy` 用于正式环境：脚本会安装 Ansible、创建 `ansible/inventory.yml` 和 `ansible/group_vars/all/settings.yml`，但不会先以本地目录模式启动 Controller。接下来将其改为 S3 配置，并在第 4 节通过 Ansible 启动服务。

只有本地单机测试才省略 `--no-deploy`。不要在已有生产配置时随意使用 `--force-config`；它会备份并重新生成 inventory 和 settings 文件。

## 4. 配置 Vault、S3 与 inventory

创建 Ansible Vault；以下三个值都应只出现在 Vault 中。`vault_video_mask_source_deploy_key` 的值是专供 Ansible 拉取仓库的 **Git Deploy Key 私钥**，不是 Terraform 的 `terraform-host.pem`：

```bash
mkdir -p /home/ubuntu/.ssh
chmod 700 /home/ubuntu/.ssh
ssh-keygen -t ed25519 -N '' \
  -C 'video-mask-ansible-deploy' \
  -f /home/ubuntu/.ssh/video-mask-source-deploy
chmod 600 /home/ubuntu/.ssh/video-mask-source-deploy
```

在 GitHub 仓库中打开 **Settings → Deploy keys → Add deploy key**，名称可填 `aws-video-mask-ansible`。将下列命令输出的**公钥**粘贴进去，保持只读，不要勾选 `Allow write access`：

```bash
cat /home/ubuntu/.ssh/video-mask-source-deploy.pub
```

然后创建 Vault：

```bash
cd /home/ubuntu/video-mask/ansible
ansible-vault create group_vars/all/vault.yml
```

在编辑器中填写内容。将 `/home/ubuntu/.ssh/video-mask-source-deploy` 的**私钥全部内容**（包括首尾 `BEGIN`/`END` 行）粘贴到 `vault_video_mask_source_deploy_key: |` 下，并且每一行缩进两个空格：

```yaml
vault_video_mask_admin_token: REPLACE_WITH_A_RANDOM_32_BYTE_HEX_TOKEN
vault_video_mask_worker_token: REPLACE_WITH_A_DIFFERENT_RANDOM_32_BYTE_HEX_TOKEN
vault_video_mask_source_deploy_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  REPLACE_WITH_THE_PRIVATE_KEY_CONTENT
  -----END OPENSSH PRIVATE KEY-----
```

生成 Token 可使用两次 `openssl rand -hex 32`。保存 Vault 密码到受控的密码管理器；不要将 Vault 密码、Git Deploy Key 私钥或 Terraform SSH 私钥填入 Git 仓库。远程 Worker 部署时，Ansible 会从 Vault 读取该私钥，并安全地复制到 Worker 的 `/home/ubuntu/.ssh/video-mask-source`，用于拉取 `video_mask_repo`。

编辑 `ansible/group_vars/all/settings.yml`，替换为实际 S3 配置：

```yaml
video_mask_repo: git@github.com:ryvengray/video-mask.git
video_mask_ref: main
video_mask_app_dir: /home/ubuntu/video-mask
video_mask_source_deploy_key: "{{ vault_video_mask_source_deploy_key }}"

video_mask_controller_url: http://CONTROLLER_PRIVATE_IP:8080
video_mask_admin_token: "{{ vault_video_mask_admin_token }}"
video_mask_worker_token: "{{ vault_video_mask_worker_token }}"

video_mask_worker_slots: 1
video_mask_storage_mode: s3
video_mask_s3_source_bucket: INPUT_BUCKET
video_mask_s3_source_prefix: source/inbox/
video_mask_s3_source_region: AWS_REGION
video_mask_s3_output_bucket: OUTPUT_BUCKET
video_mask_s3_output_prefix: outputs/
# 留空时默认使用输入 Region；输出 Bucket 在另一个 Region 时必须明确填写其 Region。
video_mask_s3_output_region: OUTPUT_AWS_REGION
# EC2 IAM Role 使用默认凭证链，因此保持为空。
video_mask_s3_profile: ''
video_mask_s3_poll_seconds: 60
video_mask_s3_presign_seconds: 86400
```

确认 Controller inventory 保持本地执行，且跳过已完成的本地源码 checkout：

```yaml
controller:
  hosts:
    controller-01:
      ansible_connection: local
      video_mask_manage_source: false

gpu_workers:
  hosts: {}
```

部署 Controller：

```bash
cd /home/ubuntu/video-mask
ansible-playbook -i ansible/inventory.yml ansible/site.yml \
  --limit controller --ask-vault-pass

curl http://127.0.0.1:8080/healthz
sudo systemctl status video-mask-controller --no-pager
```

## 5. 添加第一台 GPU Worker

将 PEM 放在运行 Ansible 的 Controller，限制权限并先验证登录。`WORKER_PRIVATE_IP` 建议使用 VPC 私网 IP：

```bash
mkdir -p /home/ubuntu/.ssh
install -m 600 /path/you/received/terraform-host.pem \
  /home/ubuntu/.ssh/terraform-host.pem
ssh -i /home/ubuntu/.ssh/terraform-host.pem ubuntu@WORKER_PRIVATE_IP \
  'sudo -n true && nvidia-smi -L'
```

编辑 `ansible/inventory.yml`：

```yaml
controller:
  hosts:
    controller-01:
      ansible_connection: local
      video_mask_manage_source: false

gpu_workers:
  vars:
    ansible_user: ubuntu
    ansible_ssh_private_key_file: /home/ubuntu/.ssh/terraform-host.pem
    ansible_ssh_common_args: "-o StrictHostKeyChecking=accept-new"
  hosts:
    worker-01:
      ansible_host: WORKER_PRIVATE_IP
      video_mask_worker_id: worker-01
      video_mask_worker_slots: 1
```

执行连通性检查和部署：

```bash
cd /home/ubuntu/video-mask
ansible -i ansible/inventory.yml gpu_workers -m ping \
  --limit worker-01 --ask-vault-pass
ansible-playbook -i ansible/inventory.yml ansible/site.yml --syntax-check \
  --ask-vault-pass
ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit worker-01 \
  --ask-vault-pass
```

Worker role 会拉取指定 Git ref、安装 CUDA 人脸流水线依赖、在缺少驱动时安装 Ubuntu 推荐 NVIDIA 驱动并自动重启一次、校验 CUDA/ONNX Runtime、向 Controller 预注册 Worker slot，并创建和启动 `video-mask-worker@slot-1.service`。

验证：

```bash
ssh -i /home/ubuntu/.ssh/terraform-host.pem ubuntu@WORKER_PRIVATE_IP \
  'sudo systemctl status video-mask-worker@slot-1 --no-pager && nvidia-smi'
curl -H "Authorization: Bearer ADMIN_TOKEN" \
  http://127.0.0.1:8080/api/workers
```

从第二台 Worker 起，只需在 `gpu_workers.hosts` 添加新主机，再将 `--limit` 改为该 Worker 名称。每个物理 Worker 都必须有不同的 `video_mask_worker_id`。

## 6. 投入运行与运维

1. 上传完成的视频到 `s3://INPUT_BUCKET/source/inbox/`。
2. Controller 每 60 秒扫描一次；空闲 Worker 自动领取任务。
3. 输出写入 `s3://OUTPUT_BUCKET/outputs/`，保存名为 `masked_<原文件名>.mp4`，并保留输入子目录结构。
4. 查看状态使用 SSH 隧道后的 `http://127.0.0.1:8080/`：任务页每页展示 50 条，可翻页查看持续累积的历史任务；完成任务会显示输入/输出文件、大小、输出时长、处理耗时、算法和参数。也可通过 Bearer Token 调用 `/api/tasks?limit=50&offset=0`、`/api/workers`。若 S3 上传首次失败（例如预签名 PUT URL 过期），Worker 会向 Controller 获取一个新的上传 URL 并重试一次，不重新下载或处理视频。

常用命令：

```bash
# Controller
sudo journalctl -u video-mask-controller -f

# 某台 Worker
sudo journalctl -u video-mask-worker@slot-1 -f

# Controller 日常任务管理：列出未完成任务、查看详情、取消或重新入队。
# 以 sudo 运行会从 /etc/video-mask-controller.env 读取 Admin Token。
sudo /home/ubuntu/video-mask/.venv/bin/python scripts/cluster_manager.py

# 或重启指定远程 Worker slot（会提示确认，并要求 Vault 密码）。
sudo /home/ubuntu/video-mask/.venv/bin/python scripts/cluster_manager.py \
  restart-slot worker-01 --slot 1

# 可一次重启 slot 1 到 15；任何运行中任务都会被中断。
sudo /home/ubuntu/video-mask/.venv/bin/python scripts/cluster_manager.py \
  restart-slot worker-01 --slot 1-15

# 部署更新后的 Worker 代码；先在 settings.yml 更新 video_mask_ref。
bash scripts/deploy_worker.sh worker-01

# 部署 Controller 配置/代码；--restart 可强制重启服务。
bash scripts/deploy_controller.sh --restart
```

当前 Controller 页面是只读状态页；Worker 上下线使用对应主机的 `systemctl start|stop video-mask-worker@slot-1`。管理脚本可立即取消 pending 任务；对于运行中的任务，会请求 Worker 在下一次心跳（最长约 15 秒）终止算法并标记为 `cancelled`。已完成、失败或取消的任务可由脚本重新入队，无需删除 Controller SQLite 数据库。

## 7. 按任务量自动启动和关闭 Worker EC2

当运维提供的 EC2 机器池文件和启停脚本都安装在 Controller 上时，可启用 Controller 侧 autoscaler。它不运行在 Worker 上，也不会将 AWS 凭证或 EC2 启停权限下发到 Worker。

它会每分钟读取 Controller 队列与 `/opt/dataai-ec2/data/ec2_host_pool.yaml`：当 `pending` 任务数大于 `ready` slot 数时，从白名单内 `stopped` 的机器中启动一台；当没有 pending 任务，且一台机器的全部已注册 slot 都持续 `ready` 达到指定时间时，才关闭该机器。`busy`、下载中、处理中、上传中、取消中，以及尚未注册到 Controller 的机器均不会被关闭。

在 `ansible/group_vars/all/settings.yml` 添加（IP 必须是你允许本系统自动启停的 Worker 私网 IP）：

```yaml
video_mask_autoscale_enabled: true
video_mask_autoscale_pool_file: /opt/dataai-ec2/data/ec2_host_pool.yaml
video_mask_autoscale_start_command: /opt/dataai-ec2/bin/start_ec2.sh
video_mask_autoscale_stop_command: /opt/dataai-ec2/bin/stop_ec2.sh
video_mask_autoscale_managed_ips:
  - 172.31.35.195 # slave-01
  - 172.31.47.141 # slave-02
video_mask_autoscale_check_seconds: 60
video_mask_autoscale_idle_shutdown_seconds: 1800 # 30 分钟
video_mask_autoscale_min_running_hosts: 0
video_mask_autoscale_max_start_per_check: 1
video_mask_autoscale_max_stop_per_check: 1
video_mask_autoscale_start_grace_seconds: 900 # EC2 启动和 Worker 注册的等待窗口
```

部署 Controller：

```bash
bash scripts/deploy_controller.sh --restart
sudo systemctl status video-mask-autoscaler --no-pager
sudo journalctl -u video-mask-autoscaler -f
```

Ansible 会为 `ubuntu` 安装最小 sudo 权限，仅允许 autoscaler 调用以下两个固定脚本并传入 `--ips`：

```text
/opt/dataai-ec2/bin/start_ec2.sh
/opt/dataai-ec2/bin/stop_ec2.sh
```

首次建议先手工以 dry-run 核验决策，确认日志只显示预期 IP 后再部署正式服务：

```bash
sudo -u ubuntu /home/ubuntu/video-mask/.venv/bin/python -m cluster.autoscaler \
  --controller-url http://127.0.0.1:8080 \
  --pool-file /opt/dataai-ec2/data/ec2_host_pool.yaml \
  --start-command /opt/dataai-ec2/bin/start_ec2.sh \
  --stop-command /opt/dataai-ec2/bin/stop_ec2.sh \
  --managed-ips 172.31.35.195,172.31.47.141 \
  --idle-shutdown-seconds 1800 --once --dry-run
```

前提：机器池 YAML 的 `status` 必须由运维脚本或其刷新机制及时更新；Worker slot 服务必须是 `enabled`，让实例启动后自动注册。若机器池状态长时间未更新，autoscaler 的 15 分钟启动等待窗口会避免重复 start，但应先修复运维侧刷新再长期启用。

## 8. 上线验收清单

- [ ] Controller 和所有 Worker 均为 Ubuntu，Controller API 仅对 Worker 安全组开放 TCP 8080。
- [ ] `nvidia-smi`、`torch.cuda.is_available()`、ONNX Runtime CUDA Provider 均在每台 Worker 可用。
- [ ] Controller EC2 IAM Role 能列举输入、读取输入、检查及写入输出；Worker 未配置长期 AWS 凭证。
- [ ] Git Deploy Key、Ansible Vault 密码、`terraform-host.pem` 不在 Git 工作区或日志中。
- [ ] 以一个短视频验证 S3 下载、打码、S3 上传、哈希和时长记录全链路。
- [ ] 再以多个视频和至少两台 Worker 验证并发领取；一台 Worker 停止后验证心跳超时重试。
- [ ] autoscaler 先以 `--once --dry-run` 验证，再用一条 pending 任务验证仅启动一个 stopped Worker；最后以空闲阈值验证不会关闭 busy 或未注册的机器。
