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
- Controller 可以是无 GPU 的 EC2；Worker 必须是带 NVIDIA GPU 的实例，且系统中已可运行 `nvidia-smi`。现有 playbook 安装 Python/CUDA 运行库，但不安装 NVIDIA 驱动。
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
      "Action": "s3:ListBucket",
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
    }
  ]
}
```

S3 Bucket Policy 还必须允许该 Controller IAM Role 访问。不要先启用要求额外上传请求头的 Bucket Policy（例如强制 SSE-KMS header）；当前 Worker 上传的预签名 PUT 不携带这些额外 header。

上传输入视频时先传到临时 key，例如 `uploading/<uuid>.partial`；校验完成后复制/移动到 `source/inbox/`。Controller 会周期扫描 `source/inbox/`，不要将未上传完成的对象直接放入此目录。

## 3. 启动 Controller 与 Ansible 控制机

以下在新的 Controller 上以 `ubuntu` 用户执行。`CONTROLLER_PRIVATE_IP` 必须是 Worker 可访问的 VPC 私网 IP：

```bash
git clone https://github.com/ryvengray/video-mask.git /home/ubuntu/video-mask
cd /home/ubuntu/video-mask
git pull --ff-only

bash scripts/bootstrap_cluster_controller.sh \
  --controller-url http://CONTROLLER_PRIVATE_IP:8080 \
  --no-deploy
```

脚本会安装 Ansible、创建 `ansible/inventory.yml` 和 `ansible/group_vars/all/settings.yml`，但不会先以本地目录模式启动 Controller。接下来将其改为 S3 正式配置。

## 4. 配置 Vault、S3 与 inventory

创建 Ansible Vault；以下三个值都应只出现在 Vault 中：

```bash
cd /home/ubuntu/video-mask/ansible
ansible-vault create group_vars/all/vault.yml
```

填写内容：

```yaml
vault_video_mask_admin_token: REPLACE_WITH_A_RANDOM_32_BYTE_HEX_TOKEN
vault_video_mask_worker_token: REPLACE_WITH_A_DIFFERENT_RANDOM_32_BYTE_HEX_TOKEN
vault_video_mask_source_deploy_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  REPLACE_WITH_READ_ONLY_GITHUB_DEPLOY_KEY
  -----END OPENSSH PRIVATE KEY-----
```

生成 Token 可使用两次 `openssl rand -hex 32`。将只读 Deploy Key 的公钥添加到 GitHub 仓库的 Deploy Keys；不要把 Terraform SSH 私钥填进 Vault。

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

Worker role 会拉取指定 Git ref、安装 CUDA 人脸流水线依赖、校验 CUDA/ONNX Runtime、向 Controller 预注册 Worker slot，并创建和启动 `video-mask-worker@slot-1.service`。

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
4. 查看状态使用 SSH 隧道后的 `http://127.0.0.1:8080/`，或以 Bearer Token 调用 `/api/tasks`、`/api/workers`。

常用命令：

```bash
# Controller
sudo journalctl -u video-mask-controller -f

# 某台 Worker
sudo journalctl -u video-mask-worker@slot-1 -f

# 部署更新后的 Worker 代码；先在 settings.yml 更新 video_mask_ref。
ansible-playbook -i ansible/inventory.yml ansible/site.yml \
  --limit worker-01 --ask-vault-pass
```

当前 Controller 页面是只读状态页；Worker 上下线使用对应主机的 `systemctl start|stop video-mask-worker@slot-1`。已完成任务不会因重新扫描自动重跑；需要保留重处理能力时，应通过新增任务或后续管理 API 实现，而不是删除 Controller SQLite 数据库。

## 7. 上线验收清单

- [ ] Controller 和所有 Worker 均为 Ubuntu，Controller API 仅对 Worker 安全组开放 TCP 8080。
- [ ] `nvidia-smi`、`torch.cuda.is_available()`、ONNX Runtime CUDA Provider 均在每台 Worker 可用。
- [ ] Controller EC2 IAM Role 能列举输入、读取输入、检查及写入输出；Worker 未配置长期 AWS 凭证。
- [ ] Git Deploy Key、Ansible Vault 密码、`terraform-host.pem` 不在 Git 工作区或日志中。
- [ ] 以一个短视频验证 S3 下载、打码、S3 上传、哈希和时长记录全链路。
- [ ] 再以多个视频和至少两台 Worker 验证并发领取；一台 Worker 停止后验证心跳超时重试。
