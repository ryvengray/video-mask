# Release 模式 Controller 启动指南

本指南用于在一台新的 Ubuntu 24.04 x86_64 机器上启动视频打码集群 Controller。Controller 以 Cython release 包运行，不 clone、不安装、不执行 `video-mask` 源码。

## 0. 部署边界

Controller 可作为 Ansible 控制端，但它只能使用 release 仓库中随发布包同步的无源码部署包，**不得** clone 当前 `video-mask` 源码仓库：

```text
构建机（持有源码，仅负责发布）
          │ push release
          ▼
客户 Controller（release + 无源码 Ansible + Vault）
          │ SSH / Ansible
          ▼
客户 Worker（release + 公开运行依赖）
```

Controller 需要能访问 GitHub release 仓库；Worker 通过内网访问 Controller 的 TCP `8080`。内网部署不需要 HTTPS，但安全组应只允许 Worker 网段访问 `8080`。

## 0.1 创建共享只读 release Deploy Key（一次性）

这把 Key 仅用于读取 `video-mask-release`，由客户 Controller 保存，并由 Ansible 自动分发给所有动态 Worker。不要使用构建服务器用于 push release 的可写 Key，也不要给它源码仓库权限。

在受信任的构建/运维机生成一次：

```bash
ssh-keygen -t ed25519 \
  -f ~/.ssh/video-mask-release-readonly \
  -C "video-mask-release-cluster-readonly" \
  -N ""

cat ~/.ssh/video-mask-release-readonly.pub
```

在 GitHub 打开 `ryvengray/video-mask-release`：

1. 进入 `Settings` → `Deploy keys` → `Add deploy key`；
2. 名称填写 `video-mask-release-cluster-readonly`；
3. 粘贴上一步输出的 **公钥**；
4. **不要勾选** `Allow write access`；
5. 保存。

随后通过受控安全渠道把私钥文件 `~/.ssh/video-mask-release-readonly` 交给客户 Controller，并在 Controller 上保存为：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# 将收到的私钥保存为下面这个文件；不要把它提交到任何 Git 仓库。
chmod 600 ~/.ssh/video-mask-release
```

Controller 后续会将同一份私钥加密保存到 Ansible Vault；Ansible 初始化每台 Worker 时，会以 `0600` 权限写入该 Worker 的 `~/.ssh/video-mask-release`。Worker 销毁时不需要去 GitHub 删除 Key。

测试 Controller 上的 Key：

```bash
ssh -i ~/.ssh/video-mask-release -o IdentitiesOnly=yes -T git@github.com
```

GitHub 提示认证成功、但不提供 shell 是正常结果。

## 1. 构建并发布 release

先在构建机完成一次发布。记下输出版本号；通常就是源码提交短 SHA。

```bash
cd ~/video-mask
git pull --ff-only
bash scripts/publish_release.sh ~/video-mask-release
```

若首次使用 release 仓库，先按 [Release构建服务器部署指南.md](./Release构建服务器部署指南.md) 启用 Git LFS。Controller 初始化时会从 Git LFS 下载该版本归档。

## 2. 准备 Ansible 配置

以下操作在客户 Controller 上进行。它使用 release 仓库里的部署包，不需要源码仓库权限。

先安装 Ansible 与 Git LFS，并 clone release 仓库。该仓库使用上一节创建的共享只读 Deploy Key。

```bash
sudo apt-get update
sudo apt-get install -y ansible git git-lfs
git lfs install

cd ~
git clone git@github.com:ryvengray/video-mask-release.git
cd ~/video-mask-release/deployment/ansible
```

下文所有 Ansible 命令均在 `~/video-mask-release/deployment/ansible` 中执行。

### 2.1 创建 inventory

从示例创建实际 inventory（实际文件不应包含任何私钥）：

```bash
cp inventory.yml.example inventory.yml
```

至少保留 Controller：

```yaml
controller:
  hosts:
    controller-01:
      ansible_user: ubuntu
      ansible_connection: local      # Ansible 与 Controller 在同一台机器

gpu_workers:
  hosts: {}
```

### 2.2 创建公开配置

```bash
cp group_vars/all.yml.example group_vars/all.yml
```

编辑 `group_vars/all.yml` 的关键字段：

```yaml
video_mask_release_repo: git@github.com:ryvengray/video-mask-release.git
video_mask_release_ref: main
video_mask_release_version: 9b55832  # 替换为第 1 步实际发布的版本
video_mask_release_dir: /home/ubuntu/video-mask-release
video_mask_release_install_root: /opt/video-mask/releases
video_mask_runtime_venv: /opt/video-mask/runtime-venv

# 使用 Controller 私网 IP，不要填写公网地址。
video_mask_controller_url: http://10.0.0.10:8080

# 测试阶段可使用本机目录；生产 S3 分发接入后再改为 s3。
video_mask_storage_mode: local
video_mask_local_source_dir: /home/ubuntu/cluster_test_sources
video_mask_local_output_dir: /home/ubuntu/cluster_test_outputs

video_mask_release_deploy_key: "{{ vault_video_mask_release_deploy_key }}"
video_mask_admin_token: "{{ vault_video_mask_admin_token }}"
video_mask_worker_token: "{{ vault_video_mask_worker_token }}"
```

`video_mask_release_version` 必须与 release 仓库中以下文件的版本完全一致：

```text
artifacts/video-mask-linux-x86_64-<版本>.tar.gz
```

### 2.3 创建 Vault 密钥文件

```bash
mkdir -p group_vars/all
ansible-vault create group_vars/all/vault.yml
```

写入以下内容。其中 release Key 是 `video-mask-release` 仓库的共享只读 Deploy Key 私钥，不是 `.pub` 文件：

```yaml
vault_video_mask_release_deploy_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  粘贴 release Deploy Key 私钥
  -----END OPENSSH PRIVATE KEY-----

vault_video_mask_admin_token: 替换为至少32位随机管理员Token
vault_video_mask_worker_token: 替换为至少32位随机WorkerToken
```

生成随机 Token 的一种方式：

```bash
openssl rand -hex 32
```

## 3. 启动 Controller

先确认 Ansible 可以连通：

```bash
ansible -i inventory.yml controller -m ping --ask-vault-pass
```

执行 Controller 初始化：

```bash
ansible-playbook -i inventory.yml site.yml \
  --limit controller \
  -K --ask-vault-pass
```

说明：

- `-K` 会询问 Controller 的 sudo 密码；若 `ubuntu` 已配置免密 sudo，可省略。
- `--ask-vault-pass` 会询问 Ansible Vault 密码；也可在受控 CI 环境改用 `--vault-password-file`。
- Playbook 自动安装 Git、Git LFS、Python 运行环境和 Controller 的公开依赖，下载 release 包，解压至 `/opt/video-mask/releases/<版本>`，并注册 `video-mask-controller.service`。

## 4. 在 Controller 上验证

SSH 登录 Controller：

```bash
ssh ubuntu@10.0.0.10
```

检查服务和健康接口：

```bash
sudo systemctl status video-mask-controller --no-pager
curl -fsS http://127.0.0.1:8080/healthz
sudo journalctl -u video-mask-controller -n 100 --no-pager
```

预期健康接口返回 JSON，且 systemd 状态为 `active (running)`。

确认只部署 release 而非源码：

```bash
ls -lah /opt/video-mask/releases
find /opt/video-mask/releases -type f -name '*.py' -print
ls -lah /home/ubuntu/video-mask-release
```

release 目录应主要包含 Cython `.so` 模块、通用启动脚本和公开依赖清单；不应出现 `video_mask_batch_fish.py`、`cluster/controller.py`、`cluster/worker_agent.py` 等核心源码文件。

## 5. 网络和浏览器访问

Controller 监听 `0.0.0.0:8080`。建议安全组规则：

| 来源 | 端口 | 用途 |
| --- | --- | --- |
| Worker 安全组/私网 CIDR | TCP 8080 | Worker 注册、领任务、上报进度 |
| 管理员 VPN 或堡垒机 | TCP 8080（可选） | 浏览器查看 Controller 页面 |

不要向全网开放 `8080`。如需浏览器访问，使用私网 VPN、堡垒机端口转发，或再配置反向代理与身份认证。

## 6. 升级到新 release

每次发布新版本后，仅修改 `group_vars/all.yml`：

```yaml
video_mask_release_version: 新版本号
```

然后重新执行：

```bash
ansible-playbook -i inventory.yml site.yml \
  --limit controller \
  -K --ask-vault-pass
```

新版本会解压到新的版本目录，并重启 Controller 服务。旧版本目录会保留，确认稳定后再人工清理，不执行自动删除。

## 7. 常见故障

### `Release ... is not present`

确认构建机已经 push release 仓库，并且 `video_mask_release_version` 与归档文件版本完全一致。也确认 Git LFS 已推送真实归档而不是仅有 LFS 指针。

### `Permission denied (publickey)`

检查 Controller 的 `~/.ssh/video-mask-release`、Vault 中的 Deploy Key 私钥、GitHub `video-mask-release` 仓库中的 Deploy Key，以及该 Key 的只读权限。Controller 不需要源码仓库 Key。

### `ModuleNotFoundError`

检查本次 Ansible 是否完成“Install public runtime dependencies”。查看服务日志：

```bash
sudo journalctl -u video-mask-controller -n 100 --no-pager
```

重新执行 Playbook 会补齐公开 Python 依赖。

### `systemctl` 服务无法启动

检查实际 release 路径与版本：

```bash
sudo systemctl cat video-mask-controller
ls -lah /opt/video-mask/releases/<版本>/bin
```

再查看完整日志：

```bash
sudo journalctl -u video-mask-controller -e --no-pager
```
