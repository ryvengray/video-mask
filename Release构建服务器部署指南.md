# Release 构建服务器部署指南

本文用于在独立 Linux 服务器上，把私有源码编译成 Linux x86_64 release 包并推送到私有 release 仓库。构建机需要读源码、写 release，不需要 GPU。

## 1. 机器要求

推荐 Ubuntu 24.04 LTS、`x86_64/amd64`、8 核 CPU、32 GB 内存、100 GB SSD。最低可用为 4 核、16 GB、80 GB SSD。不要选择 ARM。构建机与 AWS Worker 一样使用 Ubuntu 24.04 x86_64，可避免系统库兼容性问题。

以下命令均以构建用户 `ubuntu` 执行；只有标明 `sudo` 的步骤需要管理员权限。

```bash
uname -m       # 必须输出 x86_64
lsb_release -ds
df -h /
```

## 2. 安装 Docker（一次性）

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"${UBUNTU_CODENAME:-$VERSION_CODENAME}\") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
newgrp docker

docker --version
docker info
```

如果 `docker info` 仍显示权限错误，退出 SSH 后重新登录再执行即可。

## 3. GitHub 最小权限（一次性）

构建机使用两把不同 Deploy Key：

| 仓库 | 权限 | 用途 |
| --- | --- | --- |
| `ryvengray/video-mask` | 只读 | 拉取源码、编译 |
| `ryvengray/video-mask-release` | 可写 | 推送构建产物 |

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/video-mask-source -C "release-builder-source" -N ""
ssh-keygen -t ed25519 -f ~/.ssh/video-mask-release -C "release-builder-release" -N ""
cat ~/.ssh/video-mask-source.pub
cat ~/.ssh/video-mask-release.pub
```

在两个 GitHub 仓库的 `Settings` → `Deploy keys` 中添加对应公钥：源码仓库不要勾选写权限；release 仓库勾选 `Allow write access`。

创建 `~/.ssh/config`：

```sshconfig
Host github-video-mask-source
  HostName github.com
  User git
  IdentityFile ~/.ssh/video-mask-source
  IdentitiesOnly yes

Host github-video-mask-release
  HostName github.com
  User git
  IdentityFile ~/.ssh/video-mask-release
  IdentitiesOnly yes
```

```bash
chmod 600 ~/.ssh/config ~/.ssh/video-mask-source ~/.ssh/video-mask-release
ssh -T git@github-video-mask-source
ssh -T git@github-video-mask-release
```

GitHub 提示认证成功但不提供 shell 是正常结果。

## 4. 首次克隆

```bash
cd ~
git clone git@github-video-mask-source:ryvengray/video-mask.git
git clone git@github-video-mask-release:ryvengray/video-mask-release.git
```

## 5. 一键发布 release

```bash
cd ~/video-mask
bash scripts/publish_release.sh ~/video-mask-release
```

脚本会先确认两个仓库没有未提交修改并执行 `git pull --ff-only`，再启动 Docker，在固定的 Linux x86_64 Ubuntu 24.04 容器中安装编译依赖、运行 Nuitka。构建完成后自动执行 SHA256 校验、提交并推送 release 仓库。版本号默认使用当前源码提交短 SHA；同一版本不会被覆盖。

只想构建并在本地检查、不推送时：

```bash
bash scripts/publish_release.sh ~/video-mask-release --no-push
```

如需为同一份源码重新打包，显式指定新版本号：

```bash
bash scripts/publish_release.sh ~/video-mask-release --version 20260809.1
```

构建产物为：

```text
~/video-mask-release/artifacts/video-mask-linux-x86_64-<版本>.tar.gz
~/video-mask-release/artifacts/video-mask-linux-x86_64-<版本>.SHA256
~/video-mask-release/artifacts/<版本>/bin/
~/video-mask-release/manifests/<版本>.json
```

首次构建会下载系统、Python 和 AI 依赖并进行 C/C++ 编译，耗时较长；Docker 会缓存依赖，后续未修改依赖时会快很多。

构建时可另开一个终端监控：

```bash
docker ps
docker system df
top
df -h /
```

## 6. 手动校验（可选）

构建完成后：

```bash
cd ~/video-mask-release
VERSION="$(git -C ~/video-mask rev-parse --short HEAD)"  # 或指定实际发布版本
shasum -a 256 -c "artifacts/video-mask-linux-x86_64-$VERSION.SHA256"
tar -tzf "artifacts/video-mask-linux-x86_64-$VERSION.tar.gz" | head -30
```

## 7. 日常发布

```bash
cd ~/video-mask
bash scripts/publish_release.sh ~/video-mask-release
```

## 8. 常见问题

### Docker 无权限

```bash
newgrp docker
docker info
```

或退出 SSH 后重新登录。

### 下载慢或失败

确认构建机可访问 Docker Hub、PyPI 和 GitHub。构建脚本默认使用 Ubuntu 官方源；若在中国大陆构建且官方源较慢，可临时指定镜像：

```bash
UBUNTU_MIRROR=http://mirrors.aliyun.com/ubuntu bash scripts/publish_release.sh ~/video-mask-release
```

### 磁盘不足

```bash
docker system df
df -h /
```

确认没有运行中的构建且不再需要缓存后，可执行 `docker system prune` 清理未使用的镜像、容器和缓存。

## 9. 安全要求

- 构建机可读取源码，应限制登录人员并保护 `~/.ssh` 私钥。
- 客户运行服务器只读取 `video-mask-release`，不授予源码仓库权限。
- 不要将 Deploy Key、GitHub Token、S3 AK/SK 写入 Git、Ansible inventory 或日志。
- 密钥泄露后，在仓库 `Deploy keys` 页面撤销并替换该密钥。

## 10. 为弹性 Worker 配置共享 release 只读 Key

动态创建、销毁的 Worker 可共用一把只读 Deploy Key，只添加到 `video-mask-release` 仓库一次即可。该 Key 不得用于源码仓库。

在 Ansible 控制端创建加密变量文件：

```bash
cd ~/video-mask
mkdir -p ansible/group_vars/all
ansible-vault create ansible/group_vars/all/vault.yml
```

在打开的文件中填写私钥（不是 `.pub` 公钥）：

```yaml
vault_video_mask_release_deploy_key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  此处粘贴共享 release Deploy Key 私钥
  -----END OPENSSH PRIVATE KEY-----
```

在 `ansible/group_vars/all.yml`（由示例文件复制而来）中设置：

```yaml
video_mask_release_repo: git@github.com:ryvengray/video-mask-release.git
video_mask_release_ref: main
video_mask_release_dir: /home/ubuntu/video-mask-release
video_mask_release_deploy_key: "{{ vault_video_mask_release_deploy_key }}"
```

执行 Worker 初始化时带上 Vault 密码：

```bash
ansible-playbook -i ansible/inventory.yml ansible/site.yml \
  --limit gpu_workers -K --ask-vault-pass
```

角色会将私钥以 `0600` 权限写入 `/home/ubuntu/.ssh/video-mask-release`，并使用它 clone/pull release 仓库。密钥内容不会出现在 Playbook 输出中。当前步骤只是预下载 release 仓库；待编译版运行部署验证完成后，Worker 服务将从该目录启动，以彻底移除源码仓库访问。
