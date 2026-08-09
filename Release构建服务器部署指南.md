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

## 5. 构建 release

```bash
git -C ~/video-mask pull --ff-only
git -C ~/video-mask-release pull --ff-only
cd ~/video-mask
VERSION="$(git rev-parse --short HEAD)"
bash scripts/build_release_linux.sh ~/video-mask-release "$VERSION"
```

脚本会启动 Docker，在固定的 Linux x86_64 Ubuntu 24.04 容器中安装编译依赖、运行 Nuitka，并生成：

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

## 6. 校验并推送

构建完成后：

```bash
cd ~/video-mask-release
VERSION="$(git -C ~/video-mask rev-parse --short HEAD)"
shasum -a 256 -c "artifacts/video-mask-linux-x86_64-$VERSION.SHA256"
tar -tzf "artifacts/video-mask-linux-x86_64-$VERSION.tar.gz" | head -30
git add artifacts manifests
git commit -m "release: build $VERSION"
git push origin main
```

## 7. 日常发布

```bash
git -C ~/video-mask pull --ff-only
git -C ~/video-mask-release pull --ff-only
cd ~/video-mask
VERSION="$(git rev-parse --short HEAD)"
bash scripts/build_release_linux.sh ~/video-mask-release "$VERSION"
cd ~/video-mask-release
shasum -a 256 -c "artifacts/video-mask-linux-x86_64-$VERSION.SHA256"
git add artifacts manifests
git commit -m "release: build $VERSION"
git push origin main
```

## 8. 常见问题

### Docker 无权限

```bash
newgrp docker
docker info
```

或退出 SSH 后重新登录。

### 下载慢或失败

确认构建机可访问 Docker Hub、PyPI 和 GitHub。构建脚本默认使用阿里云 Ubuntu 镜像；如需改回官方源：

```bash
UBUNTU_MIRROR=http://archive.ubuntu.com/ubuntu bash scripts/build_release_linux.sh ~/video-mask-release "$VERSION"
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
