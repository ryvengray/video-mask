#!/usr/bin/env bash
# Build a source-free Linux x86_64 release archive with Docker/Nuitka.
# Usage: bash scripts/build_release_linux.sh ../video-mask-release [version]
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${1:?Usage: bash scripts/build_release_linux.sh /path/to/video-mask-release [version]}"
VERSION="${2:-$(git -C "$ROOT_DIR" describe --tags --always --dirty)}"
IMAGE="video-mask-release-builder:${VERSION//\//-}"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/video-mask-release.XXXXXX")"

cleanup() {
    rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

[[ -d "$RELEASE_DIR/.git" ]] || {
    echo "Error: release directory is not a Git repository: $RELEASE_DIR" >&2
    exit 1
}
command -v docker >/dev/null || { echo "Error: Docker is required." >&2; exit 1; }
docker info >/dev/null || { echo "Error: Docker daemon is not running." >&2; exit 1; }

echo "==> Building Linux x86_64 release image: $IMAGE"
docker build --platform linux/amd64 \
    --file "$ROOT_DIR/release/Dockerfile.build" \
    --tag "$IMAGE" \
    "$ROOT_DIR"

container="$(docker create "$IMAGE")"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true; cleanup' EXIT
docker cp "$container:/opt/release/." "$STAGE_DIR/"
docker rm "$container" >/dev/null
container=""

artifact_dir="$RELEASE_DIR/artifacts/$VERSION"
mkdir -p "$artifact_dir"
cp -R "$STAGE_DIR/bin" "$artifact_dir/"
tar -C "$artifact_dir" -czf "$RELEASE_DIR/artifacts/video-mask-linux-x86_64-$VERSION.tar.gz" bin
(
    cd "$RELEASE_DIR/artifacts"
    shasum -a 256 "video-mask-linux-x86_64-$VERSION.tar.gz" > "video-mask-linux-x86_64-$VERSION.SHA256"
)

cat > "$RELEASE_DIR/manifests/$VERSION.json" <<EOF
{
  "version": "$VERSION",
  "platform": "linux-x86_64",
  "archive": "artifacts/video-mask-linux-x86_64-$VERSION.tar.gz",
  "sha256": "artifacts/video-mask-linux-x86_64-$VERSION.SHA256",
  "binaries": ["video-mask-controller", "video-mask-worker", "video-mask-batch-fish"]
}
EOF

echo "==> Release created: $RELEASE_DIR/artifacts/video-mask-linux-x86_64-$VERSION.tar.gz"
