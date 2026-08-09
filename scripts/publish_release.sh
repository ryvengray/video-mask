#!/usr/bin/env bash
# Build, verify and publish a source-free Linux release in one command.
# Usage: bash scripts/publish_release.sh /path/to/video-mask-release [--version VERSION] [--no-push]
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=""
PUSH_RELEASE=true

usage() {
    cat <<'EOF'
Usage: bash scripts/publish_release.sh /path/to/video-mask-release [options]

Options:
  --version VERSION  Use an explicit version (default: current source commit short SHA).
  --no-push          Build, verify and commit locally without pushing the release repository.
  -h, --help         Show this help.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

RELEASE_DIR="${1:-}"
[[ $# -gt 0 ]] && shift
[[ -n "$RELEASE_DIR" ]] || { usage >&2; exit 2; }
RELEASE_DIR="$(cd "$RELEASE_DIR" && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            [[ $# -ge 2 ]] || { echo "Error: --version requires a value." >&2; exit 2; }
            VERSION="$2"
            shift 2
            ;;
        --no-push)
            PUSH_RELEASE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -d "$RELEASE_DIR/.git" ]] || {
    echo "Error: release directory is not a Git repository: $RELEASE_DIR" >&2
    exit 1
}
command -v git-lfs >/dev/null || {
    echo "Error: git-lfs is required for large release archives. Install git-lfs and run git lfs install." >&2
    exit 1
}

require_clean_tree() {
    local repo="$1"
    local label="$2"
    if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
        echo "Error: $label has uncommitted changes: $repo" >&2
        git -C "$repo" status --short >&2
        exit 1
    fi
}

require_clean_tree "$ROOT_DIR" "source repository"
require_clean_tree "$RELEASE_DIR" "release repository"

echo "==> Synchronizing source repository"
git -C "$ROOT_DIR" pull --ff-only
echo "==> Synchronizing release repository"
git -C "$RELEASE_DIR" pull --ff-only

VERSION="${VERSION:-$(git -C "$ROOT_DIR" rev-parse --short HEAD)}"
[[ "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "Error: version may contain only letters, numbers, dot, underscore and dash." >&2
    exit 2
}

LFS_ATTRIBUTE="$(git -C "$RELEASE_DIR" check-attr filter -- "artifacts/video-mask-linux-x86_64-$VERSION.tar.gz")"
[[ "$LFS_ATTRIBUTE" == *": filter: lfs" ]] || {
    echo "Error: release archives are not configured for Git LFS." >&2
    echo "Run: git lfs track 'artifacts/*.tar.gz' && git add .gitattributes && git commit -m 'chore: enable Git LFS'" >&2
    exit 1
}

ARCHIVE="$RELEASE_DIR/artifacts/video-mask-linux-x86_64-$VERSION.tar.gz"
MANIFEST="$RELEASE_DIR/manifests/$VERSION.json"
if [[ -e "$ARCHIVE" || -e "$MANIFEST" || -d "$RELEASE_DIR/artifacts/$VERSION" ]]; then
    echo "Error: release version already exists: $VERSION" >&2
    echo "Choose --version for a new build identifier; existing artifacts are never overwritten." >&2
    exit 1
fi

echo "==> Building version $VERSION"
bash "$ROOT_DIR/scripts/build_release_linux.sh" "$RELEASE_DIR" "$VERSION"

echo "==> Verifying archive"
(
    cd "$RELEASE_DIR/artifacts"
    shasum -a 256 -c "video-mask-linux-x86_64-$VERSION.SHA256"
)
tar -tzf "$ARCHIVE" >/dev/null

echo "==> Staging release files"
git -C "$RELEASE_DIR" add -- artifacts manifests
if git -C "$RELEASE_DIR" diff --cached --quiet; then
    echo "Error: build generated no files to commit." >&2
    exit 1
fi
git -C "$RELEASE_DIR" commit -m "release: build $VERSION"

if [[ "$PUSH_RELEASE" == true ]]; then
    echo "==> Pushing release repository"
    git -C "$RELEASE_DIR" push origin HEAD
else
    echo "==> Release committed locally only (--no-push)."
fi

echo "==> Published release $VERSION"
echo "    Archive: $ARCHIVE"
