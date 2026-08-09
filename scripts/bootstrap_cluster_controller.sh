#!/usr/bin/env bash
# Bootstrap a controller-only Video Mask cluster node on Ubuntu/Debian.
# Run as the ubuntu user from a checked-out video-mask repository.
set -Eeuo pipefail

REPO_URL="https://github.com/ryvengray/video-mask.git"
REPO_REF="main"
APP_DIR="/home/ubuntu/video-mask"
CONTROLLER_URL="http://127.0.0.1:8080"
STORAGE_MODE="local"
SOURCE_DIR="/home/ubuntu/cluster_test_sources"
OUTPUT_DIR="/home/ubuntu/cluster_test_outputs"
ADMIN_TOKEN=""
WORKER_TOKEN=""
DEPLOY=1
FORCE_CONFIG=0
DEPLOY_USER="ubuntu"

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_cluster_controller.sh [options]

Initializes a Controller only.  Workers can be added later through Ansible.

Options:
  --controller-url URL     Controller URL workers will use (default: http://127.0.0.1:8080)
  --app-dir PATH           Application directory (default: /home/ubuntu/video-mask)
  --repo URL               Git repository URL (default: GitHub HTTPS repository)
  --ref REF                Git ref to deploy (default: main)
  --source-dir PATH        Local-test input directory
  --output-dir PATH        Local-test output directory
  --admin-token TOKEN      Reuse a generated admin token; otherwise generate one
  --worker-token TOKEN     Reuse a generated Worker token; otherwise generate one
  --deploy-user USER       Service/deployment user (default: ubuntu)
  --force-config           Back up and recreate Ansible inventory/configuration
  --no-deploy              Generate configuration but do not run Ansible
  -h, --help               Show this help

Examples:
  bash scripts/bootstrap_cluster_controller.sh
  bash scripts/bootstrap_cluster_controller.sh \
    --controller-url http://10.0.1.25:8080

The default local directories are for Controller-only testing.  S3 task
ingestion is not configured by this script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --controller-url) CONTROLLER_URL="${2:?missing value for --controller-url}"; shift ;;
    --app-dir) APP_DIR="${2:?missing value for --app-dir}"; shift ;;
    --repo) REPO_URL="${2:?missing value for --repo}"; shift ;;
    --ref) REPO_REF="${2:?missing value for --ref}"; shift ;;
    --source-dir) SOURCE_DIR="${2:?missing value for --source-dir}"; shift ;;
    --output-dir) OUTPUT_DIR="${2:?missing value for --output-dir}"; shift ;;
    --admin-token) ADMIN_TOKEN="${2:?missing value for --admin-token}"; shift ;;
    --worker-token) WORKER_TOKEN="${2:?missing value for --worker-token}"; shift ;;
    --deploy-user) DEPLOY_USER="${2:?missing value for --deploy-user}"; shift ;;
    --force-config) FORCE_CONFIG=1 ;;
    --no-deploy) DEPLOY=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Error: this script supports Ubuntu/Debian Linux only." >&2
  exit 1
fi
if ! command -v sudo >/dev/null 2>&1; then
  echo "Error: sudo is required." >&2
  exit 1
fi

if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
  echo "Error: deployment user '$DEPLOY_USER' does not exist." >&2
  exit 1
fi

if [[ $EUID -eq 0 ]]; then
  RUN_AS_DEPLOY_USER=(sudo -u "$DEPLOY_USER")
  ANSIBLE_BECOME_ARGS=()
else
  RUN_AS_DEPLOY_USER=()
  ANSIBLE_BECOME_ARGS=(-K)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Checking sudo and installing controller prerequisites"
sudo -v
sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ansible ca-certificates curl git openssl python3 python3-pip python3-venv

if ! getent hosts github.com >/dev/null; then
  cat >&2 <<'EOF'
Error: github.com cannot be resolved.  Fix VPC DNS / Internet egress first.
For AWS, verify the subnet route (Internet Gateway or NAT Gateway), DNS support,
and outbound HTTPS/DNS rules before retrying.
EOF
  exit 1
fi

if [[ "$SOURCE_REPO" != "$APP_DIR" ]]; then
  if [[ -e "$APP_DIR" && ! -d "$APP_DIR/.git" ]]; then
    echo "Error: $APP_DIR exists but is not a Git repository." >&2
    exit 1
  fi
  if [[ ! -d "$APP_DIR/.git" ]]; then
    echo "==> Cloning $REPO_URL into $APP_DIR"
    "${RUN_AS_DEPLOY_USER[@]}" git clone --branch "$REPO_REF" "$REPO_URL" "$APP_DIR"
  fi
fi

if [[ ! -f "$APP_DIR/ansible/site.yml" ]]; then
  echo "Error: $APP_DIR does not contain the cluster Ansible files." >&2
  exit 1
fi

"${RUN_AS_DEPLOY_USER[@]}" git -C "$APP_DIR" remote set-url origin "$REPO_URL" 2>/dev/null || true
mkdir -p "$SOURCE_DIR" "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d%H%M%S)"
inventory="$APP_DIR/ansible/inventory.yml"
variables="$APP_DIR/ansible/group_vars/all.yml"

backup_if_needed() {
  local path="$1"
  if [[ -f "$path" && "$FORCE_CONFIG" -eq 1 ]]; then
    cp "$path" "$path.backup-$timestamp"
    echo "==> Backed up $path to $path.backup-$timestamp"
  fi
}

if [[ ! -f "$inventory" || "$FORCE_CONFIG" -eq 1 ]]; then
  backup_if_needed "$inventory"
  cat >"$inventory" <<'EOF'
controller:
  hosts:
    controller-01:
      ansible_connection: local
      ansible_user: ubuntu

gpu_workers:
  hosts: {}
EOF
  echo "==> Wrote Controller-only inventory: $inventory"
else
  echo "==> Keeping existing inventory: $inventory"
fi

if [[ ! -f "$variables" || "$FORCE_CONFIG" -eq 1 ]]; then
  backup_if_needed "$variables"
  [[ -n "$ADMIN_TOKEN" ]] || ADMIN_TOKEN="$(openssl rand -hex 32)"
  [[ -n "$WORKER_TOKEN" ]] || WORKER_TOKEN="$(openssl rand -hex 32)"
  umask 077
  cat >"$variables" <<EOF
video_mask_repo: $REPO_URL
video_mask_ref: $REPO_REF
video_mask_app_dir: $APP_DIR

# Use this instance's private VPC address before adding remote Workers.
video_mask_controller_url: $CONTROLLER_URL
video_mask_admin_token: $ADMIN_TOKEN
video_mask_worker_token: $WORKER_TOKEN

video_mask_storage_mode: $STORAGE_MODE
video_mask_local_source_dir: $SOURCE_DIR
video_mask_local_output_dir: $OUTPUT_DIR

video_mask_s3_source_bucket: ''
video_mask_s3_source_prefix: source/inbox/
video_mask_s3_source_region: ''
video_mask_s3_assume_role_arn: ''
video_mask_s3_output_bucket: ''
video_mask_s3_output_prefix: outputs/
video_mask_s3_output_region: ''
EOF
  echo "==> Wrote configuration: $variables (mode 0600)"
else
  echo "==> Keeping existing configuration: $variables"
fi

if [[ "$DEPLOY" -eq 0 ]]; then
  echo "Configuration created. Deploy later with:"
  echo "  cd $APP_DIR && ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit controller -K"
  exit 0
fi

echo "==> Deploying Controller"
cd "$APP_DIR"
ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit controller "${ANSIBLE_BECOME_ARGS[@]}"

echo
echo "Controller bootstrap complete. Verify with:"
echo "  curl http://127.0.0.1:8080/healthz"
echo "Add remote Workers using the Controller private URL: $CONTROLLER_URL"
