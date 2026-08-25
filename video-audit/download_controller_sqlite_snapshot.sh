#!/usr/bin/env bash
# Download a consistent snapshot of the video-mask Controller SQLite database.
#
# Example:
#   ./download_controller_sqlite_snapshot.sh \
#     --pem ~/keys/controller.pem --host controller.example.com
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./download_controller_sqlite_snapshot.sh --pem PATH --host HOST [options]

Required:
  --pem PATH             SSH private-key (.pem) file
  --host HOST             Controller host name or IP address

Options:
  --user USER             SSH user (default: ubuntu)
  --remote-db PATH        Controller SQLite path
                          (default: /var/lib/video-mask-controller/controller.sqlite3)
  --out-dir PATH          Local snapshot directory (default: ~/data/controller-snapshots)
  --delete-remote         Delete the remote temporary snapshot after checksum verification
  -h, --help              Show this help

The script uses Python's SQLite backup API on the Controller, so the live
database remains online and no -wal/-shm files need to be downloaded.
The Controller's first-seen SSH host key is saved automatically; a changed
host key is still rejected.
EOF
}

PEM_FILE=""
CONTROLLER_HOST=""
CONTROLLER_USER="ubuntu"
REMOTE_DATABASE="/var/lib/video-mask-controller/controller.sqlite3"
OUTPUT_DIR="${HOME}/data/controller-snapshots"
DELETE_REMOTE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pem) PEM_FILE="${2:?missing PEM path}"; shift 2 ;;
    --host) CONTROLLER_HOST="${2:?missing Controller host}"; shift 2 ;;
    --user) CONTROLLER_USER="${2:?missing SSH user}"; shift 2 ;;
    --remote-db) REMOTE_DATABASE="${2:?missing remote database path}"; shift 2 ;;
    --out-dir) OUTPUT_DIR="${2:?missing local output directory}"; shift 2 ;;
    --delete-remote) DELETE_REMOTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PEM_FILE" || -z "$CONTROLLER_HOST" ]]; then
  echo "--pem and --host are required." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$PEM_FILE" ]]; then
  echo "PEM file does not exist: $PEM_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_SNAPSHOT="/tmp/video-mask-controller-${TIMESTAMP}-$$.sqlite3"
LOCAL_SNAPSHOT="${OUTPUT_DIR%/}/controller-${CONTROLLER_HOST//[^A-Za-z0-9._-]/_}-${TIMESTAMP}.sqlite3"
SSH_TARGET="${CONTROLLER_USER}@${CONTROLLER_HOST}"
# Accept a first-seen host key without an interactive prompt, while still
# refusing a changed key that could indicate a host replacement or MITM.
SSH_OPTIONS=(-i "$PEM_FILE" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)

remote_quote() {
  printf '%q' "$1"
}

echo "Creating a consistent Controller snapshot on ${SSH_TARGET}…"
REMOTE_BACKUP_COMMAND="python3 -c 'import sqlite3, sys; source = sqlite3.connect(sys.argv[1]); destination = sqlite3.connect(sys.argv[2]); source.backup(destination); destination.close(); source.close()' $(remote_quote "$REMOTE_DATABASE") $(remote_quote "$REMOTE_SNAPSHOT")"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "$REMOTE_BACKUP_COMMAND"

echo "Downloading snapshot to ${LOCAL_SNAPSHOT}…"
scp "${SSH_OPTIONS[@]}" "${SSH_TARGET}:${REMOTE_SNAPSHOT}" "$LOCAL_SNAPSHOT"

REMOTE_SHA256_COMMAND="python3 -c \"import hashlib, sys; digest = hashlib.sha256(); handle = open(sys.argv[1], 'rb'); [digest.update(block) for block in iter(lambda: handle.read(1048576), b'')]; handle.close(); print(digest.hexdigest())\" $(remote_quote "$REMOTE_SNAPSHOT")"
REMOTE_SHA256="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "$REMOTE_SHA256_COMMAND")"
LOCAL_SHA256="$(shasum -a 256 "$LOCAL_SNAPSHOT" | awk '{print $1}')"
if [[ "$REMOTE_SHA256" != "$LOCAL_SHA256" ]]; then
  echo "Checksum mismatch; keeping both snapshots for investigation." >&2
  exit 1
fi

if [[ "$DELETE_REMOTE" -eq 1 ]]; then
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "rm -f $(remote_quote "$REMOTE_SNAPSHOT")"
  REMOTE_NOTICE="Remote temporary snapshot deleted."
else
  REMOTE_NOTICE="Remote temporary snapshot retained at ${REMOTE_SNAPSHOT}."
fi

echo "Snapshot downloaded and verified: ${LOCAL_SNAPSHOT}"
echo "$REMOTE_NOTICE"
