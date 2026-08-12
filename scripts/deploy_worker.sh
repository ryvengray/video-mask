#!/usr/bin/env bash
# Deploy one or more remote GPU Worker hosts from the Controller/Ansible machine.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/deploy_worker.sh WORKER_HOST [WORKER_HOST ...] [--no-vault-prompt]

Deploys one or more hosts defined in ansible/inventory.yml. Worker services may
restart if their unit/environment changes, and a missing NVIDIA driver can cause
one automatic reboot; run only after the selected Workers are drained/idle.

Examples:
  bash scripts/deploy_worker.sh worker-01
  bash scripts/deploy_worker.sh worker-01 worker-02
EOF
}

ASK_VAULT=1
HOSTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vault-prompt) ASK_VAULT=0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) HOSTS+=("$1") ;;
  esac
  shift
done

if [[ ${#HOSTS[@]} -eq 0 ]]; then
  echo "Error: provide at least one Worker inventory host." >&2
  usage >&2
  exit 2
fi

LIMIT="$(IFS=,; echo "${HOSTS[*]}")"
cd "$ROOT_DIR"
COMMAND=(ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit "$LIMIT")
[[ "$ASK_VAULT" -eq 1 ]] && COMMAND+=(--ask-vault-pass)
"${COMMAND[@]}"
