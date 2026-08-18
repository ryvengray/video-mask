#!/usr/bin/env bash
# Deploy one or more remote GPU Worker hosts from the Controller/Ansible machine.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/deploy_worker.sh WORKER_HOST [WORKER_HOST ...] [--start-stopped] [--restart-slot] [--no-vault-prompt]

Deploys one or more hosts defined in ansible/inventory.yml. Runtime source is
rsynced from this Controller/Ansible machine; Workers do not clone the Git
repository or retain its Deploy Key. With --restart-slot, restarts every
configured Worker slot on each selected host after the source has updated.
With --start-stopped, starts unreachable EC2 Workers through the Controller-
local start script, waits for SSH, then deploys. A missing NVIDIA driver can
cause one automatic reboot; run only after the selected Workers are drained/idle.

Examples:
  bash scripts/deploy_worker.sh worker-01
  bash scripts/deploy_worker.sh worker-01 worker-02
  bash scripts/deploy_worker.sh worker-01 --start-stopped
  bash scripts/deploy_worker.sh worker-01 --restart-slot
EOF
}

ASK_VAULT=1
RESTART_SLOT=0
START_STOPPED=0
VAULT_ARGUMENTS=()
VAULT_PASSWORD_FILE=""
HOSTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vault-prompt) ASK_VAULT=0 ;;
    --restart-slot) RESTART_SLOT=1 ;;
    --start-stopped) START_STOPPED=1 ;;
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

if [[ "$ASK_VAULT" -eq 1 ]]; then
  umask 077
  VAULT_PASSWORD_FILE="$(mktemp "${TMPDIR:-/tmp}/video-mask-vault.XXXXXX")"
  trap '[[ -z "$VAULT_PASSWORD_FILE" ]] || rm -f "$VAULT_PASSWORD_FILE"' EXIT
  read -r -s -p "Ansible Vault password: " vault_password
  echo
  printf '%s\n' "$vault_password" > "$VAULT_PASSWORD_FILE"
  unset vault_password
  VAULT_ARGUMENTS=(--vault-password-file "$VAULT_PASSWORD_FILE")
fi

if [[ "$START_STOPPED" -eq 1 ]]; then
  START_COMMAND="${VIDEO_MASK_WORKER_START_COMMAND:-/opt/dataai-ec2/bin/start_ec2.sh}"
  if [[ ! -f "$START_COMMAND" ]]; then
    echo "Error: EC2 start command does not exist: $START_COMMAND" >&2
    echo "Set VIDEO_MASK_WORKER_START_COMMAND if this Controller uses a different command." >&2
    exit 2
  fi
  for host in "${HOSTS[@]}"; do
    if ansible -i ansible/inventory.yml "${VAULT_ARGUMENTS[@]}" "$host" -m ansible.builtin.ping -o >/dev/null 2>&1; then
      echo "$host is already reachable; skipping EC2 start."
      continue
    fi
    host_ip="$(ansible-inventory -i ansible/inventory.yml "${VAULT_ARGUMENTS[@]}" --host "$host" | python3 -c '
import ipaddress
import json
import sys

values = json.load(sys.stdin)
address = values.get("video_mask_private_ip") or values.get("ansible_host") or sys.argv[1]
try:
    print(ipaddress.ip_address(address))
except ValueError:
    raise SystemExit(f"inventory host {sys.argv[1]!r} needs a private-IP ansible_host or video_mask_private_ip")
' "$host")"
    echo "$host is unreachable; starting EC2 instance for $host_ip."
    sudo -n "$START_COMMAND" --ips "$host_ip"
    echo "Waiting for $host SSH connection..."
    ansible -i ansible/inventory.yml "${VAULT_ARGUMENTS[@]}" "$host" -m ansible.builtin.wait_for_connection \
      -a 'timeout=900 connect_timeout=10 sleep=5'
  done
fi

COMMAND=(ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit "$LIMIT")
[[ "$RESTART_SLOT" -eq 1 ]] && COMMAND+=(--extra-vars video_mask_restart_worker=true)
COMMAND+=("${VAULT_ARGUMENTS[@]}")
"${COMMAND[@]}"
