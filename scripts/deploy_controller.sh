#!/usr/bin/env bash
# Deploy the Controller role from the Controller/Ansible machine.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/deploy_controller.sh [--no-vault-prompt] [extra ansible-playbook options]

Updates the local Controller role using ansible/inventory.yml. By default,
Ansible asks for the Vault password. The role restarts the Controller when its
service configuration changes; use --restart to force a restart after deploy.

Examples:
  bash scripts/deploy_controller.sh
  bash scripts/deploy_controller.sh --restart
EOF
}

ASK_VAULT=1
RESTART=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vault-prompt) ASK_VAULT=0 ;;
    --restart) RESTART=1 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1") ;;
  esac
  shift
done

cd "$ROOT_DIR"
COMMAND=(ansible-playbook -i ansible/inventory.yml ansible/site.yml --limit controller)
[[ "$ASK_VAULT" -eq 1 ]] && COMMAND+=(--ask-vault-pass)
COMMAND+=("${EXTRA_ARGS[@]}")
"${COMMAND[@]}"

if [[ "$RESTART" -eq 1 ]]; then
  sudo systemctl restart video-mask-controller
  if systemctl cat video-mask-autoscaler.service >/dev/null 2>&1; then
    sudo systemctl try-restart video-mask-autoscaler
  fi
fi
sudo systemctl status video-mask-controller --no-pager
curl --fail --silent --show-error http://127.0.0.1:8080/healthz
if systemctl is-enabled video-mask-autoscaler >/dev/null 2>&1; then
  sudo systemctl status video-mask-autoscaler --no-pager
fi
