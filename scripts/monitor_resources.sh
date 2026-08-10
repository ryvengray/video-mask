#!/usr/bin/env bash
# Read-only resource dashboard for a video-mask Worker or standalone server.
# It never starts, stops, or signals video-processing tasks.
set -u

INTERVAL=2
ONCE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/monitor_resources.sh [--interval SECONDS] [--once]

Options:
  --interval SECONDS  Refresh period (default: 2 seconds; minimum: 1)
  --once              Print one snapshot and exit
  -h, --help          Show this help

Ctrl+C exits the monitor only. Running video-mask tasks are not affected.
EOF
}

while (($#)); do
    case "$1" in
        --interval)
            [[ $# -ge 2 ]] || { echo "Error: --interval needs a value." >&2; exit 2; }
            INTERVAL="$2"
            shift 2
            ;;
        --once)
            ONCE=1
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

[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: interval must be a positive integer." >&2
    exit 2
}
[[ -r /proc/stat ]] || {
    echo "Error: this monitor is intended for Linux servers (/proc/stat is unavailable)." >&2
    exit 2
}

previous_total=0
previous_idle=0

cpu_percent() {
    local total idle delta_total delta_idle
    read -r total idle < <(
        awk '/^cpu / { total=0; for (i=2; i<=NF; i++) total += $i; idle=$5 + $6; print total, idle; exit }' /proc/stat
    )
    if ((previous_total == 0)); then
        previous_total="$total"
        previous_idle="$idle"
        printf 'warming up'
        return
    fi
    delta_total=$((total - previous_total))
    delta_idle=$((idle - previous_idle))
    previous_total="$total"
    previous_idle="$idle"
    if ((delta_total <= 0)); then
        printf 'n/a'
    else
        awk -v busy="$((delta_total - delta_idle))" -v total="$delta_total" \
            'BEGIN { printf "%.1f%%", (busy * 100) / total }'
    fi
}

print_processes() {
    local rows
    rows="$(ps -eo pid,ppid,%cpu,%mem,rss,etime,args --sort=-%cpu 2>/dev/null | \
        awk 'NR == 1 || /video_mask|batch_scheduler|worker_agent|ffmpeg/ { print }')"
    if [[ -n "$rows" ]]; then
        printf '%s\n' "$rows"
    else
        echo '(No matching video-mask, scheduler, Worker, or ffmpeg process found.)'
    fi
}

print_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 'NVIDIA GPU: nvidia-smi not installed or no NVIDIA driver available.'
        return
    fi

    echo 'GPU summary:'
    nvidia-smi \
        --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F ', ' '{printf "  GPU %s (%s): SM %s%% | mem-util %s%% | VRAM %s / %s MiB | %s C | %s W / %s W\n", $1,$2,$3,$4,$5,$6,$7,$8,$9}' \
        || echo '  Unable to query GPU summary.'

    echo 'GPU compute processes:'
    local processes
    processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "$processes" ]]; then
        printf '%s\n' "$processes" | sed 's/^/  /'
    else
        echo '  (No CUDA compute process reported; NVENC/NVDEC activity may not appear here.)'
    fi
}

print_snapshot() {
    local cpu load memory disk
    cpu="$(cpu_percent)"
    load="$(cut -d ' ' -f 1-3 /proc/loadavg)"
    memory="$(free -h | awk '/^Mem:/ {printf "used %s / total %s (available %s)", $3, $2, $7}')"
    disk="$(df -h / | awk 'NR == 2 {printf "used %s / total %s (%s), available %s", $3, $2, $5, $4}')"

    printf '\033[2J\033[H'
    echo '=== Video Mask Resource Monitor ==='
    printf 'Host: %s | Time: %s | Refresh: %ss\n' "$(hostname)" "$(date '+%F %T %Z')" "$INTERVAL"
    printf 'CPU: %s | Load (1/5/15m): %s\n' "$cpu" "$load"
    printf 'Memory: %s\nDisk (/): %s\n\n' "$memory" "$disk"

    print_gpu
    echo
    echo 'Video processing processes (sorted by CPU):'
    print_processes
    echo
    echo 'Ctrl+C exits monitoring only; it does not stop any task.'
}

trap 'printf "\nResource monitor stopped. Running tasks were not changed.\n"; exit 0' INT TERM

while true; do
    print_snapshot
    ((ONCE)) && exit 0
    sleep "$INTERVAL"
done
