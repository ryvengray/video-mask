#!/usr/bin/env bash
# Interactive, detached task manager for the video-mask batch scheduler.
# Run from any directory: bash scripts/task_manager.sh
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SCHEDULER="$ROOT_DIR/scripts/batch_scheduler.py"
TASK_HOME="$ROOT_DIR/.task_manager"
TASKS_DIR="$TASK_HOME/tasks"
RUNNERS_DIR="$TASK_HOME/runners"
DEFAULT_ALGORITHM="$ROOT_DIR/video_mask_batch_fish.py"
DEFAULT_EXTRA_ARGS=(--fisheye --fisheye-device pico4 --no-card --face-size 960 --face-int 5 --frame-skip 3 --face-model yolov8)
DEFAULT_EXTRA_LINE="--fisheye --fisheye-device pico4 --no-card --face-size 960 --face-int 5 --frame-skip 3 --face-model yolov8"
FISH_V1_EXTRA_ARGS=(--fisheye --fisheye-device pico4 --face-size 960 --face-int 5 --frame-skip 3 --face-model yolov8)
FISH_V1_EXTRA_LINE="--fisheye --fisheye-device pico4 --face-size 960 --face-int 5 --frame-skip 3 --face-model yolov8"

mkdir -p "$TASKS_DIR" "$RUNNERS_DIR"
LAST_INTERRUPT_AT=0
INTERRUPTED_PROMPT=0

handle_interrupt() {
    local now
    now="$(date +%s)"
    if (( LAST_INTERRUPT_AT > 0 && now - LAST_INTERRUPT_AT <= 3 )); then
        printf '\nSecond Ctrl+C received. Exiting task manager; detached tasks stay active.\n'
        trap - INT
        exit 130
    fi
    LAST_INTERRUPT_AT="$now"
    INTERRUPTED_PROMPT=1
    printf '\nCtrl+C received. Press Ctrl+C again within 3 seconds to exit; detached tasks stay active.\n'
}

trap handle_interrupt INT

die() { printf 'Error: %s\n' "$*" >&2; }
pause() { read -r -p 'Press Enter to return to the menu... ' _; }

write_meta() {
    local file="$1" id="$2" pid="$3" started="$4" algorithm="$5" source_dir="$6" target_dir="$7" log_file="$8" extra_args="$9" workers="${10}" force_reprocess="${11}"
    {
        printf 'TASK_ID=%q\n' "$id"
        printf 'PID=%q\n' "$pid"
        printf 'STARTED=%q\n' "$started"
        printf 'ALGORITHM=%q\n' "$algorithm"
        printf 'SOURCE_DIR=%q\n' "$source_dir"
        printf 'TARGET_DIR=%q\n' "$target_dir"
        printf 'LOG_FILE=%q\n' "$log_file"
    printf 'EXTRA_ARGS=%q\n' "$extra_args"
    printf 'WORKERS=%q\n' "$workers"
    printf 'FORCE_REPROCESS=%q\n' "$force_reprocess"
    } > "$file"
}

task_state() {
    local meta="$1"
    # Metadata is created by this script with shell-escaped values.
    # shellcheck disable=SC1090
    source "$meta"
    local base="${meta%.meta}"
    if task_is_alive "$meta"; then
        printf 'RUNNING'
    elif [[ -f "$base.stop_requested" ]]; then
        printf 'STOPPED'
    elif [[ -f "$base.exit_code" ]] && [[ "$(<"$base.exit_code")" == "0" ]]; then
        printf 'COMPLETED'
    elif [[ -f "$base.exit_code" ]]; then
        printf 'FAILED (exit %s)' "$(<"$base.exit_code")"
    else
        printf 'UNKNOWN'
    fi
}

task_is_alive() {
    local meta="$1" command_line
    # shellcheck disable=SC1090
    source "$meta"
    kill -0 "$PID" 2>/dev/null || return 1
    command_line="$(ps -p "$PID" -o args= 2>/dev/null || true)"
    [[ "$command_line" == *"$RUNNERS_DIR/$TASK_ID.sh"* ]]
}

print_task() {
    local meta="$1" state
    # shellcheck disable=SC1090
    source "$meta"
    state="$(task_state "$meta")"
    printf '%-24s %-16s %-20s %s\n' "$TASK_ID" "$state" "$ALGORITHM" "$STARTED"
    printf '  Source: %s\n  Target: %s\n  Log:    %s\n' "$SOURCE_DIR" "$TARGET_DIR" "$LOG_FILE"
}

list_tasks() {
    local mode="${1:-all}" meta state found=0
    shopt -s nullglob
    for meta in "$TASKS_DIR"/*.meta; do
        state="$(task_state "$meta")"
        if [[ "$mode" == "running" && "$state" != "RUNNING" ]]; then
            continue
        fi
        print_task "$meta"
        found=1
    done
    shopt -u nullglob
    [[ "$found" == 1 ]] || printf 'No %s tasks found.\n' "$mode"
}

choose_algorithm() {
    local -a files=()
    local file choice default_index=0
    while IFS= read -r file; do files+=("$file"); done < <(
        find "$ROOT_DIR" -maxdepth 1 -type f -name 'video_mask_batch*.py' -print | sort
    )
    if ((${#files[@]} == 0)); then
        die "No algorithm files matching video_mask_batch*.py were found."
        return 1
    fi
    printf '\nChoose algorithm:\n'
    for i in "${!files[@]}"; do
        printf '  %d) %s\n' "$((i + 1))" "$(basename "${files[i]}")"
        [[ "${files[i]}" == "$DEFAULT_ALGORITHM" ]] && default_index="$((i + 1))"
    done
    if ((default_index == 0)); then
        die "Default algorithm was not found: $(basename "$DEFAULT_ALGORITHM")"
        return 1
    fi
    while true; do
        read -r -p "Algorithm [1-${#files[@]}, Enter=$(basename "$DEFAULT_ALGORITHM")]: " choice
        if [[ -z "$choice" ]]; then
            CHOSEN_ALGORITHM="$DEFAULT_ALGORITHM"
            return 0
        fi
        if [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice <= ${#files[@]})); then
            CHOSEN_ALGORITHM="${files[choice - 1]}"
            return 0
        fi
        printf 'Please enter a number in the displayed range.\n'
    done
}

start_task() {
    local source_dir target_dir extra_line default_extra_line workers_line force_line force_reprocess id started log_file meta_file runner_file pid
    local workers
    local -a extra_args=("${DEFAULT_EXTRA_ARGS[@]}")
    if [[ ! -x "$PYTHON_BIN" ]]; then
        die "Python was not found: $PYTHON_BIN"
        return
    fi
    if [[ ! -f "$SCHEDULER" ]]; then
        die "Scheduler was not found: $SCHEDULER"
        return
    fi
    if ! command -v setsid >/dev/null 2>&1; then
        die "setsid is required for detached background tasks. Install util-linux."
        return
    fi
    choose_algorithm || return

    # fish_v1 is face-only and deliberately does not expose --no-card.
    # Keep the menu defaults compatible with the selected algorithm.
    if [[ "$(basename "$CHOSEN_ALGORITHM")" == "video_mask_batch_fish_v1.py" ]]; then
        extra_args=("${FISH_V1_EXTRA_ARGS[@]}")
        default_extra_line="$FISH_V1_EXTRA_LINE"
    else
        default_extra_line="$DEFAULT_EXTRA_LINE"
    fi

    read -r -p 'Source directory [/home/ubuntu/sources]: ' source_dir || return
    source_dir="${source_dir:-/home/ubuntu/sources}"
    read -r -p 'Target directory [/home/ubuntu/outputs]: ' target_dir || return
    target_dir="${target_dir:-/home/ubuntu/outputs}"
    read -r -p "Algorithm arguments [${default_extra_line}]: " extra_line || return
    if [[ -n "$extra_line" ]]; then
        # A non-empty response intentionally replaces the defaults.
        read -r -a extra_args <<< "$extra_line"
    else
        extra_line="$default_extra_line"
    fi
    read -r -p 'Force reprocess completed videos? [y/N]: ' force_line || return
    if [[ "$force_line" =~ ^[Yy]$ ]]; then
        force_reprocess=yes
    else
        force_reprocess=no
    fi
    workers=1
    [[ "$(basename "$CHOSEN_ALGORITHM")" == "video_mask_face_gpu.py" ]] && workers=2
    for arg in "${extra_args[@]}"; do
        [[ "$arg" == "--no-card" ]] && workers=2
    done
    read -r -p "Worker count [$workers]: " workers_line || return
    workers_line="${workers_line:-$workers}"
    if [[ "$workers_line" =~ ^[1-9][0-9]*$ ]]; then
        workers="$workers_line"
    else
        die "Worker count must be a positive integer."
        return
    fi

    if [[ ! -d "$source_dir" ]]; then
        die "Source directory does not exist: $source_dir"
        return
    fi
    mkdir -p "$target_dir" || { die "Cannot create target directory: $target_dir"; return; }
    source_dir="$(cd "$source_dir" && pwd)"
    target_dir="$(cd "$target_dir" && pwd)"

    started="$(date '+%Y-%m-%d %H:%M:%S %z')"
    id="$(date '+%Y%m%d-%H%M%S')-$$"
    # Keep the execution log beside its outputs, so result delivery preserves evidence too.
    mkdir -p "$target_dir/.task_manager_logs"
    log_file="$target_dir/.task_manager_logs/$id.log"
    meta_file="$TASKS_DIR/$id.meta"
    runner_file="$RUNNERS_DIR/$id.sh"

    {
        printf '#!/usr/bin/env bash\nset +e\n'
        printf 'echo "=== Video Mask Task Started ==="\n'
        printf 'printf "Task ID: %%s\\n" %q\n' "$id"
        printf 'printf "Started: %%s\\n" %q\n' "$started"
        printf 'printf "Algorithm: %%s\\n" %q\n' "$(basename "$CHOSEN_ALGORITHM")"
        printf 'printf "Source: %%s\\n" %q\n' "$source_dir"
        printf 'printf "Target: %%s\\n" %q\n' "$target_dir"
        printf 'printf "Arguments: %%s\\n" %q\n' "${extra_line:-(none)}"
        printf 'printf "Force reprocess: %%s\\n" %q\n' "$force_reprocess"
        printf 'printf "Workers: %%s\\n" %q\n' "$workers"
        printf 'echo "--- Runtime environment ---"\n'
        printf 'hostname || true\nuname -a || true\n'
        printf '%q -c %q || true\n' "$PYTHON_BIN" \
            'import sys; print("Python:", sys.version.replace("\\n", " "))'
        printf '%q -c %q || true\n' "$PYTHON_BIN" \
            'import cv2; print("OpenCV:", cv2.__version__)'
        printf '%q -c %q || true\n' "$PYTHON_BIN" \
            'import torch; print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available(), "CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
        printf '%q -c %q || true\n' "$PYTHON_BIN" \
            'import onnxruntime as ort; print("ONNX Runtime:", ort.__version__, "providers:", ort.get_available_providers())'
        printf 'command -v ffmpeg >/dev/null && ffmpeg -version 2>/dev/null | head -1 || true\n'
        printf 'command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true\n'
        printf 'echo "--- Scheduler output ---"\n'
        printf '%q -u %q %q %q %q %q %q %q ' "$PYTHON_BIN" "$SCHEDULER" "$source_dir" "$target_dir" '--algorithm' "$CHOSEN_ALGORITHM" '--workers' "$workers"
        [[ "$force_reprocess" == yes ]] && printf '%q ' '--force'
        local arg
        for arg in "${extra_args[@]}"; do
            # '=' is required when the forwarded value itself starts with '--'.
            printf '%q ' "--extra-arg=$arg"
        done
        printf '\ncode=$?\necho "--- Task finished ---"\necho "Exit code: $code"\ndate "+Finished: %%Y-%%m-%%d %%H:%%M:%%S %%z"\ncommand -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader || true\nprintf "%%s\\n" "$code" > %q\ndate "+%%Y-%%m-%%d %%H:%%M:%%S %%z" > %q\nexit "$code"\n' \
            "$TASKS_DIR/$id.exit_code" "$TASKS_DIR/$id.finished_at"
    } > "$runner_file"
    chmod 700 "$runner_file"

    # setsid gives the scheduler its own session/process group. It survives this menu and Ctrl+C.
    setsid "$runner_file" > "$log_file" 2>&1 < /dev/null &
    pid=$!
    write_meta "$meta_file" "$id" "$pid" "$started" "$(basename "$CHOSEN_ALGORITHM")" \
        "$source_dir" "$target_dir" "$log_file" "$extra_line" "$workers" "$force_reprocess"

    printf '\nTask started in the background.\n'
    printf 'Task ID: %s\nPID: %s\nLog: %s\n' "$id" "$pid" "$log_file"
    printf 'Use "View current progress" to inspect it. Ctrl+C will not stop this task.\n'
}

select_running_task() {
    local -a metas=()
    local meta choice
    shopt -s nullglob
    for meta in "$TASKS_DIR"/*.meta; do
        [[ "$(task_state "$meta")" == "RUNNING" ]] && metas+=("$meta")
    done
    shopt -u nullglob
    if ((${#metas[@]} == 0)); then
        printf 'No running tasks found.\n'
        return 1
    fi
    printf '\nRunning tasks:\n'
    for i in "${!metas[@]}"; do
        # shellcheck disable=SC1090
        source "${metas[i]}"
        printf '  %d) %s  %s -> %s\n' "$((i + 1))" "$TASK_ID" "$SOURCE_DIR" "$TARGET_DIR"
    done
    while true; do
        read -r -p "Task to stop [1-${#metas[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice <= ${#metas[@]})); then
            SELECTED_META="${metas[choice - 1]}"
            return 0
        fi
        printf 'Please enter a number in the displayed range.\n'
    done
}

stop_task() {
    local base
    select_running_task || return
    # shellcheck disable=SC1090
    source "$SELECTED_META"
    read -r -p "Stop task $TASK_ID? [y/N]: " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { printf 'Cancelled.\n'; return; }
    base="${SELECTED_META%.meta}"
    : > "$base.stop_requested"
    # Signal the isolated process group. Scheduler receives Ctrl+C semantics and can resume later.
    kill -INT -- "-$PID" 2>/dev/null || true
    printf 'Stop requested for %s. Waiting up to 15 seconds...\n' "$TASK_ID"
    for _ in {1..15}; do
        task_is_alive "$SELECTED_META" || break
        sleep 1
    done
    if task_is_alive "$SELECTED_META"; then
        printf 'Task is still running; sending TERM to its process group.\n'
        kill -TERM -- "-$PID" 2>/dev/null || true
    fi
    printf 'Stop signal sent. The task is marked STOPPED when its process exits.\n'
}

view_progress() {
    local meta base
    printf '\nCurrent tasks:\n'
    list_tasks running
    shopt -s nullglob
    for meta in "$TASKS_DIR"/*.meta; do
        [[ "$(task_state "$meta")" == "RUNNING" ]] || continue
        # shellcheck disable=SC1090
        source "$meta"
        printf '\n--- Last log lines: %s ---\n' "$TASK_ID"
        tail -n 12 "$LOG_FILE" 2>/dev/null || printf '(No log output yet.)\n'
        base="${meta%.meta}"
        [[ -f "$base.stop_requested" ]] && printf '(Stop has been requested.)\n'
    done
    shopt -u nullglob
}

view_history() {
    printf '\nTask history:\n'
    list_tasks all
}

while true; do
    printf '\n=== Video Mask Task Manager ===\n'
    printf '1) Start task\n2) Stop task\n3) View current progress\n4) View task history\n5) Exit\n'
    INTERRUPTED_PROMPT=0
    if ! read -r -p 'Choose an action [1-5]: ' action; then
        if (( INTERRUPTED_PROMPT )); then
            continue
        fi
        printf '\nInput closed. Goodbye. Running tasks remain active.\n'
        exit 0
    fi
    case "$action" in
        1) start_task; pause ;;
        2) stop_task; pause ;;
        3) view_progress; pause ;;
        4) view_history; pause ;;
        5) printf 'Goodbye. Running tasks remain active.\n'; exit 0 ;;
        *) printf 'Please choose a number from 1 to 5.\n' ;;
    esac
done
