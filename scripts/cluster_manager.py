#!/usr/bin/env python3
"""Interactive daily operations CLI for a Video Mask Cluster Controller."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UNFINISHED = ("pending", "assigned", "downloading", "processing", "uploading", "cancelling")
RESTARTABLE = ("completed", "failed", "cancelled")


class ControllerClient:
    def __init__(self, controller: str, token: str):
        self.controller = controller.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.controller + path,
            data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Controller returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Controller: {exc.reason}") from exc

    def task_page(self, statuses: tuple[str, ...] | None, offset: int, limit: int = 20) -> dict[str, Any]:
        query: dict[str, str | int] = {"limit": limit, "offset": offset}
        if statuses:
            query["status"] = ",".join(statuses)
        response = self.request("GET", "/api/tasks?" + urllib.parse.urlencode(query))
        if not {"tasks", "total", "limit", "offset"}.issubset(response):
            raise RuntimeError(
                "Controller API is outdated. Deploy the current Controller code before using cluster_manager.py."
            )
        return response

    def task(self, task_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/tasks/{urllib.parse.quote(task_id, safe='')}")

    def cancel(self, task_id: str) -> dict[str, Any]:
        return self.request("POST", f"/api/tasks/{urllib.parse.quote(task_id, safe='')}/cancel")

    def restart(self, task_id: str) -> dict[str, Any]:
        return self.request("POST", f"/api/tasks/{urllib.parse.quote(task_id, safe='')}/restart")

    def worker(self, worker_id: str) -> dict[str, Any] | None:
        response = self.request("GET", "/api/workers?limit=1000")
        return next((worker for worker in response["workers"] if worker["worker_id"] == worker_id), None)


def read_token(value: str | None, token_file: Path) -> str:
    if value:
        return value
    if os.environ.get("VIDEO_MASK_ADMIN_TOKEN"):
        return os.environ["VIDEO_MASK_ADMIN_TOKEN"]
    try:
        for line in token_file.read_text().splitlines():
            if line.startswith("VIDEO_MASK_ADMIN_TOKEN="):
                token = line.partition("=")[2].strip()
                if token:
                    return token
    except OSError:
        pass
    return getpass.getpass("Controller admin token: ").strip()


def stamp(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def duration(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(round(value), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def task_line(task: dict[str, Any]) -> str:
    progress = task.get("progress") or {}
    source = str(task.get("source_object_key") or task.get("source_url") or "-")
    processing = progress.get("processing_seconds", progress.get("elapsed_seconds"))
    return (
        f"{task['task_id']}  {task['status']:<11}  {str(task.get('assigned_worker_id') or '-'):<18} "
        f"{duration(processing):>8}  {source}"
    )


def browse_tasks(client: ControllerClient, statuses: tuple[str, ...] | None) -> None:
    offset = 0
    while True:
        page = client.task_page(statuses, offset)
        tasks = page["tasks"]
        total = page["total"]
        print(f"\nTasks {offset + 1 if total else 0}-{offset + len(tasks)} of {total}")
        print("Task ID                                Status       Worker              Process  Source")
        print("-" * 100)
        for task in tasks:
            print(task_line(task))
        if not tasks:
            return
        command = input("[n]ext, [p]revious, [q]uit: ").strip().lower() or "q"
        if command == "n" and offset + page["limit"] < total:
            offset += page["limit"]
        elif command == "p" and offset > 0:
            offset = max(0, offset - page["limit"])
        elif command == "q":
            return


def show_task(task: dict[str, Any]) -> None:
    print(json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True))


def confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def parse_slots(specification: str) -> tuple[int, ...]:
    slots: set[int] = set()
    for part in specification.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        if not match:
            raise RuntimeError("slots must use forms such as 1, 1-15, or 1,3-5")
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        if first < 1 or last < first or last > 16:
            raise RuntimeError("slot numbers must be between 1 and 16")
        slots.update(range(first, last + 1))
    return tuple(sorted(slots))


def restart_slots(client: ControllerClient, inventory: Path, host: str, slots: tuple[int, ...],
                  ask_vault_pass: bool) -> None:
    if not slots:
        raise RuntimeError("at least one slot is required")
    if not inventory.is_file():
        raise RuntimeError(f"Ansible inventory not found: {inventory}")
    worker_ids = [f"{host}-slot-{slot}" for slot in slots]
    busy = [(worker_id, client.worker(worker_id)) for worker_id in worker_ids]
    active = [(worker_id, worker) for worker_id, worker in busy if worker and worker.get("current_task_id")]
    if active:
        print("Warning: these slots have active tasks and will be interrupted:")
        for worker_id, worker in active:
            print(f"  {worker_id}: {worker['current_task_id']}")
        prompt = f"Restart {len(slots)} slot(s) anyway?"
    else:
        prompt = f"Restart {len(slots)} slot(s): {', '.join(worker_ids)}?"
    if not confirm(prompt):
        return
    units = " ".join(f"video-mask-worker@slot-{slot}.service" for slot in slots)
    command = [
        "ansible", "-i", str(inventory), host, "-b",
        "-m", "ansible.builtin.command",
        "-a", f"systemctl restart {units}",
    ]
    if ask_vault_pass:
        command.append("--ask-vault-pass")
    print("Restarting", ", ".join(worker_ids), "through Ansible...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Ansible restart failed with exit code {result.returncode}")


def interactive(client: ControllerClient, inventory: Path, ask_vault_pass: bool) -> None:
    while True:
        print("""
Video Mask Cluster Manager
  1) List unfinished tasks
  2) List recent tasks
  3) Show task details
  4) Cancel a task
  5) Restart a completed, failed, or cancelled task
  6) Restart a Worker slot through Ansible
  7) Exit""")
        choice = input("Choose: ").strip()
        try:
            if choice == "1":
                browse_tasks(client, UNFINISHED)
            elif choice == "2":
                browse_tasks(client, None)
            elif choice == "3":
                show_task(client.task(input("Full task ID: ").strip()))
            elif choice == "4":
                task_id = input("Full task ID to cancel: ").strip()
                task = client.task(task_id)
                print(task_line(task))
                if confirm("Cancel this task?"):
                    print("Updated:", client.cancel(task_id)["status"])
            elif choice == "5":
                task_id = input("Full terminal task ID to restart: ").strip()
                task = client.task(task_id)
                print(task_line(task))
                if task["status"] not in RESTARTABLE:
                    print("Only completed, failed, or cancelled tasks can be restarted.")
                elif confirm("Queue this task again?"):
                    print("Updated:", client.restart(task_id)["status"])
            elif choice == "6":
                host = input("Ansible host (for example worker-01): ").strip()
                slot_specification = input("Slot number/range (for example 1 or 1-15): ").strip()
                restart_slots(client, inventory, host, parse_slots(slot_specification), ask_vault_pass)
            elif choice == "7":
                return
            else:
                print("Choose 1-7.")
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Video Mask Controller tasks")
    parser.add_argument("action", nargs="?", default="menu",
                        choices=("menu", "list", "all", "detail", "cancel", "restart", "restart-slot"))
    parser.add_argument("task_id", nargs="?")
    parser.add_argument("--controller", default="http://127.0.0.1:8080")
    parser.add_argument("--token", help="Avoid when possible: --token is visible in shell history")
    parser.add_argument("--token-file", type=Path, default=Path("/etc/video-mask-controller.env"))
    parser.add_argument("--inventory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "ansible" / "inventory.yml")
    parser.add_argument("--slot", help="Slot number/range for restart-slot, for example 1 or 1-15")
    parser.add_argument("--no-ask-vault-pass", action="store_true",
                        help="Do not pass --ask-vault-pass to Ansible")
    args = parser.parse_args()
    client = ControllerClient(args.controller, read_token(args.token, args.token_file))
    try:
        if args.action == "menu":
            interactive(client, args.inventory, not args.no_ask_vault_pass)
        elif args.action in {"list", "all"}:
            browse_tasks(client, UNFINISHED if args.action == "list" else None)
        elif args.action == "restart-slot":
            if not args.task_id or args.slot is None:
                raise SystemExit("restart-slot requires an Ansible host and --slot N")
            restart_slots(client, args.inventory, args.task_id, parse_slots(args.slot), not args.no_ask_vault_pass)
        else:
            if not args.task_id:
                raise SystemExit(f"{args.action} requires TASK_ID")
            result = client.task(args.task_id) if args.action == "detail" else getattr(client, args.action)(args.task_id)
            show_task(result)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
