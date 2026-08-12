#!/usr/bin/env python3
"""Conservative Controller-side EC2 Worker autoscaling.

The Controller owns the queue; this process therefore runs only on the
Controller machine.  It reads the operations-maintained EC2 pool file and
calls the approved local start/stop scripts with one private IP at a time.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - deployment dependency guard
    raise SystemExit("Autoscaling requires PyYAML; install requirements-cluster.txt") from exc


ACTIVE_SLOT_STATUSES = {"assigned", "downloading", "processing", "uploading", "cancelling", "busy"}


@dataclass(frozen=True)
class PoolHost:
    private_ip: str
    status: str
    name: str


@dataclass(frozen=True)
class Action:
    kind: str
    host: PoolHost
    reason: str


def load_pool(path: Path, managed_ips: set[str]) -> list[PoolHost]:
    """Load only explicitly managed hosts from the operations pool file."""
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        raise RuntimeError(f"cannot read EC2 host pool {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("hosts"), list):
        raise RuntimeError(f"invalid EC2 host pool format: {path}")
    hosts: list[PoolHost] = []
    for value in document["hosts"]:
        if not isinstance(value, dict):
            continue
        ip = str(value.get("private_ip") or "")
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            logging.warning("Ignoring EC2 pool entry with invalid private_ip: %r", ip)
            continue
        if ip not in managed_ips:
            continue
        hosts.append(PoolHost(ip, str(value.get("status") or "").lower(),
                              str(value.get("name") or ip)))
    missing = managed_ips - {host.private_ip for host in hosts}
    if missing:
        logging.warning("Managed IPs missing from the EC2 pool file: %s", ", ".join(sorted(missing)))
    return hosts


def worker_ip(worker: dict[str, Any]) -> str:
    capabilities = worker.get("capabilities") or {}
    return str(capabilities.get("controller_seen_ip") or "")


class Autoscaler:
    def __init__(self, args: argparse.Namespace):
        self.controller_url = args.controller_url.rstrip("/")
        self.admin_token = args.admin_token or os.environ.get("VIDEO_MASK_ADMIN_TOKEN", "")
        self.pool_file = Path(args.pool_file)
        self.start_command = Path(args.start_command)
        self.stop_command = Path(args.stop_command)
        self.state_file = Path(args.state_file)
        self.managed_ips = set(args.managed_ips.split(",")) - {""}
        self.idle_shutdown_seconds = args.idle_shutdown_seconds
        self.min_running_hosts = args.min_running_hosts
        self.max_start_per_check = args.max_start_per_check
        self.max_stop_per_check = args.max_stop_per_check
        self.start_grace_seconds = args.start_grace_seconds
        self.pending_grace_seconds = args.pending_grace_seconds
        self.dry_run = args.dry_run

    def api(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.controller_url + path,
            headers={"Authorization": f"Bearer {self.admin_token}"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected Controller response from {path}")
        return payload

    def queue_and_workers(self) -> tuple[int, float | None, list[dict[str, Any]]]:
        pending = self.api("/api/tasks?status=pending&limit=1")
        workers = self.api("/api/workers?limit=1000")
        oldest = pending.get("oldest_created_at")
        return int(pending.get("total") or 0), float(oldest) if oldest is not None else None, list(workers.get("workers") or [])

    def load_state(self) -> dict[str, dict[str, float]]:
        try:
            raw = json.loads(self.state_file.read_text())
        except FileNotFoundError:
            return {"idle_since": {}, "start_requested_at": {}}
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Ignoring unreadable autoscaler state %s: %s", self.state_file, exc)
            return {"idle_since": {}, "start_requested_at": {}}
        return {
            "idle_since": {str(key): float(value) for key, value in (raw.get("idle_since") or {}).items()},
            "start_requested_at": {str(key): float(value) for key, value in (raw.get("start_requested_at") or {}).items()},
        }

    def save_state(self, state: dict[str, dict[str, float]]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
        temporary.replace(self.state_file)

    @staticmethod
    def host_workers(host: PoolHost, workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [worker for worker in workers if worker_ip(worker) == host.private_ip]

    @staticmethod
    def estimated_slots(host: PoolHost, workers: list[dict[str, Any]]) -> int:
        # A deployed, stopped Worker has historical slot rows. For a host that
        # has never registered, conservatively assume one slot.
        return max(1, len(Autoscaler.host_workers(host, workers)))

    def plan(self, hosts: list[PoolHost], workers: list[dict[str, Any]], pending: int,
             oldest_pending_created_at: float | None,
             state: dict[str, dict[str, float]], stamp: float) -> list[Action]:
        ready_slots = sum(1 for worker in workers if worker.get("status") == "ready")
        running = [host for host in hosts if host.status == "running"]
        stopped = [host for host in hosts if host.status == "stopped"]

        # Clear cooldown state once the pool file reports a completed start.
        for host in running:
            state["start_requested_at"].pop(host.private_ip, None)

        needed_slots = max(0, pending - ready_slots)
        requested_capacity = sum(
            self.estimated_slots(host, workers)
            for host in stopped
            if stamp - state["start_requested_at"].get(host.private_ip, 0) < self.start_grace_seconds
        )
        actions: list[Action] = []
        pending_age = max(0, stamp - oldest_pending_created_at) if oldest_pending_created_at else 0
        if needed_slots > requested_capacity and pending_age >= self.pending_grace_seconds:
            for host in stopped:
                if len(actions) >= self.max_start_per_check:
                    break
                last_request = state["start_requested_at"].get(host.private_ip, 0)
                if stamp - last_request < self.start_grace_seconds:
                    continue
                actions.append(Action("start", host,
                                      f"{pending} pending task(s) waited {round(pending_age)}s, "
                                      f"{ready_slots} ready slot(s)"))
                needed_slots -= self.estimated_slots(host, workers)
                if needed_slots <= requested_capacity:
                    break
            return actions  # Never scale down while the queue needs capacity.

        if pending:
            if needed_slots > requested_capacity:
                logging.info("Autoscaler is waiting for pending tasks to age: %.0fs / %ss",
                             pending_age, self.pending_grace_seconds)
            return actions

        # A host is eligible only when every registered slot is ready. An
        # unknown/booting host is deliberately not eligible for shutdown.
        candidates: list[PoolHost] = []
        for host in running:
            host_workers = self.host_workers(host, workers)
            statuses = {str(worker.get("status") or "") for worker in host_workers}
            if not host_workers or statuses != {"ready"}:
                state["idle_since"].pop(host.private_ip, None)
                continue
            state["idle_since"].setdefault(host.private_ip, stamp)
            if stamp - state["idle_since"][host.private_ip] >= self.idle_shutdown_seconds:
                candidates.append(host)

        for host in candidates:
            if len(actions) >= self.max_stop_per_check:
                break
            if len(running) - len(actions) <= self.min_running_hosts:
                break
            actions.append(Action("stop", host,
                                  f"all {len(self.host_workers(host, workers))} slot(s) idle for at least "
                                  f"{self.idle_shutdown_seconds}s"))
        return actions

    def execute(self, action: Action, state: dict[str, dict[str, float]], stamp: float) -> None:
        command = self.start_command if action.kind == "start" else self.stop_command
        if not command.is_file():
            raise RuntimeError(f"{action.kind} command does not exist: {command}")
        invocation = ["sudo", "-n", str(command), "--ips", action.host.private_ip]
        logging.warning("Autoscaler %s %s (%s): %s", action.kind, action.host.name,
                        action.reason, " ".join(invocation))
        if self.dry_run:
            return
        result = subprocess.run(invocation, text=True, capture_output=True, check=False, timeout=300)
        if result.returncode:
            diagnostic = "\n".join(value for value in (result.stdout, result.stderr) if value).strip()
            raise RuntimeError(f"{action.kind} command failed for {action.host.private_ip}: {diagnostic[-2000:]}")
        if action.kind == "start":
            state["start_requested_at"][action.host.private_ip] = stamp
        else:
            state["idle_since"].pop(action.host.private_ip, None)

    def reconcile(self) -> list[Action]:
        if not self.admin_token:
            raise RuntimeError("VIDEO_MASK_ADMIN_TOKEN is required for autoscaling")
        hosts = load_pool(self.pool_file, self.managed_ips)
        pending, oldest_pending_created_at, workers = self.queue_and_workers()
        state = self.load_state()
        stamp = time.time()
        actions = self.plan(hosts, workers, pending, oldest_pending_created_at, state, stamp)
        for action in actions:
            self.execute(action, state, stamp)
        self.save_state(state)
        logging.info("Autoscaler check: pending=%s, workers=%s, actions=%s", pending, len(workers),
                     ", ".join(f"{item.kind}:{item.host.private_ip}" for item in actions) or "none")
        return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Controller-side EC2 Worker autoscaler")
    parser.add_argument("--controller-url", required=True)
    parser.add_argument("--pool-file", required=True)
    parser.add_argument("--start-command", required=True)
    parser.add_argument("--stop-command", required=True)
    parser.add_argument("--managed-ips", required=True, help="Comma-separated private IP allowlist")
    parser.add_argument("--state-file", default="/var/lib/video-mask-autoscaler/state.json")
    parser.add_argument("--admin-token", help="Defaults to VIDEO_MASK_ADMIN_TOKEN")
    parser.add_argument("--check-seconds", type=int, default=60)
    parser.add_argument("--idle-shutdown-seconds", type=int, default=1800)
    parser.add_argument("--min-running-hosts", type=int, default=0)
    parser.add_argument("--max-start-per-check", type=int, default=1)
    parser.add_argument("--max-stop-per-check", type=int, default=1)
    parser.add_argument("--start-grace-seconds", type=int, default=900)
    parser.add_argument("--pending-grace-seconds", type=int, default=60,
                        help="Wait for a newly queued task to be claimed before starting EC2")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.check_seconds < 5 or args.idle_shutdown_seconds < 60 or args.start_grace_seconds < 60
            or args.pending_grace_seconds < 0):
        raise SystemExit("check/start grace must be at least 60 seconds and idle shutdown at least 60 seconds")
    if args.min_running_hosts < 0 or args.max_start_per_check < 1 or args.max_stop_per_check < 1:
        raise SystemExit("host limits must be non-negative, and per-check limits at least 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scaler = Autoscaler(args)
    while True:
        try:
            scaler.reconcile()
        except Exception as exc:
            logging.exception("Autoscaler check failed: %s", exc)
            # A manual --once check is an operator command: make failure
            # visible to its caller instead of returning a misleading 0.
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.check_seconds)


if __name__ == "__main__":
    main()
