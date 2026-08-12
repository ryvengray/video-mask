import argparse
import time
from pathlib import Path

from cluster.autoscaler import Autoscaler, PoolHost, load_pool


def scaler_args(tmp_path: Path, **overrides):
    values = {
        "controller_url": "http://127.0.0.1:8080",
        "admin_token": "x" * 16,
        "pool_file": str(tmp_path / "pool.yml"),
        "start_command": str(tmp_path / "start_ec2.sh"),
        "stop_command": str(tmp_path / "stop_ec2.sh"),
        "state_file": str(tmp_path / "state.json"),
        "managed_ips": "172.31.35.195,172.31.47.141",
        "idle_shutdown_seconds": 1800,
        "min_running_hosts": 0,
        "max_start_per_check": 1,
        "max_stop_per_check": 1,
        "start_grace_seconds": 900,
        "pending_grace_seconds": 60,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def worker(ip: str, status: str = "ready"):
    return {"status": status, "capabilities": {"controller_seen_ip": ip}}


def test_load_pool_keeps_only_allowlisted_valid_hosts(tmp_path: Path):
    pool = tmp_path / "pool.yml"
    pool.write_text("""hosts:
  - private_ip: 172.31.35.195
    status: stopped
    name: slave-01
  - private_ip: 172.31.47.141
    status: running
    name: slave-02
  - private_ip: not-an-ip
    status: running
""")

    hosts = load_pool(pool, {"172.31.35.195"})

    assert hosts == [PoolHost("172.31.35.195", "stopped", "slave-01")]


def test_pending_work_starts_one_stopped_host(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    hosts = [PoolHost("172.31.35.195", "stopped", "slave-01"),
             PoolHost("172.31.47.141", "running", "slave-02")]

    stamp = time.time()
    actions = scaler.plan(hosts, [worker("172.31.47.141")], pending=2,
                          oldest_pending_created_at=stamp - 61,
                          state={"idle_since": {}, "start_requested_at": {}}, stamp=stamp)

    assert [(action.kind, action.host.private_ip) for action in actions] == [("start", "172.31.35.195")]


def test_start_grace_prevents_repeating_the_same_ec2_start_request(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    hosts = [PoolHost("172.31.35.195", "stopped", "slave-01")]
    stamp = time.time()

    actions = scaler.plan(hosts, [], pending=1, oldest_pending_created_at=stamp - 1000,
                          state={"idle_since": {}, "start_requested_at": {"172.31.35.195": stamp}},
                          stamp=stamp)

    assert actions == []


def test_busy_or_unregistered_host_is_never_stopped(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    hosts = [PoolHost("172.31.35.195", "running", "slave-01"),
             PoolHost("172.31.47.141", "running", "slave-02")]
    stamp = time.time()

    actions = scaler.plan(hosts, [worker("172.31.35.195", "busy")], pending=0,
                          oldest_pending_created_at=None,
                          state={"idle_since": {"172.31.35.195": stamp - 3600},
                                 "start_requested_at": {}}, stamp=stamp)

    assert actions == []


def test_only_idle_host_can_stop_after_timeout_and_minimum_is_preserved(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path, min_running_hosts=1))
    hosts = [PoolHost("172.31.35.195", "running", "slave-01"),
             PoolHost("172.31.47.141", "running", "slave-02")]
    stamp = time.time()
    state = {"idle_since": {"172.31.35.195": stamp - 3600, "172.31.47.141": stamp - 3600},
             "start_requested_at": {}}

    actions = scaler.plan(hosts, [worker("172.31.35.195"), worker("172.31.47.141")],
                          pending=0, oldest_pending_created_at=None, state=state, stamp=stamp)

    assert len(actions) == 1
    assert actions[0].kind == "stop"


def test_newly_pending_task_waits_for_ready_worker_to_claim(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    hosts = [PoolHost("172.31.35.195", "stopped", "slave-01")]
    stamp = time.time()

    actions = scaler.plan(hosts, [], pending=1, oldest_pending_created_at=stamp - 20,
                          state={"idle_since": {}, "start_requested_at": {}}, stamp=stamp)

    assert actions == []
