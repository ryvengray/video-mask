import argparse
import time
from pathlib import Path

import pytest

from cluster.autoscaler import Autoscaler, PoolHost, load_pool


def scaler_args(tmp_path: Path, **overrides):
    values = {
        "controller_url": "http://127.0.0.1:8080",
        "admin_token": "x" * 16,
        "pool_file": str(tmp_path / "pool.yml"),
        "pool_refresh_command": str(tmp_path / "ec2_pool.sh"),
        "pool_refresh_timeout_seconds": 120,
        "start_command": str(tmp_path / "start_ec2.sh"),
        "stop_command": str(tmp_path / "stop_ec2.sh"),
        "state_file": str(tmp_path / "state.json"),
        "event_log": str(tmp_path / "events.log"),
        "host_slot": ["172.31.35.195=15", "172.31.47.141=6"],
        "idle_shutdown_seconds": 1800,
        "min_running_hosts": 0,
        "max_start_per_check": 1,
        "max_stop_per_check": 1,
        "start_grace_seconds": 900,
        "stop_grace_seconds": 1800,
        "command_timeout_seconds": 900,
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


def test_start_selection_uses_smallest_host_that_covers_pending_capacity(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    hosts = [PoolHost("172.31.35.195", "stopped", "15-slot"),
             PoolHost("172.31.47.141", "stopped", "6-slot")]
    stamp = time.time()

    small_need = scaler.plan(hosts, [], pending=4, oldest_pending_created_at=stamp - 61,
                             state={"idle_since": {}, "start_requested_at": {}}, stamp=stamp)
    large_need = scaler.plan(hosts, [], pending=7, oldest_pending_created_at=stamp - 61,
                             state={"idle_since": {}, "start_requested_at": {}}, stamp=stamp)

    assert small_need[0].host.private_ip == "172.31.47.141"
    assert large_need[0].host.private_ip == "172.31.35.195"


def test_host_slot_values_require_a_private_ip_and_slot_count(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid --host-slot"):
        Autoscaler(scaler_args(tmp_path, host_slot=["not-an-ip=3"]))


def test_at_least_one_host_slot_is_required(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one --host-slot"):
        Autoscaler(scaler_args(tmp_path, host_slot=[]))


def test_event_log_records_state_transitions_without_poll_records(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    before = {"idle_since": {}, "start_requested_at": {}, "stop_requested_at": {}, "pool_status": {}}
    after = {
        "idle_since": {"172.31.47.141": 123.0},
        "start_requested_at": {"172.31.35.195": 124.0},
        "stop_requested_at": {"172.31.47.141": 125.0},
        "pool_status": {"172.31.35.195": "starting", "172.31.47.141": "running"},
    }

    scaler.record_state_events(before, after)
    events = (tmp_path / "events.log").read_text()

    assert "event=idle_since_set ip=172.31.47.141" in events
    assert "event=start_requested ip=172.31.35.195" in events
    assert "event=stop_requested ip=172.31.47.141" in events
    assert "event=pool_status_changed ip=172.31.35.195" in events
    assert "Autoscaler check" not in events


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


def test_paused_task_dispatch_does_not_start_workers_for_queued_tasks(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))
    hosts = [PoolHost("172.31.35.195", "stopped", "slave-01")]
    stamp = time.time()

    actions = scaler.plan(hosts, [], pending=20, oldest_pending_created_at=stamp - 3600,
                          state={"idle_since": {}, "start_requested_at": {}}, stamp=stamp,
                          task_dispatch_enabled=False)

    assert actions == []


def test_queue_and_workers_requires_controller_dispatch_state(tmp_path: Path, monkeypatch):
    scaler = Autoscaler(scaler_args(tmp_path))
    monkeypatch.setattr(scaler, "api", lambda _path: {"total": 0, "workers": []})

    with pytest.raises(RuntimeError, match="task_dispatch_enabled"):
        scaler.queue_and_workers()


def test_paused_task_dispatch_allows_idle_workers_to_shut_down(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path, min_running_hosts=0))
    hosts = [PoolHost("172.31.35.195", "running", "slave-01")]
    stamp = time.time()

    actions = scaler.plan(
        hosts, [worker("172.31.35.195")], pending=20, oldest_pending_created_at=stamp - 3600,
        state={"idle_since": {"172.31.35.195": stamp - 3600}, "start_requested_at": {}}, stamp=stamp,
        task_dispatch_enabled=False,
    )

    assert [(action.kind, action.host.private_ip) for action in actions] == [("stop", "172.31.35.195")]


def test_stop_grace_skips_a_host_while_its_prior_stop_is_still_settling(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path, min_running_hosts=0))
    hosts = [PoolHost("172.31.35.195", "running", "slave-01"),
             PoolHost("172.31.47.141", "running", "slave-02")]
    stamp = time.time()
    state = {"idle_since": {"172.31.35.195": stamp - 3600, "172.31.47.141": stamp - 3600},
             "start_requested_at": {}, "stop_requested_at": {"172.31.35.195": stamp - 10}}

    actions = scaler.plan(hosts, [worker("172.31.35.195"), worker("172.31.47.141")],
                          pending=0, oldest_pending_created_at=None, state=state, stamp=stamp)

    assert [(action.kind, action.host.private_ip) for action in actions] == [("stop", "172.31.47.141")]


def test_pool_refresh_requires_an_operations_command(tmp_path: Path):
    scaler = Autoscaler(scaler_args(tmp_path))

    with pytest.raises(RuntimeError, match="refresh command does not exist"):
        scaler.refresh_pool()
