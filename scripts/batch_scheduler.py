#!/usr/bin/env python3
"""可恢复的视频打码任务调度器。

算法脚本保持独立：本调度器只负责扫描、排队、可见进度、状态持久化与重试。
默认单 worker 是刻意设计，避免多个 OWLv2 进程同时占满同一张 GPU。
"""
from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".mpg",
    ".mpeg", ".ts", ".m2ts", ".wmv", ".3gp", ".rmvb", ".rm", ".vob", ".asf",
}


@dataclass(frozen=True)
class VideoJob:
    source: Path
    relative: Path
    destination: Path
    size: int
    mtime_ns: int


class JobStore:
    """SQLite 状态库；状态与输出目录放在一起，支持中断后恢复。"""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                source TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                destination TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                duration REAL,
                error TEXT,
                updated_at REAL NOT NULL
            )
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def prepare(self, job: VideoJob, valid_output: bool, force: bool) -> str:
        key = str(job.relative)
        row = self.conn.execute(
            "SELECT size, mtime_ns, status FROM jobs WHERE source = ?", (key,)
        ).fetchone()
        changed = row is None or row[0] != job.size or row[1] != job.mtime_ns
        if force or changed:
            status = "pending"
        elif row[2] == "running":
            status = "pending"  # 上次异常退出，可安全恢复
        elif row[2] == "success" and valid_output:
            status = "success"
        elif valid_output:
            # 输出存在且 ffprobe 合格，接管旧输出，避免再次处理。
            status = "success"
        else:
            status = row[2]
        self.conn.execute("""
            INSERT INTO jobs(source, size, mtime_ns, destination, status, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                size=excluded.size, mtime_ns=excluded.mtime_ns,
                destination=excluded.destination, status=excluded.status,
                updated_at=excluded.updated_at
        """, (key, job.size, job.mtime_ns, str(job.destination), status, time.time()))
        self.conn.commit()
        return status

    def mark_running(self, job: VideoJob):
        self.conn.execute("""
            UPDATE jobs SET status='running', attempts=attempts + 1, error=NULL,
            updated_at=? WHERE source=?
        """, (time.time(), str(job.relative)))
        self.conn.commit()

    def mark_result(self, job: VideoJob, success: bool, duration: float, error: str | None = None):
        self.conn.execute("""
            UPDATE jobs SET status=?, duration=?, error=?, updated_at=? WHERE source=?
        """, ("success" if success else "failed", duration, error, time.time(), str(job.relative)))
        self.conn.commit()


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def is_valid_output(path: Path) -> bool:
    """确认输出不是中断时遗留的空/损坏容器。"""
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0 and "video" in result.stdout


def discover_jobs(source_dir: Path, target_dir: Path) -> list[VideoJob]:
    """递归扫描源目录，并在目标目录位于源目录内部时自动排除它。"""
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    excluded = None
    try:
        excluded = target_dir.relative_to(source_dir)
    except ValueError:
        pass

    jobs = []
    for path in sorted(source_dir.rglob("*")):
        relative = path.relative_to(source_dir)
        if excluded is not None and (relative == excluded or excluded in relative.parents):
            continue
        if not is_video(path):
            continue
        # 本项目的算法固定以 masked_ 前缀命名结果，避免历史输出被再次处理。
        if path.name.startswith("masked_"):
            continue
        stat = path.stat()
        destination = target_dir / relative.parent / f"masked_{path.stem}.mp4"
        jobs.append(VideoJob(path, relative, destination, stat.st_size, stat.st_mtime_ns))
    return jobs


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def run_job(job: VideoJob, algorithm: Path, extra_args: Iterable[str]) -> tuple[bool, float, str | None]:
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(algorithm), str(job.source),
           "--out-dir", str(job.destination.parent), *extra_args]
    started = time.monotonic()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"    │ {line}", flush=True)
            tail.append(line)
            tail = tail[-8:]
    code = process.wait()
    duration = time.monotonic() - started
    if code == 0 and is_valid_output(job.destination):
        return True, duration, None
    details = " | ".join(tail) or f"子进程退出码 {code}"
    return False, duration, details[-1200:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视频打码批量任务调度器（可恢复、可见进度）")
    parser.add_argument("source_dir", type=Path,
                        help="待处理视频源目录（递归扫描，自动跳过 masked_* 结果）")
    parser.add_argument("target_dir", type=Path, help="输出目录；保存结果和 SQLite 状态")
    parser.add_argument("--algorithm", type=Path,
                        default=Path(__file__).resolve().parents[1] / "video_mask_batch_flast.py",
                        help="算法脚本路径")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="SQLite 状态文件（默认：目标目录/.video_mask_jobs.sqlite3）")
    parser.add_argument("--retry-failed", action="store_true", help="重新处理上次失败的任务")
    parser.add_argument("--force", action="store_true", help="即使已有有效输出也重新处理")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="透传给算法脚本的一个参数；重复使用，例如 --extra-arg=--card-conf")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_dir = args.source_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    algorithm = args.algorithm.expanduser().resolve()
    if not source_dir.is_dir():
        print(f"[错误] 源目录不存在：{source_dir}", file=sys.stderr)
        return 2
    if not algorithm.is_file():
        print(f"[错误] 算法脚本不存在：{algorithm}", file=sys.stderr)
        return 2
    if not shutil_which("ffprobe"):
        print("[错误] 未找到 ffprobe，请安装 ffmpeg。", file=sys.stderr)
        return 2

    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / ".video_mask_scheduler.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[错误] 已有调度器在使用目标目录：{target_dir}", file=sys.stderr)
            return 2

        state_path = args.state_file or target_dir / ".video_mask_jobs.sqlite3"
        store = JobStore(state_path)
        try:
            jobs = discover_jobs(source_dir, target_dir)
            planned: list[VideoJob] = []
            skipped = failed = 0
            for job in jobs:
                status = store.prepare(job, is_valid_output(job.destination), args.force)
                if status == "pending" or (status == "failed" and args.retry_failed):
                    planned.append(job)
                elif status == "success":
                    skipped += 1
                else:
                    failed += 1

            print(f"[调度] 扫描={len(jobs)} 待处理={len(planned)} 已跳过={skipped} "
                  f"历史失败={failed} 状态库={state_path}", flush=True)
            if args.dry_run:
                for job in planned:
                    print(f"  [计划] {job.relative} -> {job.destination.relative_to(target_dir)}")
                return 0

            succeeded = newly_failed = 0
            completed_seconds = []
            all_started = time.monotonic()
            for position, job in enumerate(planned, 1):
                average = sum(completed_seconds) / len(completed_seconds) if completed_seconds else 0
                eta = average * (len(planned) - position + 1)
                print(f"\n[{position}/{len(planned)}] 开始 {job.relative} "
                      f"ETA={format_duration(eta)}", flush=True)
                store.mark_running(job)
                try:
                    success, duration, error = run_job(job, algorithm, args.extra_arg)
                except KeyboardInterrupt:
                    print("\n[停止] 收到中断；下次运行会自动恢复未完成任务。", flush=True)
                    return 130
                store.mark_result(job, success, duration, error)
                completed_seconds.append(duration)
                if success:
                    succeeded += 1
                    print(f"[完成 {position}/{len(planned)}] {job.relative} "
                          f"耗时={format_duration(duration)}", flush=True)
                else:
                    newly_failed += 1
                    print(f"[失败 {position}/{len(planned)}] {job.relative} "
                          f"耗时={format_duration(duration)} 原因={error}", flush=True)

            elapsed = time.monotonic() - all_started
            print(f"\n[汇总] 成功={succeeded} 新失败={newly_failed} 跳过={skipped} "
                  f"总耗时={format_duration(elapsed)}", flush=True)
            return 0 if newly_failed == 0 else 1
        finally:
            store.close()


def shutil_which(command: str) -> str | None:
    # 避免额外依赖；函数独立便于在最小服务器镜像运行。
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
