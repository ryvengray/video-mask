做一个本机数据库备份：
```
mkdir -p ~/data/controller-snapshots
python3 - <<'PY'
import sqlite3
from pathlib import Path
from datetime import datetime

source = Path("/var/lib/video-mask-controller/controller.sqlite3")
target = Path.home() / "data/controller-snapshots" / f"before-output-key-migration-{datetime.now():%Y%m%d-%H%M%S}.sqlite3"
with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
    src.backup(dst)
print(target)
PY
```

将失败和取消任务重新入队。若数量不多，也可启动 Controller 后在页面逐个 Restart；批量处理则执行：
```
python3 - <<'PY'
import sqlite3
import time

DB = "/var/lib/video-mask-controller/controller.sqlite3"
stamp = time.time()

with sqlite3.connect(DB) as conn:
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("""
        DELETE FROM task_logs
        WHERE task_id IN (
            SELECT task_id FROM tasks WHERE status IN ('failed', 'cancelled')
        )
    """)
    result = conn.execute("""
        UPDATE tasks
        SET status='pending', attempt_count=0, assigned_worker_id=NULL,
            progress_json='{}', output_sha256=NULL, output_duration_seconds=NULL,
            error_message=NULL, started_at=NULL, restarted_at=?,
            finished_at=NULL, face_review_owner=NULL, face_review_lease_until=NULL,
            updated_at=?
        WHERE status IN ('failed', 'cancelled')
    """, (stamp, stamp))
    conn.commit()

print(f"已重新入队 {result.rowcount} 个 failed/cancelled 任务。")
PY
```
