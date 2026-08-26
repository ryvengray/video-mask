"""Video Audit - Independent face detection review tool."""

import json
import logging
import os
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import Database

APP_DIR = Path(__file__).parent
DB = Database()
CONTROLLER_URL = os.environ.get("CONTROLLER_URL", "http://localhost:8000")
CONTROLLER_TOKEN = os.environ.get("CONTROLLER_TOKEN", "")
CONTROLLER_AUTH_USER = os.environ.get("CONTROLLER_AUTH_USER", "")
CONTROLLER_AUTH_PASS = os.environ.get("CONTROLLER_AUTH_PASS", "")
logger = logging.getLogger(__name__)


def _controller_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    token = CONTROLLER_TOKEN.strip()
    if token:
        authorization = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        # HTTP Basic authentication and Controller admin authentication both
        # use the Authorization header.  When the Controller is behind an
        # auth_basic Nginx proxy, carry the Bearer value separately; Nginx
        # restores it only for the upstream Controller request.
        header_name = (
            "X-Video-Mask-Authorization"
            if CONTROLLER_AUTH_USER and CONTROLLER_AUTH_PASS
            else "Authorization"
        )
        headers[header_name] = authorization
    return headers


def _controller_auth() -> Optional[httpx.BasicAuth]:
    if CONTROLLER_AUTH_USER and CONTROLLER_AUTH_PASS:
        return httpx.BasicAuth(CONTROLLER_AUTH_USER, CONTROLLER_AUTH_PASS)
    return None

app = FastAPI(title="Video Audit")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/projects")
async def create_project(data: dict):
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name required")
    return DB.create_project(name)


@app.get("/api/projects")
async def list_projects():
    return DB.get_projects()


@app.post("/api/projects/{project_id}/tasks")
async def add_tasks(project_id: int, data: dict):
    project = DB.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    task_ids = data.get("task_ids", [])
    if not task_ids:
        raise HTTPException(status_code=400, detail="No task IDs provided")
    return DB.create_tasks(project_id, task_ids)


@app.get("/api/projects/{project_id}/tasks")
async def list_tasks(project_id: int):
    project = DB.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return DB.get_tasks(project_id)


@app.post("/api/tasks/{task_id}/sync")
async def sync_task(task_id: int):
    task = DB.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    controller_task_id = task["controller_task_id"]
    auth = _controller_auth()
    task_info: dict = {}
    DB.update_task(task_id, status="syncing")

    # The metadata endpoint is an admin endpoint and requires the Controller
    # Bearer token.  The playback endpoint intentionally does not: Nginx
    # Basic Auth is sufficient for a video-audit deployment that only needs to
    # retrieve video bytes.  Do not make an optional metadata lookup prevent a
    # download that the playback endpoint permits.
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CONTROLLER_URL}/api/tasks/{controller_task_id}",
                headers=_controller_headers(),
                auth=auth,
            )
            if response.status_code == 200:
                task_info = response.json()
            else:
                logger.info(
                    "Controller metadata for %s was unavailable (%s); continuing with playback",
                    controller_task_id, response.status_code,
                )
    except httpx.HTTPError as exc:
        logger.info("Controller metadata lookup for %s failed: %s", controller_task_id, exc)

    local_path = APP_DIR / "videos" / f"{controller_task_id}.mp4"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_suffix(local_path.suffix + ".part")
    last_status: int | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=3600.0)) as client:
            # Processed output is preferred for review.  A source-only or
            # still-running task transparently falls back to its input video.
            for file_kind in ("output", "input"):
                url = f"{CONTROLLER_URL}/api/tasks/{controller_task_id}/play?file={file_kind}"
                async with client.stream("GET", url, headers=_controller_headers(), auth=auth) as response:
                    last_status = response.status_code
                    if not 200 <= response.status_code < 300:
                        if response.status_code in {401, 403}:
                            break
                        continue
                    with temporary_path.open("wb") as destination:
                        async for chunk in response.aiter_bytes(64 * 1024):
                            destination.write(chunk)
                    temporary_path.replace(local_path)
                    info = dict(task_info)
                    info["downloaded_file"] = file_kind
                    DB.update_task(
                        task_id,
                        status="ready",
                        local_video_path=str(local_path),
                        controller_info=json.dumps(info),
                    )
                    return DB.get_task(task_id)
    except (httpx.HTTPError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        DB.update_task(task_id, status="error")
        raise HTTPException(status_code=502, detail=f"Video download failed: {exc}") from exc

    temporary_path.unlink(missing_ok=True)
    DB.update_task(task_id, status="error")
    if last_status in {401, 403}:
        raise HTTPException(
            status_code=502,
            detail=("Controller playback rejected the configured credentials. "
                    "Set CONTROLLER_AUTH_USER and CONTROLLER_AUTH_PASS for Nginx Basic Auth."),
        )
    raise HTTPException(status_code=404, detail="No output or source video is available for this task")


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: int):
    task = DB.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/api/tasks/{task_id}/reviews")
async def create_review(task_id: int, data: dict):
    task = DB.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    review_type = data.get("review_type")
    frame_number = data.get("frame_number")
    timestamp = data.get("timestamp")
    note = data.get("note")
    if review_type not in ("missed", "false_positive"):
        raise HTTPException(status_code=400, detail="Invalid review type")
    screenshot_path = data.get("screenshot_path")
    return DB.create_review(task_id, frame_number, timestamp, review_type, note, screenshot_path)


@app.get("/api/tasks/{task_id}/reviews")
async def list_reviews(task_id: int):
    task = DB.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return DB.get_reviews(task_id)


@app.get("/api/tasks/{task_id}/video")
async def serve_video(task_id: int):
    task = DB.get_task(task_id)
    if not task or not task.get("local_video_path"):
        raise HTTPException(status_code=404, detail="Video not available")
    return FileResponse(task["local_video_path"])


@app.post("/api/tasks/{task_id}/screenshot")
async def upload_screenshot(task_id: int, screenshot: UploadFile = File(...)):
    task = DB.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    screenshot_dir = APP_DIR / "screenshots" / str(task_id)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    path = screenshot_dir / filename
    content = await screenshot.read()
    path.write_bytes(content)
    return {"path": str(path.relative_to(APP_DIR))}


@app.get("/api/stats")
async def get_stats():
    return DB.get_stats()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
