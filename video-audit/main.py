"""Video Audit - Independent face detection review tool."""

import asyncio
import json
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


def _controller_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    token = CONTROLLER_TOKEN.strip()
    if token:
        headers["Authorization"] = (
            token if token.lower().startswith("bearer ") else f"Bearer {token}"
        )
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
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CONTROLLER_URL}/api/tasks/{controller_task_id}",
                headers=_controller_headers(),
                auth=auth,
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch from controller")
            task_info = response.json()
    except Exception as exc:
        DB.update_task(task_id, status="error")
        raise HTTPException(status_code=500, detail=f"Controller fetch failed: {exc}")
    output_key = task_info.get("output_object_key") or task_info.get("source_object_key")
    if output_key:
        local_path = APP_DIR / "videos" / f"{controller_task_id}.mp4"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{CONTROLLER_URL}/api/tasks/{controller_task_id}/play?file=output",
                    headers=_controller_headers(),
                    auth=auth,
                )
                if response.status_code == 200:
                    local_path.write_bytes(response.content)
                    DB.update_task(
                        task_id,
                        status="ready",
                        local_video_path=str(local_path),
                        controller_info=json.dumps(task_info),
                    )
                else:
                    DB.update_task(task_id, status="error")
                    raise HTTPException(status_code=500, detail="Failed to download video")
        except Exception as exc:
            DB.update_task(task_id, status="error")
            raise HTTPException(status_code=500, detail=f"Video download failed: {exc}")
    else:
        DB.update_task(task_id, status="error")
        raise HTTPException(status_code=404, detail="No video available for this task")
    return DB.get_task(task_id)


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
