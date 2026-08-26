from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class Project(BaseModel):
    id: int
    name: str
    created_at: float


class TaskCreate(BaseModel):
    task_ids: list[str]


class Task(BaseModel):
    id: int
    project_id: int
    task_id: str
    controller_task_id: str
    status: str  # pending, syncing, ready, error
    local_video_path: Optional[str] = None
    controller_info: Optional[str] = None
    created_at: float
    updated_at: float


class ReviewCreate(BaseModel):
    frame_number: int
    timestamp: float
    review_type: str  # missed, false_positive
    note: Optional[str] = None


class Review(BaseModel):
    id: int
    task_id: int
    frame_number: int
    timestamp: float
    review_type: str
    note: Optional[str] = None
    screenshot_path: Optional[str] = None
    created_at: float
