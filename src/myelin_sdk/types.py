"""Pydantic response models for the Myelin REST API."""

from pydantic import BaseModel


class WorkflowInfo(BaseModel):
    id: str
    description: str
    overview: str
    content: str


class RecallResponse(BaseModel):
    session_id: str
    matched: bool
    workflow: WorkflowInfo | None = None


class ListWorkflowItem(BaseModel):
    workflow_id: str
    description: str
    sessions_passed: int = 0
    sessions_total: int = 0
    avg_reward: float | None = None


class ListWorkflowsResponse(BaseModel):
    workflows: list[ListWorkflowItem] = []
    count: int = 0


class CaptureResponse(BaseModel):
    status: str


class FeedbackResponse(BaseModel):
    session_id: str
    status: str


class FinishResponse(BaseModel):
    session_id: str
    tool_calls_recorded: int
    status: str
    workflow_id: str | None = None
    warning: str | None = None


class SyncFileResult(BaseModel):
    path: str
    status: str  # "created", "updated", "unchanged"
    workflow_id: str | None = None


class SyncResult(BaseModel):
    details: list[SyncFileResult] = []
    total: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0


