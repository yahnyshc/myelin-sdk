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


