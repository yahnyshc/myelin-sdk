"""Pydantic response models for the Myelin REST API."""

from pydantic import BaseModel


class WorkflowInfo(BaseModel):
    id: str
    description: str
    total_steps: int
    overview: str
    skeleton: str
    steps: list[str] = []


class RecallResponse(BaseModel):
    session_id: str
    matched: bool
    workflow: WorkflowInfo | None = None


class CaptureResponse(BaseModel):
    status: str


class DebriefResponse(BaseModel):
    session_id: str
    tool_calls_recorded: int
    status: str
    workflow_id: str | None = None
    warning: str | None = None


class HintResponse(BaseModel):
    session_id: str
    step_number: int
    detail: str
