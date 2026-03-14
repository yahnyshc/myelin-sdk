"""Pydantic response models for the Myelin REST API."""

from pydantic import BaseModel


class SearchMatch(BaseModel):
    workflow_id: str
    title: str
    description: str
    content: str | None = None
    score: float | None = None


class SearchResult(BaseModel):
    top_match: SearchMatch | None = None
    other_matches: list[SearchMatch] = []


class StartResult(BaseModel):
    session_id: str
    matched_workflow_id: str | None = None


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
