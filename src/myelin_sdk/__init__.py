from importlib.metadata import version as _pkg_version

from .errors import MyelinAPIError
from .redact import RedactionConfig
from .session import MyelinSession
from .types import (
    CaptureResponse,
    FeedbackResponse,
    FinishResponse,
    SearchMatch,
    SearchResult,
    StartResult,
    SyncFileResult,
    SyncResult,
)

__version__ = _pkg_version("myelin-sdk")

__all__ = [
    "MyelinAPIError",
    "MyelinSession",
    "RedactionConfig",
    "CaptureResponse",
    "FeedbackResponse",
    "FinishResponse",
    "SearchMatch",
    "SearchResult",
    "StartResult",
    "SyncFileResult",
    "SyncResult",
    "__version__",
]
