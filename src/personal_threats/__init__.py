# SPDX-License-Identifier: Apache-2.0
"""Local personal threat analysis: persistent input infrastructure."""

from .core import ContractError, Message, text_windows, utf16_span
from .store import HistoryStore, StaleJob, StoreError, WriterBusy
from .results import Fragment, PersonReference, Prediction, Scores, ThreatEvent
from .service import (
    AnalysisReader,
    AnalysisResult,
    AnalysisSettings,
    ThreatAnalyzer,
)

__version__ = "0.3.0"
__all__ = [
    "ContractError",
    "HistoryStore",
    "Message",
    "StaleJob",
    "StoreError",
    "WriterBusy",
    "text_windows",
    "utf16_span",
    "Fragment",
    "PersonReference",
    "Prediction",
    "Scores",
    "ThreatEvent",
    "AnalysisReader",
    "AnalysisResult",
    "AnalysisSettings",
    "ThreatAnalyzer",
]
