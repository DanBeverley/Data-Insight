"""Research State Manager - Persistent state for pause/resume functionality."""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

RESEARCH_SESSIONS_DIR = Path("data/research_sessions")


@dataclass
class ResearchState:
    session_id: str
    original_query: str
    start_time: float
    last_updated: float
    time_budget_seconds: int
    elapsed_at_pause: float
    status: str  # "active", "paused", "completed", "interrupted"
    findings: List[Dict[str, Any]] = field(default_factory=list)
    explored_subtopics: List[str] = field(default_factory=list)
    pending_subtopics: List[str] = field(default_factory=list)
    visited_urls: List[str] = field(default_factory=list)
    iterations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchState":
        return cls(**data)

    def time_remaining(self) -> float:
        return max(0, self.time_budget_seconds - self.elapsed_at_pause)


class ResearchStateManager:
    def __init__(self):
        RESEARCH_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        return RESEARCH_SESSIONS_DIR / f"{session_id}.json"

    def save(self, state: ResearchState) -> bool:
        state.last_updated = time.time()
        try:
            path = self._get_path(state.session_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.info(f"[RESEARCH_STATE] Saved state for {state.session_id}: {state.status}")
            return True
        except Exception as e:
            logger.error(f"[RESEARCH_STATE] Failed to save: {e}")
            return False

    def load(self, session_id: str) -> Optional[ResearchState]:
        path = self._get_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ResearchState.from_dict(data)
        except Exception as e:
            logger.error(f"[RESEARCH_STATE] Failed to load {session_id}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        path = self._get_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"[RESEARCH_STATE] Deleted state for {session_id}")
            return True
        return False

    def list_sessions(self, status_filter: Optional[str] = None) -> List[ResearchState]:
        sessions = []
        for path in RESEARCH_SESSIONS_DIR.glob("*.json"):
            state = self.load(path.stem)
            if state:
                if status_filter is None or state.status == status_filter:
                    sessions.append(state)
        return sessions

    def get_interrupted_sessions(self) -> List[ResearchState]:
        return self.list_sessions("interrupted") + self.list_sessions("active")

    def mark_interrupted_on_startup(self):
        for path in RESEARCH_SESSIONS_DIR.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "active":
                    data["status"] = "interrupted"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"[RESEARCH_STATE] Marked {path.stem} as interrupted")
            except Exception:
                pass
