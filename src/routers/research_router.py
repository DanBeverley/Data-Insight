"""Research State API Router - Endpoints for pause/resume functionality."""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

router = APIRouter(prefix="/api/research", tags=["research"])
logger = logging.getLogger(__name__)


@router.get("/{session_id}/state")
async def get_research_state(session_id: str):
    from data_scientist_chatbot.app.state.research_state_manager import ResearchStateManager

    manager = ResearchStateManager()
    state = manager.load(session_id)

    if not state:
        return {"has_state": False}

    return {
        "has_state": True,
        "status": state.status,
        "query": state.original_query,
        "findings_count": len(state.findings),
        "elapsed_seconds": int(state.elapsed_at_pause),
        "time_remaining": int(state.time_remaining()),
        "iterations": state.iterations,
    }


@router.post("/{session_id}/resume")
async def resume_research(session_id: str):
    from data_scientist_chatbot.app.agents.research_brain import ResearchBrain
    from data_scientist_chatbot.app.state.research_state_manager import ResearchStateManager

    manager = ResearchStateManager()
    state = manager.load(session_id)

    if not state or state.status not in ("paused", "interrupted"):
        raise HTTPException(status_code=404, detail="No paused research to resume")

    return {
        "status": "ready",
        "query": state.original_query,
        "findings_count": len(state.findings),
        "time_remaining": int(state.time_remaining()),
    }


@router.delete("/{session_id}/state")
async def discard_research(session_id: str):
    from data_scientist_chatbot.app.state.research_state_manager import ResearchStateManager

    manager = ResearchStateManager()
    success = manager.delete(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="No research state found")

    return {"status": "discarded", "session_id": session_id}


@router.get("/interrupted")
async def get_interrupted_sessions():
    from data_scientist_chatbot.app.state.research_state_manager import ResearchStateManager

    manager = ResearchStateManager()
    sessions = manager.get_interrupted_sessions()

    return {
        "count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "query": s.original_query,
                "status": s.status,
                "findings_count": len(s.findings),
            }
            for s in sessions
        ],
    }
