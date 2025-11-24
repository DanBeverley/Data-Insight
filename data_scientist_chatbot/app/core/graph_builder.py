"""Graph construction and workflow orchestration"""

from langgraph.graph import StateGraph, END
from typing import Optional

try:
    from .logger import logger
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger

try:
    from ..core.router import (
        route_from_router,
        route_from_brain,
        route_after_agent,
        should_continue,
        route_after_action,
    )
except ImportError:
    from core.router import route_from_router, route_from_brain, route_after_agent, should_continue, route_after_action

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    CHECKPOINTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Checkpoint import failed: {e}. Install aiosqlite: pip install aiosqlite")
    AsyncSqliteSaver = None
    CHECKPOINTER_AVAILABLE = False

DB_PATH = "context.db"
_checkpointer: Optional[AsyncSqliteSaver] = None


def set_checkpointer(checkpointer: AsyncSqliteSaver) -> None:
    global _checkpointer
    _checkpointer = checkpointer


def get_checkpointer() -> Optional[AsyncSqliteSaver]:
    return _checkpointer


async def perform_checkpoint_maintenance(force_cleanup=False):
    """Clean up old checkpoints to prevent database bloat"""
    try:
        if not _checkpointer:
            return {"status": "unavailable", "reason": "No checkpoint storage available"}

        import aiosqlite

        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM checkpoints")
            row = await cursor.fetchone()
            count = row[0] if row else 0

            if force_cleanup or count > 1000:
                await db.execute(
                    "DELETE FROM checkpoints WHERE rowid NOT IN (SELECT rowid FROM checkpoints ORDER BY rowid DESC LIMIT 500)"
                )
                await db.commit()
                return {
                    "status": "degraded",
                    "operations": [f"cleaned {count - 500} old checkpoints"],
                    "stats": {"remaining": 500},
                }

            return {"status": "healthy", "stats": {"checkpoint_count": count}}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_agent_executor(memory=None):
    """Creates the multi-agent system with Brain and Hands specialization"""
    logger.info("[GRAPH] Building agent executor graph...")

    from ..agent import (
        AgentState,
        run_router_agent,
        run_brain_agent,
        run_hands_agent,
        parse_tool_calls,
        execute_tools_node,
    )

    graph = StateGraph(AgentState)
    graph.add_node("router", run_router_agent)
    graph.add_node("brain", run_brain_agent)
    graph.add_node("hands", run_hands_agent)
    graph.add_node("parser", parse_tool_calls)
    graph.add_node("action", execute_tools_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_from_router, {"brain": "brain", "hands": "hands"})
    graph.add_conditional_edges("brain", route_from_brain, {"hands": "hands", "parser": "parser", END: END})
    graph.add_conditional_edges("hands", route_after_agent, {"parser": "parser", "brain": "brain", END: END})
    graph.add_conditional_edges("parser", should_continue, {"action": "action", "hands": "hands", END: END})
    graph.add_conditional_edges("action", route_after_action, {"brain": "brain", END: END})

    checkpointer = memory if memory else _checkpointer
    if CHECKPOINTER_AVAILABLE and checkpointer:
        logger.info("[GRAPH] Compiling graph WITH checkpointer")
        agent_executor = graph.compile(checkpointer=checkpointer)
    else:
        logger.info("[GRAPH] Compiling graph WITHOUT checkpointer")
        agent_executor = graph.compile()

    logger.info("[GRAPH] Graph compilation complete")
    return agent_executor


def create_enhanced_agent_executor(session_id: str = None):
    return create_agent_executor(memory=_checkpointer)
