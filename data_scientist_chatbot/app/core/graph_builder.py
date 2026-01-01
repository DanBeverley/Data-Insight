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
    from .state import GlobalState

    AgentState = GlobalState
except ImportError:
    from data_scientist_chatbot.app.core.state import GlobalState

    AgentState = GlobalState

from ..agent import (
    run_brain_agent,
    run_hands_agent,
    run_analyst_node,
    run_architect_node,
    run_presenter_node,
    parse_tool_calls,
    execute_tools_node,
    run_verifier_agent,
)

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    CHECKPOINTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Checkpoint import failed: {e}. Install aiosqlite: pip install aiosqlite")
    AsyncSqliteSaver = None
    CHECKPOINTER_AVAILABLE = False

DB_PATH = "data/databases/checkpoints.db"
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


def route_from_brain(state: AgentState):
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "parser"
    return END


def route_after_action(state: AgentState) -> str:
    """Route after tool execution"""
    stage = state.get("workflow_stage")
    if stage == "reporting":
        return "analyst"
    if stage == "delegating":
        return "hands"
    return "brain"


def should_continue(state: AgentState):
    # Parser logic
    return "action"


def create_agent_executor(memory=None):
    logger.info("[GRAPH] Building agent executor graph...")

    graph = StateGraph(AgentState)

    # Add Nodes
    graph.add_node("brain", run_brain_agent)
    graph.add_node("hands", run_hands_agent)
    graph.add_node("verifier", run_verifier_agent)

    # Tool Execution Nodes
    graph.add_node("parser", parse_tool_calls)
    graph.add_node("action", execute_tools_node)

    # Newsroom Nodes
    graph.add_node("analyst", run_analyst_node)
    graph.add_node("architect", run_architect_node)
    graph.add_node("presenter", run_presenter_node)

    # Set Entry Point
    graph.set_entry_point("brain")

    # Edges

    # Brain -> Tools, Hands, or End
    graph.add_conditional_edges("brain", route_from_brain, {"parser": "parser", END: END})

    # Tool Execution Loop
    graph.add_edge("parser", "action")

    # Action Routing
    graph.add_conditional_edges(
        "action", route_after_action, {"brain": "brain", "analyst": "analyst", "hands": "hands"}
    )

    # Hands -> Verifier (QA Layer)
    graph.add_edge("hands", "verifier")

    def route_from_verifier(state: AgentState):
        """Route based on verification result"""
        plan = state.get("plan", [])
        current_index = state.get("current_task_index", 0)

        if plan and current_index < len(plan):
            next_task = plan[current_index]
            if next_task.get("status") == "pending":
                retry_count = state.get("retry_count", 0)
                logger.info(f"[GRAPH] Verifier rejected. Routing to HANDS for correction (Attempt {retry_count + 1}).")
                return "hands"

        logger.info("[GRAPH] Verification successful. Returning to Brain.")
        return "brain"

    # Verifier -> Conditional Routing
    graph.add_conditional_edges("verifier", route_from_verifier, {"hands": "hands", "brain": "brain"})

    # Newsroom Pipeline
    graph.add_edge("analyst", "architect")
    graph.add_edge("architect", "presenter")
    graph.add_edge("presenter", END)

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
