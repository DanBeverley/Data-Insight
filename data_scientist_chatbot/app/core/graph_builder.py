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

from .constants import NodeName, WorkflowStage

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
            return NodeName.PARSER.value
    return END


def route_after_action(state: AgentState) -> str:
    stage = state.get("workflow_stage")
    if stage == WorkflowStage.REPORTING.value:
        return NodeName.ANALYST.value
    if stage == WorkflowStage.DELEGATING.value:
        return NodeName.HANDS.value
    return NodeName.BRAIN.value


def should_continue(state: AgentState):
    return NodeName.ACTION.value


def create_agent_executor(memory=None):
    logger.info("[GRAPH] Building agent executor graph...")

    graph = StateGraph(AgentState)

    # Add Nodes
    graph.add_node(NodeName.BRAIN.value, run_brain_agent)
    graph.add_node(NodeName.HANDS.value, run_hands_agent)
    graph.add_node(NodeName.VERIFIER.value, run_verifier_agent)

    graph.add_node(NodeName.PARSER.value, parse_tool_calls)
    graph.add_node(NodeName.ACTION.value, execute_tools_node)

    graph.add_node(NodeName.ANALYST.value, run_analyst_node)
    graph.add_node(NodeName.ARCHITECT.value, run_architect_node)
    graph.add_node(NodeName.PRESENTER.value, run_presenter_node)

    graph.set_entry_point(NodeName.BRAIN.value)

    # Edges

    # Brain -> Tools, Hands, or End
    graph.add_conditional_edges(
        NodeName.BRAIN.value, route_from_brain, {NodeName.PARSER.value: NodeName.PARSER.value, END: END}
    )

    graph.add_edge(NodeName.PARSER.value, NodeName.ACTION.value)

    graph.add_conditional_edges(
        NodeName.ACTION.value,
        route_after_action,
        {
            NodeName.BRAIN.value: NodeName.BRAIN.value,
            NodeName.ANALYST.value: NodeName.ANALYST.value,
            NodeName.HANDS.value: NodeName.HANDS.value,
        },
    )

    graph.add_edge(NodeName.HANDS.value, NodeName.VERIFIER.value)

    def route_from_verifier(state: AgentState):
        plan = state.get("plan", [])
        current_index = state.get("current_task_index", 0)

        if plan and current_index < len(plan):
            next_task = plan[current_index]
            if next_task.get("status") == "pending":
                retry_count = state.get("retry_count", 0)
                logger.info(f"[GRAPH] Verifier rejected. Routing to HANDS for correction (Attempt {retry_count + 1}).")
                return NodeName.HANDS.value

        logger.info("[GRAPH] Verification successful. Returning to Brain.")
        return NodeName.BRAIN.value

    # Verifier -> Conditional Routing
    graph.add_conditional_edges(
        NodeName.VERIFIER.value,
        route_from_verifier,
        {NodeName.HANDS.value: NodeName.HANDS.value, NodeName.BRAIN.value: NodeName.BRAIN.value},
    )

    graph.add_edge(NodeName.ANALYST.value, NodeName.ARCHITECT.value)
    graph.add_edge(NodeName.ARCHITECT.value, NodeName.PRESENTER.value)
    graph.add_edge(NodeName.PRESENTER.value, END)

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
