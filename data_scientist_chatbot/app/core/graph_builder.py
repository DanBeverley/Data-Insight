"""Graph construction and workflow orchestration"""
import sqlite3
from langgraph.graph import StateGraph, END
try:
    from ..core.router import (
        route_from_router, route_from_brain, route_after_agent,
        should_continue, route_after_action)
except ImportError:
    from core.router import (
        route_from_router, route_from_brain, route_after_agent,
        should_continue, route_after_action)
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    CHECKPOINTER_AVAILABLE = True
    db_path = "context.db"
    memory = SqliteSaver(conn=sqlite3.connect(db_path, check_same_thread=False))
    #memory = None
except ImportError:
    SqliteSaver = None
    memory = None
    CHECKPOINTER_AVAILABLE = False

def create_agent_executor(memory=None):
    """Creates the multi-agent system with Brain and Hands specialization"""

    from ..agent import (
        AgentState, run_router_agent, run_brain_agent, run_hands_agent,
        parse_tool_calls, execute_tools_node
    )
    graph = StateGraph(AgentState)
    graph.add_node("router", run_router_agent)
    graph.add_node("brain", run_brain_agent)
    graph.add_node("hands", run_hands_agent)
    graph.add_node("parser", parse_tool_calls)
    graph.add_node("action", execute_tools_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {
            "brain": "brain",
            "hands": "hands"
        }
    )
    graph.add_conditional_edges(
        "brain",
        route_from_brain,
        {
            "hands": "hands",
            "parser": "parser",
            END: END
        }
    )
    graph.add_conditional_edges(
        "hands",
        route_after_agent,
        {
            "parser": "parser",
            "brain": "brain",
            END: END
        }
    )
    graph.add_conditional_edges(
        "parser",
        should_continue,
        {
            "action": "action",
            "hands": "hands",
            END: END
        }
    )
    graph.add_conditional_edges(
        "action",
        route_after_action,
        {
            "brain": "brain",
            END: END
        }
    )

    if CHECKPOINTER_AVAILABLE and memory:
        agent_executor = graph.compile(checkpointer=memory)
    else:
        agent_executor = graph.compile()
    return agent_executor

def create_enhanced_agent_executor(session_id: str = None):
    return create_agent_executor(memory=memory)