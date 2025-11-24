"""Routing logic for agent workflow"""

import sys
import os
from typing import Dict, Any, Literal
from langgraph.graph import END
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import traceable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.helpers import get_last_message, get_message_content, has_tool_calls
from core.logger import logger

TOOL_ROUTING_MAP = {
    "delegate_coding_task": "hands",
    "python_code_interpreter": "action",
    "knowledge_graph_query": "action",
    "access_learning_data": "action",
    "retrieve_historical_patterns": "action",
}


@traceable(name="router_decision", tags=["routing", "core"])
def route_to_agent(state: Dict[str, Any]) -> str:
    last_message = get_last_message(state)
    if not last_message or isinstance(last_message, ToolMessage):
        return "brain"
    if has_tool_calls(last_message):
        tool_call = last_message.tool_calls[0]
        if tool_call.get("name") == "delegate_coding_task":
            return "hands"
    return "brain"


def should_continue(state: Dict[str, Any]) -> str:
    messages = state.get("messages", [])
    if messages is None or len(messages) == 0:
        return END

    last_message = messages[-1]
    python_executions = state.get("python_executions", 0)

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if len(messages) >= 5:
            last_tool_call = last_message.tool_calls[0]
            consecutive_identical_calls = 0
            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]
                if hasattr(msg, "type") and msg.type == "human":
                    break
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    prev_tool_call = msg.tool_calls[0]
                    if last_tool_call.get("name") == prev_tool_call.get("name") and last_tool_call.get(
                        "args"
                    ) == prev_tool_call.get("args"):
                        consecutive_identical_calls += 1
                    break
            if consecutive_identical_calls >= 2:
                logger.warning(
                    f"Termination condition met: Detected recursive tool call for '{last_tool_call.get('name')}'"
                )
                return END
        tool_name = last_message.tool_calls[0].get("name", "")
        destination = TOOL_ROUTING_MAP.get(tool_name, "action")
        logger.debug(f"Tool '{tool_name}' routing to: {destination}")
        return destination
    return END


def route_after_agent(state: Dict[str, Any]) -> str:
    """Route based on whether agent wants to use tools or just respond"""
    logger.info("=" * 80)
    logger.info("[ROUTING] ENTERED route_after_agent function")
    logger.info("=" * 80)

    last_message = get_last_message(state)
    content = get_message_content(last_message)
    current_agent = state.get("current_agent", "")

    logger.debug(f"Routing after {current_agent} agent")
    logger.info(f"[ROUTING] Current agent: {current_agent}")
    logger.debug(f"Last message type: {type(last_message).__name__}")
    logger.info(f"[ROUTING] Last message type: {type(last_message).__name__}")
    logger.debug(f"Has tool_calls? {has_tool_calls(last_message)}")
    logger.info(f"[ROUTING] Has tool_calls: {has_tool_calls(last_message)}")
    logger.debug(f"Has content? {bool(content)}")
    logger.info(f"[ROUTING] Has content: {bool(content)}")

    if current_agent == "hands":
        if content or last_message:
            logger.debug("Hands completed execution, routing to brain for interpretation")
            logger.info("[ROUTING] Hands completed, routing to BRAIN")
            logger.info("=" * 80)
            return "brain"

    if has_tool_calls(last_message):
        logger.debug("Routing to parser (tool_calls)")
        logger.info("[ROUTING] Routing to PARSER (tool_calls)")
        logger.info("=" * 80)
        return "parser"

    if (content.startswith("{") and '"name":' in content) or (
        "python_code_interpreter" in content and '"code":' in content
    ):
        logger.debug("Routing to parser (JSON format)")
        logger.info("[ROUTING] Routing to PARSER (JSON format)")
        logger.info("=" * 80)
        return "parser"

    logger.debug("Routing to END")
    logger.info("[ROUTING] Routing to END")
    logger.info("=" * 80)
    return END


def route_from_router(state: Dict[str, Any]) -> str:
    """Route based on router's binary decision"""
    decision = state.get("router_decision", "brain")
    logger.debug(f"Router routing to: {decision}")
    return decision


def route_from_brain(state: Dict[str, Any]) -> str:
    retry_count = state.get("retry_count") or 0
    last_sequence = state.get("last_agent_sequence", [])

    logger.debug(f"route_from_brain - retry_count: {retry_count}, sequence: {last_sequence}")

    if retry_count >= 3:
        logger.warning(f"Max retries reached ({retry_count}), terminating")
        return END

    if len(last_sequence) >= 4:
        recent_sequence = last_sequence[-4:]
        if recent_sequence == ["brain", "hands", "brain", "hands"]:
            logger.warning("Detected brain->hands->brain->hands loop, terminating")
            return END

    last_message = state["messages"][-1] if state["messages"] else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "parser"

    return END


def route_after_action(state: Dict[str, Any]) -> str:
    last_message = get_last_message(state)
    if last_message:
        content = get_message_content(last_message)

        if content and content.strip():
            logger.debug("Technical output completed, routing to brain for interpretation")
            return "brain"

    logger.debug("No output from action, ending flow")
    return END


def subgraph_should_continue(state: Dict[str, Any]) -> str:
    """Check if subgraph should continue executing"""
    messages = state.get("messages", [])
    if messages is None or len(messages) == 0:
        return END
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"
    return END


def subgraph_route_after_action(state: Dict[str, Any]) -> str:
    """Route after action in subgraph"""
    return "summarize"
