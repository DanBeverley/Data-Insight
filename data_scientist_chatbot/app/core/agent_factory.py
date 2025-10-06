"""Agent creation and subgraph factory functions"""
import json
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

try:
    from ..core.model_manager import ModelManager
    from ..tools.executor import execute_tool
    from ..tools.parsers import parse_message_to_tool_call
    from ..utils.sanitizers import sanitize_output
    from ..core.router import subgraph_should_continue, subgraph_route_after_action
except ImportError:
    from core.model_manager import ModelManager
    from tools.executor import execute_tool
    from tools.parsers import parse_message_to_tool_call
    from utils.sanitizers import sanitize_output
    from core.router import subgraph_should_continue, subgraph_route_after_action

model_manager = ModelManager()


def create_router_agent():
    """Create Router agent for fast binary routing decisions"""
    config = model_manager.get_ollama_config('router')
    return ChatOllama(**config)


def create_brain_agent():
    """Create Brain agent for business reasoning and planning"""
    config = model_manager.get_ollama_config('brain')
    return ChatOllama(**config)


def create_hands_agent():
    """Create Hands agent for code execution"""
    config = model_manager.get_ollama_config('hands')
    return ChatOllama(**config)


def create_status_agent():
    """Create Status agent for real-time progress updates"""
    config = model_manager.get_ollama_config('status')
    return ChatOllama(**config)


def create_hands_subgraph():
    """Create isolated hands subgraph for delegate_coding_task execution"""

    def run_hands_subgraph_agent(state):
        """Hands agent for subgraph execution"""
        session_id = state.get("session_id")
        data_context = state.get("data_context", "")

        llm = create_hands_agent()
        try:
            from ..agent import get_hands_prompt
        except ImportError:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from agent import get_hands_prompt
        prompt = get_hands_prompt()
        agent_runnable = prompt | llm

        hands_state_with_context = state.copy()
        hands_state_with_context["data_context"] = data_context

        try:
            response = agent_runnable.invoke(hands_state_with_context)
            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Hands subgraph error: {e}")]}

    def parse_subgraph_tool_calls(state):
        """Parser for subgraph tool calls"""
        last_message = state["messages"][-1]
        parse_message_to_tool_call(last_message, "subgraph")
        return state

    def execute_subgraph_tools(state):
        """Execute tools within subgraph using shared executor"""
        last_message = state["messages"][-1]

        if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            logger.warning("Subgraph action: No tool_calls found")
            return {"messages": state["messages"]}

        tool_call = last_message.tool_calls[0]
        session_id = state.get("session_id")

        if isinstance(tool_call, dict):
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', {})
            tool_id = tool_call.get('id', f"call_{tool_name}")
            logger.info(f"Subgraph action: Extracted args from dict: {list(tool_args.keys())}")
        else:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id
            logger.info(f"Subgraph action: Extracted args from object: {list(tool_args.keys()) if isinstance(tool_args, dict) else type(tool_args)}")

        try:
            content = execute_tool(tool_name, tool_args, session_id)
            python_executions = state.get("python_executions", 0) + (1 if tool_name == 'python_code_interpreter' else 0)

            tool_response = ToolMessage(content=sanitize_output(content), tool_call_id=tool_id)
            return {
                "messages": state["messages"] + [tool_response],
                "python_executions": python_executions
            }
        except ValueError as e:
            logger.error(str(e))
            return {"messages": state["messages"]}

    def summarize_hands_result(state):
        """Summarize subgraph execution result"""
        messages = state.get("messages", [])
        last_tool_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                last_tool_msg = msg
                break
        if last_tool_msg:
            summary = last_tool_msg.content
        else:
            summary = "Task completed with no output."
        summary_message = AIMessage(content=summary)
        return {"messages": [summary_message]}

    try:
        from ..agent import HandsSubgraphState
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from agent import HandsSubgraphState
    subgraph = StateGraph(HandsSubgraphState)
    subgraph.add_node("hands", run_hands_subgraph_agent)
    subgraph.add_node("parser", parse_subgraph_tool_calls)
    subgraph.add_node("action", execute_subgraph_tools)
    subgraph.add_node("summarize", summarize_hands_result)
    subgraph.set_entry_point("hands")
    subgraph.add_conditional_edges(
        "hands",
        lambda state: "parser" if (
            state["messages"] and
            hasattr(state["messages"][-1], 'content') and
            ('python_code_interpreter' in str(state["messages"][-1].content) or
             (hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls))
        ) else END,
        {"parser": "parser", END: END}
    )
    subgraph.add_conditional_edges(
        "parser",
        subgraph_should_continue,
        {"action": "action", END: END}
    )
    subgraph.add_conditional_edges(
        "action",
        subgraph_route_after_action,
        {"summarize": "summarize"}
    )
    subgraph.set_finish_point("summarize")
    return subgraph.compile()