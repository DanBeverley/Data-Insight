"""Tool execution nodes and utility functions."""

from typing import Dict, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from data_scientist_chatbot.app.core.state import GlobalState
from data_scientist_chatbot.app.core.workflow_types import WorkflowStage
from data_scientist_chatbot.app.core.logger import logger
from data_scientist_chatbot.app.tools import execute_tool
from data_scientist_chatbot.app.utils.helpers import has_tool_calls
from data_scientist_chatbot.app.utils.text_processing import parse_message_to_tool_call, sanitize_output
from data_scientist_chatbot.app.core.exceptions import ToolExecutionError


def parse_tool_calls(state: GlobalState) -> Dict[str, Any]:
    """Parse tool calls from the last message."""
    last_message = state["messages"][-1]
    parse_message_to_tool_call(last_message, "json")
    return {"messages": [last_message]}


def execute_tools_node(state: GlobalState) -> Dict[str, Any]:
    """Execute tool calls from the last message."""
    messages = state.get("messages", [])
    logger.info(f"[ACTION] execute_tools_node received {len(messages)} messages")
    msg_types = [f"{type(m).__name__}" for m in messages]
    logger.info(f"[ACTION] Message types: {msg_types}")

    if not messages:
        logger.warning("[ACTION] No messages in state")
        return {"messages": []}

    last_message = messages[-1]

    if not has_tool_calls(last_message):
        logger.debug("execute_tools: No tool calls found")
        return {"messages": messages}

    tool_calls = last_message.tool_calls or []
    logger.info(
        f"[ACTION] Found {len(tool_calls)} tool calls: {[tc.get('name', getattr(tc, 'name', 'unknown')) if isinstance(tc, dict) else tc.name for tc in tool_calls]}"
    )

    tool_responses = []
    for tool_call in tool_calls:
        session_id = state.get("session_id")

        if isinstance(tool_call, dict):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", tool_call.get("arguments", {}))
            tool_id = tool_call.get("id", f"call_{tool_name}")
        else:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

        logger.debug(f"execute_tools: Processing {tool_name} with args: {tool_args}")

        if not session_id:
            content = "Error: Session ID is missing."
        else:
            try:
                if tool_name == "generate_comprehensive_report":
                    logger.info(f"[REPORT] Triggering Newsroom Pipeline. Args: {tool_args}")
                    return {
                        "messages": [
                            ToolMessage(
                                content="[REPORT] Data analysis and report generation pipeline started.",
                                tool_call_id=tool_id,
                            )
                        ],
                        "current_agent": "action",
                        "workflow_stage": WorkflowStage.REPORTING.value,
                    }

                elif tool_name == "delegate_coding_task":
                    task_desc = tool_args.get("task_description", "")
                    logger.info(f"[ACTION] Delegating task: {task_desc}")
                    execute_tool(tool_name, tool_args, session_id)
                    return {
                        "messages": [ToolMessage(content="", tool_call_id=tool_id)],
                        "current_task_description": task_desc,
                        "workflow_stage": "delegating",
                        "current_agent": "action",
                    }

                elif tool_name == "web_search":
                    import asyncio
                    import builtins
                    from data_scientist_chatbot.app.tools.web_search import web_search as do_web_search

                    query = tool_args.get("query", "")
                    logger.info(f"[WEB_SEARCH] Executing search for: {query}")

                    search_config = {}
                    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                        search_config = builtins._session_store[session_id].get("search_config", {})
                        builtins._session_store[session_id]["search_status"] = {
                            "action": "searching",
                            "query": query,
                            "provider": search_config.get("provider", "duckduckgo"),
                        }
                    logger.info(f"[WEB_SEARCH] Using config: {search_config}")

                    try:
                        loop = asyncio.get_running_loop()
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, do_web_search(query, search_config))
                            content = future.result(timeout=30)
                    except RuntimeError:
                        content = asyncio.run(do_web_search(query, search_config))

                    import re
                    import json as json_lib

                    result_count = 0
                    json_match = re.search(r"<!-- SEARCH_DATA_JSON\n(.*?)\nSEARCH_DATA_END -->", content, re.DOTALL)
                    if json_match:
                        try:
                            search_data = json_lib.loads(json_match.group(1))
                            result_count = search_data.get("result_count", 0)
                        except:
                            result_count = content.count("[Source]")

                    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                        builtins._session_store[session_id]["search_status"] = {
                            "action": "complete",
                            "query": query,
                            "result_count": result_count,
                        }

                    logger.info(f"[WEB_SEARCH] Result length: {len(content)} chars, count: {result_count}")

                elif tool_name == "python_code_interpreter":
                    logger.debug(f"Received clean code from Pydantic schema ({len(tool_args.get('code', ''))} chars)")
                    content = execute_tool(tool_name, tool_args, session_id)
                else:
                    content = execute_tool(tool_name, tool_args, session_id)
            except Exception as e:
                logger.exception(f"Tool execution failed for {tool_name}")
                raise ToolExecutionError(f"Execution failed in tool node: {tool_name}") from e

        logger.debug(f"execute_tools: Tool {tool_name} result ({len(str(content))} chars)")

        sanitized_content = sanitize_output(content)
        tool_responses.append(ToolMessage(content=sanitized_content, tool_call_id=tool_id))

    python_executions = state.get("python_executions") or 0
    for tc in tool_calls:
        name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        if name == "python_code_interpreter":
            python_executions += 1
            break

    result_state = {
        "messages": messages + tool_responses,
        "current_agent": "action",
        "python_executions": python_executions,
        "retry_count": 0,
        "last_agent_sequence": (state.get("last_agent_sequence") or []) + ["action"],
        "artifacts": state.get("artifacts") or [],
        "agent_insights": state.get("agent_insights") or [],
        "workflow_stage": "tool_execution",
    }

    for tool_call in tool_calls:
        name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        if name == "generate_comprehensive_report":
            logger.info("[WORKFLOW] Triggering Newsroom Pipeline (Action -> Analyst)")
            result_state["workflow_stage"] = WorkflowStage.REPORTING.value
        elif name == "delegate_coding_task":
            logger.info("[WORKFLOW] Delegating to Hands Agent")
            result_state["workflow_stage"] = "delegating"

    logger.info(
        f"[ACTION] Returning state with {len(result_state['messages'])} messages (added {len(tool_responses)} ToolMessages)"
    )
    return result_state


@tool
def submit_dashboard_insights(insights_json: str, session_id: str) -> str:
    """
    Submit structured insights for the dashboard.

    Args:
        insights_json: JSON string of insights array. Each item: {label, value, type (info/warning/success)}
        session_id: The current session ID.
    """
    import json
    import builtins

    try:
        insights = json.loads(insights_json)
        if not isinstance(insights, list):
            return "Error: insights_json must be a JSON array."

        if not hasattr(builtins, "_session_store"):
            builtins._session_store = {}
        if session_id not in builtins._session_store:
            builtins._session_store[session_id] = {}

        builtins._session_store[session_id]["insights"] = insights
        logger.info(f"[TOOL] Submitted {len(insights)} insights for session {session_id}")
        return f"Successfully submitted {len(insights)} insights to dashboard."
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    except Exception as e:
        return f"Error submitting insights: {e}"
