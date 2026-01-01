import sys
import os
import re
import json
import sqlite3
import asyncio
from pathlib import Path
from typing import TypedDict, Sequence, Optional, Dict, Any, List
from difflib import SequenceMatcher

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, MessagesState, add_messages
from typing_extensions import Annotated
from langsmith import traceable

from data_scientist_chatbot.app.tools import execute_python_in_sandbox, execute_tool
from data_scientist_chatbot.app.context_manager import (
    ContextManager,
    ConversationContext,
    get_session_memory,
    record_execution,
)
from data_scientist_chatbot.app.performance_monitor import PerformanceMonitor
from data_scientist_chatbot.app.core.model_manager import ModelManager
from data_scientist_chatbot.app.core.agent_factory import (
    create_brain_agent,
    create_hands_agent,
    create_status_agent,
    create_verifier_agent,
)
from data_scientist_chatbot.app.core.state import GlobalState

from data_scientist_chatbot.app.agents.verifier import run_verifier_agent
from data_scientist_chatbot.app.utils.text_processing import (
    parse_message_to_tool_call,
    sanitize_output,
    extract_format_from_request,
)
from data_scientist_chatbot.app.tools.tool_definitions import (
    python_code_interpreter,
    retrieve_historical_patterns,
    delegate_coding_task,
    knowledge_graph_query,
    access_learning_data,
    web_search,
    zip_artifacts,
    generate_comprehensive_report,
)
from data_scientist_chatbot.app.prompts import (
    get_brain_prompt,
    get_hands_prompt,
    get_status_agent_prompt,
    get_verifier_prompt,
)
from data_scientist_chatbot.app.utils.context import get_data_context
from data_scientist_chatbot.app.utils.helpers import has_tool_calls
from data_scientist_chatbot.app.utils.semantic_matcher import get_semantic_matcher
from data_scientist_chatbot.app.core.logger import logger
from data_scientist_chatbot.app.core.exceptions import (
    LLMGenerationError,
    CodeExecutionError,
    StateManagementError,
    ToolExecutionError,
)
from data_scientist_chatbot.app.core.workflow_types import ExecutionResult, Artifact, WorkflowStage
from data_scientist_chatbot.app.core.training_decision import TrainingDecisionEngine
from data_scientist_chatbot.app.tools import execute_python_in_sandbox
from src.learning.adaptive_system import AdaptiveLearningSystem
from src.api_utils.artifact_tracker import get_artifact_tracker
from src.api_utils.session_management import session_data_manager
from src.reporting.unified_report_generator import UnifiedReportGenerator

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)
model_manager = ModelManager()


AgentState = GlobalState


class HandsSubgraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    python_executions: int
    data_context: str
    pattern_context: str
    learning_context: str
    retry_count: int
    execution_result: Optional[dict]
    artifacts: List[dict]
    workflow_stage: Optional[str]


performance_monitor = PerformanceMonitor()
context_manager = ContextManager()


def parse_tool_calls(state: AgentState):
    """Wrapper that delegates to centralized tool call parser"""
    last_message = state["messages"][-1]
    parse_message_to_tool_call(last_message, "json")
    return {"messages": [last_message]}


def execute_tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    if not has_tool_calls(last_message):
        logger.debug("execute_tools: No tool calls found")
        return {"messages": state["messages"]}

    tool_calls = last_message.tool_calls
    if tool_calls is None:
        tool_calls = []
    logger.debug(f"execute_tools: Found {len(tool_calls)} tool calls in message")

    tool_responses = []
    for tool_call in tool_calls:
        session_id = state.get("session_id")

        if isinstance(tool_call, dict):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", tool_call.get("arguments", {}))
            tool_id = tool_call.get("id", f"call_{tool_name}")
        else:
            # Handle LangChain tool call objects
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
                            ToolMessage(content="Generating dashboard with existing artifacts...", tool_call_id=tool_id)
                        ],
                        "current_agent": "action",
                        "workflow_stage": WorkflowStage.REPORTING.value,
                    }

                elif tool_name == "delegate_coding_task":
                    task_desc = tool_args.get("task_description", "")
                    logger.info(f"[ACTION] Delegating task: {task_desc}")
                    # Execute tool to get any immediate result (though usually it's just setup)
                    content = execute_tool(tool_name, tool_args, session_id)

                    # [FIX] Shorten the tool output for the frontend to prevent "Echo Bug"
                    # We pass the full task_desc to the state, but hide it from the user stream
                    display_content = ""

                    # Update state for Hands agent
                    return {
                        "messages": [ToolMessage(content=display_content, tool_call_id=tool_id)],
                        "current_task_description": task_desc,
                        "workflow_stage": "delegating",  # Triggers routing to Hands
                        "current_agent": "action",
                    }

                elif tool_name == "python_code_interpreter":
                    logger.debug(f"Received clean code from Pydantic schema ({len(tool_args.get('code', ''))} chars)")
                    content = execute_tool(tool_name, tool_args, session_id)
                else:
                    content = execute_tool(tool_name, tool_args, session_id)
            except Exception as e:
                logger.exception(f"Tool execution failed for {tool_name}")
                raise ToolExecutionError(f"Execution failed in tool node: {tool_name}") from e

        logger.debug(f"execute_tools: Tool {tool_name} result ({len(str(content))} chars)")

        # Apply semantic output sanitization to remove technical debug artifacts
        sanitized_content = sanitize_output(content)
        tool_responses.append(ToolMessage(content=sanitized_content, tool_call_id=tool_id))

    python_executions = state.get("python_executions") or 0
    for tool_name in [tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in tool_calls]:
        if tool_name == "python_code_interpreter":
            python_executions += 1
            break

    result_state = {
        "messages": state["messages"] + tool_responses,
        "python_executions": python_executions,
        "plan": state.get("plan"),
        "scratchpad": state.get("scratchpad", ""),
        "retry_count": 0,  # Reset after successful tool execution
        "last_agent_sequence": (state.get("last_agent_sequence") or []) + ["action"],
    }

    for tool_call in tool_calls:
        name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        if name == "generate_comprehensive_report":
            logger.info("[WORKFLOW] Triggering Newsroom Pipeline (Action -> Analyst)")
            result_state["workflow_stage"] = WorkflowStage.REPORTING.value
        elif name == "delegate_coding_task":
            logger.info("[WORKFLOW] Delegating to Hands Agent")
            result_state["workflow_stage"] = "delegating"

    logger.debug(f"execute_tools_node: Python executions count: {python_executions}")
    logger.debug(f"execute_tools_node: Returning state with {len(result_state['messages'])} messages")
    return result_state


from langchain_core.tools import tool
from typing import List, Dict, Any


@tool
def submit_dashboard_insights(insights: List[Dict[str, Any]], session_id: str):
    """
    Submit structured insights for the dashboard.

    Args:
        insights: List of dictionaries with keys: 'label', 'value', 'type' (info/warning/success), 'source'.
        session_id: The current session ID.
    """
    try:
        import builtins

        if not hasattr(builtins, "_session_store"):
            return "Error: Session store not initialized."

        if session_id not in builtins._session_store:
            builtins._session_store[session_id] = {}

        # Enforce Agent source and High priority
        for insight in insights:
            insight["source"] = "Agent"
            insight["priority"] = "high"

        builtins._session_store[session_id]["agent_insights"] = insights
        print(f"DEBUG: submit_dashboard_insights called for {session_id} with {len(insights)} insights")
        return "Insights successfully submitted to dashboard."
    except Exception as e:
        return f"Error submitting insights: {str(e)}"


@traceable(name="brain_execution", tags=["agent", "llm"])
def run_brain_agent(state: AgentState):
    session_id = state.get("session_id")
    enhanced_state = state.copy()
    enhanced_state["retry_count"] = enhanced_state.get("retry_count") or 0
    if "last_agent_sequence" not in enhanced_state or enhanced_state["last_agent_sequence"] is None:
        enhanced_state["last_agent_sequence"] = []

    last_sequence = enhanced_state.get("last_agent_sequence") or []
    enhanced_state["last_agent_sequence"] = last_sequence + ["brain"]

    execution_result_dict = state.get("execution_result")
    execution_result = ExecutionResult(**execution_result_dict) if execution_result_dict else None

    artifacts_dicts = state.get("artifacts") or []
    artifacts = [Artifact(**a) for a in artifacts_dicts] if artifacts_dicts else []

    logger.info(
        f"Brain agent: execution_result={'present' if execution_result else 'none'}, artifacts={len(artifacts)}"
    )

    recent_tool_result = None
    messages = enhanced_state.get("messages") or []
    if messages:
        for i in range(len(messages) - 1, max(-1, len(messages) - 3), -1):
            msg = messages[i]
            if hasattr(msg, "type") and msg.type == "tool":
                content_str = str(msg.content)
                if "Generated" in content_str and "visualization" in content_str:
                    recent_tool_result = msg.content
                    break
                elif "PLOT_SAVED" in content_str and "plot_" in content_str:
                    plot_match = re.search(r"PLOT_SAVED:([^\s]+\.(?:png|html|json))", content_str)
                    if plot_match:
                        recent_tool_result = f"Created visualization: {plot_match.group(1)}"
                        break

    plan = state.get("plan", [])
    current_index = state.get("current_task_index", 0)
    last_user_msg_content = "Provide analysis and insights."

    if plan and current_index < len(plan):
        last_user_msg_content = plan[current_index]["description"]
        logger.info(f"[BRAIN] Processing Task from Plan: {last_user_msg_content}")
    else:
        # Fallback for legacy/direct calls
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_msg_content = msg.content
                break

    data_context = get_data_context(session_id, query=last_user_msg_content)

    # Extract recent conversation history to prevent inappropriate tool usage
    conversation_history = ""
    recent_messages = messages[-3:] if messages and len(messages) >= 3 else (messages if messages else [])
    history_parts = []
    for msg in recent_messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            content_preview = str(msg.content)[:50]
            if msg.type == "human":
                history_parts.append(f"User: {content_preview}")
            elif msg.type == "ai":
                history_parts.append(f"Assistant: {content_preview}")
    if history_parts:
        conversation_history = " | ".join(history_parts)

    from data_scientist_chatbot.app.utils.artifact_formatter import format_artifact_context

    artifact_context = ""
    artifact_context = ""
    if artifacts:
        # If we have artifacts, we should show them regardless of execution_result state
        # (Hands agent might not set execution_result but produces artifacts)
        artifact_context = format_artifact_context(artifacts, execution_result)

    # Legacy logic replaced by helper function
    # total_artifacts = len(artifacts)
    # artifact_context = f"""...""" (Removed)
    logger.info(f"Artifact context integrated into system prompt ({len(artifact_context)} chars)")

    context = ""
    if data_context and data_context.strip():
        context = f"Working with data: {data_context}"
    else:
        context = "Ready to help analyze data. Need dataset upload first."

    if conversation_history and conversation_history.strip():
        context += f" | Recent: {conversation_history}"

    if artifact_context:
        context += f"\n\n{artifact_context}"

    # [INTELLIGENCE INJECTION] Pass structured insights from Hands to Brain
    agent_insights = state.get("agent_insights") or []
    logger.info(f"[BRAIN] agent_insights from state: {len(agent_insights)} items")
    if agent_insights:
        # Log first 2 insights for debugging
        for i, insight in enumerate(agent_insights[:2]):
            logger.info(
                f"[BRAIN] Insight {i+1}: label='{insight.get('label', 'N/A')}', value='{str(insight.get('value', 'N/A'))[:100]}'"
            )
        insights_str = "\n\n**KEY INSIGHTS FROM ANALYSIS:**\n"
        for insight in agent_insights:
            label = insight.get("label", "Insight")
            value = insight.get("value", "")
            insights_str += f"- **{label}:** {value}\n"
        context += insights_str
        logger.info(f"[BRAIN] Injected {len(agent_insights)} insights into context")

    if artifacts and len(artifacts) > 0:
        context += "\n\n**TASK COMPLETED BY HANDS AGENT:**\n"
        context += (
            f"The Hands agent has successfully completed the analysis and generated {len(artifacts)} artifacts.\n"
        )
        context += "Your job now is to INTERPRET these results for the user:\n"
        context += "1. Summarize the key findings based on the insights above.\n"
        context += "2. Reference the generated plots using proper markdown syntax.\n"
        context += "3. Provide actionable recommendations based on the analysis.\n"
        context += "DO NOT call delegate_coding_task again - the analysis is already complete."

    messages = enhanced_state.get("messages") or []

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    filtered_messages = []
    brain_tool_names = {
        "delegate_coding_task",
        "knowledge_graph_query",
        "access_learning_data",
        "web_search",
        "zip_artifacts",
        "generate_comprehensive_report",
    }

    for msg in messages:
        if hasattr(msg, "type"):
            if msg.type == "human":
                filtered_messages.append(msg)
            elif msg.type == "ai":
                if msg.additional_kwargs.get("internal"):
                    continue
                if has_tool_calls(msg):
                    compatible_calls = [tc for tc in msg.tool_calls if tc.get("name") in brain_tool_names]
                    if compatible_calls:
                        filtered_messages.append(msg)
                    elif msg.content:
                        filtered_messages.append(AIMessage(content=msg.content))
                else:
                    filtered_messages.append(msg)
            elif msg.type == "tool":
                filtered_messages.append(msg)

    enhanced_state_with_context = enhanced_state.copy()
    enhanced_state_with_context["messages"] = filtered_messages
    enhanced_state_with_context["dataset_context"] = context
    enhanced_state_with_context["brain_scratchpad"] = enhanced_state.get("scratchpad", "")

    if len(messages) != len(filtered_messages):
        logger.info(f"Brain agent: filtered {len(messages)} -> {len(filtered_messages)} messages")
    else:
        logger.info(f"Brain agent processing {len(filtered_messages)} messages")

    # Debug: Log message types to understand conversation structure
    msg_types = [f"{type(m).__name__}:{getattr(m, 'type', 'N/A')}" for m in filtered_messages]
    logger.info(f"[BRAIN] Message sequence: {msg_types}")

    # Determine Brain mode based on state (set by UI toggle)
    thinking_mode = state.get("thinking_mode", False)
    brain_mode = "report" if thinking_mode else "chat"
    llm = create_brain_agent(mode=brain_mode)
    brain_tools = [
        delegate_coding_task,
        knowledge_graph_query,
        access_learning_data,
        web_search,
        zip_artifacts,
        generate_comprehensive_report,
        submit_dashboard_insights,
    ]

    model_name = getattr(llm, "model", "")

    # Always bind tools - let Brain intelligently decide when to use them
    # The prompt guides Brain to:
    # - Use delegate_coding_task for analysis tasks
    # - Use generate_comprehensive_report when user wants a report AND artifacts exist
    # - Provide interpretation when artifacts exist and report not explicitly requested
    workflow_stage = state.get("workflow_stage", "")

    if "phi3" in model_name.lower():
        llm_with_tools = llm
        logger.info(f"[BRAIN] phi3 model detected - tools NOT bound")
    else:
        llm_with_tools = llm.bind_tools(brain_tools)
        logger.info(f"[BRAIN] Tools bound (stage={workflow_stage}, artifacts={len(artifacts)})")

    prompt = get_brain_prompt()
    agent_runnable = prompt | llm_with_tools

    try:
        response = agent_runnable.invoke(enhanced_state_with_context)
    except Exception as llm_error:
        logger.error(f"[BRAIN] LLM call failed: {llm_error}")
        raise

    # Debug logging for investigation
    logger.info(f"[BRAIN] Raw response type: {type(response)}")
    content_preview = getattr(response, "content", None)
    if content_preview:
        logger.info(f"[BRAIN] Response content preview: '{content_preview[:200]}'")
    else:
        logger.warning(f"[BRAIN] Response content is EMPTY or None")
        # Check for thinking content or additional_kwargs
        if hasattr(response, "additional_kwargs"):
            logger.info(f"[BRAIN] additional_kwargs: {response.additional_kwargs}")
        if hasattr(response, "response_metadata") and response.response_metadata:
            meta = response.response_metadata
            logger.info(
                f"[BRAIN] response_metadata: eval_count={meta.get('eval_count')}, done_reason={meta.get('done_reason')}"
            )
            # Log the raw message content from Ollama if available
            if "message" in meta:
                logger.info(f"[BRAIN] raw message from ollama: {meta.get('message')}")
    logger.info(f"[BRAIN] Has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(
            f"[BRAIN] Tool calls: {[tc.get('name') if isinstance(tc, dict) else tc.name for tc in response.tool_calls]}"
        )

    if not response:
        logger.error("[BRAIN] All retry attempts failed, using fallback")
        response = type(
            "obj",
            (object,),
            {
                "content": "I apologize, but I'm unable to generate a response at the moment due to high load or an internal error. Please try your request again."
            },
        )()

    if hasattr(response, "content"):
        # [FIX] Fix broken artifact links in Brain Logic
        # Brain typically writes ![Alt](filename.png). We need to point this to /static/plots/filename.png
        import re

        def fix_img_path(match):
            alt_text = match.group(1)
            path = match.group(2)
            # If path is just a filename (no slash) and looks like an image
            if "/" not in path and path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                return f"![{alt_text}](/static/plots/{path})"
            return match.group(0)

        if response.content:
            response.content = re.sub(r"!\[(.*?)\]\((.*?)\)", fix_img_path, response.content)

        if execution_result and not execution_result.success:
            response.content = f"I encountered an issue while executing the code:\n\n{execution_result.error_details}\n\nWould you like me to try a different approach?"
        elif not response.content or response.content == "":
            logger.warning(
                f"[BRAIN] Empty response detected. Workflow stage: {state.get('workflow_stage')}, Artifacts: {len(artifacts)}"
            )
            response.content = (
                "I've received your request but couldn't generate a detailed response. Please try rephrasing."
            )

        from data_scientist_chatbot.app.utils.text_processing import sanitize_output

        response.content = sanitize_output(response.content)

    has_tools = hasattr(response, "tool_calls") and response.tool_calls

    # Set workflow stage based on whether Brain is calling tools
    workflow_stage = WorkflowStage.BRAIN_INTERPRETATION.value if has_tools else WorkflowStage.COMPLETED.value

    result_state = {
        "messages": [response],
        "current_agent": "brain",
        "last_agent_sequence": enhanced_state["last_agent_sequence"],
        "retry_count": enhanced_state.get("retry_count") or 0,
        "workflow_stage": workflow_stage,
    }

    try:
        from data_scientist_chatbot.app.context_manager import record_execution

        last_user_msg = next((m.content for m in messages if hasattr(m, "type") and m.type == "human"), "")
        record_execution(
            session_id,
            {
                "user_request": last_user_msg,
                "code": "",
                "success": True,
                "output": str(response.content)[:500],
                "error": "",
                "artifacts": [],
                "self_corrected": False,
                "attempts": 1,
            },
        )
    except Exception as e:
        logger.debug(f"Session memory recording skipped: {e}")

    return result_state


@traceable(name="hands_execution", tags=["agent", "code"])
def run_hands_agent(state: AgentState):
    """
    Refactored hands agent - Direct execution without subgraph nesting
    Eliminates async/sync blocking issue by executing logic inline
    """

    session_id = state.get("session_id")

    enhanced_state = state.copy()
    last_sequence = enhanced_state.get("last_agent_sequence") or []
    enhanced_state["last_agent_sequence"] = last_sequence + ["hands"]

    retry_count = enhanced_state.get("retry_count") or 0
    if last_sequence and len(last_sequence) >= 4:
        recent_sequence = last_sequence[-4:]
        if recent_sequence == ["brain", "hands", "brain", "hands"]:
            enhanced_state["retry_count"] = retry_count + 1

    plan = state.get("plan", [])
    current_index = state.get("current_task_index", 0)
    task_description = "Perform data analysis task"

    delegated_task = state.get("current_task_description")
    if delegated_task:
        task_description = delegated_task
        logger.info(f"[HANDS] FULL DELEGATED TASK: {task_description}")
    elif plan and current_index < len(plan):
        task_description = plan[current_index]["description"]
        logger.info(f"[HANDS] Received Task from Brain: {task_description}")
    else:
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and hasattr(last_message, "content"):
            task_description = last_message.content

    existing_artifacts = state.get("artifacts") or []
    previous_execution = state.get("execution_result") or {}
    is_retry = False

    if state.get("messages"):
        last_msg = state["messages"][-1]
        content = str(last_msg.content)
        if "approved" in content and "feedback" in content:
            try:
                import json

                clean_content = content.replace("```json", "").replace("```", "").strip()
                feedback_data = json.loads(clean_content)

                if not feedback_data.get("approved", True):
                    is_retry = True
                    feedback_msg = feedback_data.get("feedback", "Unknown feedback")
                    logger.info(f"[HANDS] Detected rejection feedback: {feedback_msg}")

                    # Build execution context from previous run
                    execution_context_parts = []

                    # 1. Previous execution stdout (what Hands actually did)
                    prev_stdout = previous_execution.get("stdout", "")
                    if prev_stdout:
                        # Truncate but keep meaningful content
                        stdout_preview = prev_stdout[:4000] if len(prev_stdout) > 4000 else prev_stdout
                        execution_context_parts.append(f"**YOUR PREVIOUS EXECUTION OUTPUT:**\n{stdout_preview}")

                    # 2. Artifacts that actually exist
                    if existing_artifacts:
                        artifact_details = []
                        for artifact in existing_artifacts:
                            if isinstance(artifact, dict):
                                fname = artifact.get("filename", "unknown")
                                category = artifact.get("category", "unknown")
                                artifact_details.append(f"- {fname} ({category})")
                            else:
                                artifact_details.append(f"- {getattr(artifact, 'filename', str(artifact))}")
                        execution_context_parts.append(
                            f"**ARTIFACTS CREATED ({len(existing_artifacts)} total):**\n" + "\n".join(artifact_details)
                        )
                    else:
                        execution_context_parts.append("**ARTIFACTS CREATED:** None")

                    # 3. Agent insights from previous run (what patterns were found)
                    agent_insights = state.get("agent_insights", [])
                    if agent_insights:
                        insights_summary = []
                        for insight in agent_insights[:10]:  # Limit to prevent token overflow
                            if isinstance(insight, dict):
                                label = insight.get("label", "")
                                value = str(insight.get("value", ""))[:200]  # Truncate long values
                                insights_summary.append(f"- {label}: {value}")
                            else:
                                insights_summary.append(f"- {str(insight)[:200]}")
                        execution_context_parts.append(
                            f"**YOUR PREVIOUS INSIGHTS/FINDINGS:**\n" + "\n".join(insights_summary)
                        )

                    # 4. Verifier feedback (what needs to be fixed)
                    missing_items = feedback_data.get("missing_items", [])
                    execution_context_parts.append(f"**VERIFIER FEEDBACK:**\n{feedback_msg}")

                    execution_context = "\n\n".join(execution_context_parts)

                    targeted_task = f"""Retrying...
                                        {execution_context}
                                        Verifier said: "{feedback_msg}"

                                        {len(existing_artifacts)} artifacts already exist and are COMPLETE - DO NOT regenerate them.
                                        Only output what's missing:
                                        - If "df_info" is missing: Run `df.info()`, `df.describe()`, and print the shape
                                        - If "insights" is missing: Print the PROFILING_INSIGHTS JSON block
                                        - If a specific plot is missing: Generate ONLY that plot

                                        Write minimal code that ONLY fixes the missing item(s). Do NOT rerun any analysis or recreate any existing artifacts."""

                    task_description = targeted_task

                    logger.info(f"[HANDS] Retry mode - targeted fix for: {missing_items or feedback_msg}")
            except Exception as e:
                logger.warning(f"[HANDS] Failed to parse feedback JSON: {e}")

    data_context = get_data_context(session_id, query=task_description)

    try:
        session_data = session_data_manager.get_session(session_id)
        if session_data and session_data.get("dataframe") is not None:
            expected_df = session_data["dataframe"]
            from .tools import refresh_sandbox_data

            refresh_sandbox_data(session_id, expected_df)
            logger.info(f"[HANDS] Sandbox refreshed with {expected_df.shape[0]} rows, {expected_df.shape[1]} cols")
    except Exception as e:
        logger.warning(f"[HANDS] Sandbox refresh check failed: {e}")

    logger.info(f"[HANDS] Executing: {task_description[:100]}...")

    is_training_task = any(
        keyword in task_description.lower()
        for keyword in ["train", "model", "fit", "predict", "classify", "regression", "cluster"]
    )

    dataset_rows = 0
    feature_count = 0
    try:
        session_data = session_data_manager.get_session(session_id)
        if session_data:
            df = session_data.get("dataframe")
            if df is not None:
                dataset_rows = len(df)
                feature_count = len(df.columns)
            elif session_data.get("data_profile"):
                insights = session_data["data_profile"].dataset_insights
                dataset_rows = insights.total_records
                feature_count = insights.total_features
    except Exception as e:
        logger.warning(f"[HANDS] Could not retrieve dataset stats: {e}")

    execution_environment = "cpu"
    environment_context = ""

    if is_training_task:
        decision_engine = TrainingDecisionEngine()
        decision = decision_engine.decide(
            dataset_rows=dataset_rows, feature_count=feature_count, model_type="", code=None
        )
        execution_environment = decision.environment
        logger.info(f"[HANDS] Training decision: {decision.environment.upper()} ({decision.reasoning})")

        if execution_environment == "gpu":
            environment_context = f"""
**EXECUTION ENVIRONMENT: GPU (Azure ML / AWS SageMaker)**
Decision reasoning: {decision.reasoning}
"""
        else:
            environment_context = f"""
**EXECUTION ENVIRONMENT: CPU (E2B Sandbox)**
Dataset is pre-loaded as `df` variable - use it directly.
Decision reasoning: {decision.reasoning}
"""

    detected_format = extract_format_from_request(task_description)
    format_context = ""
    if detected_format:
        format_hints = {
            "onnx": "Use torch.onnx.export() or skl2onnx for ONNX format",
            "joblib": "Use joblib.dump() for saving",
            "pickle": "Use pickle.dump() for saving",
        }
        format_hint = format_hints.get(detected_format, f"Save in {detected_format} format")
        format_context = f"\n\n**USER REQUESTED FORMAT:** {detected_format.upper()}\n{format_hint}"

    enhanced_data_context = data_context
    if environment_context:
        enhanced_data_context = f"{data_context}\n{environment_context}"
    if format_context:
        enhanced_data_context = f"{enhanced_data_context}{format_context}"

    if is_training_task:
        session_data = session_data_manager.get_session(session_id)
        if session_data:
            session_data["training_environment"] = execution_environment
            session_data["training_decision"] = {
                "environment": decision.environment,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            }

    try:
        pattern_context = ""
        try:
            # Imports moved to top-level
            matcher = get_semantic_matcher()
            adaptive_system = AdaptiveLearningSystem()

            matcher = get_semantic_matcher()
            adaptive_system = AdaptiveLearningSystem()
            execution_history = adaptive_system.get_execution_history(success_only=True)

            if execution_history and task_description:
                pattern_context = matcher.find_relevant_patterns(task_description, execution_history, top_k=3)
        except Exception as e:
            logger.debug(f"[HANDS] Pattern retrieval skipped: {e}")

        learning_context = ""
        try:
            memory = get_session_memory(session_id)
            learning_context = memory.get_learning_context()
        except Exception as e:
            logger.debug(f"[HANDS] Learning context skipped: {e}")

        llm = create_hands_agent()

        hands_tools = [python_code_interpreter]
        llm_with_tools = llm.bind_tools(hands_tools)

        prompt = get_hands_prompt()
        agent_runnable = prompt | llm_with_tools

        # Extract plan context
        plan = state.get("plan", [])
        plan_context = ""
        if plan:
            plan_str = "\n".join([f"{i+1}. {task['description']} ({task['status']})" for i, task in enumerate(plan)])
            plan_context = f"**CURRENT PLAN:**\n{plan_str}\n\n**YOUR ASSIGNMENT:**\nExecute the pending tasks assigned to 'hands' efficiently."

        data_schema = "Schema not available"
        try:
            session_data = session_data_manager.get_session(session_id)
            if session_data:
                df = session_data.get("dataframe")
                if df is not None:
                    schema_parts = []
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        sample_vals = df[col].dropna().head(3).tolist()
                        sample_str = ", ".join([str(v)[:20] for v in sample_vals])
                        schema_parts.append(f"- {col} ({dtype}): e.g. {sample_str}")
                    data_schema = "Columns:\n" + "\n".join(schema_parts)
        except Exception as e:
            logger.debug(f"[HANDS] Schema extraction skipped: {e}")

        history_messages = [msg for msg in state.get("messages", []) if isinstance(msg, (AIMessage, ToolMessage))]

        hands_state = {
            "messages": history_messages + [HumanMessage(content=task_description)],
            "data_context": enhanced_data_context,
            "pattern_context": pattern_context,
            "learning_context": learning_context,
            "plan_context": plan_context,
            "data_schema": data_schema,
        }

        max_turns = 3
        turn_count = 0
        final_response = None

        # Initialize loop state with the initial prompt
        loop_messages = hands_state["messages"]
        prev_exec = enhanced_state.get("execution_result") or {}
        artifacts = list(enhanced_state.get("artifacts") or [])
        execution_summary = prev_exec.get("stdout", "") if isinstance(prev_exec, dict) else ""
        agent_insights = list(enhanced_state.get("agent_insights") or [])

        while turn_count < max_turns:
            logger.info(f"[HANDS] Starting Turn {turn_count + 1}/{max_turns}")

            # Update state with current history
            hands_state["messages"] = loop_messages

            # Invoke LLM
            llm_response = agent_runnable.invoke(hands_state)
            final_response = llm_response  # Keep track of the last response

            # Parse tool calls
            parse_message_to_tool_call(llm_response, "hands_direct")

            response_preview = str(llm_response.content)[:500] if llm_response.content else "EMPTY"
            logger.info(f"[HANDS] LLM Response Preview: {response_preview}")

            if not (hasattr(llm_response, "tool_calls") and llm_response.tool_calls):
                logger.info("[HANDS] No tool calls - Agent has completed its thought process.")
                break

            # Execute Tool
            tool_call = llm_response.tool_calls[0]
            if isinstance(tool_call, dict):
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{tool_name}")
            else:
                tool_name = tool_call.name
                tool_args = tool_call.args
                tool_id = tool_call.id

            if tool_name == "python_code_interpreter":
                code = tool_args.get("code", "")

                logger.info(f"[HANDS] Executing code ({len(code)} chars)...")
                logger.info(f"[HANDS] CODE PREVIEW (first 2000 chars):\n{code[:2000]}")

                result = execute_python_in_sandbox(code, session_id)

                # Capture output for the agent to observe
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")

                # [FIX] Parse artifacts to give explicit feedback to the agent
                # This prevents the agent from re-running the same code thinking it failed
                new_artifact_dicts = []
                new_artifact_names = []

                for line in stdout.split("\n"):
                    if "PLOT_SAVED:" in line:
                        fname = line.split(":")[1].strip()
                        new_artifact_names.append(fname)
                        new_artifact_dicts.append(
                            {"filename": fname, "category": "visualization", "local_path": f"/static/plots/{fname}"}
                        )
                    elif "MODEL_SAVED:" in line:
                        fname = line.split(":")[1].strip()
                        new_artifact_names.append(fname)
                        new_artifact_dicts.append(
                            {"filename": fname, "category": "model", "local_path": f"/static/models/{fname}"}
                        )

                tool_output_content = ""
                if new_artifact_names:
                    tool_output_content = (
                        f"SUCCESS. Generated {len(new_artifact_names)} artifacts: {', '.join(new_artifact_names)}.\n\n"
                    )

                tool_output_content += f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

                # Append to history for the loop
                loop_messages.append(llm_response)  # The AI's thought/tool call
                loop_messages.append(ToolMessage(content=tool_output_content, tool_call_id=tool_id))

                # Update artifacts list from parsed stdout
                artifacts.extend(new_artifact_dicts)

                # [FIX] Also sync artifacts from ArtifactTracker to catch all generated files
                try:
                    from src.api_utils.artifact_tracker import get_artifact_tracker

                    tracker = get_artifact_tracker()
                    tracker_result = tracker.get_session_artifacts(session_id)
                    tracked_artifacts = tracker_result.get("artifacts", []) if isinstance(tracker_result, dict) else []
                    if tracked_artifacts:
                        for artifact in tracked_artifacts:
                            filename = artifact.get("filename", "")
                            # Add if not already in artifacts list
                            if filename and not any(a.get("filename") == filename for a in artifacts):
                                artifacts.append(
                                    {
                                        "filename": filename,
                                        "category": artifact.get("category", "visualization"),
                                        "local_path": f"/static/plots/{filename}",
                                    }
                                )
                except Exception as tracker_error:
                    logger.warning(f"[HANDS] Could not sync artifacts from tracker: {tracker_error}")

                execution_summary += tool_output_content

                # Early exit check if the agent printed the final JSON block
                if "PROFILING_INSIGHTS_END" in stdout:
                    logger.info("[HANDS] Deep discovery completed (termination token found).")
                    break

                # Record execution for history
                record_execution(
                    session_id,
                    {
                        "user_request": f"Turn {turn_count+1}: {task_description[:50]}...",
                        "code": result.get("final_code", code),
                        "success": result.get("success", False),
                        "output": stdout[:500],
                        "error": stderr,
                        "artifacts": result.get("plots", []),
                        "self_corrected": result.get("self_corrected", False),
                        "attempts": result.get("attempts", 1),
                    },
                )

                execution_summary = execution_summary + f"\n\n--- Turn {turn_count + 1} ---\n{stdout}\n{stderr}".strip()

                # [EARLY EXIT] If the agent has output the final insights block, stop the loop.
                if "PROFILING_INSIGHTS_START" in stdout:
                    logger.info("[HANDS] Final insights detected. Completing Deep Discovery.")
                    break

            turn_count += 1

        # [END LOOP] Process final results
        logger.info(f"[HANDS] Deep Discovery Loop finished after {turn_count} turns.")

        # Extract insights from the LAST execution's output (or accumulated)
        if execution_summary:
            import json as json_module

            insights_pattern = r"PROFILING_INSIGHTS_START\s*(.*?)\s*PROFILING_INSIGHTS_END"
            insights_match = re.search(insights_pattern, execution_summary, re.DOTALL)
            if insights_match:
                try:
                    agent_insights = json_module.loads(insights_match.group(1))
                    logger.info(f"[HANDS] Extracted {len(agent_insights)} insights from final analysis")
                except Exception:
                    pass

            summary_pattern = r"REPORT_SUMMARY_START\s*(.*?)\s*REPORT_SUMMARY_END"
            summary_match = re.search(summary_pattern, execution_summary, re.DOTALL)
            if summary_match:
                try:
                    report_summary = json_module.loads(summary_match.group(1))
                except Exception:
                    pass

        # Construct final response message
        final_content = (
            final_response.content if final_response and hasattr(final_response, "content") else "Analysis completed."
        )

        # If the last message was a tool output, we need an AI message to close the conversation
        if loop_messages and isinstance(loop_messages[-1], ToolMessage):
            summary_response = AIMessage(
                content=f"Analysis complete. Generated {len(artifacts)} visualizations.",
                additional_kwargs={"internal": True},
            )
        else:
            summary_response = AIMessage(content=final_content, additional_kwargs={"internal": True})

        return {
            "messages": [summary_response],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
            "retry_count": enhanced_state.get("retry_count") or 0,
            "artifacts": artifacts,
            "agent_insights": agent_insights,
            "execution_result": {"stdout": execution_summary, "success": True},
        }

    except Exception as e:
        logger.error(f"[HANDS] Execution FAILED: {e}")
        import traceback

        logger.error(f"[HANDS] Traceback:\n{traceback.format_exc()}")
        # traceback.print_exc() # Optional: print to stderr

        return {
            "messages": [AIMessage(content=f"Hands execution failed: {e}")],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
            "retry_count": enhanced_state.get("retry_count") or 0,
        }


@traceable(name="analyst_node", tags=["reporting", "analyst"])
def run_analyst_node(state: AgentState):
    logger.info("[ANALYST] Passing to Architect...")
    artifacts = state.get("artifacts") or []
    logger.info(f"[ANALYST] {len(artifacts)} artifacts available for dashboard.")
    return {
        "messages": [AIMessage(content=f"Analyst: {len(artifacts)} artifacts ready for dashboard.")],
        "current_agent": "analyst",
        "artifacts": artifacts,
    }


@traceable(name="architect_node", tags=["reporting", "architect"])
async def run_architect_node(state: AgentState):
    """
    The Architect: Generates the UI Dashboard using UnifiedReportGenerator.
    Injects Iframe placeholders for interactivity.
    """
    logger.info(f"[ARCHITECT] Starting Architect Node for session {state.get('session_id')}...")
    logger.info("[ARCHITECT] Designing dashboard...")
    session_id = state.get("session_id")

    tracker = get_artifact_tracker()
    artifact_data = tracker.get_session_artifacts(session_id)
    artifacts = artifact_data.get("artifacts", [])

    dataset_artifacts = [a for a in artifacts if a.get("category") == "dataset"]
    dataset_path = None
    session_data = None

    if dataset_artifacts:
        dataset_path = dataset_artifacts[0].get("local_path")

    if not dataset_path:
        session_data = session_data_manager.get_session(session_id)
        dataset_path = session_data.get("dataset_path") if session_data else None

    logger.info(f"[ARCHITECT] Found {len(artifacts)} artifacts, dataset_path: {dataset_path}")

    if not dataset_path:
        if session_data is None:
            session_data = session_data_manager.get_session(session_id)

        logger.error(f"[ARCHITECT] DEBUG: Session data keys: {list(session_data.keys()) if session_data else 'None'}")
        logger.error(f"[ARCHITECT] DEBUG: Session store content: {session_data}")

    # [BRAIN-FIRST REPORTING] Retrieve the narrative report from the session
    report_content = None
    import builtins

    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
        # Try to find a comprehensive report summary
        report_content = builtins._session_store[session_id].get("report_summary")
        # If strictly JSON summary, we might want to look for the full text response from Brain
        # For now, we assume the Brain's last response might contain the narrative if it was in REPORT WRITING mode

        # Check messages for the last Brain response
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "Dataset Overview" in str(msg.content):
                report_content = msg.content
                break

    generator = UnifiedReportGenerator()

    final_html = ""
    try:
        async for chunk in generator.generate(
            session_id=session_id,
            dataset_path=dataset_path,
            artifacts=artifacts,
            report_type="general_analysis",
            report_content=report_content,  # Pass the narrative
        ):
            if chunk["section"] == "executive_dashboard":
                final_html = chunk["html"]

        def replace_placeholder(match):
            filename = match.group(1)
            artifact = next(
                (
                    a
                    for a in artifacts
                    if a.get("filename") == filename
                    or a.get("filename", "").endswith(filename)
                    or filename.endswith(a.get("filename", ""))
                ),
                None,
            )

            if not artifact:
                static_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "plots", filename)
                if os.path.exists(static_path):
                    artifact = {"filename": filename, "local_path": static_path}

            if artifact:
                path = artifact.get("file_path") or artifact.get("local_path")
                logger.debug(f"[ARCHITECT] Artifact keys: {list(artifact.keys())}, path: {path}")
                if not path:
                    logger.warning(
                        f"[ARCHITECT] Artifact {filename} has no file_path or local_path. Keys: {list(artifact.keys())}"
                    )
                    return f'<div class="chart-error">Artifact {filename} has no path</div>'
                if not os.path.exists(path):
                    logger.warning(f"[ARCHITECT] Artifact file not found: {path}")
                    return f'<div class="chart-error">Chart file not found: {filename}</div>'

                try:
                    if filename.endswith(".html"):
                        # Use direct src URL instead of srcdoc to avoid nested iframe issues
                        # This allows Plotly scripts to execute properly and charts to be interactive
                        chart_url = f"/static/plots/{filename}"
                        return f'<div class="chart-container"><iframe src="{chart_url}" style="width: 100%; height: 500px; border: none; border-radius: 12px; background: transparent;"></iframe></div>'
                    elif filename.endswith(".png"):
                        import base64

                        with open(path, "rb") as f:
                            b64_img = base64.b64encode(f.read()).decode("utf-8")
                        return f'<div class="chart-container"><img src="data:image/png;base64,{b64_img}" alt="{filename}" style="max-width: 100%; border-radius: 12px;"/></div>'
                    else:
                        return f'<div class="chart-error">Unsupported format: {filename}</div>'
                except Exception as e:
                    logger.error(f"[ARCHITECT] Failed to embed {filename}: {e}")
                    return f'<div class="chart-error">Could not load {filename}</div>'

            logger.warning(
                f"[ARCHITECT] Artifact not found: {filename}. Available: {[a.get('filename') for a in artifacts]}"
            )
            return f'<div class="chart-error">Artifact {filename} not found</div>'

        final_html = re.sub(
            r'<div[^>]*data-filename=["\']([^"\']+)["\'][^>]*>.*?</div>',
            replace_placeholder,
            final_html,
            flags=re.DOTALL,
        )

        logger.info(f"[ARCHITECT] Dashboard generated ({len(final_html)} chars).")

        report_path = generator.save_standalone_report(final_html, session_id)

        filename = os.path.basename(report_path)
        report_url = f"/reports/{filename}"

        logger.info(f"[ARCHITECT] Dashboard generated and saved to {report_path} (URL: {report_url})")

        result = {
            "messages": [
                AIMessage(
                    content=f"**Report Generated Successfully**\n\nYour comprehensive data analysis report is ready. It includes:\n- Executive summary with key findings\n- Interactive visualizations\n- Statistical insights and patterns\n- Actionable recommendations\n\n📊 [**View Interactive Dashboard**](report:{report_url})"
                )
            ],
            "current_agent": "presenter",
            "report_url": report_url,
            "report_path": report_path,
        }
        logger.info(f"[ARCHITECT] RETURNING KEYS: {list(result.keys())}")
        return result
    except Exception as e:
        logger.error(f"[ARCHITECT] Failed to generate dashboard: {e}")
        return {"messages": [AIMessage(content=f"Failed to generate dashboard: {e}")], "current_agent": "architect"}


@traceable(name="presenter_node", tags=["reporting", "presenter"])
async def run_presenter_node(state: AgentState):
    """
    The Presenter: Delivers the final response to the user.
    """
    report_url = state.get("report_url")
    return {
        "messages": [AIMessage(content="Analysis completed, but report generation failed.")],
        "current_agent": "presenter",
    }


async def warmup_models_parallel():
    """Warm up models in parallel for faster startup"""

    async def warmup_brain():
        try:
            # Determine Brain mode based on state
            thinking_mode = state.get("thinking_mode", False)
            brain_mode = "report" if thinking_mode else "chat"
            brain_agent = create_brain_agent(mode=brain_mode)
            brain_tools = [
                delegate_coding_task,
                knowledge_graph_query,
                access_learning_data,
                web_search,
                zip_artifacts,
                generate_comprehensive_report,
            ]

            model_name = getattr(brain_agent, "model", "")
            if "phi3" in model_name.lower():
                brain_with_tools = brain_agent
            else:
                brain_with_tools = brain_agent.bind_tools(brain_tools)

            await (get_brain_prompt() | brain_with_tools).ainvoke(
                {
                    "messages": [("human", "warmup")],
                    "dataset_context": "Ready to help analyze data. Need dataset upload first.",
                    "role": "data consultant",
                }
            )
        except:
            pass

    async def warmup_hands():
        try:
            hands_agent = create_hands_agent()
            await (get_hands_prompt() | hands_agent).ainvoke(
                {"messages": [("human", "warmup")], "data_context": "", "pattern_context": "", "learning_context": ""}
            )
        except:
            pass

    async def warmup_status():
        try:
            status_agent = create_status_agent()
            await (get_status_agent_prompt() | status_agent).ainvoke(
                {
                    "brain_scratchpad": "System startup...",
                    "tool_name": "None",
                    "workflow_stage": "startup",
                }
            )
        except:
            pass

    async def warmup_verifier():
        try:
            agent = create_verifier_agent()
            await agent.ainvoke([HumanMessage(content="ping")])
            logger.info("Verifier agent warmed up")
        except Exception as e:
            logger.warning(f"Verifier warmup failed: {e}")

    try:
        # Execute all warmups concurrently
        await asyncio.gather(warmup_brain(), warmup_hands(), warmup_status(), warmup_verifier())
        logger.info("Parallel model warmup completed")
    except Exception as e:
        logger.warning(f"Parallel warmup error: {e}")
