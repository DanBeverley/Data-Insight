"""Brain agent - Orchestrates analysis and interprets results."""

import re
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import traceable

from data_scientist_chatbot.app.core.state import GlobalState
from data_scientist_chatbot.app.core.agent_factory import create_brain_agent
from data_scientist_chatbot.app.core.workflow_types import ExecutionResult, Artifact, WorkflowStage
from data_scientist_chatbot.app.core.logger import logger
from data_scientist_chatbot.app.prompts import get_brain_prompt
from data_scientist_chatbot.app.utils.context import get_data_context
from data_scientist_chatbot.app.utils.helpers import has_tool_calls
from data_scientist_chatbot.app.utils.artifact_formatter import format_artifact_context
from data_scientist_chatbot.app.utils.text_processing import sanitize_output
from data_scientist_chatbot.app.context_manager import record_execution
from data_scientist_chatbot.app.tools.tool_definitions import (
    delegate_coding_task,
    knowledge_graph_query,
    access_learning_data,
    web_search,
    zip_artifacts,
    generate_comprehensive_report,
    delegate_research_task,
    save_to_knowledge,
    query_knowledge,
    save_file_to_knowledge,
    list_datasets,
    load_dataset,
    get_dataset_info,
    create_alert,
    list_my_alerts,
    delete_alert,
)


def _extract_last_user_message(messages: List) -> str:
    """Extract the most recent human message content."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return "Provide analysis and insights."


def _filter_messages_for_brain(messages: List, artifacts: List, agent_insights: List) -> List:
    """Filter messages to provide clean context without intermediate pollution."""
    from langchain_core.messages import ToolMessage

    filtered = []

    for msg in messages:
        if not hasattr(msg, "type"):
            continue

        if msg.type == "human":
            filtered.append(msg)
        elif msg.type == "ai":
            if msg.additional_kwargs.get("internal"):
                continue
            if has_tool_calls(msg):
                filtered.append(msg)
            elif msg.content and msg.content.strip():
                filtered.append(msg)
        elif msg.type == "tool" or isinstance(msg, ToolMessage):
            filtered.append(msg)

    # NOTE: Removed automatic "Present findings" injection (synthesis mode hijacking)
    # Artifacts info is available via _build_context. Brain uses tools to discover context.

    return filtered


def _build_context(
    session_id: str,
    artifacts: List[Artifact],
    execution_result: ExecutionResult,
    messages: List,
    agent_insights: List[Dict],
) -> str:
    """Build the dataset context string for Brain."""
    last_user_msg = _extract_last_user_message(messages)
    data_context = get_data_context(session_id, query=last_user_msg)

    context = ""
    if data_context and data_context.strip():
        context = f"Working with data: {data_context}"
    else:
        context = "Ready to help analyze data. Need dataset upload first."

    if artifacts:
        artifact_context = format_artifact_context(artifacts, execution_result)
        if artifact_context:
            context += f"\n\n{artifact_context}"

    if agent_insights:
        insights_str = "\n\n**KEY INSIGHTS FROM ANALYSIS:**\n"
        for insight in agent_insights:
            label = insight.get("label", "Insight")
            value = insight.get("value", "")
            insights_str += f"- **{label}:** {value}\n"
        context += insights_str
        logger.info(f"[BRAIN] Injected {len(agent_insights)} insights into context")

    return context


def _fix_artifact_paths(content: str) -> str:
    """Fix broken artifact links to point to static paths."""

    def fix_img_path(match):
        alt_text = match.group(1)
        path = match.group(2)
        if "/" not in path and path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            return f"![{alt_text}](/static/plots/{path})"
        return match.group(0)

    return re.sub(r"!\[(.*?)\]\((.*?)\)", fix_img_path, content)


@traceable(name="brain_execution", tags=["agent", "llm"])
def run_brain_agent(state: GlobalState) -> Dict[str, Any]:
    """Execute Brain agent to orchestrate analysis and interpret results."""
    session_id = state.get("session_id")
    messages = state.get("messages") or []

    enhanced_state = state.copy()
    enhanced_state["retry_count"] = enhanced_state.get("retry_count") or 0
    last_sequence = enhanced_state.get("last_agent_sequence") or []
    enhanced_state["last_agent_sequence"] = last_sequence + ["brain"]

    execution_result_dict = state.get("execution_result")
    execution_result = ExecutionResult(**execution_result_dict) if execution_result_dict else None

    artifacts_dicts = state.get("artifacts") or []
    artifacts = [Artifact(**a) for a in artifacts_dicts] if artifacts_dicts else []
    agent_insights = state.get("agent_insights") or []

    logger.info(
        f"Brain agent: execution_result={'present' if execution_result else 'none'}, artifacts={len(artifacts)}"
    )

    context = _build_context(session_id, artifacts, execution_result, messages, agent_insights)
    filtered_messages = _filter_messages_for_brain(messages, artifacts, agent_insights)

    if artifacts:
        logger.info(f"[BRAIN] Synthesis mode: {len(filtered_messages)} messages with synthesis instruction")

    logger.info(f"Brain agent processing {len(filtered_messages)} messages")
    msg_types = [f"{type(m).__name__}:{getattr(m, 'type', 'N/A')}" for m in filtered_messages]
    logger.info(f"[BRAIN] Message sequence: {msg_types}")

    thinking_mode = state.get("thinking_mode", False)
    brain_mode = "report" if thinking_mode else "chat"
    llm = create_brain_agent(mode=brain_mode)

    from data_scientist_chatbot.app.agents.tools import submit_dashboard_insights

    brain_tools = [
        delegate_coding_task,
        delegate_research_task,
        knowledge_graph_query,
        access_learning_data,
        web_search,
        zip_artifacts,
        generate_comprehensive_report,
        submit_dashboard_insights,
        save_to_knowledge,
        query_knowledge,
        save_file_to_knowledge,
        list_datasets,
        load_dataset,
        get_dataset_info,
        create_alert,
        list_my_alerts,
        delete_alert,
    ]

    model_name = getattr(llm, "model", "")
    if "phi3" in model_name.lower():
        llm_with_tools = llm
        logger.info("[BRAIN] phi3 model detected - tools NOT bound")
    else:
        llm_with_tools = llm.bind_tools(brain_tools)
        logger.info("[BRAIN] Model - tools bound")

    prompt = get_brain_prompt()
    agent_runnable = prompt | llm_with_tools

    from datetime import datetime
    import builtins

    active_modes_list = []
    if thinking_mode:
        active_modes_list.append(
            "📊 REPORT MODE: Generate comprehensive analysis with detailed visualizations and insights"
        )

    web_search_enabled = False
    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
        web_search_enabled = builtins._session_store[session_id].get("search_config") is not None
    if web_search_enabled:
        active_modes_list.append(
            "🌐 WEB SEARCH: Enabled - use web_search tool for current information, trust results over training data"
        )

    active_modes_str = (
        "\n".join(active_modes_list) if active_modes_list else "Standard chat mode - no special modes active"
    )

    uploaded_documents_str = "No documents uploaded."
    try:
        from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

        store = get_knowledge_store(session_id)
        injectable = store.list_injectable_items()
        if injectable:
            doc_lines = []
            for item in injectable[:5]:
                content = item.get("content", "")[:500]
                doc_lines.append(f"- **{item['source_name']}**: {content}")
            uploaded_documents_str = "\n".join(doc_lines)
    except Exception:
        pass

    invoke_state = {
        "messages": filtered_messages,
        "dataset_context": context,
        "uploaded_documents": uploaded_documents_str,
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "active_modes": active_modes_str,
    }

    try:
        response = agent_runnable.invoke(invoke_state)
    except Exception as llm_error:
        logger.error(f"[BRAIN] LLM call failed: {llm_error}")
        raise

    logger.info(f"[BRAIN] Raw response type: {type(response)}")
    content_preview = getattr(response, "content", None)
    if content_preview:
        logger.info(f"[BRAIN] Response content preview: '{content_preview[:200]}'")
    logger.info(f"[BRAIN] Has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")

    if not response:
        response = type(
            "obj",
            (object,),
            {"content": "I apologize, but I'm unable to generate a response at the moment. Please try again."},
        )()

    if hasattr(response, "content"):
        if response.content:
            response.content = _fix_artifact_paths(response.content)
            response.content = sanitize_output(response.content)

        if execution_result and not execution_result.success:
            response.content = f"I encountered an issue while executing the code:\n\n{execution_result.error_details}\n\nWould you like me to try a different approach?"
        elif not response.content:
            # Generate meaningful response from available artifacts/insights
            if artifacts or agent_insights:
                parts = ["## Analysis Results\n"]

                if agent_insights:
                    parts.append("\n### Key Insights\n")
                    for insight in agent_insights[:5]:
                        label = insight.get("label", "Finding")
                        value = insight.get("value", "")
                        parts.append(f"- **{label}:** {value}\n")

                if artifacts:

                    def get_cat(a):
                        return a.get("category") if isinstance(a, dict) else getattr(a, "category", None)

                    def get_fn(a):
                        return a.get("filename") if isinstance(a, dict) else getattr(a, "filename", "")

                    viz_artifacts = [a for a in artifacts if get_cat(a) == "visualization"]
                    if viz_artifacts:
                        parts.append(f"\n### Generated Visualizations ({len(viz_artifacts)} charts)\n\n")
                        for a in viz_artifacts[:8]:
                            filename = get_fn(a)
                            if filename:
                                display = filename.replace("_", " ").replace(".png", "").replace(".html", "").title()
                                if filename.endswith(".html"):
                                    parts.append(f"[📊 {display}](/{filename})\n\n")
                                else:
                                    parts.append(f"![{display}](/static/plots/{filename})\n\n")

                response.content = "".join(parts)
                logger.info(
                    f"[BRAIN] Generated fallback response from {len(artifacts)} artifacts, {len(agent_insights)} insights"
                )
            else:
                response.content = (
                    "I've received your request but couldn't generate a detailed response. Please try rephrasing."
                )

    has_tools = hasattr(response, "tool_calls") and response.tool_calls
    workflow_stage = WorkflowStage.BRAIN_INTERPRETATION.value if has_tools else WorkflowStage.COMPLETED.value

    result_state = {
        "messages": [response],
        "current_agent": "brain",
        "last_agent_sequence": enhanced_state["last_agent_sequence"],
        "retry_count": enhanced_state.get("retry_count") or 0,
        "artifacts": [a.__dict__ if hasattr(a, "__dict__") else a for a in artifacts],
        "agent_insights": agent_insights,
        "workflow_stage": workflow_stage,
    }

    try:
        last_user_msg = _extract_last_user_message(messages)
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
