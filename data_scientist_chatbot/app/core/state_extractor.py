"""State extraction utilities to eliminate duplicate patterns across agents"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, AIMessage


@dataclass
class ExtractedState:
    session_id: str
    messages: List[BaseMessage]
    artifacts: List[Dict[str, Any]]
    agent_insights: List[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    current_task_index: int
    current_task_description: Optional[str]
    workflow_stage: Optional[str]
    retry_count: int


def extract_state(state: Dict[str, Any]) -> ExtractedState:
    return ExtractedState(
        session_id=state.get("session_id", ""),
        messages=state.get("messages") or [],
        artifacts=state.get("artifacts") or [],
        agent_insights=state.get("agent_insights") or [],
        execution_result=state.get("execution_result"),
        plan=state.get("plan") or [],
        current_task_index=state.get("current_task_index") or 0,
        current_task_description=state.get("current_task_description"),
        workflow_stage=state.get("workflow_stage"),
        retry_count=state.get("retry_count") or 0,
    )


def get_last_human_message(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return None


def get_last_ai_message(messages: List[BaseMessage]) -> Optional[AIMessage]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def get_current_task(state: ExtractedState) -> Tuple[str, Dict[str, Any]]:
    if state.current_task_description:
        return state.current_task_description, {
            "id": "delegated_task",
            "description": state.current_task_description,
            "assigned_to": "hands",
            "status": "in_progress",
        }

    if state.plan and state.current_task_index < len(state.plan):
        task = state.plan[state.current_task_index]
        return task.get("description", ""), task

    last_msg = get_last_human_message(state.messages)
    return last_msg or "Unknown Task", {
        "id": "dynamic_task",
        "description": last_msg or "Unknown Task",
        "assigned_to": "hands",
        "status": "in_progress",
    }


def extract_report_url_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if hasattr(msg, "additional_kwargs"):
            report_url = msg.additional_kwargs.get("report_url")
            if report_url:
                return report_url

        if hasattr(msg, "content") and "report:" in str(msg.content):
            import re

            match = re.search(r"report:([^\s\)]+)", str(msg.content))
            if match:
                return match.group(1)
    return None
