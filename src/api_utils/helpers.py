from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


def convert_pandas_output_to_html(output_text: str) -> str:
    if not output_text or not isinstance(output_text, str):
        return output_text
    lines = output_text.strip().split("\n")
    shape_line = next((line for line in lines if "Shape:" in line), "")
    columns_line = next((line for line in lines if "Columns:" in line), "")

    if shape_line and columns_line:
        columns_text = columns_line.split(": ")[1] if ": " in columns_line else "N/A"
        return f"📊 Dataset loaded successfully!\n{shape_line}\nColumns: {columns_text}"
    return "📊 Dataset loaded successfully!"


def _get_relevant_memory_context(session_id: str, query: str) -> List:
    try:
        from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

        store = get_knowledge_store(session_id)

        relevant_turns = store.get_relevant_history(query, k=5, min_score=0.35)

        if not relevant_turns:
            recent = store.get_recent_history(k=3)
            relevant_turns = recent

        messages = []
        for turn in relevant_turns:
            content = turn.get("content", "")
            if "User:" in content and "Assistant:" in content:
                parts = content.split("\n\nAssistant:", 1)
                user_part = parts[0].replace("User:", "").strip()
                ai_part = parts[1].strip() if len(parts) > 1 else ""

                messages.append(HumanMessage(content=user_part))
                if ai_part:
                    messages.append(AIMessage(content=ai_part[:800]))

        return messages
    except Exception as e:
        logger.warning(f"[MEMORY] Failed to get context: {e}")
        return []


def create_agent_input(message: str, session_id: str, use_memory: bool = True, **kwargs) -> Dict[str, Any]:
    messages = []

    if use_memory and session_id:
        memory_messages = _get_relevant_memory_context(session_id, message)
        if memory_messages:
            messages.extend(memory_messages)
            logger.info(f"[MEMORY] Injected {len(memory_messages)} relevant history messages")

    messages.append(HumanMessage(content=message))

    return {"messages": messages, "session_id": session_id, **kwargs}


def run_agent_task(agent, message: str, session_id: str) -> Dict[str, Any]:
    try:
        response = agent.invoke(create_agent_input(message, session_id))
        messages = response.get("messages", [])
        final_message = messages[-1] if messages else None
        content = final_message.content if final_message else "Task completed"

        return {"success": True, "response": content, "session_id": session_id}
    except Exception as e:
        return {"success": False, "error": str(e), "session_id": session_id}


def create_workflow_status_context(workflow_context: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a context dictionary for the status agent based on the current workflow state.
    """
    return {
        "current_agent": workflow_context.get("current_agent", "unknown"),
        "current_action": workflow_context.get("current_action", "processing"),
        "user_goal": workflow_context.get("user_goal", ""),
        "tool_calls": workflow_context.get("tool_calls", []),
        "event_type": event.get("event", "unknown"),
        "node_name": event.get("name", "unknown"),
    }
