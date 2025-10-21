from typing import Dict, Any
from langchain_core.messages import HumanMessage


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


def create_agent_input(message: str, session_id: str) -> Dict[str, Any]:
    return {"messages": [HumanMessage(content=message)], "session_id": session_id}


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
    return {
        "user_goal": workflow_context.get("user_goal", ""),
        "current_agent": workflow_context.get("current_agent", ""),
        "current_action": workflow_context.get("current_action", ""),
        "event_type": event.get("event", ""),
        "node_name": event.get("name", ""),
    }
