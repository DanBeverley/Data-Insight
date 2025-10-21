"""State management and context building utilities"""
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)

def create_workflow_status_context(workflow_context: Dict[str, Any], current_event: Dict[str, Any]) -> Dict[str, Any]:
    """Creates rich, session-aware context for the Status Agent"""
    current_agent = workflow_context.get("current_agent", "unknown")
    current_action = workflow_context.get("current_action", "processing")
    user_goal = workflow_context.get("user_goal", "processing request")
    tool_calls = workflow_context.get("tool_calls", [])
    execution_progress = workflow_context.get("execution_progress", {})
    session_id = workflow_context.get("session_id", "")
    session_context = "No session info"
    try:
        import builtins
        if (hasattr(builtins, '_session_store') and
            session_id in builtins._session_store and
            'dataframe' in builtins._session_store[session_id]):
            df = builtins._session_store[session_id]['dataframe']
            session_context = f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns"
        else:
            session_context = "No dataset uploaded yet"
    except:
        session_context = "No dataset available"
    workflow_stage = f"{current_agent} working"

    if session_context.startswith("No dataset"):
        if current_agent == "brain":
            workflow_stage = f"guiding user (needs data upload)"
        elif current_agent == "hands":
            workflow_stage = f"attempting analysis without data"
        elif current_agent == "parser":
            workflow_stage = f"preparing code execution"
        else:
            workflow_stage = f"preparing for data upload"
    else:
        if current_agent == "router":
            workflow_stage = "analyzing request context"
        elif current_agent == "brain":
            workflow_stage = "consulting on your data" if not tool_calls else "delegating analysis task"
        elif current_agent == "hands":
            workflow_stage = "executing analysis on your dataset"
        elif current_agent == "action":
            workflow_stage = "running analysis code"
        elif current_agent == "parser":
            workflow_stage = "formatting code execution"

    active_tool = None
    if tool_calls:
        active_tool = f"{tool_calls[-1]} on {session_context}"

    context = {
        "user_goal": user_goal[:150],
        "current_agent": current_agent,
        "current_action": current_action,
        "workflow_stage": workflow_stage,
        "active_tool": active_tool or "consultation",
        "execution_progress": execution_progress.get(current_agent, session_context),
        "event_type": current_event.get("event", ""),
        "session_state": session_context
    }
    return context

def gather_status_context(state, node_name: str, node_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assembles concise, just-in-time context for the Status Agent"""
    user_goal = "Analyzing data..."
    messages = state.get("messages", [])
    if messages is None:
        messages = []
    logger.debug(Status context - found {len(messages)} messages)

    for msg in reversed(messages):
        msg_type = getattr(msg, 'type', None)
        if msg_type == 'human':
            user_goal = str(getattr(msg, 'content', ''))[:100]
            break
        elif isinstance(msg, tuple) and len(msg) == 2 and msg[0] == 'user':
            user_goal = str(msg[1])[:100]
            break

    for i, msg in enumerate(reversed(messages)):
        msg_type = getattr(msg, 'type', 'no_type')
        msg_content = getattr(msg, 'content', 'no_content')
        logger.debug(Message {i}: type={msg_type}, content={str(msg_content)[:50]}...)

        if hasattr(msg, 'type') and msg.type == 'human':
            user_goal = str(msg.content)[:100]
            logger.debug(Found user message: {user_goal})
            break

    dataset_context = "No dataset loaded."
    session_id = state.get("session_id")
    try:
        import builtins
        if (hasattr(builtins, '_session_store') and
            session_id in builtins._session_store and
            'data_profile' in builtins._session_store[session_id]):

            data_profile = builtins._session_store[session_id]['data_profile']
            column_context = data_profile.ai_agent_context['column_details']
            columns = list(column_context.keys())[:5]
            if columns:
                dataset_context = f"Dataset with columns: {', '.join(columns)}"
    except:
        pass
    tool_name = "N/A"
    tool_details = "N/A"
    if node_name == "action":
        messages = node_data.get('messages', [])
        if messages is None:
            messages = []
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                tool_call = last_msg.tool_calls[0]
                tool_name = tool_call.get('name', 'unknown_tool')
                tool_args = tool_call.get('args', {})
                if tool_name == 'python_code_interpreter':
                    code = tool_args.get('code', '')[:50]
                    tool_details = f"Running: {code}..."
    return {
        "agent_name": node_name,
        "user_goal": user_goal,
        "dataset_context": dataset_context,
        "tool_name": tool_name,
        "tool_details": tool_details
    }

def create_lean_hands_context(state, task_description: str, session_id: str) -> Dict[str, Any]:
    messages = state.get("messages", [])
    if messages is None:
        messages = []
    essential_messages = []
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, 'type') and msg.type == 'human':
            essential_messages.insert(0, msg)
            break
        elif isinstance(msg, tuple) and len(msg) == 2 and msg[0] == 'user':
            essential_messages.insert(0, msg)
            break
    brain_context = ""
    for i in range(len(messages) - 1, max(-1, len(messages) - 3), -1):
        msg = messages[i]
        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
            brain_context = f"Context: {str(msg.content)[:200]}..."
            break
    task_message = f"{brain_context}\n\nExecute this technical task: {task_description}" if brain_context else f"Execute this technical task: {task_description}"
    essential_messages.append(("user", task_message))
    lean_state = {
        "messages": essential_messages,
        "session_id": session_id,
        "python_executions": state.get("python_executions", 0),
        "current_agent": "hands",
        "last_agent_sequence": state.get("last_agent_sequence", []),
        "retry_count": state.get("retry_count", 0)
    }
    original_msg_count = len(messages)
    lean_msg_count = len(lean_state["messages"])
    reduction = ((original_msg_count - lean_msg_count) / original_msg_count * 100) if original_msg_count > 0 else 0
    logger.debug(Context pruning - Original: {original_msg_count} messages, Pruned: {lean_msg_count} messages ({reduction:.1f}% reduction))

    return lean_state