"""Shared utility helpers for message and state handling"""


def get_last_message(state):
    """Extract last message from state"""
    messages = state.get("messages", [])
    return messages[-1] if messages else None


def get_message_content(message) -> str:
    """Extract content from message"""
    if not message or not hasattr(message, 'content'):
        return ""
    return str(message.content).strip()


def has_tool_calls(message) -> bool:
    """Check if message has tool calls"""
    return hasattr(message, 'tool_calls') and bool(message.tool_calls)
