"""Message parsing utilities for tool calls"""

import json
import re


def parse_message_to_tool_call(message, tool_id_prefix="call"):
    """Shared parser logic to extract tool calls from message content"""
    if hasattr(message, 'tool_calls') and message.tool_calls:
        return True
    content_str = str(message.content).strip()
    try:
        json_str = content_str
        if '```json' in json_str:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
        if not json_str.startswith('{') and '{' in json_str:
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        tool_data = json.loads(json_str)
        if 'name' in tool_data and 'arguments' in tool_data:
            tool_name = tool_data['name']
            tool_args = tool_data['arguments']

            message.tool_calls = [{
                "name": tool_name,
                "args": tool_args,
                "id": f"{tool_id_prefix}_{tool_name}"
            }]
            message.content = ""
            return True
    except Exception:
        pass
    return False