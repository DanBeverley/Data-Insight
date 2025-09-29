"""Routing logic for agent workflow"""

from langgraph.graph import END
from langchain_core.messages import AIMessage


def get_last_message(state):
    """Extract last message from state safely"""
    messages = state.get("messages", [])
    return messages[-1] if messages else None


def get_message_content(message) -> str:
    """Extract content from message safely"""
    if not message or not hasattr(message, 'content'):
        return ""
    return str(message.content).strip()


def has_tool_calls(message) -> bool:
    """Check if message has tool calls"""
    return hasattr(message, 'tool_calls') and bool(message.tool_calls)


def route_to_agent(state):
    last_message = get_last_message(state)

    if not last_message or isinstance(last_message, ToolMessage):
        return "brain"

    if has_tool_calls(last_message):
        tool_call = last_message.tool_calls[0]
        if tool_call.get('name') == 'delegate_coding_task':
            return "hands"

    return "brain"


def should_continue(state):
    messages = state.get("messages", [])
    if messages is None or len(messages) == 0:
        return END

    last_message = messages[-1]
    python_executions = state.get("python_executions", 0)

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:

        if len(messages) >= 5:
            last_tool_call = last_message.tool_calls[0]

            consecutive_identical_calls = 0

            for i in range(len(messages) - 2, -1, -1):
                msg = messages[i]

                if hasattr(msg, 'type') and msg.type == 'human':
                    break

                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    prev_tool_call = msg.tool_calls[0]
                    if (last_tool_call.get('name') == prev_tool_call.get('name') and
                        last_tool_call.get('args') == prev_tool_call.get('args')):
                        consecutive_identical_calls += 1
                    break

            if consecutive_identical_calls >= 2:
                print(f"DEBUG: Termination condition met: Detected recursive tool call for '{last_tool_call.get('name')}'.")
                return END

        return "action"
    return END


def route_after_agent(state):
    """Route based on whether agent wants to use tools or just respond"""
    last_message = get_last_message(state)
    content = get_message_content(last_message)
    current_agent = state.get("current_agent", "")

    print(f"DEBUG: Routing after {current_agent} agent")
    print(f"DEBUG: Has tool_calls? {has_tool_calls(last_message)}")

    if has_tool_calls(last_message):
        print("DEBUG: Routing to parser (tool_calls)")
        return "parser"

    if (content.startswith("{") and '"name":' in content) or \
       ("python_code_interpreter" in content and '"code":' in content):
        print("DEBUG: Routing to parser (JSON format)")
        return "parser"

    if current_agent == "hands" and content and not content.startswith("{"):
        print("DEBUG: Hands completed execution, routing to brain for interpretation")
        return "brain"

    print("DEBUG: Routing to END")
    return END


def route_from_router(state):
    """Route based on router's binary decision"""
    decision = state.get("router_decision", "brain")
    print(f"DEBUG: Router routing to: {decision}")
    return decision


def route_from_brain(state):
    retry_count = state.get("retry_count", 0)
    last_sequence = state.get("last_agent_sequence", [])

    print(f"DEBUG: route_from_brain - retry_count: {retry_count}, sequence: {last_sequence}")

    if retry_count >= 3:
        print(f"DEBUG: Max retries reached ({retry_count}), terminating")
        return END

    if len(last_sequence) >= 4:
        recent_sequence = last_sequence[-4:]
        if recent_sequence == ["brain", "hands", "brain", "hands"]:
            print(f"DEBUG: Detected brain->hands->brain->hands loop, terminating")
            return END

    last_message = state["messages"][-1] if state["messages"] else None

    if (last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        return "parser"

    return END


def route_after_action(state):
    last_message = get_last_message(state)
    if last_message:
        content = get_message_content(last_message)

        if content and content.strip():
            print("DEBUG: Technical output completed, routing to brain for interpretation")
            return "brain"

    print("DEBUG: No output from action, ending flow")
    return END


def subgraph_should_continue(state):
    """Check if subgraph should continue executing"""
    messages = state.get("messages", [])
    if messages is None or len(messages) == 0:
        return END

    last_message = messages[-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "action"
    return END


def subgraph_route_after_action(state):
    """Route after action in subgraph"""
    return "summarize"