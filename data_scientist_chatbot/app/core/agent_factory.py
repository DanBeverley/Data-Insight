"""Agent creation and subgraph factory functions"""
import json
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model_manager import ModelManager
    from tools.executor import execute_tool
    from tools.parsers import parse_message_to_tool_call
    from utils.sanitizers import sanitize_output
    from core.router import subgraph_should_continue, subgraph_route_after_action
except ImportError as e:
    raise ImportError(f"Import error in agent_factory.py: {e}")

model_manager = ModelManager()

def create_router_agent():
    """Create Router agent for fast binary routing decisions"""
    config = model_manager.get_ollama_config('router')
    return ChatOllama(**config)

def create_brain_agent():
    """Create Brain agent for business reasoning and planning"""
    config = model_manager.get_ollama_config('brain')
    return ChatOllama(**config)

def create_hands_agent():
    """Create Hands agent for code execution"""
    config = model_manager.get_ollama_config('hands')
    return ChatOllama(**config)

def create_status_agent():
    """Create Status agent for real-time progress updates"""
    config = model_manager.get_ollama_config('status')
    return ChatOllama(**config)

def create_hands_subgraph():
    """Create isolated hands subgraph for delegate_coding_task execution"""
    def run_hands_subgraph_agent(state):
        """Hands agent with pattern matching and session learning"""
        session_id = state.get("session_id")
        data_context = state.get("data_context", "")

        user_request = ""
        messages = state.get("messages", [])

        for msg in reversed(messages):
            if hasattr(msg, 'type') and hasattr(msg, 'content') and msg.content:
                if msg.type == 'tool':
                    user_request = msg.content
                    break
                elif msg.type == 'human':
                    user_request = msg.content
                    break

        if not user_request and messages and hasattr(messages[0], 'content'):
            user_request = messages[0].content

        print(f"DEBUG: Hands task: {user_request[:150]}...")

        pattern_context = ""
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from utils.semantic_matcher import get_semantic_matcher
            matcher = get_semantic_matcher()

            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
            from src.learning.adaptive_system import AdaptiveLearningSystem

            adaptive_system = AdaptiveLearningSystem()
            execution_history = adaptive_system.get_execution_history(success_only=True)

            if execution_history and user_request:
                pattern_context = matcher.find_relevant_patterns(user_request, execution_history, top_k=3)
        except Exception as e:
            logger.debug(f"Pattern retrieval skipped: {e}")

        learning_context = ""
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from core.session_memory import get_session_memory
            memory = get_session_memory(session_id)
            learning_context = memory.get_learning_context()
        except Exception as e:
            logger.debug(f"Learning context skipped: {e}")

        llm = create_hands_agent()
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from prompts import get_hands_prompt
        prompt = get_hands_prompt()
        agent_runnable = prompt | llm

        hands_state_with_context = state.copy()
        hands_state_with_context["data_context"] = data_context
        hands_state_with_context["pattern_context"] = pattern_context
        hands_state_with_context["learning_context"] = learning_context

        try:
            response = agent_runnable.invoke(hands_state_with_context)
            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Hands subgraph error: {e}")]}

    def parse_subgraph_tool_calls(state):
        """Parser for subgraph tool calls"""
        last_message = state["messages"][-1]
        parse_message_to_tool_call(last_message, "subgraph")
        return state

    def execute_subgraph_tools(state):
        """Execute tools with LLM-driven self-correction for failures"""
        last_message = state["messages"][-1]
        if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            logger.warning("Subgraph action: No tool_calls found")
            return {"messages": state["messages"]}

        tool_call = last_message.tool_calls[0]
        session_id = state.get("session_id")

        if isinstance(tool_call, dict):
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', {})
            tool_id = tool_call.get('id', f"call_{tool_name}")
        else:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

        try:
            if tool_name == 'python_code_interpreter' and state.get("retry_count", 0) == 0:
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from core.self_correction import SelfCorrectingExecutor
                from core.session_memory import record_execution

                executor = SelfCorrectingExecutor(max_attempts=3)
                code = tool_args.get('code', '')
                context = state.get("data_context", "")

                print(f"DEBUG [Hands Subgraph]: Executing code ({len(code)} chars)")
                print(f"DEBUG [Generated Code]:\n{'='*60}\n{code[:1000]}\n{'='*60}")

                llm = create_hands_agent()

                user_request = ""
                messages = state.get("messages", [])
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'human':
                        user_request = msg.content
                        break

                result = executor.execute_with_learning(
                    initial_code=code,
                    session_id=session_id,
                    context=context,
                    llm_agent=llm
                )

                record_execution(session_id, {
                    'user_request': user_request,
                    'code': result.get('final_code', code),
                    'success': result.get('success', False),
                    'output': result.get('stdout', ''),
                    'error': result.get('stderr', ''),
                    'artifacts': result.get('plots', []) + result.get('models', []),
                    'self_corrected': result.get('self_corrected', False),
                    'attempts': result.get('attempts', 1)
                })

                stdout_output = result.get('stdout', '')
                stderr_output = result.get('stderr', '')
                plots = result.get('plots', [])
                models = result.get('models', [])

                if plots or models:
                    artifact_summary = []
                    if plots:
                        artifact_summary.append(f"Created {len(plots)} visualization(s):")
                        for plot_url in plots:
                            artifact_summary.append(f"  - {plot_url}")
                    if models:
                        artifact_summary.append(f"Saved {len(models)} model(s):")
                        for model_url in models:
                            artifact_summary.append(f"  - {model_url}")
                    content = '\n'.join(artifact_summary)
                    if stdout_output.strip():
                        content = f"{stdout_output}\n\n{content}"
                    if stderr_output.strip():
                        content = f"{content}\n\n{stderr_output}"
                elif not stdout_output.strip() and not stderr_output.strip() and result.get('success'):
                    print(f"WARNING: Code executed successfully but produced no output!")
                    print(f"WARNING: Generated code may be missing print() statements")
                    content = "⚠️ Code executed successfully but produced no output. The generated code may be missing print() statements to display results."
                else:
                    content = f"{stdout_output}\n{stderr_output}"

                task_completed = False
                if result.get('success') and (plots or models or (stdout_output and len(stdout_output) > 100)):
                    task_completed = True
                    completion_details = []
                    if plots:
                        completion_details.append(f"{len(plots)} visualization(s)")
                    if models:
                        completion_details.append(f"{len(models)} model(s)")

                    completion_marker = f"\n\nTASK_COMPLETED: Generated {', '.join(completion_details) if completion_details else 'analysis results'} successfully."
                    content += completion_marker
                    print(f"DEBUG: Task completion detected - {completion_marker.strip()}")

                python_executions = state.get("python_executions", 0) + 1
            else:
                content = execute_tool(tool_name, tool_args, session_id)
                python_executions = state.get("python_executions", 0) + (1 if tool_name == 'python_code_interpreter' else 0)

            tool_response = ToolMessage(content=sanitize_output(content), tool_call_id=tool_id)

            updated_state = {
                "messages": state["messages"] + [tool_response],
                "python_executions": python_executions
            }

            if tool_name == 'python_code_interpreter' and 'task_completed' in locals():
                updated_state["task_completed"] = task_completed

            return updated_state
        except ValueError as e:
            logger.error(str(e))
            return {"messages": state["messages"]}

    def summarize_hands_result(state):
        """Summarize subgraph execution result"""
        messages = state.get("messages", [])
        last_tool_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                last_tool_msg = msg
                break
        if last_tool_msg:
            summary = last_tool_msg.content
        else:
            summary = "Task completed with no output."
        summary_message = AIMessage(content=summary)
        return {"messages": [summary_message]}

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from agent import HandsSubgraphState
    subgraph = StateGraph(HandsSubgraphState)
    subgraph.add_node("hands", run_hands_subgraph_agent)
    subgraph.add_node("parser", parse_subgraph_tool_calls)
    subgraph.add_node("action", execute_subgraph_tools)
    subgraph.add_node("summarize", summarize_hands_result)
    subgraph.set_entry_point("hands")
    subgraph.add_conditional_edges(
        "hands",
        lambda state: "parser" if (
            state["messages"] and
            hasattr(state["messages"][-1], 'content') and
            ('python_code_interpreter' in str(state["messages"][-1].content) or
             (hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls))
        ) else END,
        {"parser": "parser", END: END}
    )
    subgraph.add_conditional_edges(
        "parser",
        subgraph_should_continue,
        {"action": "action", END: END}
    )
    subgraph.add_conditional_edges(
        "action",
        subgraph_route_after_action,
        {"summarize": "summarize"}
    )
    subgraph.set_finish_point("summarize")
    return subgraph.compile()