import os
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Sequence, Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, MessagesState, add_messages
from typing_extensions import Annotated
from langsmith import traceable

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tools import execute_python_in_sandbox, execute_tool
    from context_manager import ContextManager, ConversationContext
    from performance_monitor import PerformanceMonitor
    from core.model_manager import ModelManager
    from core.agent_factory import (
        create_brain_agent,
        create_hands_agent,
        create_router_agent,
        create_status_agent,
    )
    from utils.text_processing import parse_message_to_tool_call, sanitize_output, extract_format_from_request
    from tools.tool_definitions import (
        python_code_interpreter,
        retrieve_historical_patterns,
        delegate_coding_task,
        knowledge_graph_query,
        access_learning_data,
        web_search,
        zip_artifacts,
    )
    from prompts import get_brain_prompt, get_hands_prompt, get_router_prompt, get_status_agent_prompt
    from utils.context import get_data_context
    from utils.helpers import has_tool_calls
except ImportError:
    try:
        from .tools import execute_python_in_sandbox, execute_tool
        from .context_manager import ContextManager, ConversationContext
        from .performance_monitor import PerformanceMonitor
        from .core.model_manager import ModelManager
        from .core.agent_factory import (
            create_brain_agent,
            create_hands_agent,
            create_router_agent,
            create_status_agent,
        )
        from .utils.text_processing import parse_message_to_tool_call, sanitize_output, extract_format_from_request
        from .tools.tool_definitions import (
            python_code_interpreter,
            retrieve_historical_patterns,
            delegate_coding_task,
            knowledge_graph_query,
            access_learning_data,
            web_search,
            zip_artifacts,
        )
        from .prompts import get_brain_prompt, get_hands_prompt, get_router_prompt, get_status_agent_prompt
        from .utils.context import get_data_context
        from .utils.helpers import has_tool_calls
    except ImportError as e:
        raise ImportError(f"Could not import required modules: {e}")
import re
from difflib import SequenceMatcher
from core.logger import logger
from core.exceptions import LLMGenerationError, CodeExecutionError, StateManagementError, ToolExecutionError

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)
model_manager = ModelManager()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    python_executions: int
    plan: Optional[str]
    scratchpad: str
    business_objective: Optional[str]
    task_type: Optional[str]
    target_column: Optional[str]
    workflow_stage: Optional[str]
    current_agent: str
    business_context: Dict[str, Any]
    retry_count: int
    last_agent_sequence: List[str]
    router_decision: Optional[str]
    execution_result: Optional[dict]
    artifacts: List[dict]
    data_summary: Optional[dict]


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
    return state


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
                if tool_name == "python_code_interpreter":
                    logger.debug(f"Received clean code from Pydantic schema ({len(tool_args.get('code', ''))} chars)")

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
    logger.debug(f"execute_tools_node: Python executions count: {python_executions}")
    logger.debug(f"execute_tools_node: Returning state with {len(result_state['messages'])} messages")
    return result_state


def run_router_agent(state: AgentState):
    """Context-aware routing with complexity analysis"""
    session_id = state.get("session_id")
    last_user_message = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_message = msg
            break

    if not last_user_message:
        logger.debug("No human message found, defaulting to brain")
        return {
            "messages": state["messages"],
            "router_decision": "brain",
            "current_agent": "router",
        }

    session_context = ""
    has_dataset = False
    artifact_count = 0

    try:
        import builtins

        logger.debug(f"Router checking session_id: {session_id}")
        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            session_data = builtins._session_store[session_id]
            if "dataframe" in session_data:
                df = session_data["dataframe"]
                dataset_info = f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns"
                session_context = f"Session state: {dataset_info}"
                has_dataset = True
                logger.debug(f"Router found dataset: {dataset_info}")
            elif "data_profile" in session_data:
                session_context = "Session state: Dataset loaded"
                has_dataset = True
            else:
                session_context = "Session state: No dataset uploaded yet"
        else:
            session_context = "Session state: No dataset uploaded yet"
    except Exception as e:
        session_context = "Session state: No dataset uploaded yet"
        logger.debug(f"Router exception: {e}")

    logger.debug(f"Router context: {session_context}")
    logger.debug(f"Router analyzing message: {str(last_user_message.content)[:100]}...")

    contextual_router_state = {"messages": [last_user_message], "session_context": session_context}

    llm = create_router_agent()
    prompt = get_router_prompt()
    agent_runnable = prompt | llm

    content = None
    try:
        response = agent_runnable.invoke(contextual_router_state)
        content = str(response.content).strip()

        json_match = re.search(r'\{[^}]*"routing_decision"[^}]*\}', content)
        if not json_match:
            logger.warning("Router failed to produce JSON, defaulting to brain")
            return {
                "messages": state["messages"],
                "router_decision": "brain",
                "current_agent": "router",
            }

        json_str = json_match.group(0)
        router_data = json.loads(json_str)
        decision = router_data.get("routing_decision", "brain")

        logger.info(f"Router decision: {decision}")

        result_state = {
            "messages": state["messages"],
            "router_decision": decision,
            "current_agent": "router",
        }
        return result_state

    except Exception as e:
        logger.error(f"Router error: {e}, defaulting to brain")
        if content is not None:
            logger.debug(f"Router response content was: {repr(content)}")
        return {
            "messages": state["messages"],
            "router_decision": "brain",
            "current_agent": "router",
        }


@traceable(name="brain_execution", tags=["agent", "llm"])
def run_brain_agent(state: AgentState):
    from core.workflow_types import ExecutionResult, Artifact, WorkflowStage

    session_id = state.get("session_id")
    enhanced_state = state.copy()

    if "plan" not in enhanced_state:
        enhanced_state["plan"] = None
    if "scratchpad" not in enhanced_state:
        enhanced_state["scratchpad"] = ""
    if "current_agent" not in enhanced_state:
        enhanced_state["current_agent"] = "brain"
    if "business_context" not in enhanced_state:
        enhanced_state["business_context"] = {}

    # Ensure retry_count is always an integer (never None)
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
                    plot_match = re.search(r"PLOT_SAVED:([^\s]+\.png)", content_str)
                    if plot_match:
                        recent_tool_result = f"Created visualization: {plot_match.group(1)}"
                        break

    data_context = get_data_context(session_id)

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

    artifact_context = ""
    if execution_result and execution_result.success and artifacts:
        viz_artifacts = [a for a in artifacts if a.category == "visualization"]
        model_artifacts = [a for a in artifacts if a.category == "model"]
        dataset_artifacts = [a for a in artifacts if a.category == "dataset"]

        artifact_lines = []

        if viz_artifacts:
            artifact_lines.append(f"📊 Visualizations ({len(viz_artifacts)}):")
            for artifact in viz_artifacts[:15]:
                url = artifact.presigned_url or artifact.cloud_url or artifact.local_path
                artifact_id = (
                    artifact.artifact_id
                    if hasattr(artifact, "artifact_id") and artifact.artifact_id
                    else artifact.filename
                )
                display_name = artifact.filename.replace("_", " ").replace(".png", "")
                artifact_lines.append(f"  • {artifact.filename} (ID: {artifact_id})")
                artifact_lines.append(f"    Path: {url}")
                logger.info(f"Artifact [{artifact.filename}] - ID:{artifact_id}, Path:{url}")
            if len(viz_artifacts) > 15:
                artifact_lines.append(f"  ... and {len(viz_artifacts) - 15} more")

        if model_artifacts:
            artifact_lines.append(f"\n💾 Models ({len(model_artifacts)}):")
            for artifact in model_artifacts[:5]:
                url = artifact.presigned_url or artifact.cloud_url or artifact.local_path
                artifact_id = artifact.artifact_id if hasattr(artifact, "artifact_id") else artifact.filename
                artifact_lines.append(f"  • {artifact.filename} (ID: {artifact_id})")
                artifact_lines.append(f"    Path: {url}")
            if len(model_artifacts) > 5:
                artifact_lines.append(f"  ... and {len(model_artifacts) - 5} more")

        if dataset_artifacts:
            artifact_lines.append(f"\n📁 Datasets ({len(dataset_artifacts)}):")
            for artifact in dataset_artifacts[:5]:
                url = artifact.presigned_url or artifact.cloud_url or artifact.local_path
                artifact_id = artifact.artifact_id if hasattr(artifact, "artifact_id") else artifact.filename
                artifact_lines.append(f"  • {artifact.filename} (ID: {artifact_id})")
                artifact_lines.append(f"    Path: {url}")
            if len(dataset_artifacts) > 5:
                artifact_lines.append(f"  ... and {len(dataset_artifacts) - 5} more")

        total_artifacts = len(viz_artifacts) + len(model_artifacts) + len(dataset_artifacts)
        artifact_context = f"""
Note: The following is reference material for your use. Use the Path values to embed visualizations in your response, but do not repeat this listing.

AVAILABLE ARTIFACTS ({total_artifacts} total):
{chr(10).join(artifact_lines)}

To embed images: use ![description](Path)
To zip artifacts: use zip_artifacts tool with the ID values"""
        logger.info(f"Artifact context integrated into system prompt ({len(artifact_context)} chars)")
        logger.info(f"[ARTIFACT CONTEXT DEBUG]:\n{artifact_context}")

    context = ""
    if data_context and data_context.strip():
        context = f"Working with data: {data_context}"
    else:
        context = "Ready to help analyze data. Need dataset upload first."

    if conversation_history and conversation_history.strip():
        context += f" | Recent: {conversation_history}"

    if artifact_context:
        context += f"\n\n{artifact_context}"

    role = "business consultant" if recent_tool_result is not None else "data consultant"

    messages = enhanced_state.get("messages") or []

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    filtered_messages = []
    brain_tool_names = {
        "delegate_coding_task",
        "knowledge_graph_query",
        "access_learning_data",
        "web_search",
        "zip_artifacts",
    }

    for msg in messages:
        if hasattr(msg, "type"):
            if msg.type == "human":
                filtered_messages.append(msg)
            elif msg.type == "ai":
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

    # Add template variables to state
    enhanced_state_with_context = enhanced_state.copy()
    enhanced_state_with_context["messages"] = filtered_messages
    enhanced_state_with_context["context"] = context
    enhanced_state_with_context["role"] = role

    logger.info(f"[FULL CONTEXT DEBUG - length={len(context)}]:\n{context}")
    logger.info(f"Brain will receive {len(filtered_messages)} messages")
    for idx, msg in enumerate(filtered_messages):
        msg_type = getattr(msg, "type", "unknown")
        content_preview = str(getattr(msg, "content", ""))[:100]
        logger.debug(f"  Message {idx}: type={msg_type}, content_preview={content_preview}")

    llm = create_brain_agent()
    brain_tools = [delegate_coding_task, knowledge_graph_query, access_learning_data, web_search, zip_artifacts]

    model_name = getattr(llm, "model", "")
    if "phi3" in model_name.lower():
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(brain_tools)

    prompt = get_brain_prompt()
    agent_runnable = prompt | llm_with_tools

    try:
        # Retry logic for empty responses from LLM
        max_retries = 2
        response = None

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"[BRAIN] Calling LLM (attempt {attempt + 1}/{max_retries + 1})...")
                response = agent_runnable.invoke(enhanced_state_with_context)

                response_content = response.content if hasattr(response, "content") else ""
                logger.info(f"[BRAIN RAW RESPONSE - length={len(response_content)}]:\n{response_content[:200]}")

                # Check if response is valid
                if response_content and len(response_content.strip()) > 0:
                    logger.info(f"[BRAIN] Got valid response on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"[BRAIN] Empty response on attempt {attempt + 1}, retrying...")
                    if attempt < max_retries:
                        import time

                        time.sleep(1)  # Brief delay before retry
            except Exception as llm_error:
                logger.error(f"[BRAIN] LLM call failed on attempt {attempt + 1}: {llm_error}")
                if attempt == max_retries:
                    raise

        if not response:
            logger.error("[BRAIN] All retry attempts failed, using fallback")
            response = type("obj", (object,), {"content": ""})()

        if hasattr(response, "content"):
            if execution_result and not execution_result.success:
                response.content = f"I encountered an issue while executing the code:\n\n{execution_result.error_details}\n\nWould you like me to try a different approach?"
            elif not response.content or response.content == "":
                logger.warning("[BRAIN] Empty response after all retries, using fallback message")
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
            "workflow_stage": workflow_stage,
        }

        try:
            from context_manager import record_execution

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
    except Exception as e:
        logger.exception("Brain agent failed to generate response")
        logger.debug(f"Brain agent error type: {type(e)}")
        return {"messages": [AIMessage(content=f"Brain agent error: {e}")]}


@traceable(name="hands_execution", tags=["agent", "code"])
def run_hands_agent(state: AgentState):
    """
    Refactored hands agent - Direct execution without subgraph nesting
    Eliminates async/sync blocking issue by executing logic inline
    """
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

    logger.info("=" * 80)
    logger.info("[HANDS] run_hands_agent ENTERED (Direct Execution - No Subgraph)")
    logger.info("=" * 80)

    session_id = state.get("session_id")
    logger.info(f"[HANDS] Session ID: {session_id}")

    # Track hands agent execution
    enhanced_state = state.copy()
    last_sequence = enhanced_state.get("last_agent_sequence") or []
    enhanced_state["last_agent_sequence"] = last_sequence + ["hands"]
    logger.info(f"[HANDS] Updated agent sequence: {enhanced_state['last_agent_sequence']}")

    # Check for loop prevention
    retry_count = enhanced_state.get("retry_count") or 0
    if last_sequence and len(last_sequence) >= 4:
        recent_sequence = last_sequence[-4:]
        if recent_sequence == ["brain", "hands", "brain", "hands"]:
            logger.warning("[HANDS] Detected potential loop, incrementing retry count")
            enhanced_state["retry_count"] = retry_count + 1

    # Extract task description
    last_message = state["messages"][-1] if state["messages"] else None
    task_description = "Perform data analysis task"

    previous_agent = last_sequence[-1] if last_sequence else ""

    if previous_agent == "router":
        user_messages = [msg for msg in state["messages"] if hasattr(msg, "type") and msg.type == "human"]
        if user_messages:
            task_description = user_messages[-1].content
            logger.info(f"[HANDS] Router->hands task: {task_description[:100]}...")
    elif last_message and has_tool_calls(last_message):
        tool_call = last_message.tool_calls[0]
        if tool_call.get("name") == "delegate_coding_task":
            task_description = tool_call.get("args", {}).get("task_description", task_description)
            logger.info(f"[HANDS] Brain delegation task: {task_description[:100]}...")

    data_context = get_data_context(session_id)

    if task_description == "Perform data analysis task":
        router_decision = state.get("router_decision")
        if router_decision == "hands":
            user_messages = [msg for msg in state["messages"] if hasattr(msg, "type") and msg.type == "human"]
            if user_messages:
                task_description = user_messages[-1].content

    logger.info(f"[HANDS] Executing directly (no subgraph): {task_description[:100]}...")

    # [Rest of context building logic - training decision, format detection, etc.]
    is_training_task = any(
        keyword in task_description.lower()
        for keyword in ["train", "model", "fit", "predict", "classify", "regression", "cluster"]
    )

    execution_environment = "cpu"
    environment_context = ""

    if is_training_task:
        try:
            import builtins

            dataset_rows, feature_count = 1000, 10
            if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                session_data = builtins._session_store[session_id]
                df = session_data.get("dataframe")
                if df is not None:
                    dataset_rows, feature_count = len(df), len(df.columns)
        except Exception as e:
            logger.debug(f"[HANDS] Could not get dataset info: {e}")

        try:
            from core.training_decision import TrainingDecisionEngine
        except ImportError:
            from .core.training_decision import TrainingDecisionEngine

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

    # Store training decision
    if is_training_task:
        import builtins

        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            builtins._session_store[session_id]["training_environment"] = execution_environment
            builtins._session_store[session_id]["training_decision"] = {
                "environment": decision.environment,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            }

    # ========================================================================
    # DIRECT EXECUTION - Replaces subgraph logic
    # ========================================================================
    try:
        logger.info("[HANDS] STEP 1: Generate code with pattern matching and learning context")

        # Get pattern context (from run_hands_subgraph_agent logic)
        pattern_context = ""
        try:
            import sys, os

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from utils.semantic_matcher import get_semantic_matcher

            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
            from src.learning.adaptive_system import AdaptiveLearningSystem

            matcher = get_semantic_matcher()
            adaptive_system = AdaptiveLearningSystem()
            execution_history = adaptive_system.get_execution_history(success_only=True)

            if execution_history and task_description:
                pattern_context = matcher.find_relevant_patterns(task_description, execution_history, top_k=3)
        except Exception as e:
            logger.debug(f"[HANDS] Pattern retrieval skipped: {e}")

        # Get learning context
        learning_context = ""
        try:
            import sys, os

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from context_manager import get_session_memory

            memory = get_session_memory(session_id)
            learning_context = memory.get_learning_context()
        except Exception as e:
            logger.debug(f"[HANDS] Learning context skipped: {e}")

        # Generate code using hands LLM
        llm = create_hands_agent()
        import sys, os

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from prompts import get_hands_prompt

        prompt = get_hands_prompt()
        agent_runnable = prompt | llm

        # Create state with contexts
        hands_state = {
            "messages": [HumanMessage(content=task_description)],
            "data_context": enhanced_data_context,
            "pattern_context": pattern_context,
            "learning_context": learning_context,
        }

        logger.info("[HANDS] Invoking LLM for code generation...")
        llm_response = agent_runnable.invoke(hands_state)
        logger.info(f"[HANDS] LLM response type: {type(llm_response).__name__}")

        # ========================================================================
        # STEP 2: Parse tool calls (replaces parse_subgraph_tool_calls logic)
        # ========================================================================
        logger.info("[HANDS] STEP 2: Parse tool calls from LLM response")
        parse_message_to_tool_call(llm_response, "hands_direct")

        if not (hasattr(llm_response, "tool_calls") and llm_response.tool_calls):
            logger.warning("[HANDS] No tool calls found in LLM response")
            summary_content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

            state_messages = state.get("messages") or []
            last_msg = state_messages[-1] if state_messages else None
            is_delegation = (
                last_msg and has_tool_calls(last_msg) and last_msg.tool_calls[0].get("name") == "delegate_coding_task"
            )

            if is_delegation:
                summary_response = ToolMessage(content=summary_content, tool_call_id=last_msg.tool_calls[0].get("id"))
            else:
                summary_response = AIMessage(content=summary_content)

            return {
                "messages": [summary_response],
                "current_agent": "hands",
                "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
                "retry_count": enhanced_state.get("retry_count") or 0,
                "artifacts": [],
            }

        # ========================================================================
        # STEP 3: Execute tools with self-correction (replaces execute_subgraph_tools logic)
        # ========================================================================
        logger.info("[HANDS] STEP 3: Execute code with self-correction")

        tool_call = llm_response.tool_calls[0]
        if isinstance(tool_call, dict):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{tool_name}")
        else:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

        artifacts = []
        execution_summary = ""

        if tool_name == "python_code_interpreter":
            import sys, os

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from core.self_correction import SelfCorrectingExecutor
            from context_manager import record_execution

            executor = SelfCorrectingExecutor(max_attempts=3)
            code = tool_args.get("code", "")

            logger.info(f"[HANDS] Executing code ({len(code)} chars) with self-correction...")
            logger.debug(f"[HANDS] Code preview:\n{code[:500]}...")

            result = executor.execute_with_learning(
                initial_code=code, session_id=session_id, context=enhanced_data_context, llm_agent=llm
            )

            # Record execution
            record_execution(
                session_id,
                {
                    "user_request": task_description,
                    "code": result.get("final_code", code),
                    "success": result.get("success", False),
                    "output": result.get("stdout", ""),
                    "error": result.get("stderr", ""),
                    "artifacts": result.get("plots", []) + result.get("models", []),
                    "self_corrected": result.get("self_corrected", False),
                    "attempts": result.get("attempts", 1),
                },
            )

            stdout_output = result.get("stdout", "")
            stderr_output = result.get("stderr", "")
            plots = result.get("plots", [])
            models = result.get("models", [])

            logger.info(
                f"[HANDS] Execution complete: success={result.get('success')}, plots={len(plots)}, models={len(models)}"
            )

            # Build execution summary
            if plots or models:
                summary_parts = []
                if plots:
                    summary_parts.append(f"Created {len(plots)} visualization(s):")
                    for plot_url in plots:
                        summary_parts.append(f"  - {plot_url}")
                if models:
                    summary_parts.append(f"Saved {len(models)} model(s):")
                    for model_url in models:
                        summary_parts.append(f"  - {model_url}")
                execution_summary = "\n".join(summary_parts)
                if stdout_output.strip():
                    execution_summary = f"{stdout_output}\n\n{execution_summary}"
                if stderr_output.strip():
                    execution_summary = f"{execution_summary}\n\n{stderr_output}"

                # Build artifacts list (format matches Artifact model in workflow_types.py)
                if plots:
                    artifacts.extend(
                        [{"filename": os.path.basename(p), "category": "visualization", "local_path": p} for p in plots]
                    )
                if models:
                    artifacts.extend(
                        [{"filename": os.path.basename(m), "category": "model", "local_path": m} for m in models]
                    )
            elif not stdout_output.strip() and not stderr_output.strip() and result.get("success"):
                execution_summary = "⚠️ Code executed successfully but produced no output."
            else:
                execution_summary = f"{stdout_output}\n{stderr_output}"

            # Add completion marker
            if result.get("success") and (plots or models or (stdout_output and len(stdout_output) > 100)):
                completion_details = []
                if plots:
                    completion_details.append(f"{len(plots)} visualization(s)")
                if models:
                    completion_details.append(f"{len(models)} model(s)")
                completion_marker = f"\n\nTASK_COMPLETED: Generated {', '.join(completion_details) if completion_details else 'analysis results'} successfully."
                execution_summary += completion_marker
                logger.info(f"[HANDS] {completion_marker.strip()}")
        else:
            # Non-code tool execution
            execution_summary = execute_tool(tool_name, tool_args, session_id)
            logger.info(f"[HANDS] Executed tool: {tool_name}")

        # ========================================================================
        # STEP 4: Create final response (replaces summarize_hands_result logic)
        # ========================================================================
        logger.info("[HANDS] STEP 4: Create final response")

        summary_content = sanitize_output(execution_summary) if execution_summary else "Task completed."

        # Check if this was a delegation from brain
        state_messages = state.get("messages") or []
        last_msg = state_messages[-1] if state_messages else None
        is_delegation = (
            last_msg and has_tool_calls(last_msg) and last_msg.tool_calls[0].get("name") == "delegate_coding_task"
        )

        if is_delegation:
            summary_response = ToolMessage(content=summary_content, tool_call_id=last_msg.tool_calls[0].get("id"))
        else:
            summary_response = AIMessage(content=summary_content)

        result_state = {
            "messages": [summary_response],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
            "retry_count": enhanced_state.get("retry_count") or 0,
            "execution_result": None,
            "artifacts": artifacts,
            "workflow_stage": None,
            "python_executions": (state.get("python_executions") or 0) + 1,
        }

        logger.info("=" * 80)
        logger.info("[HANDS] Execution complete - Preparing to return result")
        logger.info(f"[HANDS] current_agent: {result_state['current_agent']}")
        logger.info(f"[HANDS] last_agent_sequence: {result_state['last_agent_sequence']}")
        logger.info(f"[HANDS] artifacts count: {len(result_state['artifacts'])}")
        logger.info(f"[HANDS] messages count: {len(result_state['messages'])}")
        logger.info(f"[HANDS] message type: {type(result_state['messages'][0]).__name__}")
        logger.info("[HANDS] RETURNING result_state now...")
        logger.info("=" * 80)

        return result_state

    except Exception as e:
        logger.error(f"[HANDS] Execution FAILED: {e}")
        import traceback

        logger.error(f"[HANDS] Traceback:\n{traceback.format_exc()}")
        traceback.print_exc()

        return {
            "messages": [AIMessage(content=f"Hands execution failed: {e}")],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
            "retry_count": enhanced_state.get("retry_count") or 0,
        }


async def warmup_models_parallel():
    """Warm up models in parallel for faster startup"""
    import asyncio

    async def warmup_brain():
        try:
            brain_agent = create_brain_agent()
            brain_tools = [delegate_coding_task, knowledge_graph_query, access_learning_data, web_search, zip_artifacts]

            model_name = getattr(brain_agent, "model", "")
            if "phi3" in model_name.lower():
                brain_with_tools = brain_agent
            else:
                brain_with_tools = brain_agent.bind_tools(brain_tools)

            await (get_brain_prompt() | brain_with_tools).ainvoke(
                {
                    "messages": [("human", "warmup")],
                    "context": "Ready to help analyze data. Need dataset upload first.",
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

    async def warmup_router():
        try:
            router_agent = create_router_agent()
            await (get_router_prompt() | router_agent).ainvoke({"messages": [("human", "warmup")]})
        except:
            pass

    async def warmup_status():
        try:
            status_agent = create_status_agent()
            await (get_status_agent_prompt() | status_agent).ainvoke(
                {
                    "agent_name": "system",
                    "user_goal": "startup",
                    "dataset_context": "",
                    "tool_name": "",
                    "tool_details": "",
                }
            )
        except:
            pass

    try:
        await asyncio.gather(warmup_brain(), warmup_hands(), warmup_router(), warmup_status())
        logger.info("All models warmed in parallel")
    except Exception as e:
        logger.warning(f"Parallel warmup error: {e}")
