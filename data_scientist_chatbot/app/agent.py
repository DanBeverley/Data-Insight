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

try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINTER_AVAILABLE = True
    db_path = "context.db"
    memory = SqliteSaver(conn=sqlite3.connect(db_path, check_same_thread=False))
except ImportError:
    SqliteSaver = None
    memory = None
    CHECKPOINTER_AVAILABLE = False

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tools import execute_python_in_sandbox
    from context_manager import ContextManager, ConversationContext
    from performance_monitor import PerformanceMonitor
    from core.model_manager import ModelManager
    from core.agent_factory import (
        create_brain_agent,
        create_hands_agent,
        create_router_agent,
        create_status_agent,
        create_hands_subgraph,
    )
    from tools.executor import execute_tool
    from tools.parsers import parse_message_to_tool_call
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
    from utils.context import get_data_context, get_artifacts_context
    from utils.sanitizers import sanitize_output
    from utils.format_parser import extract_format_from_request
    from utils.helpers import has_tool_calls
except ImportError:
    try:
        from .tools import execute_python_in_sandbox
        from .context_manager import ContextManager, ConversationContext
        from .performance_monitor import PerformanceMonitor
        from .core.model_manager import ModelManager
        from .core.agent_factory import (
            create_brain_agent,
            create_hands_agent,
            create_router_agent,
            create_status_agent,
            create_hands_subgraph,
        )
        from .tools.executor import execute_tool
        from .tools.parsers import parse_message_to_tool_call
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
        from .utils.context import get_data_context, get_artifacts_context
        from .utils.sanitizers import sanitize_output
        from .utils.format_parser import extract_format_from_request
        from .utils.helpers import has_tool_calls
    except ImportError as e:
        raise ImportError(f"Could not import required modules: {e}")
import re
import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    complexity_score: int
    complexity_reasoning: str
    route_strategy: str


class HandsSubgraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    python_executions: int
    data_context: str
    pattern_context: str
    learning_context: str
    retry_count: int


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
        print("DEBUG execute_tools: No tool calls found")
        return {"messages": state["messages"]}

    tool_calls = last_message.tool_calls
    if tool_calls is None:
        tool_calls = []
    print(f"DEBUG execute_tools: Found {len(tool_calls)} tool calls in message")

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

        print(f"DEBUG execute_tools: Processing {tool_name} with args: {tool_args}")
        if not session_id:
            content = "Error: Session ID is missing."
        else:
            try:
                if tool_name == "python_code_interpreter":
                    print(f"DEBUG: Received clean code from Pydantic schema ({len(tool_args.get('code', ''))} chars)")

                content = execute_tool(tool_name, tool_args, session_id)
            except Exception as e:
                content = f"Execution failed in tool node: {str(e)}"

        print(f"DEBUG execute_tools: Tool {tool_name} FULL result:")
        print(content)
        print("DEBUG execute_tools: End of tool result")

        # Apply semantic output sanitization to remove technical debug artifacts
        sanitized_content = sanitize_output(content)
        tool_responses.append(ToolMessage(content=sanitized_content, tool_call_id=tool_id))

    python_executions = state.get("python_executions", 0)
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
        "last_agent_sequence": state.get("last_agent_sequence", []) + ["action"],
    }
    print(f"DEBUG execute_tools_node: Python executions count: {python_executions}")
    print(f"DEBUG execute_tools_node: Returning state with {len(result_state['messages'])} messages")
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
        print("DEBUG: No human message found, defaulting to brain")
        return {
            "messages": state["messages"],
            "router_decision": "brain",
            "current_agent": "router",
            "complexity_score": 5,
            "complexity_reasoning": "No user message",
            "route_strategy": "standard",
        }

    session_context = ""
    has_dataset = False
    artifact_count = 0

    try:
        import builtins

        print(f"DEBUG: Router checking session_id: {session_id}")
        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            session_data = builtins._session_store[session_id]
            if "dataframe" in session_data:
                df = session_data["dataframe"]
                dataset_info = f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns"
                session_context = f"Session state: {dataset_info}"
                has_dataset = True
                print(f"DEBUG: Router found dataset: {dataset_info}")
            elif "data_profile" in session_data:
                session_context = "Session state: Dataset loaded"
                has_dataset = True
            else:
                session_context = "Session state: No dataset uploaded yet"
        else:
            session_context = "Session state: No dataset uploaded yet"
    except Exception as e:
        session_context = "Session state: No dataset uploaded yet"
        print(f"DEBUG: Router exception: {e}")

    complexity_assessment = None
    try:
        from core.complexity_analyzer import create_complexity_analyzer
        from core.agent_factory import create_router_agent as get_router_llm

        analyzer = create_complexity_analyzer(get_router_llm())
        complexity_assessment = analyzer.analyze(
            user_request=last_user_message.content,
            session_context={"has_dataset": has_dataset, "artifact_count": artifact_count},
        )
        print(
            f"DEBUG: Complexity - Score: {complexity_assessment.score}, Strategy: {complexity_assessment.route_strategy}"
        )
    except Exception as e:
        print(f"DEBUG: Complexity analysis failed: {e}")

    print(f"DEBUG: Router context: {session_context}")
    print(f"DEBUG: Router analyzing message: {str(last_user_message.content)[:100]}...")

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
            print("DEBUG: Router failed to produce JSON, defaulting to brain")
            return {
                "messages": state["messages"],
                "router_decision": "brain",
                "current_agent": "router",
                "complexity_score": complexity_assessment.score if complexity_assessment else 5,
                "complexity_reasoning": complexity_assessment.reasoning if complexity_assessment else "Default",
                "route_strategy": complexity_assessment.route_strategy if complexity_assessment else "standard",
            }

        json_str = json_match.group(0)
        router_data = json.loads(json_str)
        decision = router_data.get("routing_decision", "brain")

        if complexity_assessment and complexity_assessment.route_strategy == "direct" and decision == "hands":
            print(f"DEBUG: Enforcing complexity-based routing: direct → hands (no brain summary)")
        elif complexity_assessment and complexity_assessment.route_strategy == "collaborative" and decision == "brain":
            print(f"DEBUG: Complexity-based routing: collaborative strategy approved for brain")

        print(f"DEBUG: Router decision: {decision}")

        result_state = {
            "messages": state["messages"],
            "router_decision": decision,
            "current_agent": "router",
            "complexity_score": complexity_assessment.score if complexity_assessment else 5,
            "complexity_reasoning": complexity_assessment.reasoning if complexity_assessment else "Standard task",
            "route_strategy": complexity_assessment.route_strategy if complexity_assessment else "standard",
        }
        return result_state

    except Exception as e:
        print(f"Router error: {e}, defaulting to brain")
        if content is not None:
            print(f"DEBUG: Router response content was: {repr(content)}")
        return {
            "messages": state["messages"],
            "router_decision": "brain",
            "current_agent": "router",
            "complexity_score": 5,
            "complexity_reasoning": "Error in routing",
            "route_strategy": "standard",
        }


def run_brain_agent(state: AgentState):
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
    if "retry_count" not in enhanced_state:
        enhanced_state["retry_count"] = 0
    if "last_agent_sequence" not in enhanced_state:
        enhanced_state["last_agent_sequence"] = []

    last_sequence = enhanced_state.get("last_agent_sequence", [])
    enhanced_state["last_agent_sequence"] = last_sequence + ["brain"]

    recent_tool_result = None
    messages = enhanced_state.get("messages", [])
    print(f"DEBUG: Brain received {len(messages)} total messages in state")
    for i, msg in enumerate(messages):
        if hasattr(msg, "type"):
            content_preview = str(msg.content)[:80] if hasattr(msg, "content") else "no content"
            print(f"DEBUG: Message {i}: type={msg.type}, content={content_preview}")
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

    logger.debug(f"Brain agent - recent tool result found: {recent_tool_result is not None}")
    data_context = get_data_context(session_id)
    artifacts_context = get_artifacts_context(session_id)
    logger.debug(f"Brain data context length: {len(data_context) if data_context else 0} chars")
    logger.debug(f"Brain artifacts context length: {len(artifacts_context) if artifacts_context else 0} chars")

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

    # Build context and role for template variables
    context = ""
    if data_context and data_context.strip():
        context = f"Working with data: {data_context}"
    else:
        context = "Ready to help analyze data. Need dataset upload first."

    if artifacts_context and artifacts_context.strip():
        context += f"\n\n{artifacts_context}"

    if conversation_history and conversation_history.strip():
        context += f" | Recent: {conversation_history}"

    role = "business consultant" if recent_tool_result is not None else "data consultant"

    messages = enhanced_state.get("messages", [])
    if messages is None:
        messages = []

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
        original_count = len(enhanced_state.get("messages", []))
        filtered_count = len(filtered_messages)
        print(f"DEBUG: Brain agent filtered messages: {original_count} -> {filtered_count}")
        response = agent_runnable.invoke(enhanced_state_with_context)
        print(f"DEBUG: Brain agent response type: {type(response)}")
        if has_tool_calls(response):
            print(f"DEBUG: Brain agent tool calls: {[tc.get('name') for tc in response.tool_calls]}")

        result_state = {
            "messages": [response],
            "current_agent": "brain",
            "last_agent_sequence": enhanced_state["last_agent_sequence"],
            "retry_count": enhanced_state["retry_count"],
        }

        try:
            from core.session_memory import record_execution

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
        print(f"DEBUG: Brain agent full error: {str(e)}")
        print(f"DEBUG: Brain agent error type: {type(e)}")
        return {"messages": [AIMessage(content=f"Brain agent error: {e}")]}


def run_hands_agent(state: AgentState):
    session_id = state.get("session_id")

    # Track hands agent execution
    enhanced_state = state.copy()
    last_sequence = enhanced_state.get("last_agent_sequence", [])
    enhanced_state["last_agent_sequence"] = last_sequence + ["hands"]

    # Check for loop prevention
    retry_count = enhanced_state.get("retry_count", 0)
    if last_sequence and len(last_sequence) >= 4:
        recent_sequence = last_sequence[-4:]
        if recent_sequence == ["brain", "hands", "brain", "hands"]:
            print(f"DEBUG: Hands agent - detected potential loop, incrementing retry count")
            enhanced_state["retry_count"] = retry_count + 1

    # Extract task: either from Brain's delegation or direct user command
    last_message = state["messages"][-1] if state["messages"] else None
    task_description = "Perform data analysis task"

    # Check if this is a delegation from brain or direct routing from router
    current_agent = enhanced_state.get("current_agent", "")
    previous_agent = (
        enhanced_state.get("last_agent_sequence", [])[-1] if enhanced_state.get("last_agent_sequence") else ""
    )

    if previous_agent == "router":
        # Direct routing from router - use original user message
        user_messages = [msg for msg in state["messages"] if hasattr(msg, "type") and msg.type == "human"]
        if user_messages:
            task_description = user_messages[-1].content
            print(f"DEBUG: Direct router->hands task: {task_description}")
    elif last_message and has_tool_calls(last_message):
        # Delegation from brain
        tool_call = last_message.tool_calls[0]
        if tool_call.get("name") == "delegate_coding_task":
            task_description = tool_call.get("args", {}).get("task_description", task_description)
            print(f"DEBUG: Brain delegation task: {task_description}")

    # Get dataset and artifacts context for Hands agent
    data_context = get_data_context(session_id)
    artifacts_context = get_artifacts_context(session_id)
    logger.debug(f"Data context length: {len(data_context) if data_context else 0} chars")
    logger.debug(f"Artifacts context length: {len(artifacts_context) if artifacts_context else 0} chars")
    logger.debug(f"Hands agent starting with session_id: {session_id}")

    # Check if task_description was already set by Brain delegation
    # If so, don't overwrite it with router decision logic
    if task_description == "Perform data analysis task":
        router_decision = state.get("router_decision")
        if router_decision == "hands":
            user_messages = [msg for msg in state["messages"] if hasattr(msg, "type") and msg.type == "human"]
            if user_messages:
                task_description = user_messages[-1].content
                print(f"DEBUG: Direct router->hands task: {task_description}")
            else:
                task_description = "Perform data analysis task"

    print(f"DEBUG: Using subgraph for hands execution: {task_description}")

    # Detect if this is a training task and predict execution environment
    is_training_task = any(
        keyword in task_description.lower()
        for keyword in ["train", "model", "fit", "predict", "classify", "regression", "cluster"]
    )

    execution_environment = "cpu"
    environment_context = ""

    if is_training_task:
        # Get dataset info for decision
        try:
            import builtins

            dataset_rows, feature_count = 1000, 10  # defaults

            if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                session_data = builtins._session_store[session_id]
                df = session_data.get("dataframe")
                if df is not None:
                    dataset_rows, feature_count = len(df), len(df.columns)
                    print(f"DEBUG: Dataset size: {dataset_rows} rows × {feature_count} features")
        except Exception as e:
            print(f"DEBUG: Could not get dataset info: {e}")
        try:
            from core.training_decision import TrainingDecisionEngine
        except ImportError:
            from .core.training_decision import TrainingDecisionEngine
        decision_engine = TrainingDecisionEngine()
        decision = decision_engine.decide(
            dataset_rows=dataset_rows, feature_count=feature_count, model_type="", code=None
        )
        execution_environment = decision.environment
        print(f"DEBUG: Training decision - {decision.environment.upper()} ({decision.reasoning})")
        # Add environment-specific context for Hands
        if execution_environment == "gpu":
            environment_context = f"""
                                    **EXECUTION ENVIRONMENT: GPU (Azure ML / AWS SageMaker)**
                                    Your code will run on cloud GPU infrastructure, not local sandbox.

                                    **CRITICAL - DATA ACCESS FOR GPU:**
                                    - Dataset is NOT pre-loaded as `df`
                                    - You MUST load data from storage at the start of your script
                                    - Example pattern:
                                    ```python
                                    import pandas as pd
                                    import joblib
                                    from pathlib import Path

                                    # Load dataset (will be provided in environment)
                                    df = pd.read_csv('/mnt/data/dataset.csv')

                                    # Your training code here
                                    X = df[features]
                                    y = df[target]
                                    model.fit(X, y)

                                    # Save model
                                    joblib.dump(model, 'model.pkl')
                                    print("MODEL_SAVED:model.pkl")
                                    ```

                                    Decision reasoning: {decision.reasoning}
                                    """
        else:
            environment_context = f"""
                                    **EXECUTION ENVIRONMENT: CPU (E2B Sandbox)**
                                    Your code will run in the standard sandbox environment.

                                    **DATA ACCESS:**
                                    - Dataset is pre-loaded as `df` variable - use it directly
                                    - No need to load data, just use: df.head(), df['column'], etc.

                                    Decision reasoning: {decision.reasoning}
                                    """
    detected_format = extract_format_from_request(task_description)
    format_context = ""
    if detected_format:
        format_hints = {
            "onnx": "Use torch.onnx.export() or skl2onnx for ONNX format",
            "joblib": "Use joblib.dump() for saving",
            "pickle": "Use pickle.dump() for saving",
            "pytorch": "Use torch.save() for PyTorch .pt/.pth format",
            "h5": "Use model.save() for Keras .h5 format",
            "savedmodel": "Use tf.saved_model.save() for TensorFlow SavedModel",
            "json": 'Use model.save_model() with format="json" for XGBoost',
        }
        format_hint = format_hints.get(detected_format, f"Save in {detected_format} format")
        format_context = f"\n\n**USER REQUESTED FORMAT:** {detected_format.upper()}\n{format_hint}"
        print(f"DEBUG: Detected model format preference: {detected_format}")

    # Combine all context enhancements
    enhanced_data_context = data_context
    if environment_context:
        enhanced_data_context = f"{data_context}\n{environment_context}"
    if format_context:
        enhanced_data_context = f"{enhanced_data_context}{format_context}"
    if artifacts_context:
        enhanced_data_context = f"{enhanced_data_context}\n\n{artifacts_context}"

    # Store execution environment decision in session for later use
    if is_training_task:
        import builtins

        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            builtins._session_store[session_id]["training_environment"] = execution_environment
            builtins._session_store[session_id]["training_decision"] = {
                "environment": decision.environment,
                "reasoning": decision.reasoning,
                "confidence": decision.confidence,
            }
            print(
                f"DEBUG: Stored training decision: {execution_environment.upper()} (confidence: {decision.confidence:.2f})"
            )

    # Create subgraph state with task as user message
    subgraph_state = {
        "messages": [("human", task_description)],
        "session_id": session_id,
        "python_executions": state.get("python_executions", 0),
        "data_context": enhanced_data_context,
        "pattern_context": "",
        "learning_context": "",
        "retry_count": state.get("retry_count", 0),
    }

    # Execute hands subgraph to isolate execution
    try:
        hands_subgraph = create_hands_subgraph()
        result = hands_subgraph.invoke(subgraph_state)
        if result and "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                summary_content = final_message.content
            else:
                summary_content = str(final_message)
        else:
            summary_content = "Task completed but no result returned."
        print(f"DEBUG: Hands summary content: {summary_content[:20] if summary_content else 'None'}...")

        complexity_score = state.get("complexity_score", 5)

        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        is_delegation = (
            last_message
            and has_tool_calls(last_message)
            and last_message.tool_calls[0].get("name") == "delegate_coding_task"
        )

        if is_delegation:
            tool_call_id = last_message.tool_calls[0].get("id", "unknown")
            summary_response = ToolMessage(content=summary_content, tool_call_id=tool_call_id)
        elif complexity_score <= 3:
            print(f"DEBUG: Simple task (complexity={complexity_score}), skipping Brain summary")
            summary_response = AIMessage(content=summary_content)
        else:
            summary_response = AIMessage(content=summary_content)

        result_state = {
            "messages": [summary_response],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence", []),
            "retry_count": enhanced_state.get("retry_count", 0),
        }
        return result_state

    except Exception as e:
        print(f"DEBUG: Hands subgraph execution failed: {e}")
        return {"messages": [AIMessage(content=f"Hands execution failed: {e}")]}


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
        print("🚀 All models warmed in parallel")
    except Exception as e:
        print(f"⚠️ Parallel warmup error: {e}")
