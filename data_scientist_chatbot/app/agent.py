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
    from tools.executor import execute_tool
    from tools.parsers import parse_message_to_tool_call
    from utils.context import get_data_context
    from utils.sanitizers import sanitize_output
except ImportError:
    try:
        from .tools import execute_python_in_sandbox
        from .context_manager import ContextManager, ConversationContext
        from .performance_monitor import PerformanceMonitor
        from .core.model_manager import ModelManager
        from .tools.executor import execute_tool
        from .tools.parsers import parse_message_to_tool_call
        from .utils.context import get_data_context
        from .utils.sanitizers import sanitize_output
    except ImportError:
        raise ImportError("Could not import required modules")

import time
import psutil
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

class HandsSubgraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    python_executions: int
    data_context: str

performance_monitor = PerformanceMonitor()
context_manager = ContextManager()

class CodeInput(BaseModel):
    code: str = Field(description="Raw Python code to execute in the sandbox. Must be valid Python syntax.")

class PatternInput(BaseModel):
    task_description: str = Field(description="Description of the data science task to retrieve patterns for")

@tool(args_schema=CodeInput)
def python_code_interpreter(code: str) -> str:
    """
    Executes Python code in a stateful sandbox to perform data analysis,
    manipulation, and visualization. The sandbox maintains state, so you can
    define a variable or load data in one call and use it in the next.
    Always use this tool to inspect, transform, and visualize data.
    When creating plots, they will be saved automatically. Make sure to
    inform the user that you have generated a plot.
    """
    return "This is a placeholder. Execution happens in the graph node."

class GraphQueryInput(BaseModel):
    query: str = Field(description="Natural language query about data relationships, feature lineage, or past analysis patterns")

@tool(args_schema=PatternInput)
def retrieve_historical_patterns(task_description: str) -> str:
    """
    Retrieve proven successful code patterns from past sessions for similar tasks.
    Call this before performing common data science tasks like EDA, visualization,
    modeling, or data cleaning to leverage learned patterns and best practices.

    Args:
        task_description: Description of the task you're about to perform
                         (e.g., 'visualization', 'correlation_analysis', 'ml_modeling')
    """
    return "This is a placeholder. Pattern retrieval happens in the graph node."

class LearningDataInput(BaseModel):
    query: str = Field(description="Query for execution history or patterns - e.g., 'show successful visualization code' or 'get recent analysis patterns'")

class CodingTask(BaseModel):
    task_description: str = Field(description="Clear description of the coding task for the specialized coding agent")

@tool(args_schema=CodingTask)
def delegate_coding_task(task_description: str) -> str:
    """
    Delegate computational work to specialized coding agent. Use ONLY when the user has:
    - Uploaded a dataset and needs analysis, visualization, or modeling
    - Specific technical requests requiring data processing or machine learning
    - Questions that require code execution to answer properly

    DO NOT use for:
    - Conversational exchanges, greetings, or general questions
    - Requests when no dataset is available
    - Explanations that don't require computation
    """
    return "Delegation confirmed. Coding agent will execute this task."

@tool(args_schema=GraphQueryInput)
def knowledge_graph_query(query: str) -> str:
    """
    Query relationships between datasets, features, and analysis patterns from previous work.
    Use ONLY when you need to reference or compare with historical data analysis patterns.

    DO NOT use for:
    - General conversation or greetings
    - First-time questions about capabilities
    - Simple responses that don't require historical context

    Args:
        query: Natural language query about data relationships from past analyses
    """
    return "This is a placeholder. Graph query happens in the graph node."

@tool(args_schema=LearningDataInput)
def access_learning_data(query: str) -> str:
    """
    Access successful code patterns and execution strategies from previous sessions.
    Use ONLY when you need to optimize approach based on historical performance data.

    DO NOT use for:
    - Casual conversation or capability questions
    - Initial interactions without specific technical needs
    - General responses that don't require historical learning context

    Args:
        query: Description of specific learning patterns needed for current technical task
    """
    return "This is a placeholder. Learning data access happens in the graph node."





def execute_tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        print("DEBUG execute_tools: No tool calls found")
        return {"messages": state["messages"]}

    tool_calls = last_message.tool_calls
    if tool_calls is None:
        tool_calls = []
    print(f"DEBUG execute_tools: Found {len(tool_calls)} tool calls in message")
    
    tool_responses = []
    for tool_call in tool_calls:
        session_id = state.get("session_id")
        
        # Handle both dict and object formats
        if isinstance(tool_call, dict):
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', tool_call.get('arguments', {}))
            tool_id = tool_call.get('id', f"call_{tool_name}")
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
                if tool_name == 'python_code_interpreter':
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
        "last_agent_sequence": state.get("last_agent_sequence", []) + ["action"]
    }
    print(f"DEBUG execute_tools_node: Python executions count: {python_executions}")
    print(f"DEBUG execute_tools_node: Returning state with {len(result_state['messages'])} messages")
    return result_state


def get_brain_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are Insight, a {role} in a multi-agent system.

        {context}

        **Your Role:** Business consultant who provides insights and delegates technical work.

        **Available Tools:**
        - delegate_coding_task: Use when user needs data analysis, visualization, or modeling
        - knowledge_graph_query: Access past analysis patterns for similar tasks
        - access_learning_data: Get historical performance data for optimization

        **Tool Usage Rules:**
        - Use delegate_coding_task for ANY technical request (analysis, charts, modeling)
        - Use conversational responses for business advice and result interpretation
        - Do NOT execute code directly - always delegate to hands specialist
        """),
        MessagesPlaceholder(variable_name="messages")
    ])

def get_hands_prompt():
    """Technical execution prompt for Hands agent"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a senior data scientist assistant focused on technical execution with access to a dataset `df`.
                    Your primary goal is to assist the user with their data analysis tasks. You MUST respond ONLY with a valid JSON structure for the `python_code_interpreter` tool.
                    Do not include any other text, explanations, or markdown.
                    {data_context}

                    THINKING PROCESS (Internal - use for complex tasks):
                    1. **PLAN:** Outline approach for complex multi-step tasks
                    2. **EXECUTE:** Execute step-by-step using the available tools: python_code_interpreter
                    3. **SYNTHESIZE:** Note findings after each execution
                    4. **RESPOND:** Provide clear explanations to user

                    TECHNICAL EXECUTION CAPABILITIES:
                    - Generate intelligent preprocessing code based on data characteristics
                    - Create comprehensive model training and evaluation workflows
                    - Execute end-to-end ML pipelines autonomously
                    - Provide sophisticated data analysis and visualization

                    MANDATORY FORMAT - Always respond with this exact JSON structure:
                    {{{{
                    "name": "python_code_interpreter",
                    "arguments": {{{{
                        "code": "your_python_code_here"
                    }}}}
                    }}}}

                    SANDBOX EXECUTION MASTERY:
                    You execute code in a sandbox environment. Master these principles for ANY operation:

                    **OUTPUT VISIBILITY RULES:**
                    1. EXPRESSIONS that return data (df.head(), df.info(), model.summary()) → Wrap in print()
                    2. ASSIGNMENT operations (result = df.mean()) → Add print(result) on next line
                    3. FUNCTIONS with return values → Always capture and print if user needs to see
                    4. DESCRIPTIVE operations → Always make results visible with print()
                    5. COMPUTATIONS → Print intermediate steps when helpful for understanding

                    **SMART OUTPUT PATTERNS:**
                    - Data inspection: print(df.head(10)), print(df.info()), print(df.describe())
                    - Analysis results: result = df.corr(); print(result)
                    - Model outputs: print("Accuracy:", model_score)
                    - Status updates: print("Processing complete...")

                    **GENERAL PRINCIPLE:** If a user requests something they expect to SEE, ensure it's visible via print()

                    TOOLS:
                    - python_code_interpreter: Execute Python code on the dataset

                    WORKFLOW:
                    1. FIRST RESPONSE: Return ONLY valid JSON for python_code_interpreter (no explanatory text, no markdown)
                    2. AFTER EXECUTION: When the tool result comes back, provide a clear explanation in natural language

                    EXPLANATION FORMAT (after successful execution):
                    "Successfully created [visualization type] that shows [business insight]. The chart reveals [key finding] which indicates [business implication]."

                    POST-EXECUTION COMMUNICATION:
                    After tool execution completes, you MUST provide a natural explanation that:
                    - Describes what analysis/visualization was completed
                    - Highlights the key findings and patterns discovered
                    - Explains the business value and actionable insights
                    - Uses conversational language that enables business interpretation

                    RULES:
                    - When you need to execute code: Return ONLY valid JSON (no explanatory text, no markdown)
                    - After a tool executes successfully: ALWAYS respond in conversational language explaining what was created or discovered
                    - When explaining results: Use natural, conversational language that helps business users understand the value
                    - For all data analysis/visualization: Use python_code_interpreter
                    - Always use matplotlib.use('Agg') for plotting
                    - CRITICAL: Use actual column names from the dataset context above - never hardcode column names
                    - Never use plt.close() - always use plt.show() to let the plots be saved
                    - IMPORTANT: If you use markdown, ensure JSON is properly formatted within code blocks
                    - Follow the exact JSON structure shown below

                    **TASK INFERENCE PRIORITY:**
                    1. Infer the specific task from the full conversation context and user request
                    2. If the task is unclear, empty, or just a greeting, respond conversationally asking for clarification
                    3. Only proceed with code execution if you understand exactly what the user wants

                    **JSON STRUCTURE:**
                    {{{{
                    "name": "python_code_interpreter",
                    "arguments": {{{{
                        "code": "your_specific_analysis_code_here"
                    }}}}
                    }}}}"""),
                            MessagesPlaceholder(variable_name="messages")
                        ])

def get_router_prompt():
    """Hyper-focused, programmatic prompt for router agent"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Context-Aware Task Classifier. Analyze the user's request along with session context to route intelligently.

                    **YOUR INPUTS:**
                    1. User's message
                    2. Current session state (dataset availability)

                    **ROUTING LOGIC:**
                    - HANDS: Direct technical commands that require code execution (analysis, visualization, modeling, statistics)
                    - BRAIN: Conversation, planning, interpretation, discussion without immediate code execution

                    **TECHNICAL TASK INDICATORS (when dataset available → HANDS):**
                    - Data analysis: "analyze", "explore", "examine", "investigate"
                    - Visualization: "plot", "chart", "graph", "visualize", "show"
                    - Statistics: "correlation", "distribution", "summary", "statistics"
                    - Modeling: "predict", "model", "machine learning", "classification"
                    - Data operations: "clean", "transform", "filter", "group"

                    **CONTEXT AWARENESS:**
                    If session shows "No dataset uploaded yet":
                    - Route ALL requests to BRAIN (even technical-sounding ones need consultation first)

                    If session shows "Dataset loaded":
                    - Technical tasks → HANDS
                    - Conversation/questions → BRAIN

                    **OUTPUT:** JSON only: {{"routing_decision": "brain"}} or {{"routing_decision": "hands"}}

                    **EXAMPLES:**
                    Session: "No dataset uploaded yet" + User: "analyze the data" → {{"routing_decision": "brain"}}
                    Session: "Dataset loaded: 500x10" + User: "plot histogram" → {{"routing_decision": "hands"}}
                    Session: "Dataset loaded: 500x10" + User: "analyze correlations" → {{"routing_decision": "hands"}}
                    Session: "Dataset loaded: 500x10" + User: "what does this mean?" → {{"routing_decision": "brain"}}"""),
                    ("human", "Session context: {session_context}"),
                    MessagesPlaceholder(variable_name="messages")
                    ])


def get_status_agent_prompt():
    """Prompt for the dedicated status generation agent"""
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template("""Generate a quirky status update in 5-10 words maximum.

Agent: {current_agent}
Task: {user_goal}

Examples:
- "Baking your data cake 🎂"
- "Teaching numbers to dance"
- "Hunting for patterns..."
- "Crunching your numbers!"

Output only the status message, nothing else.""")






    def run_brain_agent(state: AgentState):
        session_id = state.get("session_id")
        enhanced_state = state.copy()

        if 'plan' not in enhanced_state:
            enhanced_state['plan'] = None
        if 'scratchpad' not in enhanced_state:
            enhanced_state['scratchpad'] = ""
        if 'current_agent' not in enhanced_state:
            enhanced_state['current_agent'] = "brain"
        if 'business_context' not in enhanced_state:
            enhanced_state['business_context'] = {}
        if 'retry_count' not in enhanced_state:
            enhanced_state['retry_count'] = 0
        if 'last_agent_sequence' not in enhanced_state:
            enhanced_state['last_agent_sequence'] = []

        # Track brain agent execution
        last_sequence = enhanced_state.get("last_agent_sequence", [])
        enhanced_state["last_agent_sequence"] = last_sequence + ["brain"]

        # Check for recent tool execution results
        recent_tool_result = None
        messages = enhanced_state.get("messages", [])
        for i in range(len(messages) - 1, max(-1, len(messages) - 3), -1):
            msg = messages[i]
            if hasattr(msg, 'type') and msg.type == 'tool':
                content_str = str(msg.content)
                if 'Generated' in content_str and 'visualization' in content_str:
                    recent_tool_result = msg.content
                    break
                elif 'PLOT_SAVED' in content_str and 'plot_' in content_str:
                    import re
                    plot_match = re.search(r'PLOT_SAVED:([^\s]+\.png)', content_str)
                    if plot_match:
                        recent_tool_result = f"Created visualization: {plot_match.group(1)}"
                        break

        logger.debug(f"Brain agent - recent tool result found: {recent_tool_result is not None}")
        data_context = get_data_context(session_id)
        logger.debug(f"Brain data context length: {len(data_context)} chars")

        # Extract recent conversation history to prevent inappropriate tool usage
        conversation_history = ""
        recent_messages = messages[-3:] if len(messages) >= 3 else messages  # Last 3 messages for context
        history_parts = []
        for msg in recent_messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                content_preview = str(msg.content)[:50]
                if msg.type == 'human':
                    history_parts.append(f"User: {content_preview}")
                elif msg.type == 'ai':
                    history_parts.append(f"Assistant: {content_preview}")
        if history_parts:
            conversation_history = " | ".join(history_parts)

        # Build context and role for template variables
        context = ""
        if data_context.strip():
            context = f"Working with data: {data_context}"
        else:
            context = "Ready to help analyze data. Need dataset upload first."

        if conversation_history.strip():
            context += f" | Recent: {conversation_history}"

        role = "business consultant" if recent_tool_result is not None else "data consultant"

        # Filter messages for brain agent compatibility
        filtered_messages = []
        brain_tool_names = {'delegate_coding_task', 'knowledge_graph_query', 'access_learning_data'}

        messages = enhanced_state.get("messages", [])
        if messages is None:
            messages = []

        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'human':
                    # Always include human messages
                    filtered_messages.append(msg)
                elif msg.type == 'ai':
                    # Include AI messages only if they don't have incompatible tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        # Check if tool calls are compatible with brain agent
                        tool_call = msg.tool_calls[0]
                        tool_name = tool_call.get('name', '')
                        if tool_name in brain_tool_names:
                            filtered_messages.append(msg)
                        # Skip incompatible tool calls (like python_code_interpreter)
                    else:
                        # Include conversational AI responses
                        filtered_messages.append(msg)
                elif msg.type == 'tool':
                    # Include tool results since subgraphs ensure all main graph tools are brain-compatible
                    filtered_messages.append(msg)

        # Ensure filtered_messages is never None
        if filtered_messages is None:
            filtered_messages = []

        # Add template variables to state
        enhanced_state_with_context = enhanced_state.copy()
        enhanced_state_with_context["messages"] = filtered_messages
        enhanced_state_with_context["context"] = context
        enhanced_state_with_context["role"] = role

        llm = create_brain_agent()
        brain_tools = [delegate_coding_task, knowledge_graph_query, access_learning_data]
        llm_with_tools = llm.bind_tools(brain_tools)
        prompt = get_brain_prompt()
        agent_runnable = prompt | llm_with_tools

        try:
            original_count = len(enhanced_state.get('messages', []))
            filtered_count = len(filtered_messages)
            print(f"DEBUG: Brain agent filtered messages: {original_count} -> {filtered_count}")
            response = agent_runnable.invoke(enhanced_state_with_context)
            print(f"DEBUG: Brain agent response type: {type(response)}")
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"DEBUG: Brain agent tool calls: {[tc.get('name') for tc in response.tool_calls]}")

            result_state = {
                "messages": [response],
                "current_agent": "brain",
                "last_agent_sequence": enhanced_state["last_agent_sequence"],
                "retry_count": enhanced_state["retry_count"]
            }
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
        if len(last_sequence) >= 4:
            recent_sequence = last_sequence[-4:]
            if recent_sequence == ["brain", "hands", "brain", "hands"]:
                print(f"DEBUG: Hands agent - detected potential loop, incrementing retry count")
                enhanced_state["retry_count"] = retry_count + 1

        # Extract task: either from Brain's delegation or direct user command
        last_message = state["messages"][-1] if state["messages"] else None
        task_description = "Perform data analysis task"

        # Check if this is a delegation from brain or direct routing from router
        current_agent = enhanced_state.get("current_agent", "")
        previous_agent = enhanced_state.get("last_agent_sequence", [])[-1] if enhanced_state.get("last_agent_sequence") else ""

        if previous_agent == "router":
            # Direct routing from router - use original user message
            user_messages = [msg for msg in state["messages"] if hasattr(msg, 'type') and msg.type == 'human']
            if user_messages:
                task_description = user_messages[-1].content
                print(f"DEBUG: Direct router->hands task: {task_description}")
        elif (last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            # Delegation from brain
            tool_call = last_message.tool_calls[0]
            if tool_call.get('name') == 'delegate_coding_task':
                task_description = tool_call.get('args', {}).get('task_description', task_description)
                print(f"DEBUG: Brain delegation task: {task_description}")

        # Get dataset context for Hands agent
        data_context = get_data_context(session_id)
        logger.debug(f"Data context length: {len(data_context)} chars")
        logger.debug(f"Hands agent starting with session_id: {session_id}")

        # Use subgraph for both direct routing and delegation to ensure isolation
        router_decision = state.get("router_decision")
        if router_decision == "hands":
            # Direct routing: extract user task from last human message
            user_messages = [msg for msg in state["messages"] if hasattr(msg, 'type') and msg.type == 'human']
            if user_messages:
                task_description = user_messages[-1].content
                print(f"DEBUG: Direct router->hands task: {task_description}")
            else:
                task_description = "Perform data analysis task"

        print(f"DEBUG: Using subgraph for hands execution: {task_description}")

        # Create subgraph state with task as user message
        subgraph_state = {
            "messages": [("human", task_description)],
            "session_id": session_id,
            "python_executions": state.get("python_executions", 0),
            "data_context": data_context
        }

        # Execute hands subgraph to isolate execution
        try:
            hands_subgraph = create_hands_subgraph()
            result = hands_subgraph.invoke(subgraph_state)

            # Extract summarized result
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    summary_content = final_message.content
                else:
                    summary_content = str(final_message)
            else:
                summary_content = "Task completed but no result returned."

            print(f"DEBUG: Hands subgraph execution completed")

            # Return summary as conversational response to avoid tool call conflicts
            summary_response = AIMessage(content=summary_content)

            result_state = {
                "messages": [summary_response],
                "current_agent": "hands",
                "last_agent_sequence": enhanced_state["last_agent_sequence"],
                "retry_count": enhanced_state["retry_count"]
            }
            return result_state

        except Exception as e:
            print(f"DEBUG: Hands subgraph execution failed: {e}")
            return {"messages": [AIMessage(content=f"Hands execution failed: {e}")]}

    # Routing helper functions


def create_enhanced_agent_executor(session_id: str = None):
    return create_agent_executor()


async def warmup_models_parallel():
    """Warm up models in parallel for faster startup"""
    import asyncio


    async def warmup_brain():
        try:
            brain_agent = create_brain_agent()
            brain_tools = [delegate_coding_task, knowledge_graph_query, access_learning_data]
            brain_with_tools = brain_agent.bind_tools(brain_tools)
            await (get_brain_prompt() | brain_with_tools).ainvoke({
                "messages": [("human", "warmup")],
                "context": "Ready to help analyze data. Need dataset upload first.",
                "role": "data consultant"
            })
        except: pass

    async def warmup_hands():
        try:
            hands_agent = create_hands_agent()
            await (get_hands_prompt() | hands_agent).ainvoke({
                "messages": [("human", "warmup")],
                "data_context": ""
            })
        except: pass

    async def warmup_router():
        try:
            router_agent = create_router_agent()
            await (get_router_prompt() | router_agent).ainvoke({"messages": [("human", "warmup")]})
        except: pass

    async def warmup_status():
        try:
            status_agent = create_status_agent()
            await (get_status_agent_prompt() | status_agent).ainvoke({"agent_name": "system", "user_goal": "startup", "dataset_context": "", "tool_name": "", "tool_details": ""})
        except: pass

    try:
        await asyncio.gather(warmup_brain(), warmup_hands(), warmup_router(), warmup_status())
        print("🚀 All models warmed in parallel")

    except Exception as e:
        print(f"⚠️ Parallel warmup error: {e}")

