import os
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Sequence, Optional, Dict, Any, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
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
except ImportError:
    try:
        from .tools import execute_python_in_sandbox
        from .context_manager import ContextManager, ConversationContext
        from .performance_monitor import PerformanceMonitor
    except ImportError:
        raise ImportError("Could not import required modules")

import time

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)

class ModelManager:
    def __init__(self):
        self.brain_model = 'llama3.1:8b-instruct-q4_K_M' 
        self.hands_model = 'qwen2.5-coder:14b'
        self.current_model = None
        self.switch_count = 0

    def get_model(self, agent_type: str) -> str:
        target_model = self.brain_model if agent_type == 'brain' else self.hands_model

        if self.current_model != target_model:
            print(f"🧠 Switching model: {self.current_model} → {target_model}")
            self.current_model = target_model
            self.switch_count += 1

        return self.current_model

    def _optimize_context_for_coder(self, state: dict) -> dict:
        optimized_state = state.copy()
        messages = state.get("messages", [])

        if len(messages) > 5:
            optimized_state["messages"] = messages[-5:]

        return optimized_state

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
    Delegate to specialized coding agent for Python code execution. Use for any request requiring:
    - Data analysis, visualization, plotting
    - Data preprocessing, cleaning, feature engineering
    - Model training, evaluation, machine learning workflows
    - Statistical analysis, data processing
    """
    return "Delegation confirmed. Coding agent will execute this task."

@tool(args_schema=GraphQueryInput)
def knowledge_graph_query(query: str) -> str:
    """
    Query the knowledge graph to find relationships between datasets, features, models, and analysis patterns.
    Use this to discover insights from past work, find similar datasets, or understand feature relationships.

    Args:
        query: Natural language query about data relationships
               (e.g., 'find datasets similar to housing data', 'what features correlate with price')
    """
    return "This is a placeholder. Graph query happens in the graph node."

@tool(args_schema=LearningDataInput)
def access_learning_data(query: str) -> str:
    """
    Access historical execution data and learning patterns from past sessions.
    Use this to understand what approaches worked well before, get execution time data, or find successful code patterns.

    Args:
        query: Description of what learning data you need
               (e.g., 'show recent successful executions', 'get visualization code examples')
    """
    return "This is a placeholder. Learning data access happens in the graph node."

def generate_business_insights(code: str, output: str, session_id: str) -> str:
    try:
        import sys, builtins, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        from src.insights.business_translator import BusinessTranslator, StakeholderType
        from src.insights.narrative_generator import NarrativeGenerator, ReportTone

        if hasattr(builtins, '_session_store') and session_id in builtins._session_store:
            data_profile = builtins._session_store[session_id].get('data_profile')
            if data_profile:
                translator = BusinessTranslator()
                metrics = translator.extract_business_metrics_from_analysis(code, output, data_profile)
                stakeholder_view = translator.generate_stakeholder_view(metrics, StakeholderType.BUSINESS_ANALYST)
                narrative = NarrativeGenerator().generate_insight_narrative(stakeholder_view, ReportTone.BUSINESS)
                return narrative[:200] + "..." if len(narrative) > 200 else narrative
    except:
        pass
    return ""

def generate_explainability_insights(code: str, session_id: str) -> str:
    try:
        import sys, builtins, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        from src.explainability.bias_detector import BiasDetector

        if hasattr(builtins, '_session_store') and session_id in builtins._session_store:
            df = builtins._session_store[session_id].get('dataframe')
            if df is not None:
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    detector = BiasDetector(sensitive_attributes=categorical_cols[:1])
                    bias_results = detector.detect_data_bias(df)
                    if bias_results:
                        return f"Data analysis: {bias_results[0].description}"
    except:
        pass
    return ""

def knowledge_graph_query_logic(query: str, session_id: str) -> str:
    try:
        import sys, os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        from src.knowledge_graph.service import KnowledgeGraphService, SessionDataStorage

        # Use session storage directly for lightweight operations
        storage = SessionDataStorage()

        # Get structured data from storage
        graph_data = storage.get_all_data()

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="graph_query",
            value=1.0,
            context={"query": query}
        )

        # Return structured data for agent to interpret
        if graph_data:
            return f"Knowledge graph data found: {graph_data}"
        else:
            return "No relevant historical patterns or relationships found in the knowledge graph."

    except Exception as e:
        return f"Graph query failed: {str(e)}"

def access_learning_data_logic(query: str, session_id: str) -> str:
    try:
        import sys, os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        from src.learning.adaptive_system import AdaptiveLearningSystem

        adaptive_system = AdaptiveLearningSystem()
        execution_history = adaptive_system.get_execution_history(success_only=True)

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="learning_data_access",
            value=1.0,
            context={"query": query}
        )

        if execution_history:
            return f"Learning data found: {len(execution_history)} successful executions. Recent examples: {execution_history[-5:]}"
        else:
            return "No learning data available yet. Execute some code first to build learning history."

    except Exception as e:
        return f"Learning data access failed: {str(e)}"

def retrieve_historical_patterns_logic(task_description: str, session_id: str) -> str:
    try:
        patterns = context_manager.get_cross_session_patterns(
            pattern_type=task_description, 
            limit=3
        )
        
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="pattern_retrieval",
            value=len(patterns),
            context={"task_description": task_description}
        )
        
        if not patterns:
            return f"No historical patterns found for '{task_description}'. Now execute the task using python_code_interpreter tool with standard best practices."
        
        pattern_summaries = []
        for pattern in patterns:
            summary = f"**{pattern['type']}** (confidence: {pattern['confidence']:.2f}, used {pattern['success_count']} times)"
            if pattern.get('data'):
                data_preview = str(pattern['data'])[:200]
                summary += f"\nPattern details: {data_preview}{'...' if len(str(pattern['data'])) > 200 else ''}"
            pattern_summaries.append(summary)
        
        result = f"Found {len(patterns)} proven patterns for '{task_description}':\n\n" + "\n\n".join(pattern_summaries)
        result += "\n\nNow use python_code_interpreter tool to implement the task, adapting these successful patterns."
        
        return result
        
    except Exception as e:
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="pattern_retrieval_error",
            value=1.0,
            context={"error": str(e), "task_description": task_description}
        )
        return f"Error retrieving patterns: {str(e)}"

def execute_tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        print("DEBUG execute_tools: No tool calls found")
        return {"messages": state["messages"]}

    tool_calls = last_message.tool_calls
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
                    code = tool_args["code"]
                    print(f"DEBUG: Received clean code from Pydantic schema ({len(code)} chars)")
                    start_time = time.time()

                    # Start performance monitoring
                    from src.mlops.monitoring import PerformanceMonitor, MetricType
                    monitor = PerformanceMonitor()

                    result = execute_python_in_sandbox(code, session_id)

                    # Record performance metrics
                    execution_time = time.time() - start_time
                    monitor.record_metric(
                        deployment_id=session_id,
                        metric_type=MetricType.LATENCY,
                        value=execution_time * 1000,  # Convert to ms
                        metadata={"code_length": len(code), "tool": "python_code_interpreter"}
                    )

                    # Adaptive learning - capture execution data
                    try:
                        from src.learning.adaptive_system import AdaptiveLearningSystem
                        adaptive_system = AdaptiveLearningSystem()
                        adaptive_system.capture_execution_data(
                            session_id=session_id,
                            code=code,
                            execution_time=execution_time,
                            success=result.get('success', False),
                            output=result.get('stdout', ''),
                            error=result.get('stderr', ''),
                            context={'tool': 'python_code_interpreter'}
                        )
                    except Exception as adaptive_error:
                        print(f"Adaptive learning capture failed: {adaptive_error}")
                    execution_time = time.time() - start_time
                    performance_monitor.record_metric(
                        session_id=session_id, 
                        metric_name="code_execution_time", 
                        value=execution_time,
                        context={"code_length": len(code), "success": True}
                    )
                    
                    stdout_content = result.get('stdout', '')
                    stderr_content = result.get('stderr', '')
                    plots = result.get('plots', [])

                    output = []
                    if stdout_content:
                        output.append(stdout_content)
                    if stderr_content:
                        output.append(f"Error: {stderr_content}")
                    if plots:
                        output.append(f"\n📊 Generated {len(plots)} visualization(s)")

                        # Add business insights for visualizations
                        try:
                            insights = generate_business_insights(code, result.get('stdout', ''), session_id)
                            if insights:
                                output.append(f"\n💡 {insights}")
                        except Exception as e:
                            print(f"Insight generation failed: {e}")

                    try:
                        explanation = generate_explainability_insights(code, session_id)
                        if explanation:
                            output.append(f"\n🔍 {explanation}")
                    except:
                        pass

                    if output:
                        content = "\n".join(output)
                    else:
                        stdout_content = result.get('stdout', '').strip()
                        if stdout_content:
                            content = f"Code executed successfully.\n\n{stdout_content}"
                        else:
                            content = "Code executed successfully, but no output was generated."

                elif tool_name == 'retrieve_historical_patterns':
                    task_description = tool_args["task_description"]
                    content = retrieve_historical_patterns_logic(task_description, session_id)
                elif tool_name == 'knowledge_graph_query':
                    query = tool_args["query"]
                    content = knowledge_graph_query_logic(query, session_id)
                elif tool_name == 'access_learning_data':
                    query = tool_args["query"]
                    content = access_learning_data_logic(query, session_id)
                else:
                    content = f"Error: Unknown tool '{tool_name}'"
            except Exception as e:
                content = f"Execution failed in tool node: {str(e)}"
                
        print(f"DEBUG execute_tools: Tool {tool_name} FULL result:")
        print(content)
        print("DEBUG execute_tools: End of tool result")
        tool_responses.append(ToolMessage(content=content, tool_call_id=tool_id))
    
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

def create_brain_agent():
    """Create Brain agent for business reasoning and planning"""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_manager.get_model('brain'),
        base_url="http://localhost:11434",
        temperature=0.1
    )

def create_hands_agent():
    """Create Hands agent for code execution"""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_manager.get_model('hands'),
        base_url="http://localhost:11434",
        temperature=0.0
    )

def get_brain_prompt(data_context: str = "", has_recent_results: bool = False):
    """Enhanced business consultant prompt with dataset awareness"""
    result_awareness = ""
    if has_recent_results:
        result_awareness = """
TASK COMPLETION AWARENESS:
Your technical specialist has completed the requested work. Focus on:
1. Acknowledge the completed visualization/analysis
2. Interpret the results and provide business insights
3. Reference the specific output that was generated
4. Avoid providing example code since the task is already complete
Your role is to explain what was accomplished and its business value."""

    return ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert business data consultant with autonomous workflow intelligence and dataset awareness.

{data_context}

{result_awareness}

STATUS REPORTING PROTOCOL:
ALWAYS begin your response with a status line in this format:
STATUS: [Brief description of what you're currently doing]

Examples:
- STATUS: Analyzing request for sales trend visualization and determining optimal approach
- STATUS: Interpreting correlation analysis results and generating business insights
- STATUS: Planning comprehensive machine learning workflow for predictive modeling
- STATUS: Reviewing completed visualization and providing strategic recommendations

AUTONOMOUS WORKFLOW DETECTION:
You intelligently detect business objectives requiring complete ML workflows:
- "Understand drivers of X" → Comprehensive analysis workflow
- "Predict Y outcomes" → End-to-end modeling pipeline
- "Optimize Z performance" → Complete ML solution
- "Find patterns" → Exploratory + advanced analytics

INTELLIGENT DELEGATION:
For ANY technical work, use the delegate_coding_task tool with clear instructions.

TOOLS AVAILABLE:
- delegate_coding_task: For all coding, analysis, modeling, visualization tasks
- knowledge_graph_query: Access historical patterns and relationships
- access_learning_data: Get successful code patterns from past sessions

BUSINESS WORKFLOW INTELLIGENCE:
You can plan and execute complete ML workflows autonomously:
1. Detect business intent from user requests
2. Plan appropriate technical approach based on data characteristics
3. Delegate comprehensive technical implementation
4. Interpret results for business impact
5. Generate actionable recommendations

COMMUNICATION APPROACH:
- Business consultant tone focused on outcomes
- Detect intent and plan workflows without asking technical approval
- Delegate technical execution with clear business context
- Interpret results as actionable business insights"""),
        MessagesPlaceholder(variable_name="messages")
    ])

def get_hands_prompt(data_context: str = ""):
    """Technical execution prompt for Hands agent"""
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior data scientist assistant focused on technical execution with access to a dataset `df`.
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

                    STATUS REPORTING PROTOCOL:
                    Include a status comment at the start of your Python code to describe what you're implementing.
                    Use this format: # STATUS: [Brief description of what you're implementing]

                    Examples:
                    - # STATUS: Creating interactive pie charts for all dataset features
                    - # STATUS: Implementing correlation analysis with heatmap visualization
                    - # STATUS: Building machine learning pipeline with feature engineering
                    - # STATUS: Developing comprehensive data cleaning workflow

                    MANDATORY FORMAT - Always respond with this exact JSON structure:
                    {{{{
                    "name": "python_code_interpreter",
                    "arguments": {{{{
                        "code": "# STATUS: [Your status description]\\nyour_python_code_here"
                    }}}}
                    }}}}

                    TOOLS:
                    - python_code_interpreter: Execute Python code on the dataset

                    RULES:
                    - When you need to execute code: Return ONLY valid JSON (no explanatory text, no markdown)
                    - After a tool executes successfully: ALWAYS respond in conversational language explaining what was created or discovered
                    - When explaining results: Use natural, conversational language
                    - For all data analysis/visualization: Use python_code_interpreter
                    - Always use matplotlib.use('Agg') for plotting
                    - CRITICAL: Use actual column names from the dataset context above - never hardcode column names
                    - Never use plt.close() - always use plt.show() to let the plots be saved
                    - Put all Python code in the "code" field
                    - IMPORTANT: If you use markdown, ensure JSON is properly formatted within code blocks
                    - Follow the exact JSON structure shown in the example below
                    - Remember: First respond with JSON, then explain after execution

                    EXAMPLE:
                    {{{{
                    "name": "python_code_interpreter",
                    "arguments": {{{{
                        "code": "# STATUS: Creating correlation heatmap for numerical features\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\ncorr_matrix = df.corr()\\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm')\\nplt.title('Correlation Heatmap')\\nplt.show()"
                    }}}}
                    }}}}"""),
                            MessagesPlaceholder(variable_name="messages")
                        ])

def extract_brain_status(content: str) -> Optional[str]:
    """Extract status message from brain agent response."""
    import re

    match = re.search(r'STATUS:\s*(.+)', content)
    if match:
        status = match.group(1).strip()
        # Clean up the status message
        if status.endswith('.'):
            status = status[:-1]
        return f"🧠 {status}"
    return None

def extract_hands_status(content: str) -> Optional[str]:
    """Extract status message from hands agent code or response."""
    import re

    # Try to extract from JSON code field first
    json_match = re.search(r'"code":\s*"([^"]*#\s*STATUS:\s*[^"]*)"', content)
    if json_match:
        code_content = json_match.group(1)
        status_match = re.search(r'#\s*STATUS:\s*(.+?)\\n', code_content)
        if status_match:
            status = status_match.group(1).strip()
            return f"👨‍💻 {status}"

    # Try to extract from plain text
    status_match = re.search(r'#\s*STATUS:\s*(.+)', content)
    if status_match:
        status = status_match.group(1).strip()
        if status.endswith('\\n'):
            status = status[:-2]
        return f"👨‍💻 {status}"

    return None

def extract_parser_status(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Generate status message for parser/tool execution."""
    if tool_name == "python_code_interpreter":
        return "⚙️ Executing Python code in secure sandbox..."
    elif tool_name == "delegate_coding_task":
        task = tool_args.get("task_description", "task")
        return f"🔄 Delegating to specialist: {task[:50]}..."
    elif tool_name == "knowledge_graph_query":
        return "🔍 Querying knowledge graph for insights..."
    elif tool_name == "access_learning_data":
        return "📚 Accessing historical learning patterns..."
    else:
        return f"⚙️ Processing {tool_name}..."

def create_lean_hands_context(state: AgentState, task_description: str, session_id: str) -> Dict[str, Any]:
    messages = state.get("messages", [])
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
        if hasattr(msg, 'content') and 'STATUS:' in str(msg.content):
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

    print(f"DEBUG: Context pruning - Original: {original_msg_count} messages, Pruned: {lean_msg_count} messages ({reduction:.1f}% reduction)")

    return lean_state

def create_agent_executor():
    """Creates the multi-agent system with Brain and Hands specialization."""

    from langchain_ollama import ChatOllama
    
    
    def parse_tool_calls(state: AgentState):
        """
        Parses tool calls from the last AI message's content and adds them
        to the .tool_calls attribute.
        """
        import json
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return state

        content_str = str(last_message.content).strip()
        
        try:
            import re
            json_str = content_str
            print(f"DEBUG: Parser input: {content_str[:150]}...")

            # Handle markdown code blocks more robustly
            if '```json' in json_str:
                # Extract JSON from ```json blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    print(f"DEBUG: Extracted JSON from markdown: {json_str[:100]}...")
                else:
                    # Try without closing ```
                    json_match = re.search(r'```json\s*(\{.*)', json_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).rstrip('`').strip()
                        print(f"DEBUG: Extracted JSON (no closing): {json_str[:100]}...")

            elif '```python' in json_str:
                # Extract Python code and auto-wrap in JSON
                code_match = re.search(r'```python\s*(.*?)\s*```', json_str, re.DOTALL)
                if code_match:
                    code_content = code_match.group(1).strip()
                    json_str = f'{{"name": "python_code_interpreter", "arguments": {{"code": "{code_content.replace(chr(34), chr(92)+chr(34)).replace(chr(10), chr(92)+"n")}"}}}}'
                    print(f"DEBUG: Auto-wrapped Python code in JSON")

            # Try to find JSON object in the content if not already found
            if not json_str.startswith('{') and '{' in json_str:
                json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    print(f"DEBUG: Found JSON object: {json_str[:100]}...")
            
            tool_data = json.loads(json_str)
            
            if 'name' in tool_data and 'arguments' in tool_data:
                tool_name = tool_data['name']
                tool_args = tool_data['arguments']
                
                new_message = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"json_{tool_name}"
                    }]
                )
                state["messages"][-1] = new_message
                return state
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"DEBUG: JSON parsing failed: {e}")
            print(f"DEBUG: Content: {content_str[:200]}...")
        
        # Fallback: Handle ReAct format: Thought: ... Action: ... Action Input: ...
        if "Action:" in content_str and "Action Input:" in content_str:
            import re
            
            # Extract Action (tool name)
            action_match = re.search(r'Action:\s*([^\n]+)', content_str)
            if not action_match:
                return state
            
            tool_name = action_match.group(1).strip()
            
            # Extract Action Input (code/query) - improved to handle multiline code
            action_input_match = re.search(r'Action Input:\s*(.*?)(?=\n(?:Thought|Action|Observation)|\Z)', content_str, re.DOTALL)
            if not action_input_match:
                return state
                
            action_input = action_input_match.group(1).strip()
            
            # Create tool call based on ReAct format
            if tool_name == "python_code_interpreter":
                tool_args = {"code": action_input}
            elif tool_name == "retrieve_historical_patterns":
                tool_args = {"task_description": action_input}
            else:
                return state
                
            new_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": tool_args,
                    "id": f"react_{tool_name}"
                }]
            )
            state["messages"][-1] = new_message
            return state
        
        # Fallback: Handle JSON format if present
        if content_str.startswith("{") and '"name":' in content_str:
            try:
                # Handle multiline strings by using regex to extract name and args
                import re
                
                # Extract tool name
                name_match = re.search(r'"name":\s*"([^"]+)"', content_str)
                if not name_match:
                    raise ValueError("No tool name found")
                
                tool_name = name_match.group(1)
                
                # Extract arguments section
                args_match = re.search(r'"arguments":\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', content_str, re.DOTALL)
                if not args_match:
                    # Try simpler pattern for code argument
                    code_match = re.search(r'"code":\s*"""([^"]+(?:"""[^"]*"""[^"]*)*?)"""', content_str, re.DOTALL)
                    if code_match:
                        tool_args = {"code": code_match.group(1)}
                    else:
                        tool_args = {}
                else:
                    # Try to parse the arguments
                    args_str = args_match.group(1)
                    try:
                        tool_args = json.loads("{" + args_str + "}")
                    except:
                        # Fallback: extract code from triple quotes
                        code_match = re.search(r'"code":\s*"""([^"]+(?:"""[^"]*"""[^"]*)*?)"""', content_str, re.DOTALL)
                        if code_match:
                            tool_args = {"code": code_match.group(1)}
                        else:
                            tool_args = {}
                
                new_message = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"call_{tool_name}"
                    }]
                )
                state["messages"][-1] = new_message
            except Exception as e:
                pass
            
        return state

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

        print(f"DEBUG: Brain agent - recent tool result found: {recent_tool_result is not None}")
        if recent_tool_result:
            print(f"DEBUG: Tool result preview: {str(recent_tool_result)[:200]}...")

        data_context = ""
        try:
            import builtins
            if (hasattr(builtins, '_session_store') and
                session_id in builtins._session_store and
                'data_profile' in builtins._session_store[session_id]):

                data_profile = builtins._session_store[session_id]['data_profile']
                column_context = data_profile.ai_agent_context['column_details']

                data_context = f"""
                                DATASET CONTEXT:
                                Available dataset with columns:
                                {chr(10).join(f'• {col}: {info["dtype"]} ({info["semantic_type"]}) - {info["null_count"]} nulls, {info["unique_count"]} unique values'
                                                            for col, info in column_context.items())}

                                BUSINESS INSIGHTS FROM DATA:
                                • Numeric columns: {', '.join(data_profile.ai_agent_context['code_generation_hints']['numeric_columns'])}
                                • Categorical columns: {', '.join(data_profile.ai_agent_context['code_generation_hints']['categorical_columns'])}
                                • Suggested targets: {', '.join(data_profile.ai_agent_context['code_generation_hints']['suggested_target_columns'])}
                                """
        except Exception as e:
            print(f"WARNING: Brain agent failed to get dataset context: {e}")
            print(f"DEBUG: Brain session_id: {session_id}")

        print(f"DEBUG: Brain data context length: {len(data_context)} chars")
        if data_context:
            print(f"DEBUG: Brain context preview: {data_context[:200]}...")
        else:
            print("DEBUG: Brain has NO data context")

        llm = create_brain_agent()
        brain_tools = [delegate_coding_task, knowledge_graph_query, access_learning_data]
        llm_with_tools = llm.bind_tools(brain_tools)
        prompt = get_brain_prompt(data_context, has_recent_results=(recent_tool_result is not None))
        agent_runnable = prompt | llm_with_tools

        try:
            response = agent_runnable.invoke(enhanced_state)
            print(f"DEBUG: Brain agent response type: {type(response)}")
            print(f"DEBUG: Brain agent content preview: {str(response.content)[:200]}...")
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

        # Extract delegation task from Brain's tool call
        last_message = state["messages"][-1] if state["messages"] else None
        task_description = "Perform data analysis task"

        if (last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            tool_call = last_message.tool_calls[0]
            if tool_call.get('name') == 'delegate_coding_task':
                task_description = tool_call.get('args', {}).get('task_description', task_description)

        # Get dataset context for Hands agent
        data_context = ""
        try:
            import builtins
            if (hasattr(builtins, '_session_store') and
                session_id in builtins._session_store and
                'data_profile' in builtins._session_store[session_id]):

                data_profile = builtins._session_store[session_id]['data_profile']
                column_context = data_profile.ai_agent_context['column_details']

                data_context = f"""
                                DATASET CONTEXT:
                                Available dataset with columns:
                                {chr(10).join(f'• {col}: {info["dtype"]} ({info["semantic_type"]}) - {info["null_count"]} nulls, {info["unique_count"]} unique values'
                                                            for col, info in column_context.items())}

                                TECHNICAL INSIGHTS:
                                • Numeric columns: {', '.join(data_profile.ai_agent_context['code_generation_hints']['numeric_columns'])}
                                • Categorical columns: {', '.join(data_profile.ai_agent_context['code_generation_hints']['categorical_columns'])}
                                • Suggested targets: {', '.join(data_profile.ai_agent_context['code_generation_hints']['suggested_target_columns'])}
                                """
        except Exception as e:
            print(f"WARNING: Failed to get dataset context: {e}")
            print(f"DEBUG: Session ID: {session_id}")
            print(f"DEBUG: Has _session_store: {hasattr(builtins, '_session_store') if 'builtins' in locals() else 'builtins not imported'}")
            data_context = "No dataset context available"

        print(f"DEBUG: Data context length: {len(data_context)} chars")
        print(f"DEBUG: Data context preview: {data_context[:200]}...")
        print(f"DEBUG: Hands agent starting with session_id: {session_id}")

        # Create lean context for faster inference
        hands_state = create_lean_hands_context(state, task_description, session_id)

        llm = create_hands_agent()
        prompt = get_hands_prompt(data_context)
        agent_runnable = prompt | llm

        try:
            response = agent_runnable.invoke(hands_state)
            print(f"DEBUG: Hands agent response type: {type(response)}")
            print(f"DEBUG: Hands agent content preview: {str(response.content)[:300]}...")
            print(f"DEBUG: Hands response starts with JSON? {str(response.content).strip().startswith('{')}")
            print(f"DEBUG: Hands response contains 'python_code_interpreter'? {'python_code_interpreter' in str(response.content)}")

            result_state = {
                "messages": [response],
                "current_agent": "hands",
                "last_agent_sequence": enhanced_state["last_agent_sequence"],
                "retry_count": enhanced_state["retry_count"]
            }
            return result_state
        except Exception as e:
            return {"messages": [AIMessage(content=f"Hands agent error: {e}")]}

    def route_to_agent(state: AgentState):
        last_message = state["messages"][-1] if state["messages"] else None

        if not last_message:
            return "brain"

        if isinstance(last_message, ToolMessage):
            return "brain"

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call.get('name') == 'delegate_coding_task':
                return "hands"

        return "brain"


    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        python_executions = state.get("python_executions", 0)

        if python_executions >= 7:
            print("DEBUG: Termination condition met: Maximum python executions reached.")
            return END

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:

            if len(messages) >= 5:  # Need more messages to detect true recursion
                last_tool_call = last_message.tool_calls[0]

                # Check if there was a human message between the last two tool calls
                # True recursion only happens when AI makes consecutive identical tool calls
                # without new human input
                consecutive_identical_calls = 0

                for i in range(len(messages) - 2, -1, -1):
                    msg = messages[i]

                    # If we find a human message, reset the counter (new user input breaks recursion)
                    if hasattr(msg, 'type') and msg.type == 'human':
                        break

                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        prev_tool_call = msg.tool_calls[0]
                        if (last_tool_call.get('name') == prev_tool_call.get('name') and
                            last_tool_call.get('args') == prev_tool_call.get('args')):
                            consecutive_identical_calls += 1
                        break

                # Only terminate if we have multiple consecutive identical calls (true recursion)
                if consecutive_identical_calls >= 2:
                    print(f"DEBUG: Termination condition met: Detected recursive tool call for '{last_tool_call.get('name')}'.")
                    return END

            return "action"
        return END 

    def route_after_agent(state: AgentState):
        """Route based on whether agent wants to use tools or just respond"""
        last_message = state["messages"][-1]
        content = str(last_message.content).strip()

        print(f"DEBUG: Routing after agent. Content preview: {content[:100]}...")
        print(f"DEBUG: Has tool_calls? {hasattr(last_message, 'tool_calls') and last_message.tool_calls}")
        print(f"DEBUG: Starts with {{? {content.startswith('{')}")

        name_check = '"name":' in content
        print(f"DEBUG: Contains 'name'? {name_check}")
        print(f"DEBUG: Contains python_code_interpreter? {'python_code_interpreter' in content}")

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print("DEBUG: Routing to parser (tool_calls)")
            return "parser"

        if (content.startswith("{") and '"name":' in content) or \
           ("python_code_interpreter" in content and '"code":' in content):
            print("DEBUG: Routing to parser (JSON format)")
            return "parser"

        print("DEBUG: Routing to END")
        return END

    graph = StateGraph(AgentState)
    graph.add_node("brain", run_brain_agent)
    graph.add_node("hands", run_hands_agent)
    graph.add_node("parser", parse_tool_calls)
    graph.add_node("action", execute_tools_node)
    graph.set_entry_point("brain")

    def route_from_brain(state: AgentState):
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
            tool_call = last_message.tool_calls[0]
            if tool_call.get('name') == 'delegate_coding_task':
                return "hands"
            else:
                return "parser"

        return END

    graph.add_conditional_edges(
        "brain",
        route_from_brain,
        {
            "hands": "hands",
            "parser": "parser",
            END: END
        }
    )

    graph.add_conditional_edges(
        "hands",
        route_after_agent,
        {
            "parser": "parser",
            END: END
        }
    )

    graph.add_conditional_edges(
        "parser",
        should_continue,
        {
            "action": "action",
            END: END
        }
    )

    graph.add_edge("action", "brain")
    
    if CHECKPOINTER_AVAILABLE and memory:
        agent_executor = graph.compile(checkpointer=memory)
    else:
        agent_executor = graph.compile()
    
    return agent_executor

def create_enhanced_agent_executor(session_id: str = None):
    return create_agent_executor()