import os
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Sequence, Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, MessagesState, add_messages
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

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    session_id: str
    python_executions: int
    plan: Optional[str]
    scratchpad: str

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

def generate_business_insights(code: str, output: str, session_id: str) -> str:
    try:
        import sys, os, builtins
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        from insights.business_translator import BusinessTranslator, StakeholderType
        from insights.narrative_generator import NarrativeGenerator, ReportTone

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
        import sys, os, builtins
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        from explainability.bias_detector import BiasDetector

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
    content_str = str(last_message.content) if last_message.content else ""
    print(f"DEBUG execute_tools: Parsing content: {content_str[:200]}...")
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_calls = last_message.tool_calls
        print(f"DEBUG execute_tools: Found {len(tool_calls)} tool calls in message")
    else:
        tool_calls = []
        import json
        import re
        
        try:
            json_start = content_str.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(content_str[json_start:], json_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                json_str = content_str[json_start:json_end]
                print(f"DEBUG execute_tools: Attempting to parse JSON: {json_str[:200]}...")
                
                if '"""' in json_str:
                    print("DEBUG execute_tools: Fixing triple-quoted strings for JSON parsing")
                    import re
                    code_match = re.search(r'"code":\s*"""([^"]+(?:"[^"]+)*)"""', json_str, re.DOTALL)
                    if code_match:
                        code_content = code_match.group(1).strip()
                        escaped_code = json.dumps(code_content)
                        json_str = re.sub(r'"code":\s*"""[^"]+(?:"[^"]+)*"""', f'"code": {escaped_code}', json_str, flags=re.DOTALL)
                        print(f"DEBUG execute_tools: Fixed JSON: {json_str[:200]}...")
                
                try:
                    tool_data = json.loads(json_str)
                    if 'name' in tool_data and 'arguments' in tool_data:
                        tool_name = tool_data['name']
                        args_data = tool_data['arguments']
                        
                        print(f"DEBUG execute_tools: Tool name: {tool_name}")
                        print(f"DEBUG execute_tools: Arguments type: {type(args_data)}")
                        
                        tool_calls.append({
                            "name": tool_name,
                            "args": args_data,
                            "id": f"call_{len(tool_calls)}"
                        })
                        print(f"DEBUG execute_tools: Successfully extracted tool call: {tool_name}")
                        
                except json.JSONDecodeError as e:
                    print(f"DEBUG execute_tools: JSON parse error: {e}")
                    print(f"DEBUG execute_tools: Failed JSON: {json_str}")
                
        except Exception as e:
            print(f"DEBUG execute_tools: Failed to parse tool calls: {e}")
            return {"messages": state["messages"]}
    
    if not tool_calls:
        print("DEBUG execute_tools: No tool calls found")
        return {"messages": state["messages"]}
    
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
                    result = execute_python_in_sandbox(code, session_id)
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
        "scratchpad": state.get("scratchpad", "")
    }
    print(f"DEBUG execute_tools_node: Python executions count: {python_executions}")
    print(f"DEBUG execute_tools_node: Returning state with {len(result_state['messages'])} messages")
    return result_state

def create_agent_executor():
    """Creates the stateful agent by manually constructing the graph."""
    
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(
        model="qwen2.5-coder:14b",
        base_url="http://localhost:11434",
        temperature=0.0
    )
    tools = [python_code_interpreter, retrieve_historical_patterns]
    llm_with_tools = llm  

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior data scientist assistant with access to a dataset `df`.

            THINKING PROCESS (Internal - use for complex tasks):
            1. **PLAN:** Outline approach for complex multi-step tasks
            2. **EXECUTE:** Execute step-by-step using python_code_interpreter
            3. **SYNTHESIZE:** Note findings after each execution
            4. **RESPOND:** Provide clear explanations to user

            When you need to analyze data or create visualizations, use python_code_interpreter tool.
            After the tool executes successfully, provide a clear explanation of what was created.

            TOOLS:
            - python_code_interpreter: Execute Python code on the dataset

            RULES:
            - Think step-by-step and plan your approach for complex tasks
            - For all data analysis/visualization: Use python_code_interpreter
            - Always use matplotlib.use('Agg') for plotting
            - Use actual column names from the data
            - Never use plt.close() - always use plt.show()
            - After tool execution, explain results clearly to the user"""),
            MessagesPlaceholder(variable_name="messages"),])
    agent_runnable = prompt | llm_with_tools
    
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
            json_str = content_str
            
            # Remove any markdown code blocks if present
            if json_str.startswith('```json'):
                import re
                json_str = re.sub(r'^```json\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)
            elif json_str.startswith('```'):
                import re
                json_str = re.sub(r'^```\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)
            
            # Try to find JSON object in the content
            if not json_str.startswith('{'):
                import re
                json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
            
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

    def run_agent(state: AgentState):
        """Invoke the agent with the current state and inject data profile context"""
        python_executions = state.get("python_executions", 0)
        session_id = state.get("session_id")
        
        # Inject data profile context if available
        enhanced_state = state.copy()

        # Initialize plan and scratchpad if not present
        if 'plan' not in enhanced_state:
            enhanced_state['plan'] = None
        if 'scratchpad' not in enhanced_state:
            enhanced_state['scratchpad'] = ""
        
        try:
            import builtins
            if (hasattr(builtins, '_session_store') and 
                session_id in builtins._session_store and 
                'data_profile' in builtins._session_store[session_id]):
                
                data_profile = builtins._session_store[session_id]['data_profile']
                column_context = data_profile.ai_agent_context['column_details']
                
                context_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are a senior data scientist assistant with access to a dataset `df`.

                    THINKING PROCESS (Internal - use for complex tasks):
                    1. **PLAN:** Outline approach for complex multi-step tasks
                    2. **EXECUTE:** Execute step-by-step using python_code_interpreter
                    3. **SYNTHESIZE:** Note findings after each execution
                    4. **RESPOND:** Provide clear explanations to user

                    Current plan: {state.get('plan', 'None set yet')}
                    Current scratchpad: {state.get('scratchpad', 'Empty')}

                    DATASET CONTEXT:
                    You have access to a dataset with the following columns and their details:
                    {chr(10).join(f'• {col}: {info["dtype"]} ({info["semantic_type"]}) - {info["null_count"]} nulls, {info["unique_count"]} unique values'
                                for col, info in column_context.items())}

                    VISUALIZATION SUGGESTIONS:
                    {chr(10).join(f'• {sug["type"]}: {sug["description"]}' for sug in data_profile.ai_agent_context.get('visualization_suggestions', [])[:3])}

                    CODE GENERATION HINTS:
                    • Numeric columns: {', '.join(data_profile.ai_agent_context['code_generation_hints']['numeric_columns'])}
                    • Categorical columns: {', '.join(data_profile.ai_agent_context['code_generation_hints']['categorical_columns'])}
                    • Suggested targets: {', '.join(data_profile.ai_agent_context['code_generation_hints']['suggested_target_columns'])}

                    MANDATORY FORMAT - Always respond with this exact JSON structure:

                    {{{{
                    "name": "python_code_interpreter",
                    "arguments": {{{{
                        "code": "your_python_code_here"
                    }}}}
                    }}}}

                    TOOLS:
                    - python_code_interpreter: Execute Python code on the dataset

                    RULES:
                    - IMPORTANT: After a tool executes successfully, ALWAYS respond in conversational language explaining what was created or discovered
                    - When you need to execute code: Return ONLY valid JSON
                    - When explaining results: Use natural, conversational language
                    - For all data analysis/visualization: Use python_code_interpreter
                    - Always use matplotlib.use('Agg') for plotting
                    - Use actual column names from the dataset context above
                    - Never use plt.close() - always use plt.show() to let the plots be saved
                    - Put all Python code in the "code" field

                    EXAMPLE:
                    {{{{
                    "name": "python_code_interpreter",
                    "arguments": {{{{
                        "code": "import matplotlib.pyplot as plt\\nimport seaborn as sns\\ncorr_matrix = df.corr()\\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm')\\nplt.title('Correlation Heatmap')\\nplt.show()"
                    }}}}
                    }}}}"""),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                
                enhanced_agent = context_prompt | llm_with_tools
                agent_outcome = enhanced_agent.invoke(enhanced_state)
                
            else:
                agent_outcome = agent_runnable.invoke(enhanced_state)
                
        except Exception as e:
            agent_outcome = agent_runnable.invoke(enhanced_state)
            
        return {
            "messages": enhanced_state["messages"] + [agent_outcome],
            "python_executions": enhanced_state.get("python_executions", 0),
            "plan": enhanced_state.get("plan"),
            "scratchpad": enhanced_state.get("scratchpad", "")
        }

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        python_executions = state.get("python_executions", 0)

        if python_executions >= 7:
            print("DEBUG: Termination condition met: Maximum python executions reached.")
            return END

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:

            if len(messages) >= 3:
                last_tool_call = last_message.tool_calls[0]
                # Find the previous AI message with tool calls
                for i in range(len(messages) - 2, -1, -1):
                    msg = messages[i]
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        prev_tool_call = msg.tool_calls[0]
                        # If the tool name and arguments are the same, it's a loop
                        if (last_tool_call.get('name') == prev_tool_call.get('name') and
                            last_tool_call.get('args') == prev_tool_call.get('args')):
                            print(f"DEBUG: Termination condition met: Detected recursive tool call for '{last_tool_call.get('name')}'.")
                            return END
                        break  # Found the last tool call, no need to look further back

            return "action"
        return END 

    def route_after_agent(state: AgentState):
        """Route based on whether agent wants to use tools or just respond"""
        last_message = state["messages"][-1]
        content = str(last_message.content).strip()

        # Check if response contains tool usage patterns (including markdown wrapped JSON)
        if (content.startswith("{") and '"name":' in content) or \
           ("python_code_interpreter" in content and '"code":' in content) or \
           (content.startswith("```json") and '"name":' in content) or \
           (content.startswith("```") and '"name":' in content):
            return "parser"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", run_agent)
    graph.add_node("parser", parse_tool_calls)
    graph.add_node("action", execute_tools_node)
    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
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
    graph.add_edge("action", "agent")
    
    if CHECKPOINTER_AVAILABLE and memory:
        agent_executor = graph.compile(checkpointer=memory)
    else:
        agent_executor = graph.compile()
    
    return agent_executor

def create_enhanced_agent_executor(session_id: str = None):
    return create_agent_executor()