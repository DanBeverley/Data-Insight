import os
import json
import sqlite3
from pathlib import Path
from typing import TypedDict, Sequence, Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, MessagesState, add_messages
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    CHECKPOINTER_AVAILABLE = True
except ImportError:
    SqliteSaver = None
    CHECKPOINTER_AVAILABLE = False
from .tools import execute_python_in_sandbox
from .context_manager import ContextManager, ConversationContext
from .performance_monitor import PerformanceMonitor

import time

load_dotenv()

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    session_id: str

performance_monitor = PerformanceMonitor()
context_manager = ContextManager()

@tool
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

@tool
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
            return f"No relevant historical patterns found for '{task_description}'. Proceeding with standard approach."
        
        pattern_summaries = []
        for pattern in patterns:
            summary = f"**{pattern['type']}** (confidence: {pattern['confidence']:.2f}, used {pattern['success_count']} times)"
            if pattern.get('data'):
                data_preview = str(pattern['data'])[:200]
                summary += f"\nPattern details: {data_preview}{'...' if len(str(pattern['data'])) > 200 else ''}"
            pattern_summaries.append(summary)
        
        result = f"Found {len(patterns)} proven patterns for '{task_description}':\n\n" + "\n\n".join(pattern_summaries)
        result += "\n\nConsider adapting these successful patterns for your current task."
        
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
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": state["messages"]}
    
    tool_responses = []
    for tool_call in last_message.tool_calls:
        session_id = state.get("session_id")
        tool_name = tool_call['name']
        
        if not session_id:
            content = "Error: Session ID is missing."
        else:
            try:
                if tool_name == 'python_code_interpreter':
                    code = tool_call["args"]["code"]
                    start_time = time.time()
                    result = execute_python_in_sandbox(code, session_id)
                    execution_time = time.time() - start_time
                    performance_monitor.record_metric(
                        session_id=session_id, 
                        metric_name="code_execution_time", 
                        value=execution_time,
                        context={"code_length": len(code), "success": True}
                    )
                    output = []
                    if result.get("stdout"): 
                        output.append(result['stdout'])
                    if result.get("stderr"): 
                        output.append(f"Error: {result['stderr']}")
                    if result.get("plots"): 
                        output.append(f"\n📊 Generated {len(result['plots'])} visualization(s)")
                    content = "\n".join(output) or "Code executed successfully."

                elif tool_name == 'retrieve_historical_patterns':
                    task_description = tool_call["args"]["task_description"]
                    content = retrieve_historical_patterns_logic(task_description, session_id)
                else:
                    content = f"Error: Unknown tool '{tool_name}'"
            except Exception as e:
                content = f"Execution failed in tool node: {str(e)}"
                
        tool_responses.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
    
    return {"messages": state["messages"] + tool_responses}

def create_agent_executor():
    """Creates the stateful agent by manually constructing the graph."""
    
    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434",
        temperature=0.0
    )
    tools = [python_code_interpreter, retrieve_historical_patterns]
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data scientist AI assistant.
        - When you need to execute code, use the `python_code_interpreter` tool.
        - The user's dataset is always available in a pandas DataFrame called `df`.
        - When asked to create a plot, generate the code and inform the user that a visualization has been created. The plot will be displayed automatically.
        - Before complex tasks like EDA or modeling, consider using the `retrieve_historical_patterns` tool to see how similar problems were solved successfully in the past.
        - Analyze the results of your code execution and provide a clear, concise explanation to the user."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    agent_runnable = prompt | llm_with_tools

    def run_agent(state: AgentState):
        """Invoke the agent with the current state"""
        agent_outcome = agent_runnable.invoke(state)
        return {"messages": [agent_outcome]}

    def should_continue(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        if len(messages) > 25:
            return "end"
            
        last_message = messages[-1]
        has_tool_calls = (hasattr(last_message, 'tool_calls') and 
                         last_message.tool_calls and 
                         len(last_message.tool_calls) > 0)
        
        return "continue" if has_tool_calls else "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", run_agent)
    graph.add_node("action", execute_tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END},
    )
    graph.add_edge("action", "agent")
    class AgentState(MessagesState):
        session_id:str
    agent_executor = graph.compile()
    
    return agent_executor

def create_enhanced_agent_executor(session_id: str = None):
    return create_agent_executor()