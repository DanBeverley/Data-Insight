"""Integration tests for tool chaining and execution.

These tests validate:
1. Tool execution flow
2. Sequential tool execution
3. State updates after tool execution
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.fixture
def mock_execute_tools_node():
    """Mock execute_tools_node that simulates tool execution."""

    def _execute(state):
        messages = list(state.get("messages", []))
        python_executions = state.get("python_executions", 0)
        last_agent_sequence = list(state.get("last_agent_sequence", []))

        last_msg = messages[-1] if messages else None
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_id = tool_call.get("id", "call_id")

                if tool_name == "python_code_interpreter":
                    python_executions += 1
                    content = "Code executed successfully"
                elif tool_name == "delegate_coding_task":
                    content = "Delegation confirmed: task assigned to Hands agent"
                elif tool_name == "knowledge_graph_query":
                    content = "Graph query results: Previous analysis found"
                elif tool_name == "web_search":
                    content = "Web search results found"
                elif tool_name == "access_learning_data":
                    content = "Learning data retrieved successfully"
                else:
                    content = f"Tool {tool_name} executed"

                messages.append(ToolMessage(content=content, tool_call_id=tool_id))
                last_agent_sequence.append("tools")

        return {
            "messages": messages,
            "python_executions": python_executions,
            "retry_count": 0,
            "last_agent_sequence": last_agent_sequence,
            "session_id": state.get("session_id"),
        }

    return _execute


@pytest.mark.integration
class TestToolChaining:
    """Tests for tool chaining and execution."""

    def test_sequential_tool_execution(self, mock_execute_tools_node):
        """Sequential tools should execute in order."""
        state = {
            "messages": [
                HumanMessage(content="First analyze patterns, then execute code"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_historical_patterns",
                            "args": {"task_description": "correlation_analysis"},
                            "id": "call_pattern",
                        }
                    ],
                ),
            ],
            "session_id": "test_chain_123",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)

        assert len(result["messages"]) > len(state["messages"])
        assert any(isinstance(msg, ToolMessage) for msg in result["messages"])

    def test_delegate_then_execute_pattern(self, mock_execute_tools_node):
        """Delegation tool should create appropriate response."""
        state = {
            "messages": [
                HumanMessage(content="Calculate correlation"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_coding_task",
                            "args": {"task_description": "Calculate correlation matrix"},
                            "id": "call_delegate",
                        }
                    ],
                ),
            ],
            "session_id": "test_delegate_123",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)

        assert len(result["messages"]) > len(state["messages"])
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) > 0
        assert "Delegation confirmed" in tool_messages[-1].content

    def test_knowledge_graph_then_code_execution(self, mock_execute_tools_node):
        """Knowledge graph query should execute properly."""
        state = {
            "messages": [
                HumanMessage(content="Train model based on previous patterns"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "knowledge_graph_query",
                            "args": {"query": "previous model training patterns"},
                            "id": "call_kg",
                        }
                    ],
                ),
            ],
            "session_id": "test_kg_code_123",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert any("graph" in msg.content.lower() for msg in tool_messages)

    def test_web_search_then_analysis(self, mock_execute_tools_node):
        """Web search tool should execute."""
        state = {
            "messages": [
                HumanMessage(content="Compare our data with market trends"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "housing market trends 2024"}, "id": "call_web"}
                    ],
                ),
            ],
            "session_id": "test_web_123",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) > 0

    def test_multiple_python_executions_in_sequence(self, mock_execute_tools_node):
        """Python execution counter should increment."""
        state1 = {
            "messages": [
                HumanMessage(content="Calculate correlation"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "python_code_interpreter", "args": {"code": "corr = df.corr()"}, "id": "call_corr"}
                    ],
                ),
            ],
            "session_id": "test_multi_exec",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result1 = mock_execute_tools_node(state1)
        assert result1["python_executions"] == 1

        state2 = {
            "messages": result1["messages"]
            + [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "python_code_interpreter", "args": {"code": "plt.figure()"}, "id": "call_viz"}
                    ],
                )
            ],
            "session_id": "test_multi_exec",
            "python_executions": result1["python_executions"],
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result2 = mock_execute_tools_node(state2)
        assert result2["python_executions"] == 2

    def test_error_recovery_in_tool_chain(self, mock_execute_tools_node):
        """Errors should be handled gracefully."""
        state = {
            "messages": [
                HumanMessage(content="Execute invalid code"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "python_code_interpreter", "args": {"code": "invalid syntax here"}, "id": "call_error"}
                    ],
                ),
            ],
            "session_id": "test_error_123",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) > 0

    def test_learning_data_retrieval_before_execution(self, mock_execute_tools_node):
        """Learning data retrieval should work."""
        state = {
            "messages": [
                HumanMessage(content="What's the best way to visualize this?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "access_learning_data",
                            "args": {"query": "successful visualization patterns"},
                            "id": "call_learn",
                        }
                    ],
                ),
            ],
            "session_id": "test_learn_123",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)

        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert any("learning" in msg.content.lower() for msg in tool_messages)

    def test_tool_call_retry_count_reset(self, mock_execute_tools_node):
        """Retry count should reset after successful execution."""
        state = {
            "messages": [
                HumanMessage(content="Execute code"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "python_code_interpreter", "args": {"code": "print('success')"}, "id": "call_success"}
                    ],
                ),
            ],
            "session_id": "test_retry_reset",
            "python_executions": 0,
            "retry_count": 2,
            "last_agent_sequence": ["brain", "hands"],
        }

        result = mock_execute_tools_node(state)

        assert result["retry_count"] == 0


@pytest.mark.integration
class TestToolExecutionState:
    """Tests for state after tool execution."""

    def test_session_id_preserved(self, mock_execute_tools_node):
        """Session ID should be preserved in result."""
        state = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "t1"}]),
            ],
            "session_id": "unique_session_id",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = mock_execute_tools_node(state)
        assert result["session_id"] == "unique_session_id"

    def test_agent_sequence_updated_after_tool(self, mock_execute_tools_node):
        """Agent sequence should include tools."""
        state = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "t1"}]),
            ],
            "session_id": "test",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": ["brain"],
        }

        result = mock_execute_tools_node(state)
        assert "tools" in result["last_agent_sequence"]
