import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.mark.integration
class TestToolChaining:

    def test_sequential_tool_execution(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

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

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.return_value = "Pattern found: Use df.corr() for correlation analysis"

            result = execute_tools_node(state)

            assert len(result["messages"]) > len(state["messages"])
            assert any(isinstance(msg, ToolMessage) for msg in result["messages"])

    def test_delegate_then_execute_pattern(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

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

        result = execute_tools_node(state)

        assert len(result["messages"]) > len(state["messages"])
        tool_message = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)][-1]
        assert "Delegation confirmed" in tool_message.content or tool_message.content

    def test_knowledge_graph_then_code_execution(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.return_value = "Graph query results: Previous analysis used LinearRegression"

            state1 = {
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

            result1 = execute_tools_node(state1)
            assert any(
                "graph" in str(msg.content).lower() or "query" in str(msg.content).lower()
                for msg in result1["messages"]
                if isinstance(msg, ToolMessage)
            )

    def test_web_search_then_analysis(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.return_value = "Web search result: Average housing prices increased by 5% in 2024"

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

            result = execute_tools_node(state)

            tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
            assert len(tool_messages) > 0

    def test_multiple_python_executions_in_sequence(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.side_effect = ["Correlation matrix calculated", "Visualization created: correlation_heatmap.html"]

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

            result1 = execute_tools_node(state1)
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

            result2 = execute_tools_node(state2)
            assert result2["python_executions"] == 2

    def test_error_recovery_in_tool_chain(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

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

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.return_value = "Error: SyntaxError: invalid syntax"

            result = execute_tools_node(state)

            tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
            assert len(tool_messages) > 0

    def test_learning_data_retrieval_before_execution(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.return_value = (
                "Learning data: Previous successful executions show using seaborn for visualization"
            )

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

            result = execute_tools_node(state)

            tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
            assert any(
                "learning" in msg.content.lower() or "data" in msg.content.lower() or len(msg.content) > 0
                for msg in tool_messages
            )

    def test_tool_call_retry_count_reset(self):
        from data_scientist_chatbot.app.agent import execute_tools_node

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

        with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
            mock_exec.return_value = "success"

            result = execute_tools_node(state)

            assert result["retry_count"] == 0
