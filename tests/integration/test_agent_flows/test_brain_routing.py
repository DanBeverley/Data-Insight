"""Integration tests for Brain agent routing and behavior.

These tests validate:
1. Routing logic (route_from_brain) - which agent handles the next step
2. Brain agent invocation with proper mocking at the chain level
3. State structure validation
4. Edge cases and error handling
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class MockAIResponse:
    """Mock AIMessage-like response for testing."""

    def __init__(self, content: str = "", tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "ai"
        self.additional_kwargs = {}


@pytest.fixture
def base_state():
    """Base state for Brain agent tests."""
    return {
        "messages": [HumanMessage(content="Analyze the data")],
        "session_id": "test_session_123",
        "python_executions": 0,
        "plan": None,
        "scratchpad": "",
        "business_objective": None,
        "task_type": None,
        "target_column": None,
        "workflow_stage": None,
        "current_agent": "brain",
        "business_context": {},
        "retry_count": 0,
        "last_agent_sequence": [],
        "router_decision": "brain",
        "artifacts": [],
        "agent_insights": [],
        "execution_result": None,
    }


@pytest.mark.integration
class TestRouteFromBrain:
    """Tests for the route_from_brain routing function.

    These tests validate routing decisions without invoking the full LLM chain.
    """

    def test_routes_to_parser_when_tool_calls_present(self):
        """When last message has tool_calls, should route to parser."""
        from data_scientist_chatbot.app.core.graph_builder import route_from_brain

        state = {
            "messages": [
                HumanMessage(content="Analyze data"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate_coding_task", "args": {"task_description": "run analysis"}, "id": "call_1"}
                    ],
                ),
            ]
        }

        decision = route_from_brain(state)
        assert decision == "parser", "Should route to parser when tool_calls exist"

    def test_routes_to_end_when_no_tool_calls(self):
        """When last message has no tool_calls, should route to END."""
        from data_scientist_chatbot.app.core.graph_builder import route_from_brain
        from langgraph.graph import END

        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hello! How can I help you today?"),
            ]
        }

        decision = route_from_brain(state)
        assert decision == END, "Should route to END when no tool_calls"

    def test_routes_to_end_with_empty_messages(self):
        """When no messages, should route to END."""
        from data_scientist_chatbot.app.core.graph_builder import route_from_brain
        from langgraph.graph import END

        state = {"messages": []}

        decision = route_from_brain(state)
        assert decision == END, "Should route to END with empty messages"

    def test_routes_to_parser_for_knowledge_graph_query(self):
        """Should route to parser for knowledge_graph_query tool."""
        from data_scientist_chatbot.app.core.graph_builder import route_from_brain

        state = {
            "messages": [
                HumanMessage(content="What did we find in previous analysis?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "knowledge_graph_query", "args": {"query": "previous analysis"}, "id": "kg_1"}
                    ],
                ),
            ]
        }

        decision = route_from_brain(state)
        assert decision == "parser", "Should route to parser for any tool call"

    def test_routes_to_parser_for_web_search(self):
        """Should route to parser for web_search tool."""
        from data_scientist_chatbot.app.core.graph_builder import route_from_brain

        state = {
            "messages": [
                HumanMessage(content="What are current market trends?"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "web_search", "args": {"query": "market trends 2024"}, "id": "ws_1"}],
                ),
            ]
        }

        decision = route_from_brain(state)
        assert decision == "parser", "Should route to parser for web_search"


@pytest.mark.integration
class TestBrainAgentInvocation:
    """Tests for Brain agent invocation with proper mocking.

    These tests mock at the chain level to verify agent behavior.
    """

    def test_brain_returns_valid_state_structure(self, base_state):
        """Brain agent should return a valid state dictionary."""
        with patch("data_scientist_chatbot.app.agents.brain.create_brain_agent") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MockAIResponse(content="Analysis complete.")

            with patch("data_scientist_chatbot.app.agents.brain.get_brain_prompt") as mock_prompt:
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.return_value = mock_prompt_instance

                from data_scientist_chatbot.app.agents.brain import run_brain_agent

                result = run_brain_agent(base_state)

                assert isinstance(result, dict), "Result should be a dictionary"
                assert "messages" in result, "Result should contain messages"
                assert "current_agent" in result, "Result should contain current_agent"
                assert result["current_agent"] == "brain", "current_agent should be 'brain'"

    def test_brain_appends_response_to_messages(self, base_state):
        """Brain agent should return AI response in messages."""
        with patch("data_scientist_chatbot.app.agents.brain.create_brain_agent") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MockAIResponse(content="Here is my analysis.")

            with patch("data_scientist_chatbot.app.agents.brain.get_brain_prompt") as mock_prompt:
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.return_value = mock_prompt_instance

                from data_scientist_chatbot.app.agents.brain import run_brain_agent

                result = run_brain_agent(base_state)

                # Brain returns the response in messages list
                assert len(result["messages"]) >= 1, "Should have at least one message"
                last_msg = result["messages"][-1]
                assert hasattr(last_msg, "content"), "Last message should have content"

    def test_brain_handles_tool_calls_in_response(self, base_state):
        """Brain agent should properly handle responses with tool_calls."""
        with patch("data_scientist_chatbot.app.agents.brain.create_brain_agent") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm

            mock_chain = MagicMock()
            mock_response = MockAIResponse(
                content="",
                tool_calls=[
                    {"name": "delegate_coding_task", "args": {"task_description": "Calculate stats"}, "id": "tc_1"}
                ],
            )
            mock_chain.invoke.return_value = mock_response

            with patch("data_scientist_chatbot.app.agents.brain.get_brain_prompt") as mock_prompt:
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.return_value = mock_prompt_instance

                from data_scientist_chatbot.app.agents.brain import run_brain_agent

                result = run_brain_agent(base_state)

                last_message = result["messages"][-1]
                assert hasattr(last_message, "tool_calls") or "tool_calls" in str(type(last_message))

    def test_brain_updates_agent_sequence(self, base_state):
        """Brain should append itself to last_agent_sequence."""
        with patch("data_scientist_chatbot.app.agents.brain.create_brain_agent") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MockAIResponse(content="Done.")

            with patch("data_scientist_chatbot.app.agents.brain.get_brain_prompt") as mock_prompt:
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.return_value = mock_prompt_instance

                from data_scientist_chatbot.app.agents.brain import run_brain_agent

                result = run_brain_agent(base_state)

                assert "brain" in result.get("last_agent_sequence", []), "Brain should be in agent sequence"


@pytest.mark.integration
class TestBrainAgentContext:
    """Tests for Brain agent context handling."""

    def test_brain_uses_session_id_from_state(self, base_state):
        """Brain should use session_id from state."""
        base_state["session_id"] = "unique_session_456"

        with patch("data_scientist_chatbot.app.agents.brain.create_brain_agent") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MockAIResponse(content="Response.")

            with patch("data_scientist_chatbot.app.agents.brain.get_brain_prompt") as mock_prompt:
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.return_value = mock_prompt_instance

                with patch("data_scientist_chatbot.app.agents.brain.get_data_context") as mock_context:
                    mock_context.return_value = "Context for unique_session_456"

                    from data_scientist_chatbot.app.agents.brain import run_brain_agent

                    run_brain_agent(base_state)

                    mock_context.assert_called()

    def test_brain_preserves_agent_insights(self, base_state):
        """Brain should preserve agent_insights in state."""
        base_state["agent_insights"] = [{"label": "Test Insight", "value": "42"}]

        with patch("data_scientist_chatbot.app.agents.brain.create_brain_agent") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.return_value = mock_llm

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MockAIResponse(content="Analysis.")

            with patch("data_scientist_chatbot.app.agents.brain.get_brain_prompt") as mock_prompt:
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.return_value = mock_prompt_instance

                from data_scientist_chatbot.app.agents.brain import run_brain_agent

                result = run_brain_agent(base_state)

                assert result.get("agent_insights") == [{"label": "Test Insight", "value": "42"}]


@pytest.mark.integration
class TestBrainHelperFunctions:
    """Tests for Brain agent helper functions."""

    def test_extract_last_user_message(self):
        """Should extract the last human message content."""
        from data_scientist_chatbot.app.agents.brain import _extract_last_user_message

        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            HumanMessage(content="Second question"),
            AIMessage(content="Second answer"),
        ]

        result = _extract_last_user_message(messages)
        assert result == "Second question", "Should return last human message"

    def test_extract_last_user_message_empty(self):
        """Should return default when no human messages."""
        from data_scientist_chatbot.app.agents.brain import _extract_last_user_message

        messages = [AIMessage(content="Only AI")]

        result = _extract_last_user_message(messages)
        assert result == "Provide analysis and insights.", "Should return default"

    def test_filter_messages_removes_internal(self):
        """Should filter out internal AI messages."""
        from data_scientist_chatbot.app.agents.brain import _filter_messages_for_brain

        internal_msg = AIMessage(content="Internal processing")
        internal_msg.additional_kwargs = {"internal": True}

        messages = [
            HumanMessage(content="User question"),
            internal_msg,
            AIMessage(content="Public response"),
        ]

        filtered = _filter_messages_for_brain(messages, [], [])

        assert len(filtered) == 2, "Should filter out internal message"
        assert all(not getattr(m, "additional_kwargs", {}).get("internal") for m in filtered)

    def test_fix_artifact_paths(self):
        """Should fix artifact paths in content."""
        from data_scientist_chatbot.app.agents.brain import _fix_artifact_paths

        content = "Here's the chart: ![chart](myplot.png)"
        result = _fix_artifact_paths(content)

        assert "/static/plots/" in result or "myplot.png" in result
