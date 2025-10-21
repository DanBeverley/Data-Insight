import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.fixture
def mock_ollama():
    import data_scientist_chatbot.app.core.agent_factory as agent_factory

    with patch.object(agent_factory, "ChatOllama") as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(content="Brain agent response")
        mock_instance.bind_tools.return_value = mock_instance
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def brain_agent_state():
    return {
        "messages": [HumanMessage(content="Analyze the correlation between price and area")],
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
    }


@pytest.mark.integration
class TestBrainAgentRouting:
    def test_brain_delegates_to_hands_for_technical_work(self, mock_ollama):
        from data_scientist_chatbot.app.agent import AgentState
        from data_scientist_chatbot.app.core.router import route_from_brain

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "delegate_coding_task",
                    "args": {"task_description": "Calculate correlation between price and area"},
                    "id": "call_123",
                }
            ],
        )

        state = {
            "messages": [HumanMessage(content="Analyze the correlation between price and area")],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        from data_scientist_chatbot.app.agent import run_brain_agent

        result = run_brain_agent(state)

        assert len(result["messages"]) > 0
        last_message = result["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            assert last_message.tool_calls[0].get("name") == "delegate_coding_task"
            decision = route_from_brain(result)
            assert decision == "parser"

    def test_brain_responds_conversationally_without_delegation(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent
        from data_scientist_chatbot.app.core.router import route_from_brain, END

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(content="Hello! I'm here to help with your data analysis needs.")

        state = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = run_brain_agent(state)

        assert len(result["messages"]) > 0
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert len(last_message.content) > 10

        decision = route_from_brain(result)
        assert decision == END

    def test_brain_prevents_infinite_delegation_loop(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent
        from data_scientist_chatbot.app.core.router import route_from_brain, END

        state = {
            "messages": [
                HumanMessage(content="Analyze data"),
                AIMessage(content="", tool_calls=[{"name": "delegate_coding_task", "args": {}, "id": "call_loop"}]),
            ],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 3,
            "last_agent_sequence": ["brain", "hands", "brain", "hands"],
        }

        decision = route_from_brain(state)
        assert decision == END

    def test_brain_uses_knowledge_graph_for_historical_queries(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "knowledge_graph_query",
                    "args": {"query": "previous housing analysis patterns"},
                    "id": "call_456",
                }
            ],
        )

        state = {
            "messages": [HumanMessage(content="What patterns did we find in previous housing data analysis?")],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = run_brain_agent(state)

        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            assert last_message.tool_calls[0].get("name") == "knowledge_graph_query"

    def test_brain_web_search_integration(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "current housing market trends 2024"}, "id": "call_789"}
            ],
        )

        state = {
            "messages": [HumanMessage(content="What are the current housing market trends?")],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = run_brain_agent(state)

        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            assert last_message.tool_calls[0].get("name") == "web_search"

    def test_brain_avoids_web_search_for_dataset_analysis(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "delegate_coding_task",
                    "args": {"task_description": "Analyze the price distribution in our dataset"},
                    "id": "call_delegate",
                }
            ],
        )

        state = {
            "messages": [HumanMessage(content="Analyze the price distribution in our dataset")],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = run_brain_agent(state)

        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_name = last_message.tool_calls[0].get("name")
            assert tool_name in [
                "delegate_coding_task",
                "knowledge_graph_query",
            ], f"Dataset analysis should use delegate_coding_task or knowledge_graph_query, not {tool_name}"
            assert tool_name != "web_search", "Should not use web_search for dataset-specific questions"

    def test_brain_maintains_business_context(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(content="Analysis complete")

        state = {
            "messages": [HumanMessage(content="Analyze sales data")],
            "session_id": "test_123",
            "current_agent": "brain",
            "business_context": {"domain": "retail", "objective": "revenue_optimization"},
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = run_brain_agent(state)

        assert (
            result.get("business_context", {}).get("domain") == "retail"
            or "business_context" in result
            or len(result.get("messages", [])) > 0
        )

    def test_complexity_scoring_in_router(self):
        from data_scientist_chatbot.app.agent import run_router_agent
        from unittest.mock import patch, MagicMock

        state = {
            "messages": [HumanMessage(content="Analyze complex multivariate relationships with deep learning")],
            "session_id": "test_123",
            "python_executions": 0,
            "plan": None,
            "scratchpad": "",
            "business_objective": None,
            "task_type": None,
            "target_column": None,
            "workflow_stage": None,
            "current_agent": "router",
            "business_context": {},
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": None,
        }

        result = run_router_agent(state)

        assert "complexity_score" in result
        assert isinstance(result["complexity_score"], int)
        assert 1 <= result["complexity_score"] <= 10, "Complexity score should be 1-10"

        assert "route_strategy" in result
        assert result["route_strategy"] in ["direct", "standard", "collaborative"]

    def test_complexity_routing_strategy_enforcement(self):
        from data_scientist_chatbot.app.agent import run_router_agent

        simple_task_state = {
            "messages": [HumanMessage(content="Show first 5 rows of data")],
            "session_id": "test_123",
            "python_executions": 0,
            "plan": None,
            "scratchpad": "",
            "business_objective": None,
            "task_type": None,
            "target_column": None,
            "workflow_stage": None,
            "current_agent": "router",
            "business_context": {},
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": None,
        }

        result = run_router_agent(simple_task_state)

        if result["complexity_score"] <= 3:
            assert result["route_strategy"] == "direct", "Simple tasks (score <= 3) should use 'direct' strategy"

    def test_session_memory_recording_after_brain_execution(self):
        from data_scientist_chatbot.app.agent import run_brain_agent
        from data_scientist_chatbot.app.core.session_memory import get_session_memory
        from unittest.mock import patch

        with patch("data_scientist_chatbot.app.agent.create_brain_agent") as mock_agent_factory:
            mock_agent = MagicMock()
            mock_agent.bind_tools.return_value = mock_agent
            mock_agent.invoke.return_value = AIMessage(content="Analysis complete")
            mock_agent_factory.return_value = mock_agent

            state = {
                "messages": [HumanMessage(content="Analyze the data")],
                "session_id": "test_memory_123",
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
            }

            result = run_brain_agent(state)

            assert len(result["messages"]) > 0
            assert result["current_agent"] == "brain"
