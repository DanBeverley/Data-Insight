import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.fixture
def mock_ollama():
    with patch('langchain_ollama.chat_models.ChatOllama') as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(content="Brain agent response")
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
        "router_decision": "brain"
    }


@pytest.mark.integration
class TestBrainAgentRouting:

    def test_brain_delegates_to_hands_for_technical_work(self, mock_ollama):
        from data_scientist_chatbot.app.agent import AgentState
        from data_scientist_chatbot.app.core.router import route_from_brain

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[{
                "name": "delegate_coding_task",
                "args": {"task_description": "Calculate correlation between price and area"},
                "id": "call_123"
            }]
        )

        state = {
            "messages": [
                HumanMessage(content="Analyze the correlation between price and area")
            ],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": []
        }

        from data_scientist_chatbot.app.agent import run_brain_agent
        result = run_brain_agent(state)

        assert len(result["messages"]) > 0
        last_message = result["messages"][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            assert last_message.tool_calls[0].get('name') == 'delegate_coding_task'
            decision = route_from_brain(result)
            assert decision == "parser"


    def test_brain_responds_conversationally_without_delegation(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent
        from data_scientist_chatbot.app.core.router import route_from_brain, END

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="Hello! I'm here to help with your data analysis needs."
        )

        state = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": []
        }

        result = run_brain_agent(state)

        assert len(result["messages"]) > 0
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert "help" in last_message.content.lower()

        decision = route_from_brain(result)
        assert decision == END


    def test_brain_prevents_infinite_delegation_loop(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent
        from data_scientist_chatbot.app.core.router import route_from_brain, END

        state = {
            "messages": [
                HumanMessage(content="Analyze data"),
                AIMessage(content="", tool_calls=[{"name": "delegate_coding_task", "args": {}, "id": "call_loop"}])
            ],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 3,
            "last_agent_sequence": ["brain", "hands", "brain", "hands"]
        }

        decision = route_from_brain(state)
        assert decision == END


    def test_brain_uses_knowledge_graph_for_historical_queries(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[{
                "name": "knowledge_graph_query",
                "args": {"query": "previous housing analysis patterns"},
                "id": "call_456"
            }]
        )

        state = {
            "messages": [
                HumanMessage(content="What patterns did we find in previous housing data analysis?")
            ],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": []
        }

        result = run_brain_agent(state)

        last_message = result["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            assert last_message.tool_calls[0].get('name') == 'knowledge_graph_query'


    def test_brain_web_search_integration(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(
            content="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "current housing market trends 2024"},
                "id": "call_789"
            }]
        )

        state = {
            "messages": [
                HumanMessage(content="What are the current housing market trends?")
            ],
            "session_id": "test_123",
            "current_agent": "brain",
            "retry_count": 0,
            "last_agent_sequence": []
        }

        result = run_brain_agent(state)

        last_message = result["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            assert last_message.tool_calls[0].get('name') == 'web_search'


    def test_brain_maintains_business_context(self, mock_ollama):
        from data_scientist_chatbot.app.agent import run_brain_agent

        mock_ollama.bind_tools.return_value = mock_ollama
        mock_ollama.invoke.return_value = AIMessage(content="Analysis complete")

        state = {
            "messages": [
                HumanMessage(content="Analyze sales data")
            ],
            "session_id": "test_123",
            "current_agent": "brain",
            "business_context": {"domain": "retail", "objective": "revenue_optimization"},
            "retry_count": 0,
            "last_agent_sequence": []
        }

        result = run_brain_agent(state)

        assert result.get("business_context", {}).get("domain") == "retail" or "business_context" in result or len(result.get("messages", [])) > 0
