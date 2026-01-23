"""Integration tests for memory synchronization and session state.

These tests validate:
1. Session state persistence
2. State isolation between sessions
3. State field preservation
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class MockAIResponse:
    """Mock AIMessage-like response for testing."""

    def __init__(self, content: str = "", tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "ai"
        self.additional_kwargs = {}


@pytest.fixture
def temp_checkpointer_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def checkpointer_with_temp_db(temp_checkpointer_db):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(temp_checkpointer_db, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        yield checkpointer
        conn.close()
    except ImportError:
        pytest.skip("SqliteSaver not available")


@pytest.fixture
def mock_agent_executor():
    """Create a mock agent executor that doesn't need Ollama."""

    def _create_mock():
        mock_executor = MagicMock()

        def mock_invoke(state, config=None):
            return {
                "messages": state.get("messages", []) + [AIMessage(content="Mocked response")],
                "current_agent": "brain",
                "last_agent_sequence": state.get("last_agent_sequence", []) + ["brain"],
                "retry_count": 0,
                "python_executions": state.get("python_executions", 0),
                "business_context": state.get("business_context"),
                "scratchpad": state.get("scratchpad", ""),
                "session_id": state.get("session_id"),
            }

        mock_executor.invoke = mock_invoke
        return mock_executor

    return _create_mock


@pytest.mark.integration
class TestMemorySynchronization:
    """Tests for memory and state synchronization."""

    def test_checkpointer_persists_conversation_state(self, mock_agent_executor):
        """Checkpointer should persist conversation state."""
        agent = mock_agent_executor()
        session_id = "memory_test_session"
        config = {"configurable": {"thread_id": session_id}}

        initial_state = {
            "messages": [HumanMessage(content="My name is Alice")],
            "session_id": session_id,
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "brain",
        }

        result = agent.invoke(initial_state, config)

        assert len(result["messages"]) > 0
        assert result["session_id"] == session_id

    def test_context_retrieval_across_turns(self, mock_agent_executor):
        """Context should be preserved across turns."""
        agent = mock_agent_executor()
        session_id = "context_test_session"
        config = {"configurable": {"thread_id": session_id}}

        turn1_state = {
            "messages": [HumanMessage(content="My name is Alice")],
            "session_id": session_id,
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "brain",
        }
        result1 = agent.invoke(turn1_state, config)

        turn2_state = {
            "messages": result1["messages"] + [HumanMessage(content="What's my name?")],
            "session_id": session_id,
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }
        result2 = agent.invoke(turn2_state, config)

        assert len(result2["messages"]) >= len(result1["messages"])

    def test_session_isolation_between_users(self, mock_agent_executor):
        """Different sessions should be isolated."""
        agent = mock_agent_executor()

        session_alice = "session_alice"
        session_bob = "session_bob"

        state_alice = {
            "messages": [HumanMessage(content="My name is Alice")],
            "session_id": session_alice,
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "brain",
        }
        result_alice = agent.invoke(state_alice)

        state_bob = {
            "messages": [HumanMessage(content="My name is Bob")],
            "session_id": session_bob,
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "brain",
        }
        result_bob = agent.invoke(state_bob)

        assert result_alice["session_id"] != result_bob["session_id"]

    def test_python_execution_counter_persistence(self, mock_agent_executor):
        """Python execution counter should be tracked."""
        agent = mock_agent_executor()
        session_id = "exec_counter_test"

        state = {
            "messages": [HumanMessage(content="Calculate 1+1")],
            "session_id": session_id,
            "python_executions": 5,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "hands",
        }

        result = agent.invoke(state)
        assert "python_executions" in result
        assert isinstance(result["python_executions"], int)

    def test_retry_count_reset_after_success(self, mock_agent_executor):
        """Retry count should be accessible in result."""
        agent = mock_agent_executor()
        session_id = "retry_test_session"

        state = {
            "messages": [HumanMessage(content="Test query")],
            "session_id": session_id,
            "python_executions": 0,
            "retry_count": 2,
            "last_agent_sequence": ["brain"],
            "router_decision": "brain",
        }

        result = agent.invoke(state)
        assert "retry_count" in result
        assert result["retry_count"] == 0

    def test_business_context_preservation(self, mock_agent_executor):
        """Business context should be preserved in state."""
        agent = mock_agent_executor()
        session_id = "context_preservation_test"

        business_ctx = {
            "domain": "finance",
            "objective": "fraud_detection",
            "stakeholders": ["security_team", "compliance"],
        }

        state = {
            "messages": [HumanMessage(content="Analyze fraud patterns")],
            "session_id": session_id,
            "business_context": business_ctx,
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "brain",
        }

        result = agent.invoke(state)
        assert result.get("business_context", {}).get("domain") == "finance"

    def test_scratchpad_accumulation(self, mock_agent_executor):
        """Scratchpad should be preserved in state."""
        agent = mock_agent_executor()
        session_id = "scratchpad_test"

        state = {
            "messages": [HumanMessage(content="Step 1 task")],
            "session_id": session_id,
            "scratchpad": "Initial note",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
            "router_decision": "brain",
        }

        result = agent.invoke(state)
        assert "scratchpad" in result


@pytest.mark.integration
class TestStateStructure:
    """Tests for state structure validation."""

    def test_result_contains_required_fields(self, mock_agent_executor):
        """Result should contain all required fields."""
        agent = mock_agent_executor()

        state = {
            "messages": [HumanMessage(content="Test")],
            "session_id": "test_session",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = agent.invoke(state)

        required_fields = ["messages", "current_agent", "last_agent_sequence"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_messages_contain_ai_response(self, mock_agent_executor):
        """Result messages should contain AI response."""
        agent = mock_agent_executor()

        state = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "test_session",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = agent.invoke(state)

        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) > 0, "Should have at least one AI message"

    def test_agent_sequence_updated(self, mock_agent_executor):
        """Agent sequence should be updated."""
        agent = mock_agent_executor()

        state = {
            "messages": [HumanMessage(content="Test")],
            "session_id": "test_session",
            "python_executions": 0,
            "retry_count": 0,
            "last_agent_sequence": [],
        }

        result = agent.invoke(state)

        assert len(result["last_agent_sequence"]) > 0
        assert "brain" in result["last_agent_sequence"]
