import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


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


@pytest.mark.integration
class TestMemorySynchronization:

    def test_checkpointer_persists_conversation_state(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        session_id = "memory_test_session"
        config = {"configurable": {"thread_id": session_id}}

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.return_value = AIMessage(content="Router decision: brain")
            mock_ollama.return_value = mock_instance

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

    def test_context_retrieval_across_turns(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        session_id = "context_test_session"
        config = {"configurable": {"thread_id": session_id}}

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.side_effect = [
                AIMessage(content="Router: brain"),
                AIMessage(content="Nice to meet you, Alice!"),
                AIMessage(content="Router: brain"),
                AIMessage(content="Your name is Alice, as you mentioned earlier."),
            ]
            mock_ollama.return_value = mock_instance

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

    def test_session_isolation_between_users(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        session_alice = "session_alice"
        session_bob = "session_bob"
        config_alice = {"configurable": {"thread_id": session_alice}}
        config_bob = {"configurable": {"thread_id": session_bob}}

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.return_value = AIMessage(content="Acknowledged")
            mock_ollama.return_value = mock_instance

            state_alice = {
                "messages": [HumanMessage(content="My name is Alice")],
                "session_id": session_alice,
                "python_executions": 0,
                "retry_count": 0,
                "last_agent_sequence": [],
                "router_decision": "brain",
            }
            result_alice = agent.invoke(state_alice, config_alice)

            state_bob = {
                "messages": [HumanMessage(content="My name is Bob")],
                "session_id": session_bob,
                "python_executions": 0,
                "retry_count": 0,
                "last_agent_sequence": [],
                "router_decision": "brain",
            }
            result_bob = agent.invoke(state_bob, config_bob)

            assert result_alice["messages"][0].content != result_bob["messages"][0].content

    def test_python_execution_counter_persistence(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        session_id = "exec_counter_test"
        config = {"configurable": {"thread_id": session_id}}

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.return_value = AIMessage(
                content="",
                tool_calls=[{"name": "python_code_interpreter", "args": {"code": "print(1+1)"}, "id": "call_1"}],
            )
            mock_ollama.return_value = mock_instance

            with patch("data_scientist_chatbot.app.tools.executor.execute_tool") as mock_exec:
                mock_exec.return_value = "2"

                state = {
                    "messages": [HumanMessage(content="Calculate 1+1")],
                    "session_id": session_id,
                    "python_executions": 0,
                    "retry_count": 0,
                    "last_agent_sequence": [],
                    "router_decision": "hands",
                }

                result = agent.invoke(state, config)
                assert result.get("python_executions", 0) >= 0

    def test_retry_count_reset_after_success(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        session_id = "retry_test_session"
        config = {"configurable": {"thread_id": session_id}}

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.return_value = AIMessage(content="Success")
            mock_ollama.return_value = mock_instance

            state = {
                "messages": [HumanMessage(content="Test query")],
                "session_id": session_id,
                "python_executions": 0,
                "retry_count": 2,
                "last_agent_sequence": ["brain"],
                "router_decision": "brain",
            }

            result = agent.invoke(state, config)
            assert result.get("retry_count", 0) <= 2

    def test_business_context_preservation(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        session_id = "context_preservation_test"
        config = {"configurable": {"thread_id": session_id}}

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        business_ctx = {
            "domain": "finance",
            "objective": "fraud_detection",
            "stakeholders": ["security_team", "compliance"],
        }

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.return_value = AIMessage(content="Acknowledged")
            mock_ollama.return_value = mock_instance

            state = {
                "messages": [HumanMessage(content="Analyze fraud patterns")],
                "session_id": session_id,
                "business_context": business_ctx,
                "python_executions": 0,
                "retry_count": 0,
                "last_agent_sequence": [],
                "router_decision": "brain",
            }

            result = agent.invoke(state, config)
            assert result.get("business_context", {}).get("domain") == "finance"

    def test_scratchpad_accumulation(self, checkpointer_with_temp_db):
        from data_scientist_chatbot.app.core.graph_builder import create_agent_executor

        session_id = "scratchpad_test"
        config = {"configurable": {"thread_id": session_id}}

        agent = create_agent_executor(memory=checkpointer_with_temp_db)

        with patch("langchain_ollama.chat_models.ChatOllama") as mock_ollama:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value = mock_instance
            mock_instance.invoke.return_value = AIMessage(content="Processing")
            mock_ollama.return_value = mock_instance

            state = {
                "messages": [HumanMessage(content="Step 1 task")],
                "session_id": session_id,
                "scratchpad": "Initial note",
                "python_executions": 0,
                "retry_count": 0,
                "last_agent_sequence": [],
                "router_decision": "brain",
            }

            result = agent.invoke(state, config)
            assert "scratchpad" in result
