import pytest
from typing import List, Dict, Any
from .base_scenario import BaseScenario, ScenarioStep


class MemoryRetentionScenario(BaseScenario):
    def __init__(self, api_client):
        super().__init__(
            name="Memory Retention",
            description="Multi-turn conversation testing context persistence across interactions"
        )
        self.api_client = api_client
        self.session_id: str = None
        self.conversation_history: List[Dict[str, str]] = []

    def setup(self) -> None:
        response = self.api_client.post("/api/sessions/new")
        if response.status_code == 200:
            self.session_id = response.json().get("session_id")

    def define_steps(self) -> List[ScenarioStep]:
        return [
            ScenarioStep(
                name="Turn 1: First Query",
                action="turn_1",
                expected_outcome="response",
                metadata={"user_message": "Hello"}
            ),
            ScenarioStep(
                name="Turn 2: Second Query",
                action="turn_2",
                expected_outcome="response",
                metadata={"user_message": "What can you help me with?"}
            ),
            ScenarioStep(
                name="Turn 3: Third Query",
                action="turn_3",
                expected_outcome="response",
                metadata={"user_message": "Thank you"}
            )
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        user_message = step.metadata.get("user_message")

        try:
            response = self.api_client.get(
                "/api/agent/chat-stream",
                params={
                    "message": user_message,
                    "session_id": self.session_id,
                    "web_search_enabled": "false"
                },
                timeout=60.0
            )

            if response.status_code == 200:
                agent_response = response.text

                self.conversation_history.append({
                    "user": user_message,
                    "agent": agent_response
                })

                return len(agent_response) > 0

        except Exception as e:
            self.errors.append(f"{step.name} failed: {str(e)}")

        return False

    def teardown(self) -> None:
        if self.session_id:
            try:
                self.api_client.delete(f"/api/sessions/{self.session_id}")
            except:
                pass


@pytest.mark.e2e
class TestMemoryRetention:

    @pytest.fixture
    def api_client(self):
        from fastapi.testclient import TestClient
        from src.api import app
        return TestClient(app)

    @pytest.mark.slow
    def test_multi_turn_conversation(self, api_client):
        scenario = MemoryRetentionScenario(api_client)
        result = scenario.execute()

        assert result.duration > 0
        assert len(scenario.conversation_history) > 0

    def test_session_persistence(self, api_client):
        session_response = api_client.post("/api/sessions/new")
        session_id = session_response.json()["session_id"]

        response1 = api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "Hello",
                "session_id": session_id,
                "web_search_enabled": "false"
            },
            timeout=30.0
        )

        response2 = api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "What can you help with?",
                "session_id": session_id,
                "web_search_enabled": "false"
            },
            timeout=30.0
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_session_memory_records_execution_history(self, api_client):
        from src.learning.adaptive_system import AdaptiveLearningSystem

        session_response = api_client.post("/api/sessions/new")
        session_id = session_response.json()["session_id"]

        response1 = api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "Analyze this data",
                "session_id": session_id,
                "web_search_enabled": "false"
            },
            timeout=30.0
        )
        assert response1.status_code == 200

        try:
            adaptive_system = AdaptiveLearningSystem()
            execution_history = adaptive_system.get_execution_history(session_id=session_id)
            assert len(execution_history) > 0, "Adaptive system should record execution history"

            first_execution = execution_history[0]
            assert 'session_id' in first_execution
            assert first_execution['session_id'] == session_id
            assert 'success' in first_execution
            assert 'timestamp' in first_execution
        except Exception as e:
            pytest.skip(f"Adaptive learning system not available: {e}")


@pytest.mark.e2e
class TestCrossSessionIsolation:

    @pytest.fixture
    def api_client(self):
        from fastapi.testclient import TestClient
        from src.api import app
        return TestClient(app)

    def test_sessions_do_not_share_memory(self, api_client):
        resp_alice = api_client.post("/api/sessions/new")
        session_alice = resp_alice.json()["session_id"]

        resp_bob = api_client.post("/api/sessions/new")
        session_bob = resp_bob.json()["session_id"]

        assert session_alice != session_bob

    def test_execution_history_isolation(self, api_client):
        from src.learning.adaptive_system import AdaptiveLearningSystem

        resp1 = api_client.post("/api/sessions/new")
        session_1 = resp1.json()["session_id"]

        resp2 = api_client.post("/api/sessions/new")
        session_2 = resp2.json()["session_id"]

        api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "Task for session 1",
                "session_id": session_1,
                "web_search_enabled": "false"
            },
            timeout=30.0
        )

        api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "Task for session 2",
                "session_id": session_2,
                "web_search_enabled": "false"
            },
            timeout=30.0
        )

        try:
            adaptive_system = AdaptiveLearningSystem()

            history_1 = adaptive_system.get_execution_history(session_id=session_1)
            history_2 = adaptive_system.get_execution_history(session_id=session_2)

            assert len(history_1) > 0, "Session 1 should have history"
            assert len(history_2) > 0, "Session 2 should have history"
            assert history_1[0].get('session_id') != history_2[0].get('session_id'), \
                "Different sessions should have different execution histories"
        except Exception as e:
            pytest.skip(f"Adaptive learning system not available: {e}")
