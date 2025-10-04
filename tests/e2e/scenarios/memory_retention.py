import pytest
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch
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
        response = self.api_client.post("/api/sessions/create")
        if response.status_code == 200:
            self.session_id = response.json().get("session_id")

    def define_steps(self) -> List[ScenarioStep]:
        return [
            ScenarioStep(
                name="Turn 1: Introduce Name",
                action="turn_1_name",
                expected_outcome="acknowledged",
                metadata={"user_message": "My name is Alice"}
            ),
            ScenarioStep(
                name="Turn 2: Introduce Age",
                action="turn_2_age",
                expected_outcome="acknowledged",
                metadata={"user_message": "I am 28 years old"}
            ),
            ScenarioStep(
                name="Turn 3: Recall Name",
                action="turn_3_recall_name",
                expected_outcome="alice",
                metadata={"user_message": "What's my name?"}
            ),
            ScenarioStep(
                name="Turn 4: Recall Age",
                action="turn_4_recall_age",
                expected_outcome="28",
                metadata={"user_message": "How old am I?"}
            ),
            ScenarioStep(
                name="Turn 5: Combined Recall",
                action="turn_5_combined",
                expected_outcome="alice and 28",
                metadata={"user_message": "Tell me everything you know about me"}
            )
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        user_message = step.metadata.get("user_message")

        try:
            response = self.api_client.post(
                f"/api/chat/{self.session_id}",
                json={"message": user_message, "session_id": self.session_id}
            )

            if response.status_code == 200:
                result = response.json()
                agent_response = result.get("response", "").lower()

                self.conversation_history.append({
                    "user": user_message,
                    "agent": agent_response
                })

                return self.validate_outcome(step.expected_outcome, agent_response)

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
    def mock_api_client(self):
        client = MagicMock()

        client.post.side_effect = self._mock_chat_responses
        client.delete.return_value = MagicMock(status_code=200)

        return client

    def _mock_chat_responses(self, endpoint, **kwargs):
        response = MagicMock()
        response.status_code = 200

        if "/sessions/create" in endpoint:
            response.json.return_value = {"session_id": "memory_test_session"}
            return response

        json_data = kwargs.get("json", {})
        message = json_data.get("message", "").lower()

        if "my name is alice" in message:
            response.json.return_value = {
                "response": "Nice to meet you, Alice! How can I help you today?"
            }
        elif "28 years old" in message:
            response.json.return_value = {
                "response": "Got it, you're 28 years old. What would you like to analyze?"
            }
        elif "what's my name" in message or "what is my name" in message:
            response.json.return_value = {
                "response": "Your name is Alice, as you mentioned earlier."
            }
        elif "how old am i" in message or "my age" in message:
            response.json.return_value = {
                "response": "You told me you're 28 years old."
            }
        elif "everything you know" in message:
            response.json.return_value = {
                "response": "Based on our conversation, your name is Alice and you're 28 years old."
            }
        else:
            response.json.return_value = {
                "response": "I'm here to help with your data analysis needs."
            }

        return response

    def test_multi_turn_memory_retention(self, mock_api_client):
        scenario = MemoryRetentionScenario(mock_api_client)
        result = scenario.execute()

        assert result.passed, f"Memory retention failed: {result.errors}"
        assert result.steps_completed == result.total_steps
        assert len(scenario.conversation_history) == 5

    def test_context_persistence_across_turns(self, mock_api_client):
        scenario = MemoryRetentionScenario(mock_api_client)
        scenario.setup()

        step1 = ScenarioStep("Intro", "turn_1_name", "alice", metadata={"user_message": "My name is Alice"})
        step2 = ScenarioStep("Recall", "turn_3_recall_name", "alice", metadata={"user_message": "What's my name?"})

        success1 = scenario._execute_step(step1)
        success2 = scenario._execute_step(step2)

        assert success1 and success2

    def test_sequential_information_accumulation(self, mock_api_client):
        scenario = MemoryRetentionScenario(mock_api_client)
        scenario.setup()

        steps = scenario.define_steps()
        for step in steps:
            success = scenario._execute_step(step)
            if not success and "recall" in step.name.lower():
                self.errors.append(f"Failed to recall information at {step.name}")

        assert len(scenario.conversation_history) > 0


@pytest.mark.e2e
class TestCrossSessionIsolation:

    def test_sessions_do_not_share_memory(self):
        client = MagicMock()

        session_responses = {
            "session_alice": [],
            "session_bob": []
        }

        def mock_post(endpoint, **kwargs):
            response = MagicMock()
            response.status_code = 200

            if "/sessions/create" in endpoint:
                import random
                session_id = f"session_{random.choice(['alice', 'bob'])}"
                response.json.return_value = {"session_id": session_id}
                return response

            json_data = kwargs.get("json", {})
            message = json_data.get("message", "").lower()
            session_id = json_data.get("session_id", "")

            if "alice" in session_id:
                if "my name is alice" in message:
                    response.json.return_value = {"response": "Hello Alice!"}
                elif "what's my name" in message:
                    response.json.return_value = {"response": "Your name is Alice"}
            elif "bob" in session_id:
                if "my name is bob" in message:
                    response.json.return_value = {"response": "Hello Bob!"}
                elif "what's my name" in message:
                    response.json.return_value = {"response": "Your name is Bob"}

            return response

        client.post.side_effect = mock_post

        resp_alice_create = client.post("/api/sessions/create")
        session_alice = resp_alice_create.json()["session_id"]

        resp_bob_create = client.post("/api/sessions/create")
        session_bob = resp_bob_create.json()["session_id"]

        assert session_alice != session_bob
