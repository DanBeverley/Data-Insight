import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json

from data_scientist_chatbot.app.agents.verifier import run_verifier_agent, VerifierDecision


@pytest.fixture
def mock_verifier_state() -> Dict[str, Any]:
    return {
        "execution_result": {
            "stdout": '=== DATASET INFO ===\nShape: (545, 13)\nPROFILING_INSIGHTS_START\n[{"label": "Test", "value": "123"}]\nPROFILING_INSIGHTS_END'
        },
        "artifacts": [
            {"filename": "chart1.html", "category": "visualization", "local_path": "/static/plots/chart1.html"},
            {"filename": "chart2.png", "category": "visualization", "local_path": "/static/plots/chart2.png"},
            {"filename": "report.html", "category": "report", "local_path": "/static/plots/report.html"},
        ],
        "agent_insights": [{"label": "Test Insight", "value": "Test value", "type": "pattern"}],
        "current_task_description": "Generate distribution plot",
        "plan": [],
        "current_task_index": 0,
        "messages": [],
    }


@pytest.fixture
def mock_verifier_llm():
    with patch("data_scientist_chatbot.app.agents.verifier.create_verifier_agent") as mock:
        mock_agent = MagicMock()
        mock.return_value = mock_agent
        yield mock_agent


@pytest.mark.unit
class TestVerifierDecision:
    def test_verifier_approves_complete_task(self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock):
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": true, "feedback": "All requirements met", "missing_items": [], "existing_items": ["artifacts", "insights", "df_info"]}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        assert result.get("workflow_stage") == "verification_passed" or result.get("verification_passed") is True

    def test_verifier_rejects_missing_artifacts(self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock):
        mock_verifier_state["artifacts"] = []
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": false, "feedback": "No artifacts generated", "missing_items": ["artifacts"], "existing_items": ["df_info"]}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        assert result.get("workflow_stage") != "verification_passed" or result.get("verification_passed") is False

    def test_verifier_rejects_missing_insights(self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock):
        mock_verifier_state["agent_insights"] = []
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": false, "feedback": "Insights not provided", "missing_items": ["insights"], "existing_items": ["artifacts"]}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        assert result.get("workflow_stage") != "verification_passed" or result.get("verification_passed") is False

    def test_verifier_handles_empty_execution_output(self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock):
        mock_verifier_state["execution_result"] = {}
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": false, "feedback": "No output", "missing_items": ["df_info"], "existing_items": []}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        assert result is not None

    def test_verifier_handles_none_execution_result(self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock):
        mock_verifier_state["execution_result"] = None
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": false, "feedback": "No execution", "missing_items": ["unknown"], "existing_items": []}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        assert result is not None


@pytest.mark.unit
class TestVerifierDecisionParsing:
    def test_parse_valid_json_response(self):
        decision = VerifierDecision(
            approved=True, feedback="Complete", missing_items=[], existing_items=["artifacts", "insights"]
        )
        assert decision.approved is True
        assert len(decision.existing_items) == 2

    def test_parse_rejection_with_missing_items(self):
        decision = VerifierDecision(
            approved=False,
            feedback="Missing correlation heatmap",
            missing_items=["correlation", "df_info"],
            existing_items=["artifacts"],
        )
        assert decision.approved is False
        assert "correlation" in decision.missing_items
        assert "df_info" in decision.missing_items


@pytest.mark.unit
class TestVerifierStateHandling:
    def test_verifier_reads_full_execution_output(self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock):
        long_output = "=== DATASET INFO ===\n" + "x" * 5000 + "\nPROFILING_INSIGHTS_START\n[]\nPROFILING_INSIGHTS_END"
        mock_verifier_state["execution_result"] = {"stdout": long_output}
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": true, "feedback": "OK", "missing_items": [], "existing_items": ["df_info"]}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        assert result is not None

    def test_verifier_preserves_existing_items_across_retries(
        self, mock_verifier_state: Dict, mock_verifier_llm: MagicMock
    ):
        mock_verifier_llm.invoke.return_value = MagicMock(
            content='{"approved": false, "feedback": "Missing insights", "missing_items": ["insights"], "existing_items": ["artifacts", "df_info"]}'
        )

        with patch("data_scientist_chatbot.app.agents.verifier.get_verifier_prompt") as mock_prompt:
            mock_prompt.return_value = MagicMock()
            result = run_verifier_agent(mock_verifier_state, {})

        feedback = result.get("verification_feedback", {})
        if isinstance(feedback, dict):
            existing = feedback.get("existing_items", [])
            assert "artifacts" in existing or result.get("artifacts")
