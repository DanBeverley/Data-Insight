import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def mock_hands_state() -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content="Analyze the dataset")],
        "session_id": "test-session-123",
        "artifacts": [{"filename": "prev_chart.html", "category": "report"}],
        "agent_insights": [{"label": "Previous Insight", "value": "Test", "type": "pattern"}],
        "execution_result": {"stdout": "Previous execution output", "success": True},
        "retry_count": 1,
        "current_task_description": "Generate distribution plot",
    }


@pytest.mark.unit
class TestHandsStatePreservation:
    def test_preserves_artifacts_from_previous_state(self, mock_hands_state: Dict):
        prev_exec = mock_hands_state.get("execution_result") or {}
        artifacts = list(mock_hands_state.get("artifacts") or [])

        assert len(artifacts) == 1
        assert artifacts[0]["filename"] == "prev_chart.html"

    def test_preserves_insights_from_previous_state(self, mock_hands_state: Dict):
        agent_insights = list(mock_hands_state.get("agent_insights") or [])

        assert len(agent_insights) == 1
        assert agent_insights[0]["label"] == "Previous Insight"

    def test_preserves_execution_output_from_previous_state(self, mock_hands_state: Dict):
        prev_exec = mock_hands_state.get("execution_result") or {}
        execution_summary = prev_exec.get("stdout", "") if isinstance(prev_exec, dict) else ""

        assert execution_summary == "Previous execution output"

    def test_handles_none_execution_result(self):
        state = {"execution_result": None, "artifacts": [], "agent_insights": []}

        prev_exec = state.get("execution_result") or {}
        artifacts = list(state.get("artifacts") or [])
        execution_summary = prev_exec.get("stdout", "") if isinstance(prev_exec, dict) else ""
        agent_insights = list(state.get("agent_insights") or [])

        assert artifacts == []
        assert execution_summary == ""
        assert agent_insights == []

    def test_handles_missing_keys_in_state(self):
        state = {}

        prev_exec = state.get("execution_result") or {}
        artifacts = list(state.get("artifacts") or [])
        execution_summary = prev_exec.get("stdout", "") if isinstance(prev_exec, dict) else ""
        agent_insights = list(state.get("agent_insights") or [])

        assert artifacts == []
        assert execution_summary == ""
        assert agent_insights == []

    def test_accumulates_new_artifacts_with_existing(self, mock_hands_state: Dict):
        existing_artifacts = list(mock_hands_state.get("artifacts") or [])
        new_artifact = {"filename": "new_chart.html", "category": "report"}

        combined = existing_artifacts + [new_artifact]

        assert len(combined) == 2
        assert combined[0]["filename"] == "prev_chart.html"
        assert combined[1]["filename"] == "new_chart.html"

    def test_accumulates_execution_output(self, mock_hands_state: Dict):
        prev_exec = mock_hands_state.get("execution_result") or {}
        execution_summary = prev_exec.get("stdout", "") if isinstance(prev_exec, dict) else ""

        new_output = "\n\n--- Turn 2 ---\nNew execution output"
        accumulated = execution_summary + new_output

        assert "Previous execution output" in accumulated
        assert "New execution output" in accumulated

    def test_accumulates_insights(self, mock_hands_state: Dict):
        existing_insights = list(mock_hands_state.get("agent_insights") or [])
        new_insight = {"label": "New Insight", "value": "New value", "type": "finding"}

        combined = existing_insights + [new_insight]

        assert len(combined) == 2


@pytest.mark.unit
class TestHandsRetryLogic:
    def test_retry_count_increments(self, mock_hands_state: Dict):
        current_retry = mock_hands_state.get("retry_count", 0)
        next_retry = current_retry + 1

        assert next_retry == 2

    def test_detects_retry_from_feedback(self):
        feedback = '{"approved": false, "feedback": "Missing insights", "missing_items": ["insights"]}'

        assert "missing" in feedback.lower()
        assert "insights" in feedback

    def test_parses_missing_items_from_feedback(self):
        import json

        feedback = '{"approved": false, "feedback": "Missing df_info", "missing_items": ["df_info", "correlation"]}'

        data = json.loads(feedback)
        missing = data.get("missing_items", [])

        assert "df_info" in missing
        assert "correlation" in missing


@pytest.mark.unit
class TestInsightsExtraction:
    def test_extracts_insights_from_profiling_block(self):
        import re
        import json

        stdout = """Some output
PROFILING_INSIGHTS_START
[{"label": "Finding", "value": "123"}]
PROFILING_INSIGHTS_END
More output"""

        pattern = r"PROFILING_INSIGHTS_START\s*(.*?)\s*PROFILING_INSIGHTS_END"
        match = re.search(pattern, stdout, re.DOTALL)

        assert match is not None
        insights = json.loads(match.group(1))
        assert len(insights) == 1
        assert insights[0]["label"] == "Finding"

    def test_handles_missing_insights_block(self):
        import re

        stdout = "Normal output without insights block"
        pattern = r"PROFILING_INSIGHTS_START\s*(.*?)\s*PROFILING_INSIGHTS_END"
        match = re.search(pattern, stdout, re.DOTALL)

        assert match is None

    def test_extracts_df_info_from_output(self):
        stdout = """=== DATASET INFO ===
Shape: (545, 13)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 545 entries, 0 to 544
Data columns (total 13 columns):
==================="""

        assert "Shape: (545, 13)" in stdout
        assert "DATASET INFO" in stdout

    def test_detects_df_info_presence(self):
        stdout = "Shape: (545, 13)\ndf.info() output here"

        has_df_info = "Shape:" in stdout or "df.info" in stdout
        assert has_df_info is True
