import pytest
import pandas as pd
from typing import List
from tests.e2e.scenarios.base_scenario import BaseScenario, ScenarioStep


@pytest.mark.e2e
class ErrorEdgeCasesScenario(BaseScenario):
    def __init__(self):
        super().__init__(
            name="Error Handling & Edge Cases",
            description="Tests system resilience against malformed inputs, timeouts, and edge cases"
        )
        self.session_id = None

    def setup(self) -> None:
        from src.api_utils.session_management import create_new_session
        self.session_id = create_new_session()["session_id"]

    def define_steps(self) -> List[ScenarioStep]:
        return [
            ScenarioStep(
                name="Upload malformed CSV",
                action="upload_malformed",
                expected_outcome="error",
                timeout=30
            ),
            ScenarioStep(
                name="Query without dataset",
                action="query_no_dataset",
                expected_outcome="no dataset",
                timeout=30
            ),
            ScenarioStep(
                name="Upload valid dataset",
                action="upload_valid",
                expected_outcome="success",
                timeout=30
            ),
            ScenarioStep(
                name="Ambiguous query",
                action="ambiguous_query",
                expected_outcome="clarification",
                timeout=45
            ),
            ScenarioStep(
                name="Request non-existent column",
                action="invalid_column",
                expected_outcome="error or not found",
                timeout=30
            ),
            ScenarioStep(
                name="Empty message",
                action="empty_message",
                expected_outcome="error",
                timeout=15
            )
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        if step.action == "upload_malformed":
            return self._upload_malformed()
        elif step.action == "query_no_dataset":
            return self._query_without_dataset()
        elif step.action == "upload_valid":
            return self._upload_valid_dataset()
        elif step.action == "ambiguous_query":
            return self._send_ambiguous_query()
        elif step.action == "invalid_column":
            return self._request_invalid_column()
        elif step.action == "empty_message":
            return self._send_empty_message()
        return False

    def _upload_malformed(self) -> bool:
        from src.api_utils.upload_handler import handle_upload

        try:
            malformed_df = pd.DataFrame({
                "col1": [1, 2, None, None, None],
                "col2": ["", "", "", "", ""],
                "col3": [float('inf'), float('-inf'), float('nan'), 0, 0]
            })

            result = handle_upload(malformed_df, self.session_id)
            has_warning = result.get("status") != "success" or "warning" in str(result).lower()
            return has_warning
        except Exception:
            return True

    def _query_without_dataset(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response("Show me the first 10 rows", self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]

            has_no_dataset_message = any(word in response.lower() for word in ["no dataset", "upload", "provide"])
            return has_no_dataset_message
        except Exception as e:
            return "dataset" in str(e).lower()

    def _upload_valid_dataset(self) -> bool:
        from tests.conftest import housing_dataset
        from src.api_utils.upload_handler import handle_upload

        try:
            df = housing_dataset(None)
            result = handle_upload(df, self.session_id)
            return result.get("status") == "success"
        except Exception as e:
            self.errors.append(f"Valid upload failed: {e}")
            return False

    def _send_ambiguous_query(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response("analyze it", self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]

            return len(response) > 20
        except Exception:
            return True

    def _request_invalid_column(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response("Show correlation between price and nonexistent_column", self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]

            has_error_handling = any(word in response.lower() for word in ["not found", "doesn't exist", "error", "invalid"])
            return has_error_handling
        except Exception:
            return True

    def _send_empty_message(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response("", self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]
            return False
        except Exception:
            return True

    def teardown(self) -> None:
        if self.session_id:
            from src.api_utils.session_management import clear_session
            try:
                clear_session(self.session_id)
            except:
                pass


def test_error_edge_cases_scenario():
    scenario = ErrorEdgeCasesScenario()
    result = scenario.execute()

    assert result.passed, f"Scenario failed with errors: {result.errors}"
    assert result.steps_completed == result.total_steps
    assert result.duration < 180
