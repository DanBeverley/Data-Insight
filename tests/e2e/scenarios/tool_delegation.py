import pytest
from typing import List
from tests.e2e.scenarios.base_scenario import BaseScenario, ScenarioStep


@pytest.mark.e2e
class ToolDelegationScenario(BaseScenario):
    def __init__(self):
        super().__init__(
            name="Tool Delegation Flow",
            description="Tests agent's ability to delegate tasks across brain→hands→tools chain",
        )
        self.session_id = None
        self.dataset_uploaded = False
        self.correlation_result = None

    def setup(self) -> None:
        from src.api_utils.session_management import create_new_session

        self.session_id = create_new_session()["session_id"]

    def define_steps(self) -> List[ScenarioStep]:
        return [
            ScenarioStep(
                name="Upload dataset",
                action="upload_housing_data",
                expected_outcome="dataset uploaded successfully",
                timeout=30,
            ),
            ScenarioStep(
                name="Request correlation analysis",
                action="ask_correlation_price_area",
                expected_outcome="correlation",
                timeout=60,
            ),
            ScenarioStep(name="Request visualization", action="ask_for_heatmap", expected_outcome="plot", timeout=60),
            ScenarioStep(
                name="Follow-up with derived insight",
                action="ask_which_feature_strongest",
                expected_outcome="area",
                timeout=45,
            ),
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        if step.action == "upload_housing_data":
            return self._upload_dataset()
        elif step.action == "ask_correlation_price_area":
            return self._ask_correlation()
        elif step.action == "ask_for_heatmap":
            return self._ask_visualization()
        elif step.action == "ask_which_feature_strongest":
            return self._ask_followup()
        return False

    def _upload_dataset(self) -> bool:
        from src.api_utils.upload_handler import handle_upload
        import pandas as pd

        try:
            df = pd.DataFrame(
                {
                    "price": [300000, 450000, 250000, 500000, 350000],
                    "area": [1500, 2000, 1200, 2200, 1800],
                    "bedrooms": [3, 4, 2, 4, 3],
                    "bathrooms": [2, 3, 1, 3, 2],
                }
            )
            result = handle_upload(df, self.session_id)
            self.dataset_uploaded = result.get("status") == "success"
            return self.dataset_uploaded
        except Exception as e:
            self.errors.append(f"Upload failed: {e}")
            return False

    def _ask_correlation(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response("Show correlation between price and area", self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]

            has_correlation = "correlation" in response.lower() or "0." in response
            if has_correlation:
                self.correlation_result = response
            return has_correlation
        except Exception as e:
            self.errors.append(f"Correlation query failed: {e}")
            return False

    def _ask_visualization(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            has_plot = False
            for chunk in stream_agent_response("Create a correlation heatmap", self.session_id, False):
                if "plots" in chunk and chunk["plots"]:
                    has_plot = True
                if "content" in chunk:
                    response += chunk["content"]

            return has_plot or "heatmap" in response.lower()
        except Exception as e:
            self.errors.append(f"Visualization request failed: {e}")
            return False

    def _ask_followup(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response(
                "Which feature has the strongest correlation with price?", self.session_id, False
            ):
                if "content" in chunk:
                    response += chunk["content"]

            return "area" in response.lower()
        except Exception as e:
            self.errors.append(f"Follow-up query failed: {e}")
            return False

    def teardown(self) -> None:
        if self.session_id:
            from src.api_utils.session_management import clear_session

            try:
                clear_session(self.session_id)
            except:
                pass


@pytest.mark.tool_delegation
def test_tool_delegation_scenario():
    scenario = ToolDelegationScenario()
    result = scenario.execute()

    assert result.passed, f"Scenario failed with errors: {result.errors}"
    assert result.steps_completed == result.total_steps
    assert result.duration < 180
