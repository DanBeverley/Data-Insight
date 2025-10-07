import pytest
from typing import List
from tests.e2e.scenarios.base_scenario import BaseScenario, ScenarioStep


@pytest.mark.e2e
class WebSearchIntegrationScenario(BaseScenario):
    def __init__(self):
        super().__init__(
            name="Web Search Integration",
            description="Tests agent's ability to use web search for domain knowledge augmentation"
        )
        self.session_id = None
        self.search_used = False

    def setup(self) -> None:
        from src.api_utils.session_management import create_new_session
        self.session_id = create_new_session()["session_id"]

    def define_steps(self) -> List[ScenarioStep]:
        return [
            ScenarioStep(
                name="Query requiring domain knowledge",
                action="ask_domain_question",
                expected_outcome="search or external",
                timeout=60
            ),
            ScenarioStep(
                name="Upload dataset and ask context-specific question",
                action="upload_and_ask_context",
                expected_outcome="analysis",
                timeout=90
            ),
            ScenarioStep(
                name="Request comparison with external benchmark",
                action="ask_benchmark_comparison",
                expected_outcome="comparison",
                timeout=60
            )
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        if step.action == "ask_domain_question":
            return self._ask_domain_knowledge()
        elif step.action == "upload_and_ask_context":
            return self._upload_and_query()
        elif step.action == "ask_benchmark_comparison":
            return self._ask_benchmark()
        return False

    def _ask_domain_knowledge(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response(
                "What are the key factors affecting housing prices in 2024?",
                self.session_id,
                web_search_enabled=True
            ):
                if "content" in chunk:
                    response += chunk["content"]
                if "tool_calls" in chunk:
                    for tool in chunk.get("tool_calls", []):
                        if tool.get("name") == "web_search":
                            self.search_used = True

            return len(response) > 50 and ("location" in response.lower() or "market" in response.lower())
        except Exception as e:
            self.errors.append(f"Domain knowledge query failed: {e}")
            return False

    def _upload_and_query(self) -> bool:
        from tests.conftest import housing_dataset
        from src.api_utils.upload_handler import handle_upload
        from src.api_utils.agent_response import stream_agent_response

        try:
            df = housing_dataset(None)
            upload_result = handle_upload(df, self.session_id)
            if upload_result.get("status") != "success":
                return False

            response = ""
            for chunk in stream_agent_response(
                "Analyze this housing dataset and explain the price trends",
                self.session_id,
                web_search_enabled=False
            ):
                if "content" in chunk:
                    response += chunk["content"]

            return "price" in response.lower() and len(response) > 100
        except Exception as e:
            self.errors.append(f"Context query failed: {e}")
            return False

    def _ask_benchmark(self) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response(
                "How does the average price in this dataset compare to national housing market averages?",
                self.session_id,
                web_search_enabled=True
            ):
                if "content" in chunk:
                    response += chunk["content"]

            return len(response) > 50
        except Exception as e:
            self.errors.append(f"Benchmark comparison failed: {e}")
            return False

    def teardown(self) -> None:
        if self.session_id:
            from src.api_utils.session_management import clear_session
            try:
                clear_session(self.session_id)
            except:
                pass


def test_web_search_integration_scenario():
    scenario = WebSearchIntegrationScenario()
    result = scenario.execute()

    assert result.passed, f"Scenario failed with errors: {result.errors}"
    assert result.steps_completed == result.total_steps
    assert result.duration < 240
