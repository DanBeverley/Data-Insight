import json
import ollama
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class LLMTestGenerator:
    PROMPTS = {
        "adversarial": """You are a QA engineer testing an AI data science agent. Generate 10 queries designed to break the system.

Focus on:
- Infinite loops (circular references, recursive delegation)
- Tool delegation failures (invalid arguments, non-existent tools)
- Memory corruption (context overflow, session conflicts)
- Edge cases (empty data, malformed inputs, extreme values)

Output format: JSON array
[
  {
    "query": "User query that might break the system",
    "expected_failure_type": "infinite_loop | tool_error | memory_error | validation_error",
    "description": "Brief explanation of why this might fail"
  }
]

Generate diverse, realistic failure scenarios.""",

        "coverage": """You are a QA engineer creating comprehensive test coverage for a data science AI agent.

Generate 20 diverse queries covering:
- Exploratory data analysis (correlations, distributions, outliers)
- Statistical tests (t-tests, ANOVA, chi-square)
- Machine learning (regression, classification, clustering)
- Visualization (scatter plots, heatmaps, time series)
- Data cleaning (missing values, duplicates, normalization)

Use realistic domain-specific terminology (housing, sales, finance, healthcare).

Output format: JSON array
[
  {
    "query": "Realistic user query",
    "category": "eda | statistics | ml | visualization | cleaning",
    "domain": "housing | sales | finance | healthcare",
    "expected_tool": "python_code_interpreter | web_search | knowledge_graph"
  }
]""",

        "conversation": """You are simulating a multi-turn conversation between a data analyst and an AI assistant.

Generate a 10-turn conversation that includes:
- Context building (upload data, ask about columns)
- Follow-up questions (based on previous answers)
- Clarification requests
- Memory references ("like you showed earlier", "the correlation we calculated")
- Progressive complexity (start simple, become more complex)

Output format: JSON array
[
  {
    "turn": 1,
    "speaker": "user | ai",
    "message": "Message content",
    "requires_memory": false,
    "references_turn": null
  }
]

Make it realistic and coherent.""",

        "regression": """You are generating test variants for known bug scenarios.

Based on this bug pattern: {bug_pattern}

Generate 5 variations that might trigger the same or similar issues:
- Same intent, different phrasing
- Edge cases near the boundary
- Combined with other features
- Extreme values or empty inputs

Output format: JSON array
[
  {
    "query": "Variation query",
    "similarity_score": 0.8,
    "variation_type": "phrasing | edge_case | combination | extreme"
  }
]"""
    }

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        self.output_dir = Path("tests/e2e/scenarios/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_scenarios(self, mode: str, count: int = 10, bug_pattern: str = None) -> List[Dict[str, Any]]:
        if mode not in self.PROMPTS:
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self.PROMPTS.keys())}")

        prompt = self.PROMPTS[mode]
        if mode == "regression" and bug_pattern:
            prompt = prompt.format(bug_pattern=bug_pattern)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.8,
                    "num_predict": 2000,
                    "seed": None
                }
            )

            scenarios = self._parse_json_response(response["response"])
            return scenarios[:count] if scenarios else []

        except Exception as e:
            print(f"LLM generation failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)
        except json.JSONDecodeError:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except:
                    pass
            return []

    def generate_test_file(self, scenario: Dict[str, Any], scenario_id: str) -> Path:
        if "query" in scenario:
            test_code = self._create_single_query_test(scenario, scenario_id)
        elif "turn" in scenario:
            test_code = self._create_conversation_test(scenario, scenario_id)
        else:
            return None

        output_path = self.output_dir / f"{scenario_id}.py"
        output_path.write_text(test_code)
        return output_path

    def _create_single_query_test(self, scenario: Dict[str, Any], scenario_id: str) -> str:
        query = scenario.get("query", "")
        category = scenario.get("category", "general")
        expected_tool = scenario.get("expected_tool", "python_code_interpreter")

        return f'''import pytest
from tests.e2e.scenarios.base_scenario import BaseScenario, ScenarioStep
from typing import List


@pytest.mark.e2e
@pytest.mark.generated
class {scenario_id.replace("_", " ").title().replace(" ", "")}Scenario(BaseScenario):
    def __init__(self):
        super().__init__(
            name="{scenario_id}",
            description="Generated test: {category}"
        )
        self.session_id = None

    def setup(self) -> None:
        from src.api_utils.session_management import create_new_session
        self.session_id = create_new_session()["session_id"]

    def define_steps(self) -> List[ScenarioStep]:
        return [
            ScenarioStep(
                name="Execute query",
                action="execute_query",
                expected_outcome="response",
                timeout=60
            )
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        try:
            response = ""
            for chunk in stream_agent_response("{query}", self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]

            return len(response) > 10
        except Exception as e:
            self.errors.append(str(e))
            return False

    def teardown(self) -> None:
        if self.session_id:
            from src.api_utils.session_management import clear_session
            try:
                clear_session(self.session_id)
            except:
                pass


def test_{scenario_id}():
    scenario = {scenario_id.replace("_", " ").title().replace(" ", "")}Scenario()
    result = scenario.execute()
    assert result.passed, f"Scenario failed: {{result.errors}}"
'''

    def _create_conversation_test(self, conversation: List[Dict[str, Any]], scenario_id: str) -> str:
        return f'''import pytest
from tests.e2e.scenarios.base_scenario import BaseScenario, ScenarioStep
from typing import List


@pytest.mark.e2e
@pytest.mark.generated
class {scenario_id.replace("_", " ").title().replace(" ", "")}Scenario(BaseScenario):
    def __init__(self):
        super().__init__(
            name="{scenario_id}",
            description="Generated multi-turn conversation test"
        )
        self.session_id = None
        self.conversation = {json.dumps(conversation, indent=8)}

    def setup(self) -> None:
        from src.api_utils.session_management import create_new_session
        self.session_id = create_new_session()["session_id"]

    def define_steps(self) -> List[ScenarioStep]:
        user_turns = [turn for turn in self.conversation if turn["speaker"] == "user"]
        return [
            ScenarioStep(
                name=f"Turn {{i+1}}: {{turn['message'][:50]}}",
                action=f"turn_{{i}}",
                expected_outcome="response",
                timeout=45
            )
            for i, turn in enumerate(user_turns)
        ]

    def _execute_step(self, step: ScenarioStep) -> bool:
        from src.api_utils.agent_response import stream_agent_response

        turn_index = int(step.action.split("_")[1])
        user_turn = [t for t in self.conversation if t["speaker"] == "user"][turn_index]

        try:
            response = ""
            for chunk in stream_agent_response(user_turn["message"], self.session_id, False):
                if "content" in chunk:
                    response += chunk["content"]

            return len(response) > 10
        except Exception as e:
            self.errors.append(str(e))
            return False

    def teardown(self) -> None:
        if self.session_id:
            from src.api_utils.session_management import clear_session
            try:
                clear_session(self.session_id)
            except:
                pass


def test_{scenario_id}():
    scenario = {scenario_id.replace("_", " ").title().replace(" ", "")}Scenario()
    result = scenario.execute()
    assert result.passed, f"Scenario failed: {{result.errors}}"
'''

    def generate_batch(self, mode: str, count: int) -> List[Path]:
        scenarios = self.generate_scenarios(mode, count)
        generated_files = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, scenario in enumerate(scenarios):
            scenario_id = f"{mode}_{timestamp}_{i:03d}"
            file_path = self.generate_test_file(scenario, scenario_id)
            if file_path:
                generated_files.append(file_path)

        return generated_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate test scenarios using LLM")
    parser.add_argument("--mode", choices=["adversarial", "coverage", "conversation", "regression"], required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--bug-pattern", help="Bug pattern for regression mode")

    args = parser.parse_args()

    generator = LLMTestGenerator(model=args.model)

    if args.mode == "regression" and not args.bug_pattern:
        print("Error: --bug-pattern required for regression mode")
        return

    print(f"Generating {args.count} {args.mode} scenarios...")
    files = generator.generate_batch(args.mode, args.count)

    print(f"Generated {len(files)} test files:")
    for file in files:
        print(f"  - {file}")


if __name__ == "__main__":
    main()
