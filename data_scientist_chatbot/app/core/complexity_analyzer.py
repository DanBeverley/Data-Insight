"""LLM-driven task complexity analysis for intelligent routing"""

from typing import Dict, Any
from dataclasses import dataclass
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplexityAssessment:
    score: int
    reasoning: str
    route_strategy: str
    needs_planning: bool


class ComplexityAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    def analyze(self, user_request: str, session_context: Dict[str, Any]) -> ComplexityAssessment:
        """Assess task complexity and determine optimal routing strategy"""

        has_dataset = session_context.get("has_dataset", False)
        previous_artifacts = session_context.get("artifact_count", 0)

        prompt = f"""Analyze the complexity of this data science request:

                    REQUEST: "{user_request}"

                    CONTEXT:
                    - Dataset loaded: {has_dataset}
                    - Previous artifacts: {previous_artifacts}

                    Rate complexity (1-10) and suggest routing:

                    COMPLEXITY LEVELS:
                    1-3 (Simple): Single operation, no analysis needed
                    Examples: "show first 5 rows", "describe the data", "list columns"
                    Route: direct - execute immediately, minimal summary

                    4-7 (Medium): Standard analysis or modeling task
                    Examples: "build a classifier", "plot correlation", "train random forest"
                    Route: standard - execute then summarize results

                    8-10 (Complex): Multi-step workflow requiring planning
                    Examples: "compare 3 models and recommend best", "full EDA with insights", "optimize hyperparameters and validate"
                    Route: collaborative - plan steps, then execute iteratively

                    Respond ONLY with JSON:
                    {{"complexity": <1-10>, "reasoning": "<one sentence>", "route": "direct|standard|collaborative"}}"""

        try:
            response = self.llm.invoke([("human", prompt)])
            content = response.content if hasattr(response, "content") else str(response)

            assessment_data = self._parse_response(content)

            return ComplexityAssessment(
                score=assessment_data["complexity"],
                reasoning=assessment_data["reasoning"],
                route_strategy=assessment_data["route"],
                needs_planning=assessment_data["complexity"] >= 8,
            )
        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}, defaulting to standard")
            return ComplexityAssessment(
                score=5,
                reasoning="Default: complexity analysis unavailable",
                route_strategy="standard",
                needs_planning=False,
            )

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""

        json_match = re.search(r"\{[^}]+\}", content)
        if json_match:
            try:
                data = json.loads(json_match.group(0))

                if "complexity" in data and "route" in data:
                    return {
                        "complexity": int(data["complexity"]),
                        "reasoning": data.get("reasoning", "No reasoning provided"),
                        "route": data["route"],
                    }
            except (json.JSONDecodeError, ValueError):
                pass

        if any(kw in content.lower() for kw in ["simple", "trivial", "direct"]):
            return {"complexity": 2, "reasoning": "Simple task detected", "route": "direct"}
        elif any(kw in content.lower() for kw in ["complex", "multi-step", "planning"]):
            return {"complexity": 8, "reasoning": "Complex task detected", "route": "collaborative"}
        else:
            return {"complexity": 5, "reasoning": "Standard task", "route": "standard"}


def create_complexity_analyzer(llm) -> ComplexityAnalyzer:
    return ComplexityAnalyzer(llm)
