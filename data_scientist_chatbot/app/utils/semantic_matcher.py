"""Semantic pattern matching for intelligent code retrieval"""

import numpy as np
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
from datetime import datetime


class SemanticPatternMatcher:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        embedding_model = (
            config.get("advanced_features", {}).get("semantic_grouping", {}).get("embedding_model", "all-MiniLM-L6-v2")
        )

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(embedding_model)
        except ImportError:
            print("[SemanticMatcher] sentence-transformers not installed, pattern matching disabled")
            self.model = None

    def find_relevant_patterns(
        self, user_request: str, execution_history: List[Dict[str, Any]], top_k: int = 3, min_similarity: float = 0.5
    ) -> str:
        if not self.model or not execution_history:
            return ""

        request_embedding = self.model.encode(user_request, convert_to_numpy=True)

        scored_executions = []
        for execution in execution_history:
            task_description = self._extract_task_description(execution)
            if not task_description:
                continue

            task_embedding = self.model.encode(task_description, convert_to_numpy=True)
            similarity = self._cosine_similarity(request_embedding, task_embedding)

            if similarity >= min_similarity:
                quality_score = self._calculate_quality_score(execution)
                combined_score = (similarity * 0.7) + (quality_score * 0.3)
                scored_executions.append((combined_score, similarity, execution))

        if not scored_executions:
            return ""

        scored_executions.sort(reverse=True, key=lambda x: x[0])
        top_patterns = [(sim, exec) for _, sim, exec in scored_executions[:top_k]]

        return self._format_patterns(top_patterns)

    def _extract_task_description(self, execution: Dict[str, Any]) -> str:
        code = execution.get("code", "")
        context = execution.get("context", {})

        if context and "task_description" in context:
            return context["task_description"]

        lines = code.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        if non_empty_lines:
            return " ".join(non_empty_lines[:3])

        return code[:200]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _calculate_quality_score(self, execution: Dict[str, Any]) -> float:
        """Score pattern quality based on recency, success, and usage"""
        score = 0.0

        timestamp_str = execution.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - timestamp).days
                recency_score = max(0, 1.0 - (age_days / 30))
                score += recency_score * 0.4
            except:
                score += 0.2

        success = execution.get("success", False)
        score += 0.3 if success else 0.0

        usage_count = execution.get("usage_count", 0)
        usage_score = min(usage_count / 10, 0.3)
        score += usage_score

        return score

    def _format_patterns(self, top_patterns: List[tuple]) -> str:
        if not top_patterns:
            return ""

        formatted = "PROVEN APPROACHES FROM PAST SUCCESSFUL EXECUTIONS:\n\n"

        for i, (similarity, execution) in enumerate(top_patterns, 1):
            task_desc = self._extract_task_description(execution)
            code = execution.get("code", "")
            output = execution.get("output", "")

            code_preview = code[:400] + "..." if len(code) > 400 else code

            formatted += f"{i}. Similar task (confidence: {similarity:.2f})\n"
            formatted += f"   Description: {task_desc[:150]}\n"
            formatted += f"   Working code:\n```python\n{code_preview}\n```\n"

            if output:
                output_preview = output[:200]
                formatted += f"   Result: {output_preview}\n"

            formatted += "\n"

        formatted += "Adapt these proven patterns to the current task.\n"

        return formatted


def get_semantic_matcher() -> SemanticPatternMatcher:
    if not hasattr(get_semantic_matcher, "_instance"):
        get_semantic_matcher._instance = SemanticPatternMatcher()
    return get_semantic_matcher._instance
