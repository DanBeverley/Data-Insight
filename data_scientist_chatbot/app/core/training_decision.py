"""Training environment decision engine - Hybrid approach"""

from typing import Optional
from dataclasses import dataclass
import re
import time


@dataclass
class TrainingDecision:
    environment: str
    reasoning: str
    confidence: float
    data_points: int


class TrainingDecisionEngine:
    """
    Hybrid decision engine for CPU vs GPU training environment

    Strategy:
    1. Fast path: Obvious cases based on data points
    2. Deep learning detection: Always GPU
    3. Borderline cases: Quick profiling on sample
    """

    def __init__(self):
        self.obvious_cpu_threshold = 1_000_000
        self.obvious_gpu_threshold = 100_000_000
        self.profile_timeout = 10

    def decide(
        self, dataset_rows: int, feature_count: int, model_type: str = "", code: Optional[str] = None
    ) -> TrainingDecision:
        """
        Main decision logic using hybrid approach

        Args:
            dataset_rows: Number of rows in dataset
            feature_count: Number of features/columns
            model_type: Model type from user request
            code: Training code to analyze

        Returns:
            TrainingDecision with environment and reasoning
        """
        data_points = dataset_rows * feature_count

        # Fast path 1: Obvious CPU case
        if data_points < self.obvious_cpu_threshold:
            return TrainingDecision(
                environment="cpu",
                reasoning=f"Small dataset ({dataset_rows:,} rows × {feature_count} features = {data_points:,} data points)",
                confidence=0.95,
                data_points=data_points,
            )

        # Fast path 2: Obvious GPU case
        if data_points > self.obvious_gpu_threshold:
            return TrainingDecision(
                environment="gpu",
                reasoning=f"Very large dataset ({dataset_rows:,} rows × {feature_count} features = {data_points:,} data points)",
                confidence=0.95,
                data_points=data_points,
            )

        # Fast path 3: Deep learning always GPU
        if self._is_deep_learning(code, model_type):
            return TrainingDecision(
                environment="gpu",
                reasoning="Deep learning model detected (neural networks require GPU acceleration)",
                confidence=0.98,
                data_points=data_points,
            )

        # Borderline case: Profile on sample
        if code:
            profile_result = self._profile_and_decide(code, dataset_rows, feature_count, data_points)
            if profile_result:
                return profile_result

        # Fallback: Use simple analytical heuristic
        return self._analytical_fallback(dataset_rows, feature_count, data_points, model_type)

    def _is_deep_learning(self, code: Optional[str], model_type: str) -> bool:
        """Detect deep learning frameworks and models"""
        if not code and not model_type:
            return False

        dl_patterns = [
            r"import\s+torch",
            r"from\s+torch",
            r"import\s+tensorflow",
            r"from\s+tensorflow",
            r"import\s+keras",
            r"from\s+keras",
            r"nn\.Module",
            r"Sequential\(",
            r"Conv\d+D",
            r"LSTM",
            r"Dense\(",
        ]

        dl_keywords = ["neural", "deep", "cnn", "rnn", "lstm", "transformer", "bert", "resnet"]

        if code:
            for pattern in dl_patterns:
                if re.search(pattern, code):
                    return True

        model_lower = model_type.lower()
        return any(keyword in model_lower for keyword in dl_keywords)

    def _profile_and_decide(
        self, code: str, dataset_rows: int, feature_count: int, data_points: int
    ) -> Optional[TrainingDecision]:
        """
        Profile training on small sample to estimate complexity
        Returns None if profiling fails
        """
        try:
            sample_size = min(1000, max(100, dataset_rows // 100))

            complexity_indicators = self._analyze_code_complexity(code)

            # High complexity code → GPU
            if complexity_indicators["score"] > 5:
                return TrainingDecision(
                    environment="gpu",
                    reasoning=f"Complex training detected: {', '.join(complexity_indicators['reasons'])}",
                    confidence=0.85,
                    data_points=data_points,
                )

            # Medium dataset with medium complexity
            if data_points > 10_000_000 and complexity_indicators["score"] > 2:
                return TrainingDecision(
                    environment="gpu",
                    reasoning=f"Moderate complexity with substantial data ({data_points:,} data points)",
                    confidence=0.75,
                    data_points=data_points,
                )

            # Otherwise CPU is sufficient
            return TrainingDecision(
                environment="cpu",
                reasoning=f"Moderate dataset ({data_points:,} data points) with standard complexity",
                confidence=0.80,
                data_points=data_points,
            )

        except Exception as e:
            return None

    def _analyze_code_complexity(self, code: str) -> dict:
        """
        Analyze code for computational complexity indicators
        Returns score and reasons
        """
        score = 0
        reasons = []

        # Gradient boosting with many trees
        xgb_trees = re.search(r"n_estimators\s*=\s*(\d+)", code)
        if xgb_trees and int(xgb_trees.group(1)) > 500:
            score += 2
            reasons.append(f"many boosting iterations ({xgb_trees.group(1)} trees)")

        # Random forest with many trees
        rf_trees = re.search(r"n_estimators\s*=\s*(\d+)", code)
        if rf_trees and int(rf_trees.group(1)) > 200:
            score += 1
            reasons.append(f"ensemble with {rf_trees.group(1)} trees")

        # Cross-validation or grid search
        if re.search(r"GridSearchCV|RandomizedSearchCV|cross_val", code):
            score += 3
            reasons.append("hyperparameter tuning with cross-validation")

        # Multiple model training
        fit_count = code.count(".fit(")
        if fit_count > 5:
            score += 2
            reasons.append(f"training {fit_count} models")

        # Large iteration loops
        large_loop = re.search(r"for\s+\w+\s+in\s+range\s*\(\s*(\d+)", code)
        if large_loop and int(large_loop.group(1)) > 10000:
            score += 2
            reasons.append("extensive iteration loops")

        # Nested loops
        if code.count("for ") > 3:
            score += 1
            reasons.append("nested iteration patterns")

        return {"score": score, "reasons": reasons}

    def _analytical_fallback(
        self, dataset_rows: int, feature_count: int, data_points: int, model_type: str
    ) -> TrainingDecision:
        """
        Simple analytical decision when profiling unavailable
        Based purely on data scale
        """
        # Use data points as primary metric
        if data_points > 50_000_000:
            return TrainingDecision(
                environment="gpu",
                reasoning=f"Large-scale dataset ({data_points:,} data points)",
                confidence=0.70,
                data_points=data_points,
            )

        # Check for known GPU-friendly algorithms
        model_lower = model_type.lower()
        gpu_friendly = ["xgboost", "lightgbm", "catboost"]

        if any(algo in model_lower for algo in gpu_friendly) and data_points > 10_000_000:
            return TrainingDecision(
                environment="gpu",
                reasoning=f"Gradient boosting with substantial data ({data_points:,} data points)",
                confidence=0.75,
                data_points=data_points,
            )

        # Default to CPU
        return TrainingDecision(
            environment="cpu",
            reasoning=f"Standard training task ({data_points:,} data points)",
            confidence=0.70,
            data_points=data_points,
        )

    def should_use_gpu(
        self, dataset_rows: int, feature_count: int, model_type: str = "", code: Optional[str] = None
    ) -> bool:
        """Simple boolean check for GPU usage"""
        decision = self.decide(dataset_rows, feature_count, model_type, code)
        return decision.environment == "gpu"
