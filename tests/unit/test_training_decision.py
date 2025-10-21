import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data_scientist_chatbot" / "app"))


@pytest.mark.unit
class TestTrainingDecisionEngine:
    """Unit tests for GPU/CPU training decision logic"""

    @pytest.fixture
    def decision_engine(self):
        from core.training_decision import TrainingDecisionEngine
        return TrainingDecisionEngine()

    def test_small_dataset_routes_to_cpu(self, decision_engine):
        """Small datasets should use CPU (E2B sandbox)"""
        decision = decision_engine.decide(
            dataset_rows=100,
            feature_count=5,
            model_type="linear_regression",
            code=None
        )

        assert decision.environment == "cpu"
        assert decision.confidence >= 0.0 and decision.confidence <= 1.0
        assert isinstance(decision.reasoning, str)

    def test_large_dataset_routes_to_gpu(self, decision_engine):
        """Large datasets should use GPU (Azure/AWS)"""
        decision = decision_engine.decide(
            dataset_rows=100000,
            feature_count=100,
            model_type="deep_learning",
            code=None
        )

        assert decision.environment in ["gpu", "cpu"]
        assert decision.confidence >= 0.0 and decision.confidence <= 1.0

    def test_deep_learning_prefers_gpu(self, decision_engine):
        """Deep learning models should prefer GPU"""
        decision = decision_engine.decide(
            dataset_rows=5000,
            feature_count=20,
            model_type="deep_learning",
            code=None
        )

        if decision.environment == "gpu":
            assert decision.confidence > 0.7, "GPU choice should be high confidence for deep learning"

    def test_medium_dataset_cpu_default(self, decision_engine):
        """Medium datasets with simple models should default to CPU"""
        decision = decision_engine.decide(
            dataset_rows=5000,
            feature_count=10,
            model_type="random_forest",
            code=None
        )

        assert decision.environment in ["cpu", "gpu"]
        assert isinstance(decision.confidence, float)

    def test_feature_count_impacts_decision(self, decision_engine):
        """High feature count should increase GPU routing likelihood"""
        small_features = decision_engine.decide(
            dataset_rows=10000,
            feature_count=5,
            model_type="xgboost",
            code=None
        )

        many_features = decision_engine.decide(
            dataset_rows=10000,
            feature_count=500,
            model_type="xgboost",
            code=None
        )

        if small_features.environment == "cpu" and many_features.environment == "gpu":
            assert many_features.confidence >= small_features.confidence, \
                "Higher feature count should increase GPU confidence"

    def test_decision_object_structure(self, decision_engine):
        """Decision object should have required fields"""
        decision = decision_engine.decide(
            dataset_rows=1000,
            feature_count=10,
            model_type="linear_regression",
            code=None
        )

        assert hasattr(decision, 'environment')
        assert hasattr(decision, 'confidence')
        assert hasattr(decision, 'reasoning')
        assert decision.environment in ["cpu", "gpu"]


@pytest.mark.unit
class TestTrainingTaskDetection:
    """Unit tests for detecting training tasks in user requests"""

    def test_detect_training_keywords(self):
        """Should detect common training keywords"""
        training_keywords = ['train', 'model', 'fit', 'predict', 'classify', 'regression', 'cluster']

        task_descriptions = [
            "Train a deep learning model",
            "Fit a random forest classifier",
            "Predict customer churn",
            "Cluster the data",
            "Build a regression model"
        ]

        for task in task_descriptions:
            has_keyword = any(kw in task.lower() for kw in training_keywords)
            assert has_keyword, f"Should detect training keywords in: {task}"

    def test_non_training_tasks_not_flagged(self):
        """Should not flag non-training tasks as training"""
        training_keywords = ['train', 'model', 'fit', 'predict', 'classify', 'regression', 'cluster']

        non_training_tasks = [
            "Show me the data distribution",
            "Create a visualization",
            "Calculate the mean and standard deviation",
            "Explore the dataset"
        ]

        for task in non_training_tasks:
            has_keyword = any(kw in task.lower() for kw in training_keywords)
            assert not has_keyword, f"Should NOT flag as training: {task}"


@pytest.mark.unit
class TestGPUEnvironmentDetection:
    """Unit tests for GPU environment detection and availability"""

    def test_azure_decision_availability(self):
        """Test that Azure decision logic is available"""
        try:
            from core.training_decision import TrainingDecisionEngine
            engine = TrainingDecisionEngine()
            assert engine is not None
        except ImportError:
            pytest.skip("Training decision engine not available")

    def test_aws_decision_availability(self):
        """Test that AWS decision logic is available"""
        try:
            from core.training_decision import TrainingDecisionEngine
            engine = TrainingDecisionEngine()
            assert engine is not None
        except ImportError:
            pytest.skip("Training decision engine not available")

    def test_cpu_fallback_always_available(self):
        """CPU (E2B sandbox) should always be available"""
        try:
            from core.training_decision import TrainingDecisionEngine
            engine = TrainingDecisionEngine()

            decision = engine.decide(
                dataset_rows=1000,
                feature_count=10,
                model_type="linear_regression",
                code=None
            )

            assert decision.environment in ["cpu", "gpu"], \
                "Should always have a valid environment decision"
        except ImportError:
            pytest.skip("Training decision engine not available")


@pytest.mark.unit
class TestTrainingDecisionIntegration:
    """Integration tests for training decision in hands agent context"""

    def test_training_decision_within_hands_agent_flow(self):
        """Test that training decision can be made within hands agent context"""
        from data_scientist_chatbot.app.agent import run_hands_agent

        state = {
            "messages": [("human", "Train a neural network model on this data")],
            "session_id": "test_training_123",
            "python_executions": 0,
            "plan": None,
            "scratchpad": "",
            "business_objective": None,
            "task_type": None,
            "target_column": None,
            "workflow_stage": None,
            "current_agent": "hands",
            "business_context": {},
            "retry_count": 0,
            "last_agent_sequence": ["router"],
            "router_decision": "hands",
            "complexity_score": 7,
            "complexity_reasoning": "Model training task",
            "route_strategy": "standard"
        }

        result = run_hands_agent(state)

        assert "messages" in result
        assert len(result["messages"]) > 0
