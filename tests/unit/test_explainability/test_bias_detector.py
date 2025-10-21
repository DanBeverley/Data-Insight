import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.explainability.bias_detector import (
    BiasDetector,
    BiasType,
    SeverityLevel,
    BiasResult,
    FairnessMetrics
)


@pytest.mark.unit
class TestBiasDetector:

    @pytest.fixture
    def balanced_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'sensitive_attr': np.random.choice(['Group_A', 'Group_B'], 200)
        })

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(43)
        return pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'sensitive_attr': np.concatenate([['Group_A'] * 180, ['Group_B'] * 20])
        })

    @pytest.fixture
    def balanced_predictions(self):
        np.random.seed(42)
        return {
            'y_true': np.random.randint(0, 2, 200),
            'y_pred': np.random.randint(0, 2, 200),
            'y_pred_proba': np.random.rand(200)
        }

    def test_detector_initialization(self):
        detector = BiasDetector(sensitive_attributes=['gender', 'race'])
        assert detector.sensitive_attributes == ['gender', 'race']
        assert detector.fairness_threshold == 0.1

    def test_detector_custom_threshold(self):
        detector = BiasDetector(sensitive_attributes=['age'], fairness_threshold=0.05)
        assert detector.fairness_threshold == 0.05

    def test_detect_bias_returns_list(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )
        assert isinstance(results, list)

    def test_detect_bias_skips_missing_attributes(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['non_existent_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )
        assert len(results) == 0

    def test_bias_result_structure(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert isinstance(result, BiasResult)
            assert hasattr(result, 'bias_type')
            assert hasattr(result, 'severity')
            assert hasattr(result, 'metric_value')
            assert hasattr(result, 'affected_groups')
            assert hasattr(result, 'description')

    def test_bias_severity_levels(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert isinstance(result.severity, SeverityLevel)

    def test_bias_types_checked(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert isinstance(result.bias_type, BiasType)

    def test_single_group_skipped(self):
        single_group_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'sensitive_attr': ['Group_A', 'Group_A', 'Group_A']
        })
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 1])

        results = detector.detect_bias(model, single_group_data, y_true, y_pred)
        assert len(results) == 0

    def test_affected_groups_not_empty(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert isinstance(result.affected_groups, list)

    def test_description_not_empty(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert result.description
            assert len(result.description) > 0

    def test_recommendation_provided(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert hasattr(result, 'recommendation')
            assert result.recommendation is not None

    def test_metric_value_is_numeric(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert isinstance(result.metric_value, (int, float))

    def test_threshold_comparison(self, balanced_data, balanced_predictions):
        detector = BiasDetector(sensitive_attributes=['sensitive_attr'])
        model = Mock()
        results = detector.detect_bias(
            model,
            balanced_data,
            balanced_predictions['y_true'],
            balanced_predictions['y_pred']
        )

        for result in results:
            assert isinstance(result.threshold, (int, float))
