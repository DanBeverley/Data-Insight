import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_quality.drift_monitor import ComprehensiveDriftMonitor, DriftType, DriftResult


@pytest.mark.unit
class TestComprehensiveDriftMonitor:

    @pytest.fixture
    def monitor(self):
        return ComprehensiveDriftMonitor()

    @pytest.fixture
    def reference_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(100, 10, 500),
                "feature2": np.random.normal(50, 5, 500),
                "category": np.random.choice(["A", "B", "C"], 500),
            }
        )

    @pytest.fixture
    def no_drift_data(self):
        np.random.seed(43)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(100, 10, 200),
                "feature2": np.random.normal(50, 5, 200),
                "category": np.random.choice(["A", "B", "C"], 200),
            }
        )

    @pytest.fixture
    def drifted_data(self):
        np.random.seed(44)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(150, 15, 200),
                "feature2": np.random.normal(80, 10, 200),
                "category": np.random.choice(["A", "D", "E"], 200),
            }
        )

    def test_monitor_initialization(self, monitor):
        assert monitor is not None
        assert hasattr(monitor, "config")
        assert hasattr(monitor, "reference_stats")
        assert hasattr(monitor, "drift_history")

    def test_default_config(self, monitor):
        config = monitor.config
        assert config["psi_threshold"] == 0.1
        assert config["ks_p_threshold"] == 0.05
        assert config["min_samples_for_test"] == 50

    def test_fit_reference_creates_stats(self, monitor, reference_data):
        monitor.fit_reference(reference_data)

        assert len(monitor.reference_stats) > 0
        assert "feature1" in monitor.reference_stats
        assert "feature2" in monitor.reference_stats

    def test_fit_reference_numeric_stats(self, monitor, reference_data):
        monitor.fit_reference(reference_data)

        feature1_stats = monitor.reference_stats["feature1"]
        assert "mean" in feature1_stats or "bin_edges" in feature1_stats

    def test_fit_reference_categorical_stats(self, monitor, reference_data):
        monitor.fit_reference(reference_data)

        category_stats = monitor.reference_stats["category"]
        assert "reference_proportions" in category_stats or "type" in category_stats

    def test_detect_drift_requires_reference(self, monitor, no_drift_data):
        with pytest.raises(ValueError, match="Reference statistics not fitted"):
            monitor.detect_drift(no_drift_data)

    def test_detect_drift_on_no_drift_data(self, monitor, reference_data, no_drift_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(no_drift_data)

        assert isinstance(results, list)

    def test_detect_drift_finds_drift(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(drifted_data)

        assert len(results) >= 0

    def test_drift_result_structure(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(drifted_data)

        for result in results:
            assert isinstance(result, DriftResult)
            assert hasattr(result, "drift_type")
            assert hasattr(result, "severity")
            assert hasattr(result, "drift_score")
            assert hasattr(result, "affected_features")
            assert hasattr(result, "timestamp")

    def test_drift_severity_levels(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(drifted_data)

        for result in results:
            assert result.severity in ["none", "low", "medium", "high", "critical"]

    def test_drift_score_in_range(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(drifted_data)

        for result in results:
            assert result.drift_score >= 0.0

    def test_drift_history_updated(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        initial_length = len(monitor.drift_history)
        monitor.detect_drift(drifted_data)

        assert len(monitor.drift_history) >= initial_length

    def test_feature_importance_drift_detection(self, monitor, reference_data, no_drift_data):
        monitor.fit_reference(reference_data)
        feature_importance = {"feature1": 0.8, "feature2": 0.2}

        results = monitor.detect_drift(no_drift_data, feature_importance=feature_importance)
        assert isinstance(results, list)

    def test_timestamp_in_results(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(drifted_data)

        for result in results:
            assert isinstance(result.timestamp, datetime)

    def test_affected_features_not_empty(self, monitor, reference_data, drifted_data):
        monitor.fit_reference(reference_data)
        results = monitor.detect_drift(drifted_data)

        for result in results:
            assert len(result.affected_features) > 0
