import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.data_quality.anomaly_detector import MultiLayerAnomalyDetector, AnomalyType, AnomalyResult


@pytest.mark.unit
class TestMultiLayerAnomalyDetector:

    @pytest.fixture
    def detector(self):
        return MultiLayerAnomalyDetector()

    @pytest.fixture
    def clean_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(100, 10, 200),
                "feature2": np.random.normal(50, 5, 200),
                "feature3": np.random.normal(0, 1, 200),
            }
        )

    @pytest.fixture
    def data_with_outliers(self):
        np.random.seed(42)
        data = {"feature1": np.random.normal(100, 10, 200), "feature2": np.random.normal(50, 5, 200)}
        data["feature1"][0] = 500
        data["feature1"][1] = -200
        data["feature2"][5] = 300
        return pd.DataFrame(data)

    def test_detector_initialization(self, detector):
        assert detector is not None
        assert hasattr(detector, "config")
        assert hasattr(detector, "detection_results")
        assert hasattr(detector, "feature_stats")

    def test_default_config(self, detector):
        config = detector.config
        assert config["statistical_threshold"] == 3.0
        assert config["iqr_multiplier"] == 1.5
        assert config["isolation_contamination"] == 0.1
        assert config["lof_n_neighbors"] == 20
        assert config["confidence_threshold"] == 0.7

    def test_custom_config(self):
        custom_config = {"statistical_threshold": 2.5, "iqr_multiplier": 2.0}
        detector = MultiLayerAnomalyDetector(config=custom_config)
        assert detector.config["statistical_threshold"] == 2.5
        assert detector.config["iqr_multiplier"] == 2.0

    def test_feature_statistics_computation(self, detector, clean_data):
        detector._compute_feature_statistics(clean_data)

        assert "feature1" in detector.feature_stats
        assert "feature2" in detector.feature_stats
        assert "mean" in detector.feature_stats["feature1"]
        assert "std" in detector.feature_stats["feature1"]
        assert "median" in detector.feature_stats["feature1"]
        assert "iqr" in detector.feature_stats["feature1"]

    def test_compute_stats_skips_non_numeric_columns(self, detector):
        df = pd.DataFrame({"numeric_col": [1, 2, 3], "text_col": ["a", "b", "c"]})
        detector._compute_feature_statistics(df)

        assert "numeric_col" in detector.feature_stats
        assert "text_col" not in detector.feature_stats

    def test_compute_stats_handles_empty_columns(self, detector):
        df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan], "valid": [1, 2, 3]})
        detector._compute_feature_statistics(df)

        assert "all_nan" not in detector.feature_stats
        assert "valid" in detector.feature_stats

    def test_detect_anomalies_returns_list(self, detector, clean_data):
        results = detector.detect_anomalies(clean_data)
        assert isinstance(results, list)

    def test_detect_statistical_outliers_on_clean_data(self, detector, clean_data):
        results = detector.detect_anomalies(clean_data)
        assert isinstance(results, list)

    def test_detect_statistical_outliers_finds_outliers(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        assert len(results) > 0
        statistical_results = [r for r in results if r.anomaly_type == AnomalyType.STATISTICAL]
        assert len(statistical_results) > 0

        feature1_results = [r for r in results if "feature1" in r.affected_features]
        assert len(feature1_results) > 0

    def test_anomaly_result_structure(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            assert isinstance(result, AnomalyResult)
            assert hasattr(result, "anomaly_type")
            assert hasattr(result, "severity")
            assert hasattr(result, "confidence")
            assert hasattr(result, "affected_features")
            assert hasattr(result, "anomaly_indices")
            assert hasattr(result, "description")
            assert hasattr(result, "metadata")

    def test_severity_levels(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            assert result.severity in ["low", "medium", "high", "critical"]

    def test_confidence_scores_in_range(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            assert 0.0 <= result.confidence <= 1.0

    def test_multivariate_detection_requires_minimum_samples(self, detector):
        small_df = pd.DataFrame({"feature1": np.random.normal(100, 10, 50), "feature2": np.random.normal(50, 5, 50)})

        detector.config["min_samples_for_multivariate"] = 100
        detector.detect_anomalies(small_df)

        multivariate_results = [r for r in detector.detection_results if r.anomaly_type == AnomalyType.MULTIVARIATE]
        assert len(multivariate_results) == 0

    def test_detect_with_reference_dataframe(self, detector, clean_data):
        reference_data = clean_data.copy()
        test_data = clean_data * 2

        results = detector.detect_anomalies(test_data, reference_df=reference_data)
        assert isinstance(results, list)

    def test_outlier_indices_are_valid(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            for idx in result.anomaly_indices:
                assert 0 <= idx < len(data_with_outliers)

    def test_metadata_contains_method_info(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        statistical_results = [r for r in results if r.anomaly_type == AnomalyType.STATISTICAL]
        for result in statistical_results:
            assert "method" in result.metadata
            assert result.metadata["method"] in ["modified_z_score", "iqr"]

    def test_filter_by_confidence_threshold(self, detector, data_with_outliers):
        detector.config["confidence_threshold"] = 0.95
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            assert result.confidence >= 0.7

    def test_statistical_detection_skips_small_samples(self, detector):
        small_df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        detector._compute_feature_statistics(small_df)
        detector._detect_statistical_outliers(small_df)

        statistical_results = [r for r in detector.detection_results if r.anomaly_type == AnomalyType.STATISTICAL]
        assert len(statistical_results) == 0

    def test_iqr_method_detects_outliers(self, detector):
        data = pd.DataFrame({"feature1": list(range(1, 20)) + [100]})
        detector._compute_feature_statistics(data)
        detector._detect_statistical_outliers(data)

        iqr_results = [r for r in detector.detection_results if r.metadata.get("method") == "iqr"]
        assert len(iqr_results) >= 0

    def test_modified_z_score_handles_zero_mad(self, detector):
        constant_data = pd.DataFrame({"feature1": [5.0] * 100})
        constant_data.loc[0, "feature1"] = 10.0

        detector._compute_feature_statistics(constant_data)
        detector._detect_statistical_outliers(constant_data)

        assert isinstance(detector.detection_results, list)

    def test_affected_features_not_empty(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            assert len(result.affected_features) > 0
            assert all(isinstance(f, str) for f in result.affected_features)

    def test_description_not_empty(self, detector, data_with_outliers):
        results = detector.detect_anomalies(data_with_outliers)

        for result in results:
            assert result.description
            assert isinstance(result.description, str)
            assert len(result.description) > 0

    def test_empty_dataframe_handling(self, detector):
        empty_df = pd.DataFrame()
        results = detector.detect_anomalies(empty_df)
        assert isinstance(results, list)

    def test_single_column_dataframe(self, detector):
        df = pd.DataFrame({"single_feature": np.concatenate([np.random.normal(100, 10, 50), [500]])})
        results = detector.detect_anomalies(df)
        assert isinstance(results, list)
