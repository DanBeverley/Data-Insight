"""Comprehensive Drift Detection System for DataInsight AI"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.spatial.distance import jensenshannon


class DriftType(Enum):
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"
    MODEL_BASED = "model_based"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class DriftResult:
    drift_type: DriftType
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    drift_score: float  # 0-1 drift intensity
    affected_features: List[str]
    p_value: Optional[float]
    description: str
    metadata: Dict[str, Any]
    timestamp: datetime


class ComprehensiveDriftMonitor:
    """Advanced drift detection with multiple statistical and model-based methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.reference_stats: Dict[str, Dict] = {}
        self.drift_history: List[DriftResult] = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for drift detection."""
        return {
            "psi_threshold": 0.1,  # Population Stability Index threshold
            "ks_p_threshold": 0.05,  # KS test p-value threshold
            "js_threshold": 0.1,  # Jensen-Shannon divergence threshold
            "domain_classifier_threshold": 0.7,  # Domain classifier AUC threshold
            "feature_importance_threshold": 0.1,  # Feature importance drift threshold
            "min_samples_for_test": 50,  # Minimum samples for statistical tests
            "categorical_max_unique": 50,  # Max unique values for categorical analysis
            "drift_severity_thresholds": {"low": 0.1, "medium": 0.25, "high": 0.5, "critical": 0.75},
        }

    def fit_reference(self, reference_df: pd.DataFrame) -> None:
        """Fit reference statistics for drift detection."""
        logging.info(f"Fitting reference statistics on {len(reference_df)} samples")

        self.reference_stats = {}

        for column in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[column]):
                self._compute_numeric_reference_stats(reference_df[column], column)
            else:
                self._compute_categorical_reference_stats(reference_df[column], column)

        logging.info(f"Reference statistics computed for {len(self.reference_stats)} features")

    def detect_drift(
        self, current_df: pd.DataFrame, feature_importance: Optional[Dict[str, float]] = None
    ) -> List[DriftResult]:
        """
        Detect drift between current data and reference statistics.

        Args:
            current_df: Current dataset to check for drift
            feature_importance: Optional feature importance from a model

        Returns:
            List of detected drift results
        """
        if not self.reference_stats:
            raise ValueError("Reference statistics not fitted. Call fit_reference() first.")

        logging.info(f"Starting drift detection on {len(current_df)} samples")

        drift_results = []
        timestamp = datetime.now()

        # Statistical drift tests
        for column in current_df.columns:
            if column not in self.reference_stats:
                continue

            if pd.api.types.is_numeric_dtype(current_df[column]):
                results = self._detect_numeric_drift(current_df[column], column, timestamp)
            else:
                results = self._detect_categorical_drift(current_df[column], column, timestamp)

            drift_results.extend(results)

        # Model-based drift detection (domain classifier approach)
        domain_drift = self._detect_domain_drift(current_df, timestamp)
        if domain_drift:
            drift_results.append(domain_drift)

        # Feature importance drift
        if feature_importance:
            importance_drift = self._detect_feature_importance_drift(feature_importance, timestamp)
            if importance_drift:
                drift_results.append(importance_drift)

        # Store in history
        self.drift_history.extend(drift_results)

        # Filter by severity and return
        significant_drifts = [d for d in drift_results if d.severity != "none"]

        logging.info(f"Detected {len(significant_drifts)} significant drifts out of {len(drift_results)} tests")
        return significant_drifts

    def _compute_numeric_reference_stats(self, series: pd.Series, column: str) -> None:
        """Compute reference statistics for numeric features."""
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return

        # Create bins for Population Stability Index
        try:
            # Use quantile-based binning for better distribution
            n_bins = min(10, len(clean_series.unique()))
            if n_bins < 2:
                bins = [clean_series.min(), clean_series.max()]
                bin_edges = np.array(bins)
            else:
                bin_edges = np.unique(np.percentile(clean_series, np.linspace(0, 100, n_bins + 1)))

            # Ensure proper bin edges
            if len(bin_edges) < 2:
                bin_edges = np.array([clean_series.min() - 1, clean_series.max() + 1])

            # Compute reference distribution
            ref_counts, _ = np.histogram(clean_series, bins=bin_edges)
            ref_proportions = ref_counts / ref_counts.sum()

            # Avoid zero proportions (add small epsilon)
            ref_proportions = np.maximum(ref_proportions, 1e-6)
            ref_proportions = ref_proportions / ref_proportions.sum()  # Renormalize

            self.reference_stats[column] = {
                "type": "numeric",
                "bin_edges": bin_edges,
                "reference_proportions": ref_proportions,
                "mean": float(clean_series.mean()),
                "std": float(clean_series.std()),
                "median": float(clean_series.median()),
                "q25": float(clean_series.quantile(0.25)),
                "q75": float(clean_series.quantile(0.75)),
                "sample_size": len(clean_series),
            }
        except Exception as e:
            logging.warning(f"Failed to compute reference stats for {column}: {e}")

    def _compute_categorical_reference_stats(self, series: pd.Series, column: str) -> None:
        """Compute reference statistics for categorical features."""
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return

        value_counts = clean_series.value_counts()

        # Limit to most frequent categories if too many unique values
        if len(value_counts) > self.config["categorical_max_unique"]:
            value_counts = value_counts.head(self.config["categorical_max_unique"])

        total_count = value_counts.sum()
        proportions = value_counts / total_count

        self.reference_stats[column] = {
            "type": "categorical",
            "reference_proportions": proportions.to_dict(),
            "total_categories": len(value_counts),
            "sample_size": total_count,
        }

    def _detect_numeric_drift(self, current_series: pd.Series, column: str, timestamp: datetime) -> List[DriftResult]:
        """Detect drift in numeric features."""
        results = []
        ref_stats = self.reference_stats[column]
        current_clean = current_series.dropna()

        if len(current_clean) < self.config["min_samples_for_test"]:
            return results

        # Population Stability Index (PSI)
        try:
            current_counts, _ = np.histogram(current_clean, bins=ref_stats["bin_edges"])
            current_proportions = current_counts / current_counts.sum()
            current_proportions = np.maximum(current_proportions, 1e-6)
            current_proportions = current_proportions / current_proportions.sum()

            # PSI calculation
            psi = np.sum(
                (current_proportions - ref_stats["reference_proportions"])
                * np.log(current_proportions / ref_stats["reference_proportions"])
            )

            severity = self._calculate_drift_severity(psi, self.config["psi_threshold"])

            if psi > self.config["psi_threshold"]:
                results.append(
                    DriftResult(
                        drift_type=DriftType.STATISTICAL,
                        severity=severity,
                        drift_score=float(psi),
                        affected_features=[column],
                        p_value=None,
                        description=f"Population Stability Index drift in {column}",
                        metadata={
                            "method": "psi",
                            "psi_value": float(psi),
                            "threshold": self.config["psi_threshold"],
                            "reference_sample_size": ref_stats["sample_size"],
                            "current_sample_size": len(current_clean),
                        },
                        timestamp=timestamp,
                    )
                )
        except Exception as e:
            logging.warning(f"PSI calculation failed for {column}: {e}")

        # Kolmogorov-Smirnov test
        try:
            # Create reference sample for comparison
            ref_sample = np.random.normal(
                ref_stats["mean"], ref_stats["std"], min(len(current_clean), 1000)  # Limit size for efficiency
            )

            ks_statistic, ks_p_value = stats.ks_2samp(current_clean, ref_sample)

            if ks_p_value < self.config["ks_p_threshold"]:
                severity = self._calculate_drift_severity(ks_statistic, 0.1)

                results.append(
                    DriftResult(
                        drift_type=DriftType.DISTRIBUTION,
                        severity=severity,
                        drift_score=float(ks_statistic),
                        affected_features=[column],
                        p_value=float(ks_p_value),
                        description=f"Distribution drift detected in {column} (KS test)",
                        metadata={
                            "method": "kolmogorov_smirnov",
                            "ks_statistic": float(ks_statistic),
                            "ks_p_value": float(ks_p_value),
                            "threshold": self.config["ks_p_threshold"],
                        },
                        timestamp=timestamp,
                    )
                )
        except Exception as e:
            logging.warning(f"KS test failed for {column}: {e}")

        return results

    def _detect_categorical_drift(
        self, current_series: pd.Series, column: str, timestamp: datetime
    ) -> List[DriftResult]:
        """Detect drift in categorical features."""
        results = []
        ref_stats = self.reference_stats[column]
        current_clean = current_series.dropna()

        if len(current_clean) < self.config["min_samples_for_test"]:
            return results

        current_value_counts = current_clean.value_counts()
        current_proportions = current_value_counts / len(current_clean)

        # Population Stability Index for categorical
        try:
            ref_props = ref_stats["reference_proportions"]
            psi = 0.0

            all_categories = set(ref_props.keys()) | set(current_proportions.index)

            for category in all_categories:
                current_prop = current_proportions.get(category, 1e-6)
                ref_prop = ref_props.get(category, 1e-6)

                psi += (current_prop - ref_prop) * np.log(current_prop / ref_prop)

            severity = self._calculate_drift_severity(psi, self.config["psi_threshold"])

            if psi > self.config["psi_threshold"]:
                results.append(
                    DriftResult(
                        drift_type=DriftType.STATISTICAL,
                        severity=severity,
                        drift_score=float(psi),
                        affected_features=[column],
                        p_value=None,
                        description=f"Categorical distribution drift in {column}",
                        metadata={
                            "method": "categorical_psi",
                            "psi_value": float(psi),
                            "threshold": self.config["psi_threshold"],
                            "new_categories": list(set(current_proportions.index) - set(ref_props.keys())),
                            "missing_categories": list(set(ref_props.keys()) - set(current_proportions.index)),
                        },
                        timestamp=timestamp,
                    )
                )
        except Exception as e:
            logging.warning(f"Categorical PSI calculation failed for {column}: {e}")

        # Chi-square test for categorical drift
        try:
            ref_props = ref_stats["reference_proportions"]

            # Align categories
            common_categories = list(set(ref_props.keys()) & set(current_proportions.index))

            if len(common_categories) > 1:
                expected_counts = np.array([ref_props[cat] * len(current_clean) for cat in common_categories])
                observed_counts = np.array([current_value_counts.get(cat, 0) for cat in common_categories])

                # Ensure minimum expected count
                if all(expected_counts >= 5):
                    chi2_stat, chi2_p_value = stats.chisquare(observed_counts, expected_counts)

                    if chi2_p_value < self.config["ks_p_threshold"]:
                        severity = self._calculate_drift_severity(chi2_stat / len(common_categories), 1.0)

                        results.append(
                            DriftResult(
                                drift_type=DriftType.DISTRIBUTION,
                                severity=severity,
                                drift_score=float(chi2_stat),
                                affected_features=[column],
                                p_value=float(chi2_p_value),
                                description=f"Categorical frequency drift in {column} (Chi-square test)",
                                metadata={
                                    "method": "chi_square",
                                    "chi2_statistic": float(chi2_stat),
                                    "chi2_p_value": float(chi2_p_value),
                                    "degrees_of_freedom": len(common_categories) - 1,
                                },
                                timestamp=timestamp,
                            )
                        )
        except Exception as e:
            logging.warning(f"Chi-square test failed for {column}: {e}")

        return results

    def _detect_domain_drift(self, current_df: pd.DataFrame, timestamp: datetime) -> Optional[DriftResult]:
        """Detect drift using domain classifier approach."""
        try:
            # We need reference data for domain classifier
            # This is a simplified approach - in practice, you'd store reference data
            logging.info("Domain drift detection requires reference dataset storage - skipping")
            return None

        except Exception as e:
            logging.warning(f"Domain drift detection failed: {e}")
            return None

    def _detect_feature_importance_drift(
        self, current_importance: Dict[str, float], timestamp: datetime
    ) -> Optional[DriftResult]:
        """Detect drift in feature importance rankings."""
        if not hasattr(self, "reference_importance"):
            # Store as reference for next time
            self.reference_importance = current_importance.copy()
            return None

        try:
            ref_importance = self.reference_importance

            # Compare feature importance rankings
            common_features = set(ref_importance.keys()) & set(current_importance.keys())

            if len(common_features) < 2:
                return None

            # Calculate rank correlation
            ref_ranks = {
                feat: i for i, feat in enumerate(sorted(common_features, key=lambda x: ref_importance[x], reverse=True))
            }
            curr_ranks = {
                feat: i
                for i, feat in enumerate(sorted(common_features, key=lambda x: current_importance[x], reverse=True))
            }

            ref_rank_values = [ref_ranks[feat] for feat in common_features]
            curr_rank_values = [curr_ranks[feat] for feat in common_features]

            rank_correlation, p_value = stats.spearmanr(ref_rank_values, curr_rank_values)

            # Drift score is inverse of correlation (1 = no correlation, 0 = perfect correlation)
            drift_score = 1.0 - abs(rank_correlation)

            if drift_score > self.config["feature_importance_threshold"]:
                severity = self._calculate_drift_severity(drift_score, self.config["feature_importance_threshold"])

                # Update reference for next comparison
                self.reference_importance = current_importance.copy()

                return DriftResult(
                    drift_type=DriftType.FEATURE_IMPORTANCE,
                    severity=severity,
                    drift_score=float(drift_score),
                    affected_features=list(common_features),
                    p_value=float(p_value) if not np.isnan(p_value) else None,
                    description="Feature importance ranking has shifted significantly",
                    metadata={
                        "method": "feature_importance_rank",
                        "rank_correlation": float(rank_correlation),
                        "drift_score": float(drift_score),
                        "common_features": list(common_features),
                    },
                    timestamp=timestamp,
                )
        except Exception as e:
            logging.warning(f"Feature importance drift detection failed: {e}")
            return None

    def _calculate_drift_severity(self, drift_score: float, threshold: float) -> str:
        """Calculate drift severity based on score and threshold."""
        thresholds = self.config["drift_severity_thresholds"]

        if drift_score < threshold:
            return "none"

        # Normalize score relative to threshold
        normalized_score = min(1.0, drift_score / threshold)

        if normalized_score >= thresholds["critical"]:
            return "critical"
        elif normalized_score >= thresholds["high"]:
            return "high"
        elif normalized_score >= thresholds["medium"]:
            return "medium"
        else:
            return "low"

    def get_drift_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        results = self.drift_history

        if time_window:
            cutoff_time = datetime.now() - time_window
            results = [r for r in results if r.timestamp >= cutoff_time]

        if not results:
            return {"total_drifts": 0, "by_type": {}, "by_severity": {}, "affected_features": []}

        by_type = {}
        by_severity = {}
        affected_features = set()

        for result in results:
            # Count by type
            type_name = result.drift_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by severity
            by_severity[result.severity] = by_severity.get(result.severity, 0) + 1

            # Track affected features
            affected_features.update(result.affected_features)

        return {
            "total_drifts": len(results),
            "time_window": str(time_window) if time_window else "all_time",
            "by_type": by_type,
            "by_severity": by_severity,
            "affected_features": list(affected_features),
            "avg_drift_score": np.mean([r.drift_score for r in results]),
            "max_drift_score": max([r.drift_score for r in results]) if results else 0,
            "latest_drift_time": max([r.timestamp for r in results]) if results else None,
        }

    def clear_history(self) -> None:
        """Clear drift detection history."""
        self.drift_history.clear()
        logging.info("Drift detection history cleared")


def monitor_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_importance: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[DriftResult], Dict[str, Any]]:
    """
    Convenience function for drift monitoring.

    Args:
        reference_df: Reference dataset
        current_df: Current dataset to check for drift
        feature_importance: Optional feature importance scores
        config: Optional configuration overrides

    Returns:
        Tuple of (drift_results, summary_stats)
    """
    monitor = ComprehensiveDriftMonitor(config)
    monitor.fit_reference(reference_df)
    results = monitor.detect_drift(current_df, feature_importance)
    summary = monitor.get_drift_summary()

    return results, summary
