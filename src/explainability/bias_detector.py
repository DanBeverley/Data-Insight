import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class BiasType(Enum):
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_OPPORTUNITY = "equalized_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasResult:
    bias_type: BiasType
    severity: SeverityLevel
    metric_value: float
    threshold: float
    affected_groups: List[str]
    description: str
    recommendation: str


@dataclass
class FairnessMetrics:
    demographic_parity: float
    equalized_opportunity: float
    equalized_odds: float
    calibration_score: float
    overall_fairness_score: float


class BiasDetector:
    def __init__(self, sensitive_attributes: List[str], fairness_threshold: float = 0.1):
        self.sensitive_attributes = sensitive_attributes
        self.fairness_threshold = fairness_threshold

    def detect_bias(
        self, model, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
    ) -> List[BiasResult]:
        bias_results = []

        for attribute in self.sensitive_attributes:
            if attribute not in X.columns:
                continue

            group_results = self._analyze_attribute_bias(model, X, y_true, y_pred, y_pred_proba, attribute)
            bias_results.extend(group_results)

        return bias_results

    def _analyze_attribute_bias(
        self,
        model,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        attribute: str,
    ) -> List[BiasResult]:
        results = []
        unique_groups = X[attribute].unique()

        if len(unique_groups) < 2:
            return results

        group_stats = self._calculate_group_statistics(X, y_true, y_pred, y_pred_proba, attribute)

        results.append(self._check_demographic_parity(group_stats, attribute))
        results.append(self._check_equalized_opportunity(group_stats, attribute))
        results.append(self._check_equalized_odds(group_stats, attribute))

        if y_pred_proba is not None:
            results.append(self._check_calibration(group_stats, attribute))

        return [r for r in results if r is not None]

    def _calculate_group_statistics(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        attribute: str,
    ) -> Dict[str, Dict[str, float]]:
        stats = {}

        for group in X[attribute].unique():
            mask = X[attribute] == group
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]

            if len(group_y_true) == 0:
                continue

            positive_rate = np.mean(group_y_pred)

            if len(np.unique(group_y_true)) > 1:
                true_positive_rate = np.mean(group_y_pred[group_y_true == 1]) if np.any(group_y_true == 1) else 0
                false_positive_rate = np.mean(group_y_pred[group_y_true == 0]) if np.any(group_y_true == 0) else 0
                true_negative_rate = 1 - false_positive_rate
                false_negative_rate = 1 - true_positive_rate
            else:
                true_positive_rate = false_positive_rate = true_negative_rate = false_negative_rate = 0

            accuracy = np.mean(group_y_true == group_y_pred)

            group_stats = {
                "positive_rate": positive_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "true_negative_rate": true_negative_rate,
                "false_negative_rate": false_negative_rate,
                "accuracy": accuracy,
                "size": len(group_y_true),
                "base_rate": np.mean(group_y_true),
            }

            if y_pred_proba is not None:
                group_proba = y_pred_proba[mask]
                if len(group_proba.shape) > 1:
                    group_proba = group_proba[:, 1]

                group_stats["avg_confidence"] = np.mean(group_proba)
                group_stats["calibration_error"] = self._calculate_calibration_error(group_y_true, group_proba)

            stats[str(group)] = group_stats

        return stats

    def _calculate_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            calibration_error = 0
            total_samples = len(y_true)

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return calibration_error

        except Exception:
            return 0.0

    def _check_demographic_parity(
        self, group_stats: Dict[str, Dict[str, float]], attribute: str
    ) -> Optional[BiasResult]:
        positive_rates = [stats["positive_rate"] for stats in group_stats.values()]

        if len(positive_rates) < 2:
            return None

        max_rate = max(positive_rates)
        min_rate = min(positive_rates)
        parity_diff = max_rate - min_rate

        severity = self._assess_severity(parity_diff)

        affected_groups = [
            group
            for group, stats in group_stats.items()
            if abs(stats["positive_rate"] - np.mean(positive_rates)) > self.fairness_threshold
        ]

        return BiasResult(
            bias_type=BiasType.DEMOGRAPHIC_PARITY,
            severity=severity,
            metric_value=parity_diff,
            threshold=self.fairness_threshold,
            affected_groups=affected_groups,
            description=f"Demographic parity violation: {parity_diff:.3f} difference in positive prediction rates across {attribute} groups",
            recommendation="Consider rebalancing training data or applying fairness constraints during model training",
        )

    def _check_equalized_opportunity(
        self, group_stats: Dict[str, Dict[str, float]], attribute: str
    ) -> Optional[BiasResult]:
        tpr_rates = [stats["true_positive_rate"] for stats in group_stats.values()]

        if len(tpr_rates) < 2:
            return None

        max_tpr = max(tpr_rates)
        min_tpr = min(tpr_rates)
        opportunity_diff = max_tpr - min_tpr

        severity = self._assess_severity(opportunity_diff)

        affected_groups = [
            group
            for group, stats in group_stats.items()
            if abs(stats["true_positive_rate"] - np.mean(tpr_rates)) > self.fairness_threshold
        ]

        return BiasResult(
            bias_type=BiasType.EQUALIZED_OPPORTUNITY,
            severity=severity,
            metric_value=opportunity_diff,
            threshold=self.fairness_threshold,
            affected_groups=affected_groups,
            description=f"Equalized opportunity violation: {opportunity_diff:.3f} difference in true positive rates across {attribute} groups",
            recommendation="Ensure equal benefit rates across groups through threshold optimization or post-processing",
        )

    def _check_equalized_odds(self, group_stats: Dict[str, Dict[str, float]], attribute: str) -> Optional[BiasResult]:
        tpr_rates = [stats["true_positive_rate"] for stats in group_stats.values()]
        fpr_rates = [stats["false_positive_rate"] for stats in group_stats.values()]

        if len(tpr_rates) < 2 or len(fpr_rates) < 2:
            return None

        tpr_diff = max(tpr_rates) - min(tpr_rates)
        fpr_diff = max(fpr_rates) - min(fpr_rates)
        odds_violation = max(tpr_diff, fpr_diff)

        severity = self._assess_severity(odds_violation)

        affected_groups = []
        mean_tpr = np.mean(tpr_rates)
        mean_fpr = np.mean(fpr_rates)

        for group, stats in group_stats.items():
            if (
                abs(stats["true_positive_rate"] - mean_tpr) > self.fairness_threshold
                or abs(stats["false_positive_rate"] - mean_fpr) > self.fairness_threshold
            ):
                affected_groups.append(group)

        return BiasResult(
            bias_type=BiasType.EQUALIZED_ODDS,
            severity=severity,
            metric_value=odds_violation,
            threshold=self.fairness_threshold,
            affected_groups=affected_groups,
            description=f"Equalized odds violation: {odds_violation:.3f} maximum difference in error rates across {attribute} groups",
            recommendation="Balance both true positive and false positive rates across groups through algorithmic debiasing",
        )

    def _check_calibration(self, group_stats: Dict[str, Dict[str, float]], attribute: str) -> Optional[BiasResult]:
        calibration_errors = [stats.get("calibration_error", 0) for stats in group_stats.values()]

        if not calibration_errors or max(calibration_errors) == 0:
            return None

        max_error = max(calibration_errors)
        avg_error = np.mean(calibration_errors)
        calibration_diff = max_error - min(calibration_errors)

        severity = self._assess_severity(calibration_diff)

        affected_groups = [
            group
            for group, stats in group_stats.items()
            if stats.get("calibration_error", 0) > avg_error + self.fairness_threshold
        ]

        return BiasResult(
            bias_type=BiasType.CALIBRATION,
            severity=severity,
            metric_value=calibration_diff,
            threshold=self.fairness_threshold,
            affected_groups=affected_groups,
            description=f"Calibration bias detected: {calibration_diff:.3f} difference in calibration errors across {attribute} groups",
            recommendation="Apply calibration techniques like Platt scaling or isotonic regression per group",
        )

    def _assess_severity(self, metric_value: float) -> SeverityLevel:
        if metric_value < self.fairness_threshold:
            return SeverityLevel.LOW
        elif metric_value < self.fairness_threshold * 2:
            return SeverityLevel.MEDIUM
        elif metric_value < self.fairness_threshold * 3:
            return SeverityLevel.HIGH
        else:
            return SeverityLevel.CRITICAL

    def calculate_fairness_metrics(
        self, model, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
    ) -> FairnessMetrics:
        all_bias_results = self.detect_bias(model, X, y_true, y_pred, y_pred_proba)

        bias_by_type = {}
        for result in all_bias_results:
            if result.bias_type not in bias_by_type:
                bias_by_type[result.bias_type] = []
            bias_by_type[result.bias_type].append(result.metric_value)

        demographic_parity = np.mean(bias_by_type.get(BiasType.DEMOGRAPHIC_PARITY, [0]))
        equalized_opportunity = np.mean(bias_by_type.get(BiasType.EQUALIZED_OPPORTUNITY, [0]))
        equalized_odds = np.mean(bias_by_type.get(BiasType.EQUALIZED_ODDS, [0]))
        calibration_score = np.mean(bias_by_type.get(BiasType.CALIBRATION, [0]))

        overall_fairness_score = 1 - np.mean(
            [demographic_parity, equalized_opportunity, equalized_odds, calibration_score]
        )

        return FairnessMetrics(
            demographic_parity=demographic_parity,
            equalized_opportunity=equalized_opportunity,
            equalized_odds=equalized_odds,
            calibration_score=calibration_score,
            overall_fairness_score=max(0, overall_fairness_score),
        )

    def suggest_mitigation_strategies(self, bias_results: List[BiasResult]) -> List[Dict[str, Any]]:
        strategies = []

        critical_biases = [r for r in bias_results if r.severity == SeverityLevel.CRITICAL]
        high_biases = [r for r in bias_results if r.severity == SeverityLevel.HIGH]

        if critical_biases:
            strategies.append(
                {
                    "strategy": "immediate_retraining",
                    "priority": "critical",
                    "description": "Model shows critical bias and should be retrained with fairness constraints",
                    "affected_attributes": list(set([r.affected_groups for r in critical_biases])),
                    "estimated_effort": "high",
                }
            )

        if high_biases or critical_biases:
            strategies.append(
                {
                    "strategy": "data_rebalancing",
                    "priority": "high",
                    "description": "Rebalance training data to ensure equal representation across sensitive groups",
                    "techniques": ["oversampling", "undersampling", "synthetic_data_generation"],
                    "estimated_effort": "medium",
                }
            )

        if any(r.bias_type == BiasType.CALIBRATION for r in bias_results):
            strategies.append(
                {
                    "strategy": "calibration_adjustment",
                    "priority": "medium",
                    "description": "Apply group-specific calibration to improve prediction reliability",
                    "techniques": ["platt_scaling", "isotonic_regression"],
                    "estimated_effort": "low",
                }
            )

        if any(r.bias_type in [BiasType.DEMOGRAPHIC_PARITY, BiasType.EQUALIZED_OPPORTUNITY] for r in bias_results):
            strategies.append(
                {
                    "strategy": "threshold_optimization",
                    "priority": "medium",
                    "description": "Optimize decision thresholds per group to achieve fairness",
                    "techniques": ["group_specific_thresholds", "pareto_optimization"],
                    "estimated_effort": "low",
                }
            )

        return strategies

    def generate_bias_report(
        self, model, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        bias_results = self.detect_bias(model, X, y_true, y_pred, y_pred_proba)
        fairness_metrics = self.calculate_fairness_metrics(model, X, y_true, y_pred, y_pred_proba)
        mitigation_strategies = self.suggest_mitigation_strategies(bias_results)

        severity_counts = {}
        for severity in SeverityLevel:
            severity_counts[severity.value] = len([r for r in bias_results if r.severity == severity])

        return {
            "bias_detection_summary": {
                "total_biases_detected": len(bias_results),
                "severity_breakdown": severity_counts,
                "attributes_analyzed": self.sensitive_attributes,
                "fairness_threshold": self.fairness_threshold,
            },
            "detailed_bias_results": [
                {
                    "bias_type": result.bias_type.value,
                    "severity": result.severity.value,
                    "metric_value": result.metric_value,
                    "affected_groups": result.affected_groups,
                    "description": result.description,
                    "recommendation": result.recommendation,
                }
                for result in bias_results
            ],
            "fairness_metrics": {
                "demographic_parity": fairness_metrics.demographic_parity,
                "equalized_opportunity": fairness_metrics.equalized_opportunity,
                "equalized_odds": fairness_metrics.equalized_odds,
                "calibration_score": fairness_metrics.calibration_score,
                "overall_fairness_score": fairness_metrics.overall_fairness_score,
            },
            "mitigation_strategies": mitigation_strategies,
            "recommendations": self._generate_overall_recommendations(bias_results, fairness_metrics),
        }

    def _generate_overall_recommendations(
        self, bias_results: List[BiasResult], fairness_metrics: FairnessMetrics
    ) -> List[str]:
        recommendations = []

        if fairness_metrics.overall_fairness_score < 0.7:
            recommendations.append("Overall fairness score is low - comprehensive bias mitigation required")

        if any(r.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH] for r in bias_results):
            recommendations.append("High-severity biases detected - immediate action required before deployment")

        if fairness_metrics.demographic_parity > 0.2:
            recommendations.append("Significant demographic parity violations - review data collection and sampling")

        if fairness_metrics.calibration_score > 0.15:
            recommendations.append("Poor calibration across groups - apply group-specific calibration techniques")

        if not recommendations:
            recommendations.append("Model shows acceptable fairness levels - continue monitoring in production")

        return recommendations
