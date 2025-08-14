import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TrustLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

@dataclass
class TrustScore:
    overall_trust: float
    reliability_score: float
    consistency_score: float
    robustness_score: float
    calibration_score: float
    uncertainty_score: float
    trust_level: TrustLevel

@dataclass
class ReliabilityMetrics:
    prediction_consistency: float
    cross_validation_stability: float
    feature_importance_stability: float
    explanation_consistency: float

class TrustMetricsCalculator:
    def __init__(self, confidence_threshold: float = 0.8, stability_threshold: float = 0.1):
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
    
    def calculate_trust_score(self, model, X: pd.DataFrame, y_true: np.ndarray, 
                            y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                            explanation_engine=None) -> TrustScore:
        
        reliability_score = self._calculate_reliability(model, X, y_true, y_pred)
        consistency_score = self._calculate_consistency(model, X, y_pred_proba, explanation_engine)
        robustness_score = self._calculate_robustness(model, X, y_pred)
        calibration_score = self._calculate_calibration_quality(y_true, y_pred_proba)
        uncertainty_score = self._calculate_uncertainty_quality(y_pred_proba)
        
        overall_trust = np.mean([
            reliability_score, consistency_score, robustness_score, 
            calibration_score, uncertainty_score
        ])
        
        trust_level = self._determine_trust_level(overall_trust)
        
        return TrustScore(
            overall_trust=overall_trust,
            reliability_score=reliability_score,
            consistency_score=consistency_score,
            robustness_score=robustness_score,
            calibration_score=calibration_score,
            uncertainty_score=uncertainty_score,
            trust_level=trust_level
        )
    
    def _calculate_reliability(self, model, X: pd.DataFrame, y_true: np.ndarray, 
                              y_pred: np.ndarray) -> float:
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score, r2_score
            
            if len(np.unique(y_true)) == 2:
                base_score = accuracy_score(y_true, y_pred)
                scoring = 'accuracy'
            else:
                base_score = r2_score(y_true, y_pred)
                scoring = 'r2'
            
            if len(X) > 100:
                cv_scores = cross_val_score(model, X.sample(min(200, len(X))), 
                                          y_true[:min(200, len(X))], cv=3, scoring=scoring)
                cv_stability = 1 - np.std(cv_scores)
                avg_cv_score = np.mean(cv_scores)
                
                reliability = 0.6 * base_score + 0.4 * avg_cv_score * cv_stability
            else:
                reliability = base_score
            
            return max(0, min(1, reliability))
            
        except Exception:
            return 0.5
    
    def _calculate_consistency(self, model, X: pd.DataFrame, y_pred_proba: Optional[np.ndarray], 
                              explanation_engine) -> float:
        consistency_scores = []
        
        prediction_consistency = self._calculate_prediction_consistency(model, X, y_pred_proba)
        consistency_scores.append(prediction_consistency)
        
        if explanation_engine:
            explanation_consistency = self._calculate_explanation_consistency(explanation_engine, X)
            consistency_scores.append(explanation_consistency)
        
        feature_consistency = self._calculate_feature_importance_consistency(model, X)
        consistency_scores.append(feature_consistency)
        
        return np.mean(consistency_scores)
    
    def _calculate_prediction_consistency(self, model, X: pd.DataFrame, 
                                        y_pred_proba: Optional[np.ndarray]) -> float:
        try:
            if y_pred_proba is None:
                return 0.5
            
            if len(y_pred_proba.shape) > 1:
                confidence_scores = np.max(y_pred_proba, axis=1)
            else:
                confidence_scores = np.abs(y_pred_proba - 0.5) * 2
            
            high_confidence_rate = np.mean(confidence_scores > self.confidence_threshold)
            avg_confidence = np.mean(confidence_scores)
            confidence_std = np.std(confidence_scores)
            
            consistency = (high_confidence_rate * 0.4 + avg_confidence * 0.4 + 
                          (1 - min(confidence_std, 0.5)) * 0.2)
            
            return max(0, min(1, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_explanation_consistency(self, explanation_engine, X: pd.DataFrame) -> float:
        try:
            if len(X) < 10:
                return 0.5
            
            sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
            explanations = []
            
            for idx in sample_indices:
                explanation = explanation_engine.explain_local(X.iloc[idx])
                explanations.append(explanation.feature_contributions)
            
            if not explanations:
                return 0.5
            
            feature_consistency_scores = []
            feature_names = list(explanations[0].keys())
            
            for feature in feature_names:
                feature_contributions = [exp.get(feature, 0) for exp in explanations]
                if len(set(feature_contributions)) > 1:
                    consistency = 1 - (np.std(feature_contributions) / (np.mean(np.abs(feature_contributions)) + 1e-6))
                else:
                    consistency = 1.0
                feature_consistency_scores.append(max(0, consistency))
            
            return np.mean(feature_consistency_scores)
            
        except Exception:
            return 0.5
    
    def _calculate_feature_importance_consistency(self, model, X: pd.DataFrame) -> float:
        try:
            if hasattr(model, 'feature_importances_'):
                base_importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                base_importances = np.abs(model.coef_).flatten()
            else:
                return 0.5
            
            if len(X) < 50:
                return 0.7
            
            bootstrap_importances = []
            n_bootstrap = 3
            
            for _ in range(n_bootstrap):
                try:
                    sample_indices = np.random.choice(len(X), min(100, len(X)), replace=True)
                    X_sample = X.iloc[sample_indices]
                    
                    if hasattr(model, 'fit'):
                        from sklearn.base import clone
                        model_copy = clone(model)
                        y_sample = model.predict(X_sample)
                        model_copy.fit(X_sample, y_sample)
                        
                        if hasattr(model_copy, 'feature_importances_'):
                            bootstrap_importances.append(model_copy.feature_importances_)
                        elif hasattr(model_copy, 'coef_'):
                            bootstrap_importances.append(np.abs(model_copy.coef_).flatten())
                except Exception:
                    continue
            
            if not bootstrap_importances:
                return 0.7
            
            importance_std = np.std(bootstrap_importances, axis=0)
            importance_mean = np.mean(bootstrap_importances, axis=0)
            
            relative_std = importance_std / (importance_mean + 1e-6)
            consistency = 1 - np.mean(relative_std)
            
            return max(0, min(1, consistency))
            
        except Exception:
            return 0.5
    
    def _calculate_robustness(self, model, X: pd.DataFrame, y_pred: np.ndarray) -> float:
        try:
            if len(X) < 20:
                return 0.5
            
            sample_size = min(50, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            original_predictions = y_pred[sample_indices]
            
            noise_levels = [0.01, 0.05, 0.1]
            robustness_scores = []
            
            for noise_level in noise_levels:
                try:
                    X_noisy = X_sample.copy()
                    numeric_columns = X_sample.select_dtypes(include=[np.number]).columns
                    
                    for col in numeric_columns:
                        noise = np.random.normal(0, X_sample[col].std() * noise_level, len(X_sample))
                        X_noisy[col] = X_sample[col] + noise
                    
                    noisy_predictions = model.predict(X_noisy)
                    
                    if len(np.unique(original_predictions)) == 2:
                        agreement = np.mean(original_predictions == noisy_predictions)
                    else:
                        relative_error = np.abs(original_predictions - noisy_predictions) / (np.abs(original_predictions) + 1e-6)
                        agreement = 1 - np.mean(relative_error)
                    
                    robustness_scores.append(max(0, agreement))
                    
                except Exception:
                    robustness_scores.append(0.5)
            
            return np.mean(robustness_scores)
            
        except Exception:
            return 0.5
    
    def _calculate_calibration_quality(self, y_true: np.ndarray, 
                                      y_pred_proba: Optional[np.ndarray]) -> float:
        try:
            if y_pred_proba is None or len(np.unique(y_true)) != 2:
                return 0.5
            
            if len(y_pred_proba.shape) > 1:
                y_proba = y_pred_proba[:, 1]
            else:
                y_proba = y_pred_proba
            
            n_bins = min(10, len(y_true) // 5)
            if n_bins < 3:
                return 0.5
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_errors = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                    calibration_errors.append(calibration_error * prop_in_bin)
            
            if not calibration_errors:
                return 0.5
            
            expected_calibration_error = np.sum(calibration_errors)
            calibration_quality = 1 - expected_calibration_error
            
            return max(0, min(1, calibration_quality))
            
        except Exception:
            return 0.5
    
    def _calculate_uncertainty_quality(self, y_pred_proba: Optional[np.ndarray]) -> float:
        try:
            if y_pred_proba is None:
                return 0.5
            
            if len(y_pred_proba.shape) > 1:
                entropies = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-10), axis=1)
                max_entropy = np.log(y_pred_proba.shape[1])
                normalized_entropies = entropies / max_entropy
            else:
                normalized_entropies = -((y_pred_proba * np.log(y_pred_proba + 1e-10)) + 
                                       ((1 - y_pred_proba) * np.log(1 - y_pred_proba + 1e-10))) / np.log(2)
            
            uncertainty_distribution = np.std(normalized_entropies)
            mean_uncertainty = np.mean(normalized_entropies)
            
            well_calibrated_uncertainty = 0.5 * (1 - abs(mean_uncertainty - 0.5)) + 0.5 * uncertainty_distribution
            
            return max(0, min(1, well_calibrated_uncertainty))
            
        except Exception:
            return 0.5
    
    def _determine_trust_level(self, overall_trust: float) -> TrustLevel:
        if overall_trust >= 0.9:
            return TrustLevel.VERY_HIGH
        elif overall_trust >= 0.8:
            return TrustLevel.HIGH
        elif overall_trust >= 0.6:
            return TrustLevel.MEDIUM
        elif overall_trust >= 0.4:
            return TrustLevel.LOW
        else:
            return TrustLevel.VERY_LOW
    
    def calculate_reliability_metrics(self, model, X: pd.DataFrame, y_true: np.ndarray,
                                    explanation_engine=None) -> ReliabilityMetrics:
        
        prediction_consistency = self._calculate_prediction_consistency(model, X, None)
        
        try:
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X.sample(min(100, len(X))), 
                                      y_true[:min(100, len(X))], cv=3)
            cv_stability = 1 - (np.std(cv_scores) / (np.mean(cv_scores) + 1e-6))
        except Exception:
            cv_stability = 0.5
        
        feature_importance_stability = self._calculate_feature_importance_consistency(model, X)
        
        if explanation_engine:
            explanation_consistency = self._calculate_explanation_consistency(explanation_engine, X)
        else:
            explanation_consistency = 0.5
        
        return ReliabilityMetrics(
            prediction_consistency=prediction_consistency,
            cross_validation_stability=max(0, cv_stability),
            feature_importance_stability=feature_importance_stability,
            explanation_consistency=explanation_consistency
        )
    
    def assess_out_of_distribution_detection(self, model, X_train: pd.DataFrame, 
                                           X_test: pd.DataFrame) -> Dict[str, float]:
        try:
            train_predictions = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else model.predict(X_train)
            test_predictions = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                train_confidence = np.max(train_predictions, axis=1)
                test_confidence = np.max(test_predictions, axis=1)
            else:
                train_confidence = np.abs(train_predictions - np.mean(train_predictions))
                test_confidence = np.abs(test_predictions - np.mean(test_predictions))
            
            confidence_shift = np.mean(train_confidence) - np.mean(test_confidence)
            distribution_similarity = 1 - min(1, abs(confidence_shift))
            
            feature_shift_scores = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                if col in X_test.columns:
                    train_mean = X_train[col].mean()
                    test_mean = X_test[col].mean()
                    train_std = X_train[col].std()
                    
                    if train_std > 0:
                        standardized_shift = abs(train_mean - test_mean) / train_std
                        shift_score = 1 / (1 + standardized_shift)
                        feature_shift_scores.append(shift_score)
            
            feature_distribution_similarity = np.mean(feature_shift_scores) if feature_shift_scores else 0.5
            
            return {
                'confidence_distribution_similarity': max(0, distribution_similarity),
                'feature_distribution_similarity': feature_distribution_similarity,
                'overall_ood_detection_quality': np.mean([distribution_similarity, feature_distribution_similarity])
            }
            
        except Exception:
            return {
                'confidence_distribution_similarity': 0.5,
                'feature_distribution_similarity': 0.5,
                'overall_ood_detection_quality': 0.5
            }
    
    def generate_trust_report(self, model, X: pd.DataFrame, y_true: np.ndarray, 
                            y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None,
                            explanation_engine=None) -> Dict[str, Any]:
        
        trust_score = self.calculate_trust_score(model, X, y_true, y_pred, y_pred_proba, explanation_engine)
        reliability_metrics = self.calculate_reliability_metrics(model, X, y_true, explanation_engine)
        
        trust_interpretation = self._interpret_trust_score(trust_score)
        improvement_recommendations = self._generate_trust_improvement_recommendations(trust_score)
        
        return {
            'trust_score_summary': {
                'overall_trust': trust_score.overall_trust,
                'trust_level': trust_score.trust_level.value,
                'interpretation': trust_interpretation
            },
            'detailed_trust_metrics': {
                'reliability_score': trust_score.reliability_score,
                'consistency_score': trust_score.consistency_score,
                'robustness_score': trust_score.robustness_score,
                'calibration_score': trust_score.calibration_score,
                'uncertainty_score': trust_score.uncertainty_score
            },
            'reliability_breakdown': {
                'prediction_consistency': reliability_metrics.prediction_consistency,
                'cross_validation_stability': reliability_metrics.cross_validation_stability,
                'feature_importance_stability': reliability_metrics.feature_importance_stability,
                'explanation_consistency': reliability_metrics.explanation_consistency
            },
            'improvement_recommendations': improvement_recommendations,
            'trust_thresholds': {
                'confidence_threshold': self.confidence_threshold,
                'stability_threshold': self.stability_threshold
            }
        }
    
    def _interpret_trust_score(self, trust_score: TrustScore) -> str:
        if trust_score.trust_level == TrustLevel.VERY_HIGH:
            return "Model demonstrates exceptional trustworthiness across all metrics. Suitable for high-stakes deployment."
        elif trust_score.trust_level == TrustLevel.HIGH:
            return "Model shows strong trustworthiness with minor areas for improvement. Ready for production deployment."
        elif trust_score.trust_level == TrustLevel.MEDIUM:
            return "Model shows moderate trustworthiness. Consider improvements before critical deployment."
        elif trust_score.trust_level == TrustLevel.LOW:
            return "Model has significant trust issues. Substantial improvements needed before deployment."
        else:
            return "Model demonstrates poor trustworthiness. Not recommended for deployment without major improvements."
    
    def _generate_trust_improvement_recommendations(self, trust_score: TrustScore) -> List[str]:
        recommendations = []
        
        if trust_score.reliability_score < 0.7:
            recommendations.append("Improve model reliability through better training data quality and cross-validation")
        
        if trust_score.consistency_score < 0.7:
            recommendations.append("Enhance prediction consistency through ensemble methods or model regularization")
        
        if trust_score.robustness_score < 0.7:
            recommendations.append("Increase model robustness through adversarial training or data augmentation")
        
        if trust_score.calibration_score < 0.7:
            recommendations.append("Improve calibration using Platt scaling or isotonic regression")
        
        if trust_score.uncertainty_score < 0.7:
            recommendations.append("Better uncertainty quantification through Bayesian methods or ensemble approaches")
        
        if not recommendations:
            recommendations.append("Model demonstrates good trustworthiness - maintain through continuous monitoring")
        
        return recommendations