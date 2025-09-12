"""Production Model Validation and Performance Assessment System"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report, mean_absolute_percentage_error
)
from sklearn.base import clone
from scipy import stats

class ValidationStrategy(Enum):
    CROSS_VALIDATION = "cross_validation"
    HOLDOUT = "holdout"
    TIME_SERIES = "time_series"
    BOOTSTRAP = "bootstrap"
    ADAPTIVE = "adaptive"

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

@dataclass
class ValidationMetrics:
    primary_metric: str
    primary_score: float
    primary_std: float
    all_metrics: Dict[str, float]
    all_std: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class ModelPerformance:
    algorithm_name: str
    validation_metrics: ValidationMetrics
    training_time: float
    prediction_time: float
    memory_usage_mb: float
    hyperparameters: Dict[str, Any]
    validation_strategy: str
    stability_score: float
    robustness_score: float

@dataclass
class ValidationConfig:
    strategy: ValidationStrategy = ValidationStrategy.ADAPTIVE
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    scoring_metrics: Optional[List[str]] = None
    confidence_level: float = 0.95
    stability_threshold: float = 0.1
    min_samples_per_fold: int = 30

class ProductionModelValidator:
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      task_type: TaskType, algorithm_name: str = "") -> ModelPerformance:
        
        validation_strategy = self._select_validation_strategy(X, y, task_type)
        scoring_metrics = self._get_scoring_metrics(task_type)
        
        start_time = time.time()
        validation_metrics = self._execute_validation(
            model, X, y, validation_strategy, scoring_metrics, task_type
        )
        training_time = time.time() - start_time
        
        prediction_time = self._measure_prediction_time(model, X.head(100))
        memory_usage = self._estimate_memory_usage(model)
        stability_score = self._calculate_stability_score(validation_metrics)
        robustness_score = self._assess_robustness(model, X, y, task_type)
        
        return ModelPerformance(
            algorithm_name=algorithm_name,
            validation_metrics=validation_metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage_mb=memory_usage,
            hyperparameters=model.get_params(),
            validation_strategy=validation_strategy.value,
            stability_score=stability_score,
            robustness_score=robustness_score
        )
    
    def _select_validation_strategy(self, X: pd.DataFrame, y: pd.Series, 
                                   task_type: TaskType) -> ValidationStrategy:
        
        if self.config.strategy != ValidationStrategy.ADAPTIVE:
            return self.config.strategy
        
        n_samples = len(X)
        
        if n_samples < 100:
            return ValidationStrategy.BOOTSTRAP
        elif n_samples < 500:
            return ValidationStrategy.HOLDOUT
        elif self._is_time_series_data(X):
            return ValidationStrategy.TIME_SERIES
        else:
            return ValidationStrategy.CROSS_VALIDATION
    
    def _is_time_series_data(self, X: pd.DataFrame) -> bool:
        date_columns = []
        for col in X.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)
        
        if date_columns:
            try:
                for col in date_columns:
                    pd.to_datetime(X[col].head(10))
                    return True
            except:
                return False
        
        return False
    
    def _get_scoring_metrics(self, task_type: TaskType) -> List[str]:
        if self.config.scoring_metrics:
            return self.config.scoring_metrics
        
        if task_type == TaskType.CLASSIFICATION:
            return ['accuracy', 'f1', 'precision', 'recall']
        elif task_type == TaskType.REGRESSION:
            return ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        else:
            return ['silhouette']
    
    def _execute_validation(self, model, X: pd.DataFrame, y: pd.Series,
                           strategy: ValidationStrategy, metrics: List[str],
                           task_type: TaskType) -> ValidationMetrics:
        
        if strategy == ValidationStrategy.CROSS_VALIDATION:
            return self._cross_validation(model, X, y, metrics, task_type)
        elif strategy == ValidationStrategy.HOLDOUT:
            return self._holdout_validation(model, X, y, metrics, task_type)
        elif strategy == ValidationStrategy.TIME_SERIES:
            return self._time_series_validation(model, X, y, metrics, task_type)
        elif strategy == ValidationStrategy.BOOTSTRAP:
            return self._bootstrap_validation(model, X, y, metrics, task_type)
        else:
            return self._cross_validation(model, X, y, metrics, task_type)
    
    def _cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                         metrics: List[str], task_type: TaskType) -> ValidationMetrics:
        
        if task_type == TaskType.CLASSIFICATION and len(y.unique()) > 1:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, 
                      random_state=self.config.random_state)
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=metrics, 
                                  return_train_score=False, n_jobs=1)
        
        all_metrics = {}
        all_std = {}
        cv_scores = {}
        confidence_intervals = {}
        
        primary_metric = metrics[0]
        primary_key = f'test_{primary_metric}'
        
        for metric in metrics:
            key = f'test_{metric}'
            if key in cv_results:
                scores = cv_results[key]
                all_metrics[metric] = scores.mean()
                all_std[metric] = scores.std()
                cv_scores[metric] = scores.tolist()
                
                ci_lower, ci_upper = self._calculate_confidence_interval(
                    scores, self.config.confidence_level
                )
                confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return ValidationMetrics(
            primary_metric=primary_metric,
            primary_score=all_metrics.get(primary_metric, 0.0),
            primary_std=all_std.get(primary_metric, 0.0),
            all_metrics=all_metrics,
            all_std=all_std,
            cv_scores=cv_scores,
            confidence_intervals=confidence_intervals
        )
    
    def _holdout_validation(self, model, X: pd.DataFrame, y: pd.Series,
                           metrics: List[str], task_type: TaskType) -> ValidationMetrics:
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y if task_type == TaskType.CLASSIFICATION and len(y.unique()) > 1 else None
        )
        
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        all_metrics = {}
        all_std = {}
        cv_scores = {}
        confidence_intervals = {}
        
        for metric in metrics:
            score = self._calculate_single_metric(y_test, y_pred, metric, task_type)
            all_metrics[metric] = score
            all_std[metric] = 0.0
            cv_scores[metric] = [score]
            confidence_intervals[metric] = (score, score)
        
        return ValidationMetrics(
            primary_metric=metrics[0],
            primary_score=all_metrics.get(metrics[0], 0.0),
            primary_std=0.0,
            all_metrics=all_metrics,
            all_std=all_std,
            cv_scores=cv_scores,
            confidence_intervals=confidence_intervals
        )
    
    def _time_series_validation(self, model, X: pd.DataFrame, y: pd.Series,
                               metrics: List[str], task_type: TaskType) -> ValidationMetrics:
        
        cv = TimeSeriesSplit(n_splits=min(5, len(X) // 100))
        cv_results = cross_validate(model, X, y, cv=cv, scoring=metrics, 
                                  return_train_score=False, n_jobs=1)
        
        return self._process_cv_results(cv_results, metrics)
    
    def _bootstrap_validation(self, model, X: pd.DataFrame, y: pd.Series,
                             metrics: List[str], task_type: TaskType) -> ValidationMetrics:
        
        n_bootstrap = min(self.config.cv_folds, 20)
        bootstrap_scores = {metric: [] for metric in metrics}
        
        for i in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            oob_indices = list(set(range(len(X))) - set(indices))
            if len(oob_indices) < 10:
                continue
                
            X_oob = X.iloc[oob_indices]
            y_oob = y.iloc[oob_indices]
            
            model_clone = clone(model)
            model_clone.fit(X_boot, y_boot)
            y_pred = model_clone.predict(X_oob)
            
            for metric in metrics:
                score = self._calculate_single_metric(y_oob, y_pred, metric, task_type)
                bootstrap_scores[metric].append(score)
        
        all_metrics = {}
        all_std = {}
        cv_scores = {}
        confidence_intervals = {}
        
        for metric in metrics:
            scores = bootstrap_scores[metric]
            if scores:
                all_metrics[metric] = np.mean(scores)
                all_std[metric] = np.std(scores)
                cv_scores[metric] = scores
                ci_lower, ci_upper = self._calculate_confidence_interval(
                    np.array(scores), self.config.confidence_level
                )
                confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return ValidationMetrics(
            primary_metric=metrics[0],
            primary_score=all_metrics.get(metrics[0], 0.0),
            primary_std=all_std.get(metrics[0], 0.0),
            all_metrics=all_metrics,
            all_std=all_std,
            cv_scores=cv_scores,
            confidence_intervals=confidence_intervals
        )
    
    def _process_cv_results(self, cv_results: Dict, metrics: List[str]) -> ValidationMetrics:
        all_metrics = {}
        all_std = {}
        cv_scores = {}
        confidence_intervals = {}
        
        for metric in metrics:
            key = f'test_{metric}'
            if key in cv_results:
                scores = cv_results[key]
                all_metrics[metric] = scores.mean()
                all_std[metric] = scores.std()
                cv_scores[metric] = scores.tolist()
                
                ci_lower, ci_upper = self._calculate_confidence_interval(
                    scores, self.config.confidence_level
                )
                confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return ValidationMetrics(
            primary_metric=metrics[0],
            primary_score=all_metrics.get(metrics[0], 0.0),
            primary_std=all_std.get(metrics[0], 0.0),
            all_metrics=all_metrics,
            all_std=all_std,
            cv_scores=cv_scores,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_single_metric(self, y_true, y_pred, metric: str, task_type: TaskType) -> float:
        try:
            if task_type == TaskType.CLASSIFICATION:
                if metric == 'accuracy':
                    return accuracy_score(y_true, y_pred)
                elif metric == 'f1':
                    return f1_score(y_true, y_pred, average='weighted')
                elif metric == 'precision':
                    return precision_score(y_true, y_pred, average='weighted')
                elif metric == 'recall':
                    return recall_score(y_true, y_pred, average='weighted')
            
            elif task_type == TaskType.REGRESSION:
                if metric == 'r2':
                    return r2_score(y_true, y_pred)
                elif metric in ['neg_mean_squared_error', 'mse']:
                    return -mean_squared_error(y_true, y_pred)
                elif metric in ['neg_mean_absolute_error', 'mae']:
                    return -mean_absolute_error(y_true, y_pred)
                elif metric == 'mape':
                    return -mean_absolute_percentage_error(y_true, y_pred)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_confidence_interval(self, scores: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        alpha = 1 - confidence_level
        return stats.t.interval(confidence_level, len(scores) - 1,
                              loc=scores.mean(), scale=stats.sem(scores))
    
    def _measure_prediction_time(self, model, sample_data: pd.DataFrame) -> float:
        if len(sample_data) == 0:
            return 0.0
        
        try:
            start_time = time.time()
            model.predict(sample_data)
            end_time = time.time()
            
            time_per_sample = (end_time - start_time) / len(sample_data)
            return time_per_sample * 1000
        except Exception:
            return 0.0
    
    def _estimate_memory_usage(self, model) -> float:
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, metrics: ValidationMetrics) -> float:
        cv_scores = metrics.cv_scores
        if not cv_scores:
            return 0.0
        
        stability_scores = []
        for metric, scores in cv_scores.items():
            if len(scores) > 1:
                cv = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else 1
                stability = max(0, 1 - cv)
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _assess_robustness(self, model, X: pd.DataFrame, y: pd.Series, 
                          task_type: TaskType) -> float:
        try:
            if len(X) < 100:
                return 0.5
            
            original_score = self._quick_score(model, X, y, task_type)
            
            X_noisy = X.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    noise = np.random.normal(0, X_noisy[col].std() * 0.1, len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise
            
            noisy_score = self._quick_score(model, X_noisy, y, task_type)
            
            robustness = 1 - abs(original_score - noisy_score) / max(abs(original_score), 0.01)
            return max(0, min(1, robustness))
            
        except Exception:
            return 0.5
    
    def _quick_score(self, model, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> float:
        try:
            model_clone = clone(model)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            if task_type == TaskType.CLASSIFICATION:
                return accuracy_score(y_test, y_pred)
            else:
                return r2_score(y_test, y_pred)
        except Exception:
            return 0.0
    
    def compare_models(self, performances: List[ModelPerformance]) -> Dict[str, Any]:
        if len(performances) < 2:
            return {'error': 'Need at least 2 models to compare'}
        
        performance_ranking = []
        for perf in performances:
            score = perf.validation_metrics.primary_score
            performance_ranking.append((perf.algorithm_name, score))
        
        performance_ranking.sort(key=lambda x: x[1], reverse=True)
        best_algorithm = performance_ranking[0][0]
        
        statistical_tests = self._perform_statistical_tests(performances)
        recommendation = self._generate_recommendation(performances, performance_ranking)
        
        return {
            'best_algorithm': best_algorithm,
            'performance_ranking': performance_ranking,
            'statistical_tests': statistical_tests,
            'recommendation': recommendation,
            'detailed_comparison': self._detailed_comparison(performances)
        }
    
    def _perform_statistical_tests(self, performances: List[ModelPerformance]) -> Dict[str, Any]:
        if len(performances) < 2:
            return {}
        
        tests = {}
        primary_metric = performances[0].validation_metrics.primary_metric
        
        for i, perf1 in enumerate(performances[:-1]):
            for j, perf2 in enumerate(performances[i+1:], i+1):
                scores1 = perf1.validation_metrics.cv_scores.get(primary_metric, [])
                scores2 = perf2.validation_metrics.cv_scores.get(primary_metric, [])
                
                if len(scores1) > 1 and len(scores2) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)
                        test_key = f"{perf1.algorithm_name}_vs_{perf2.algorithm_name}"
                        tests[test_key] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': abs(np.mean(scores1) - np.mean(scores2))
                        }
                    except Exception:
                        continue
        
        return tests
    
    def _generate_recommendation(self, performances: List[ModelPerformance], 
                               ranking: List[Tuple[str, float]]) -> str:
        
        if not performances:
            return "No models to evaluate"
        
        best_perf = next(p for p in performances if p.algorithm_name == ranking[0][0])
        
        if best_perf.stability_score > 0.8 and best_perf.robustness_score > 0.7:
            reliability = "highly reliable"
        elif best_perf.stability_score > 0.6 and best_perf.robustness_score > 0.5:
            reliability = "moderately reliable"
        else:
            reliability = "less reliable"
        
        primary_score = best_perf.validation_metrics.primary_score
        
        recommendation = f"Recommend {best_perf.algorithm_name} "
        recommendation += f"(score: {primary_score:.3f}, {reliability})"
        
        if best_perf.training_time > 10:
            recommendation += ". Note: High training time"
        
        if best_perf.memory_usage_mb > 100:
            recommendation += ". Note: High memory usage"
        
        return recommendation
    
    def _detailed_comparison(self, performances: List[ModelPerformance]) -> Dict[str, Dict[str, float]]:
        comparison = {}
        
        for perf in performances:
            comparison[perf.algorithm_name] = {
                'primary_score': perf.validation_metrics.primary_score,
                'stability': perf.stability_score,
                'robustness': perf.robustness_score,
                'training_time': perf.training_time,
                'prediction_time': perf.prediction_time,
                'memory_mb': perf.memory_usage_mb
            }
            
            comparison[perf.algorithm_name].update(perf.validation_metrics.all_metrics)
        
        return comparison

def validate_model_performance(model, X: pd.DataFrame, y: pd.Series,
                             task_type: TaskType, algorithm_name: str = "",
                             config: Optional[ValidationConfig] = None) -> ModelPerformance:
    
    validator = ProductionModelValidator(config)
    return validator.validate_model(model, X, y, task_type, algorithm_name)