import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from abc import ABC, abstractmethod

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ValidationMetrics:
    primary_metric: str
    primary_score: float
    primary_std: float
    all_metrics: Dict[str, float]
    all_std: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None
    feature_importance: Optional[Dict[str, float]] = None
    
@dataclass 
class ModelPerformance:
    algorithm_name: str
    validation_metrics: ValidationMetrics
    training_time: float
    prediction_time: float
    model_size_mb: float
    hyperparameters: Dict[str, Any]
    validation_method: str
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ComparisonResult:
    best_algorithm: str
    performance_ranking: List[Tuple[str, float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    significance_level: float
    recommendation: str
    confidence_interval: Dict[str, Tuple[float, float]]

class BaseValidator(ABC):
    """Base class for performance validation strategies"""
    
    @abstractmethod
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, **kwargs) -> ValidationMetrics:
        pass

class CrossValidationValidator(BaseValidator):
    """Cross-validation based performance validation"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      task_type: str = 'auto', **kwargs) -> ValidationMetrics:
        """Validate model using cross-validation with multiple metrics"""
        
        # Determine task type
        if task_type == 'auto':
            task_type = 'classification' if len(y.unique()) <= 20 else 'regression'
        
        # Create CV strategy
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Define scoring metrics based on task type
        if task_type == 'classification':
            if len(y.unique()) == 2:  # Binary classification
                scoring_metrics = {
                    'accuracy': 'accuracy',
                    'precision': 'precision',
                    'recall': 'recall', 
                    'f1': 'f1',
                    'roc_auc': 'roc_auc'
                }
                primary_metric = 'f1'
            else:  # Multiclass classification
                scoring_metrics = {
                    'accuracy': 'accuracy',
                    'precision_weighted': 'precision_weighted',
                    'recall_weighted': 'recall_weighted',
                    'f1_weighted': 'f1_weighted'
                }
                primary_metric = 'f1_weighted'
        else:  # Regression
            scoring_metrics = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
            }
            primary_metric = 'r2'
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring_metrics, 
            return_train_score=False, n_jobs=-1
        )
        
        # Extract results
        all_metrics = {}
        all_std = {}
        cv_scores = {}
        
        for metric_name, sklearn_name in scoring_metrics.items():
            test_scores = cv_results[f'test_{sklearn_name}']
            cv_scores[metric_name] = test_scores.tolist()
            all_metrics[metric_name] = np.mean(test_scores)
            all_std[metric_name] = np.std(test_scores)
        
        # Handle negative metrics (convert to positive)
        for metric in ['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']:
            if metric in all_metrics:
                all_metrics[metric] = -all_metrics[metric]  # Convert to positive
        
        # Get additional metrics for classification
        confusion_mat = None
        class_report = None
        
        if task_type == 'classification':
            # Fit model on full data for additional metrics
            model.fit(X, y)
            y_pred = model.predict(X)
            
            confusion_mat = confusion_matrix(y, y_pred)
            class_report = classification_report(y, y_pred, output_dict=True)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim == 1:
                feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            else:  # Multiclass case
                feature_importance = dict(zip(X.columns, np.abs(model.coef_).mean(axis=0)))
        
        return ValidationMetrics(
            primary_metric=primary_metric,
            primary_score=all_metrics[primary_metric],
            primary_std=all_std[primary_metric],
            all_metrics=all_metrics,
            all_std=all_std,
            cv_scores=cv_scores,
            confusion_matrix=confusion_mat,
            classification_report=class_report,
            feature_importance=feature_importance
        )

class HoldoutValidator(BaseValidator):
    """Holdout validation for large datasets"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                      task_type: str = 'auto', **kwargs) -> ValidationMetrics:
        """Validate model using holdout method"""
        
        from sklearn.model_selection import train_test_split
        
        # Determine task type
        if task_type == 'auto':
            task_type = 'classification' if len(y.unique()) <= 20 else 'regression'
        
        # Split data
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, 
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        all_metrics = {}
        
        if task_type == 'classification':
            all_metrics['accuracy'] = accuracy_score(y_test, y_pred)
            
            if len(y.unique()) == 2:  # Binary classification
                all_metrics['precision'] = precision_score(y_test, y_pred)
                all_metrics['recall'] = recall_score(y_test, y_pred)
                all_metrics['f1'] = f1_score(y_test, y_pred)
                
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    all_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                
                primary_metric = 'f1'
            else:  # Multiclass
                all_metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
                all_metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
                all_metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
                primary_metric = 'f1_weighted'
                
            confusion_mat = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
        else:  # Regression
            all_metrics['r2'] = r2_score(y_test, y_pred)
            all_metrics['neg_mean_squared_error'] = mean_squared_error(y_test, y_pred)
            all_metrics['neg_mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
            all_metrics['neg_root_mean_squared_error'] = np.sqrt(mean_squared_error(y_test, y_pred))
            primary_metric = 'r2'
            confusion_mat = None
            class_report = None
        
        # Since this is single split, std is 0
        all_std = {metric: 0.0 for metric in all_metrics}
        cv_scores = {metric: [score] for metric, score in all_metrics.items()}
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim == 1:
                feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            else:
                feature_importance = dict(zip(X.columns, np.abs(model.coef_).mean(axis=0)))
        
        return ValidationMetrics(
            primary_metric=primary_metric,
            primary_score=all_metrics[primary_metric],
            primary_std=all_std[primary_metric],
            all_metrics=all_metrics,
            all_std=all_std,
            cv_scores=cv_scores,
            confusion_matrix=confusion_mat,
            classification_report=class_report,
            feature_importance=feature_importance
        )

class PerformanceValidator:
    """Comprehensive performance validation and comparison system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_history: List[ModelPerformance] = []
        
    def validate_single_model(self, model, algorithm_name: str, X: pd.DataFrame, y: pd.Series,
                            hyperparameters: Dict[str, Any], validation_method: str = 'cv',
                            **kwargs) -> ModelPerformance:
        """Validate a single model comprehensively"""
        
        # Choose validation strategy
        if validation_method == 'cv':
            cv_folds = kwargs.get('cv_folds', 5)
            validator = CrossValidationValidator(cv_folds=cv_folds)
        elif validation_method == 'holdout':
            test_size = kwargs.get('test_size', 0.2)
            validator = HoldoutValidator(test_size=test_size)
        else:
            raise ValueError(f"Unknown validation method: {validation_method}")
        
        # Measure training time
        start_time = time.time()
        validation_metrics = validator.validate_model(model, X, y, **kwargs)
        training_time = time.time() - start_time
        
        # Measure prediction time
        start_pred_time = time.time()
        _ = model.predict(X.iloc[:min(100, len(X))])  # Sample for speed
        prediction_time_per_sample = (time.time() - start_pred_time) / min(100, len(X))
        
        # Estimate model size
        model_size_mb = self._estimate_model_size(model)
        
        # Create dataset info
        dataset_info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'task_type': 'classification' if len(y.unique()) <= 20 else 'regression',
            'target_distribution': y.value_counts().to_dict() if len(y.unique()) <= 20 else {'mean': float(y.mean()), 'std': float(y.std())}
        }
        
        performance = ModelPerformance(
            algorithm_name=algorithm_name,
            validation_metrics=validation_metrics,
            training_time=training_time,
            prediction_time=prediction_time_per_sample,
            model_size_mb=model_size_mb,
            hyperparameters=hyperparameters,
            validation_method=validation_method,
            dataset_info=dataset_info
        )
        
        # Store in history
        self.validation_history.append(performance)
        
        return performance
    
    def validate_multiple_models(self, models_info: List[Tuple], X: pd.DataFrame, y: pd.Series,
                                validation_config: Optional[Dict[str, Any]] = None) -> List[ModelPerformance]:
        """Validate multiple models and return performance comparison"""
        
        validation_config = validation_config or {}
        results = []
        
        for model, algorithm_name, hyperparameters in models_info:
            logging.info(f"Validating {algorithm_name}...")
            
            try:
                performance = self.validate_single_model(
                    model, algorithm_name, X, y, hyperparameters, **validation_config
                )
                results.append(performance)
                
            except Exception as e:
                logging.error(f"Validation failed for {algorithm_name}: {e}")
                continue
        
        return results
    
    def compare_models(self, performances: List[ModelPerformance], 
                      significance_level: float = 0.05) -> ComparisonResult:
        """Compare multiple models with statistical significance testing"""
        
        if len(performances) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Extract primary scores
        primary_scores = []
        model_names = []
        cv_scores_dict = {}
        
        for perf in performances:
            model_names.append(perf.algorithm_name)
            primary_scores.append(perf.validation_metrics.primary_score)
            cv_scores_dict[perf.algorithm_name] = perf.validation_metrics.cv_scores[perf.validation_metrics.primary_metric]
        
        # Rank models by performance
        performance_ranking = list(zip(model_names, primary_scores))
        performance_ranking.sort(key=lambda x: x[1], reverse=True)
        
        best_algorithm = performance_ranking[0][0]
        
        # Statistical significance testing
        statistical_tests = self._perform_statistical_tests(cv_scores_dict, significance_level)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(performances, statistical_tests, significance_level)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for perf in performances:
            scores = perf.validation_metrics.cv_scores[perf.validation_metrics.primary_metric]
            mean_score = np.mean(scores)
            sem = stats.sem(scores)
            ci = stats.t.interval(1-significance_level, len(scores)-1, loc=mean_score, scale=sem)
            confidence_intervals[perf.algorithm_name] = ci
        
        return ComparisonResult(
            best_algorithm=best_algorithm,
            performance_ranking=performance_ranking,
            statistical_tests=statistical_tests,
            significance_level=significance_level,
            recommendation=recommendation,
            confidence_interval=confidence_intervals
        )
    
    def _perform_statistical_tests(self, cv_scores_dict: Dict[str, List[float]], 
                                 significance_level: float) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests between models"""
        
        statistical_tests = {}
        model_names = list(cv_scores_dict.keys())
        
        # Pairwise t-tests
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                scores1 = cv_scores_dict[model1]
                scores2 = cv_scores_dict[model2]
                
                # Paired t-test (since same CV folds)
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                test_key = f"{model1}_vs_{model2}"
                statistical_tests[test_key] = {
                    'test_type': 'paired_t_test',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < significance_level,
                    'effect_size': float(np.mean(scores1) - np.mean(scores2)),
                    'winner': model1 if np.mean(scores1) > np.mean(scores2) else model2
                }
        
        # ANOVA test if more than 2 models
        if len(model_names) > 2:
            scores_lists = [cv_scores_dict[name] for name in model_names]
            f_stat, p_value = stats.f_oneway(*scores_lists)
            
            statistical_tests['anova'] = {
                'test_type': 'one_way_anova',
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < significance_level
            }
        
        return statistical_tests
    
    def _generate_recommendation(self, performances: List[ModelPerformance], 
                               statistical_tests: Dict[str, Dict[str, Any]],
                               significance_level: float) -> str:
        """Generate human-readable recommendation"""
        
        if len(performances) == 1:
            return f"Single model evaluated: {performances[0].algorithm_name}"
        
        # Find best performing model
        best_perf = max(performances, key=lambda p: p.validation_metrics.primary_score)
        best_name = best_perf.algorithm_name
        best_score = best_perf.validation_metrics.primary_score
        
        # Check if best model is significantly better than others
        significant_wins = 0
        total_comparisons = 0
        
        for test_key, test_result in statistical_tests.items():
            if test_key.startswith(best_name) or test_key.endswith(best_name):
                if test_result.get('winner') == best_name and test_result.get('significant', False):
                    significant_wins += 1
                total_comparisons += 1
        
        if significant_wins == total_comparisons and total_comparisons > 0:
            recommendation = (f"Strong recommendation: {best_name} (score: {best_score:.4f}) "
                            f"significantly outperforms all other models.")
        elif significant_wins > total_comparisons / 2:
            recommendation = (f"Moderate recommendation: {best_name} (score: {best_score:.4f}) "
                            f"outperforms most other models significantly.")
        else:
            # Look at practical considerations
            fastest_model = min(performances, key=lambda p: p.training_time)
            smallest_model = min(performances, key=lambda p: p.model_size_mb)
            
            if best_perf.training_time < np.median([p.training_time for p in performances]):
                recommendation = (f"Balanced recommendation: {best_name} (score: {best_score:.4f}) "
                                f"offers best performance with reasonable training time.")
            else:
                recommendation = (f"Performance vs Speed tradeoff: {best_name} (score: {best_score:.4f}) "
                                f"has highest accuracy, but {fastest_model.algorithm_name} "
                                f"is fastest to train. Consider use case requirements.")
        
        return recommendation
    
    def _estimate_model_size(self, model) -> float:
        """Estimate model size in MB"""
        try:
            import pickle
            import sys
            
            # Serialize model to estimate size
            pickled_model = pickle.dumps(model)
            size_bytes = len(pickled_model)
            size_mb = size_bytes / (1024 * 1024)
            
            return size_mb
            
        except Exception as e:
            logging.debug(f"Could not estimate model size: {e}")
            return 0.0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation history"""
        
        if not self.validation_history:
            return {'message': 'No validation history available'}
        
        summary = {
            'total_validations': len(self.validation_history),
            'algorithms_evaluated': list(set(p.algorithm_name for p in self.validation_history)),
            'best_performance': None,
            'average_metrics': {},
            'training_time_stats': {},
            'model_size_stats': {}
        }
        
        # Find best performance
        best_perf = max(self.validation_history, key=lambda p: p.validation_metrics.primary_score)
        summary['best_performance'] = {
            'algorithm': best_perf.algorithm_name,
            'score': best_perf.validation_metrics.primary_score,
            'metric': best_perf.validation_metrics.primary_metric
        }
        
        # Calculate average metrics across all validations
        all_primary_scores = [p.validation_metrics.primary_score for p in self.validation_history]
        summary['average_metrics'] = {
            'mean_primary_score': np.mean(all_primary_scores),
            'std_primary_score': np.std(all_primary_scores),
            'min_primary_score': np.min(all_primary_scores),
            'max_primary_score': np.max(all_primary_scores)
        }
        
        # Training time statistics
        training_times = [p.training_time for p in self.validation_history]
        summary['training_time_stats'] = {
            'mean_seconds': np.mean(training_times),
            'std_seconds': np.std(training_times),
            'min_seconds': np.min(training_times),
            'max_seconds': np.max(training_times)
        }
        
        # Model size statistics
        model_sizes = [p.model_size_mb for p in self.validation_history]
        summary['model_size_stats'] = {
            'mean_mb': np.mean(model_sizes),
            'std_mb': np.std(model_sizes),
            'min_mb': np.min(model_sizes),
            'max_mb': np.max(model_sizes)
        }
        
        return summary