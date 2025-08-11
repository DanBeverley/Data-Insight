"""Intelligent AutoML Model Selection System"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')

from .algorithm_portfolio import IntelligentAlgorithmPortfolio, TaskType, AlgorithmRecommendation
from .hyperparameter_optimizer import IntelligentHyperparameterOptimizer, OptimizationResult
from .performance_validator import ProductionModelValidator, ModelPerformance, TaskType as ValidatorTaskType
from ..learning.adaptive_system import AdaptiveLearningSystem, AdaptiveConfig

@dataclass
class AutoMLResult:
    best_model: Any
    best_algorithm: str
    best_score: float
    best_params: Dict[str, Any]
    all_results: List[Dict[str, Any]]
    selection_time: float
    recommendation: str
    performance_details: Dict[str, Any]

@dataclass
class AutoMLConfig:
    max_algorithms: int = 5
    max_optimization_time: int = 300
    cv_folds: int = 5
    test_size: float = 0.2
    enable_ensemble: bool = True
    enable_adaptive_learning: bool = True
    random_state: int = 42

class IntelligentAutoMLSystem:
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.algorithm_portfolio = IntelligentAlgorithmPortfolio()
        self.hyperparameter_optimizer = IntelligentHyperparameterOptimizer()
        self.performance_validator = ProductionModelValidator()
        
        if self.config.enable_adaptive_learning:
            self.adaptive_system = AdaptiveLearningSystem()
        else:
            self.adaptive_system = None
    
    def select_best_model(self, X: pd.DataFrame, y: pd.Series, 
                         task_type: Optional[TaskType] = None) -> AutoMLResult:
        
        start_time = time.time()
        
        if task_type is None:
            task_type = TaskType.CLASSIFICATION if y.nunique() <= 20 else TaskType.REGRESSION
        
        data_characteristics = self.algorithm_portfolio.analyze_data_characteristics(X, y)
        algorithm_recommendations = self.algorithm_portfolio.recommend_algorithms(
            data_characteristics, task_type, self.config.max_algorithms
        )
        
        all_results = []
        best_model = None
        best_score = -np.inf if task_type == TaskType.REGRESSION else 0
        best_algorithm = ""
        best_params = {}
        
        for i, recommendation in enumerate(algorithm_recommendations):
            try:
                result = self._evaluate_algorithm(
                    recommendation, X, y, task_type, data_characteristics
                )
                all_results.append(result)
                
                if result['final_score'] > best_score:
                    best_score = result['final_score']
                    best_model = result['optimized_model']
                    best_algorithm = result['algorithm_name']
                    best_params = result['best_params']
                    
            except Exception as e:
                logging.warning(f"Failed to evaluate {recommendation.algorithm_name}: {e}")
                continue
        
        if self.config.enable_ensemble and len(all_results) >= 2:
            ensemble_result = self._create_ensemble(all_results, X, y, task_type)
            if ensemble_result['final_score'] > best_score:
                best_model = ensemble_result['model']
                best_score = ensemble_result['final_score']
                best_algorithm = "ensemble"
                best_params = ensemble_result['params']
                all_results.append(ensemble_result)
        
        selection_time = time.time() - start_time
        recommendation = self._generate_recommendation(best_algorithm, best_score, all_results)
        performance_details = self._compile_performance_details(all_results, best_algorithm)
        
        return AutoMLResult(
            best_model=best_model,
            best_algorithm=best_algorithm,
            best_score=best_score,
            best_params=best_params,
            all_results=all_results,
            selection_time=selection_time,
            recommendation=recommendation,
            performance_details=performance_details
        )
    
    def _evaluate_algorithm(self, recommendation: AlgorithmRecommendation,
                           X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                           data_characteristics) -> Dict[str, Any]:
        
        model = self.algorithm_portfolio.create_model_instance(recommendation)
        algorithm_name = recommendation.algorithm_name
        
        baseline_performance = self.algorithm_portfolio.quick_evaluation(
            model, X, y, task_type
        )
        
        optimization_result = self.hyperparameter_optimizer.optimize(
            model, X, y, algorithm_name, task_type.value
        )
        
        if optimization_result.best_params:
            optimized_model = self.algorithm_portfolio.create_model_instance(
                recommendation, optimization_result.best_params
            )
        else:
            optimized_model = model
        
        validator_task_type = ValidatorTaskType.CLASSIFICATION if task_type == TaskType.CLASSIFICATION else ValidatorTaskType.REGRESSION
        performance = self.performance_validator.validate_model(
            optimized_model, X, y, validator_task_type, algorithm_name
        )
        
        return {
            'algorithm_name': algorithm_name,
            'baseline_score': baseline_performance.get('mean_accuracy', baseline_performance.get('mean_r2', 0)),
            'optimized_model': optimized_model,
            'best_params': optimization_result.best_params,
            'optimization_score': optimization_result.best_score,
            'final_score': performance.validation_metrics.primary_score,
            'suitability_score': recommendation.suitability_score,
            'reasoning': recommendation.reasoning,
            'computational_cost': recommendation.computational_cost,
            'interpretability': recommendation.interpretability,
            'stability': performance.stability_score,
            'robustness': performance.robustness_score,
            'training_time': performance.training_time,
            'memory_usage': performance.memory_usage_mb,
            'optimization_history': optimization_result.optimization_history
        }
    
    def _create_ensemble(self, algorithm_results: List[Dict[str, Any]],
                        X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> Dict[str, Any]:
        
        top_models = sorted(algorithm_results, key=lambda x: x['final_score'], reverse=True)[:3]
        
        models = [result['optimized_model'] for result in top_models]
        model_names = [result['algorithm_name'] for result in top_models]
        
        ensemble = VotingEnsemble(models, model_names, task_type)
        
        validator_task_type = ValidatorTaskType.CLASSIFICATION if task_type == TaskType.CLASSIFICATION else ValidatorTaskType.REGRESSION
        performance = self.performance_validator.validate_model(
            ensemble, X, y, validator_task_type, "ensemble"
        )
        
        return {
            'algorithm_name': 'ensemble',
            'model': ensemble,
            'final_score': performance.validation_metrics.primary_score,
            'params': {'component_algorithms': model_names},
            'stability': performance.stability_score,
            'robustness': performance.robustness_score,
            'training_time': performance.training_time,
            'memory_usage': performance.memory_usage_mb,
            'component_scores': [result['final_score'] for result in top_models]
        }
    
    def _generate_recommendation(self, best_algorithm: str, best_score: float,
                               all_results: List[Dict[str, Any]]) -> str:
        
        if not all_results:
            return "No models could be evaluated successfully"
        
        best_result = next(r for r in all_results if r['algorithm_name'] == best_algorithm)
        
        recommendation = f"Recommend {best_algorithm} (score: {best_score:.3f})"
        
        if best_result.get('stability', 0) > 0.8:
            recommendation += " - highly stable"
        elif best_result.get('stability', 0) < 0.5:
            recommendation += " - low stability, monitor carefully"
        
        if best_result.get('training_time', 0) > 30:
            recommendation += " - long training time"
        
        if best_result.get('interpretability') == 'high':
            recommendation += " - interpretable"
        elif best_result.get('interpretability') == 'low':
            recommendation += " - black box model"
        
        score_gap = best_score - sorted([r['final_score'] for r in all_results], reverse=True)[1] if len(all_results) > 1 else 0
        if score_gap < 0.02:
            recommendation += " - marginal improvement over alternatives"
        
        return recommendation
    
    def _compile_performance_details(self, all_results: List[Dict[str, Any]],
                                   best_algorithm: str) -> Dict[str, Any]:
        
        if not all_results:
            return {}
        
        performance_summary = {
            'algorithms_evaluated': len(all_results),
            'best_algorithm': best_algorithm,
            'performance_ranking': sorted(
                [(r['algorithm_name'], r['final_score']) for r in all_results],
                key=lambda x: x[1], reverse=True
            ),
            'optimization_efficiency': {
                r['algorithm_name']: {
                    'improvement': r['optimization_score'] - r['baseline_score']
                    if r.get('optimization_score') and r.get('baseline_score') else 0,
                    'evaluations': len(r.get('optimization_history', []))
                }
                for r in all_results
            },
            'computational_costs': {
                r['algorithm_name']: {
                    'training_time': r.get('training_time', 0),
                    'memory_mb': r.get('memory_usage', 0),
                    'cost_category': r.get('computational_cost', 'unknown')
                }
                for r in all_results
            },
            'reliability_scores': {
                r['algorithm_name']: {
                    'stability': r.get('stability', 0),
                    'robustness': r.get('robustness', 0),
                    'interpretability': r.get('interpretability', 'unknown')
                }
                for r in all_results
            }
        }
        
        return performance_summary
    
    def continuous_learning(self, current_model, X_new: pd.DataFrame, y_new: pd.Series,
                           algorithm_name: str, task_type: TaskType) -> Dict[str, Any]:
        
        if not self.adaptive_system:
            return {'status': 'adaptive_learning_disabled'}
        
        validator_task_type = ValidatorTaskType.CLASSIFICATION if task_type == TaskType.CLASSIFICATION else ValidatorTaskType.REGRESSION
        
        adaptation_results = self.adaptive_system.monitor_and_adapt(
            current_model, X_new, y_new, validator_task_type, algorithm_name
        )
        
        if adaptation_results.get('success'):
            logging.info(f"Model adapted successfully: {adaptation_results.get('improvement', 0):.4f} improvement")
        
        return adaptation_results
    
    def get_model_insights(self, model, X: pd.DataFrame, algorithm_name: str) -> Dict[str, Any]:
        
        insights = {
            'algorithm_name': algorithm_name,
            'feature_count': len(X.columns),
            'sample_count': len(X),
            'model_complexity': self._assess_model_complexity(model, algorithm_name)
        }
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            insights['feature_importance'] = dict(sorted(feature_importance.items(), 
                                                        key=lambda x: x[1], reverse=True)[:10])
        
        if hasattr(model, 'coef_'):
            coef_dict = dict(zip(X.columns, model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_))
            insights['coefficients'] = dict(sorted(coef_dict.items(), 
                                                  key=lambda x: abs(x[1]), reverse=True)[:10])
        
        insights['memory_footprint_mb'] = self._estimate_model_size(model)
        insights['prediction_complexity'] = self._assess_prediction_complexity(model, algorithm_name)
        
        return insights
    
    def _assess_model_complexity(self, model, algorithm_name: str) -> str:
        
        if algorithm_name in ['linear_regression', 'logistic_regression']:
            return 'low'
        elif algorithm_name in ['decision_tree', 'naive_bayes']:
            return 'low'
        elif algorithm_name in ['random_forest', 'knn']:
            return 'medium'
        elif algorithm_name in ['gradient_boosting', 'svm', 'neural_network']:
            return 'high'
        else:
            return 'medium'
    
    def _assess_prediction_complexity(self, model, algorithm_name: str) -> str:
        
        if algorithm_name in ['linear_regression', 'logistic_regression']:
            return 'O(n)'
        elif algorithm_name in ['decision_tree']:
            return 'O(log n)'
        elif algorithm_name in ['random_forest']:
            return 'O(k * log n)'
        elif algorithm_name in ['knn']:
            return 'O(n * d)'
        else:
            return 'O(n)'
    
    def _estimate_model_size(self, model) -> float:
        
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except:
            return 0.0
    
    def export_model_config(self, automl_result: AutoMLResult) -> Dict[str, Any]:
        
        return {
            'model_selection': {
                'best_algorithm': automl_result.best_algorithm,
                'best_score': automl_result.best_score,
                'best_parameters': automl_result.best_params,
                'selection_time': automl_result.selection_time
            },
            'alternatives': [
                {
                    'algorithm': result['algorithm_name'],
                    'score': result['final_score'],
                    'parameters': result.get('best_params', {}),
                    'reasoning': result.get('reasoning', [])
                }
                for result in automl_result.all_results[:5]
            ],
            'recommendation': automl_result.recommendation,
            'performance_analysis': automl_result.performance_details,
            'config_used': self.config.__dict__,
            'timestamp': time.time()
        }

class VotingEnsemble:
    
    def __init__(self, models: List, model_names: List[str], task_type: TaskType):
        self.models = models
        self.model_names = model_names
        self.task_type = task_type
    
    def predict(self, X):
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except:
                continue
        
        if not predictions:
            return np.zeros(len(X))
        
        predictions = np.array(predictions)
        
        if self.task_type == TaskType.CLASSIFICATION:
            from scipy import stats
            ensemble_pred = stats.mode(predictions, axis=0)[0].flatten()
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def get_params(self, deep=True):
        return {
            'models': self.model_names,
            'ensemble_type': 'voting',
            'task_type': self.task_type.value
        }

def run_automl(X: pd.DataFrame, y: pd.Series, 
              task_type: Optional[TaskType] = None,
              config: Optional[AutoMLConfig] = None) -> AutoMLResult:
    
    automl = IntelligentAutoMLSystem(config)
    return automl.select_best_model(X, y, task_type)