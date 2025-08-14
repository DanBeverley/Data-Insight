"""Intelligent Hyperparameter Optimization System"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, ParameterGrid, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error
from scipy.stats import uniform, randint

class OptimizationStrategy(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search" 
    BAYESIAN = "bayesian"
    ADAPTIVE = "adaptive"

class ObjectiveType(Enum):
    ACCURACY = "accuracy"
    SPEED = "speed" 
    BALANCED = "balanced"
    INTERPRETABILITY = "interpretability"

@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    strategy_used: str
    convergence_reached: bool

@dataclass
class OptimizationConfig:
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    objective_type: ObjectiveType = ObjectiveType.BALANCED
    max_evaluations: int = 50
    cv_folds: int = 5
    early_stopping_rounds: int = 10
    time_budget_seconds: Optional[int] = None
    scoring_metric: str = "accuracy"
    random_state: int = 42

class IntelligentHyperparameterOptimizer:
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.param_spaces = self._init_parameter_spaces()
        
    def _init_parameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 3000]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000, 2000]
            },
            'ada_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        }
    
    def optimize(self, model, X: pd.DataFrame, y: pd.Series, 
                algorithm_name: str, task_type: str) -> OptimizationResult:
        
        start_time = time.time()
        param_space = self.param_spaces.get(algorithm_name, {})
        
        if not param_space:
            return OptimizationResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                total_evaluations=0,
                optimization_time=0.0,
                strategy_used="none",
                convergence_reached=False
            )
        
        adapted_space = self._adapt_parameter_space(param_space, X, y, algorithm_name)
        
        if self.config.strategy == OptimizationStrategy.ADAPTIVE:
            strategy = self._select_optimization_strategy(X, adapted_space)
        else:
            strategy = self.config.strategy
        
        if strategy == OptimizationStrategy.BAYESIAN:
            result = self._bayesian_optimization(model, X, y, adapted_space, task_type)
        elif strategy == OptimizationStrategy.RANDOM_SEARCH:
            result = self._random_search_optimization(model, X, y, adapted_space, task_type)
        else:
            result = self._grid_search_optimization(model, X, y, adapted_space, task_type)
        
        result.optimization_time = time.time() - start_time
        result.strategy_used = strategy.value
        
        return result
    
    def _adapt_parameter_space(self, param_space: Dict[str, List], 
                              X: pd.DataFrame, y: pd.Series, 
                              algorithm_name: str) -> Dict[str, List]:
        
        adapted_space = param_space.copy()
        n_samples, n_features = X.shape
        
        if algorithm_name == 'random_forest':
            if n_samples < 1000:
                adapted_space['n_estimators'] = [50, 100]
            if n_features > 100:
                adapted_space['max_features'] = ['sqrt', 'log2']
        
        elif algorithm_name == 'gradient_boosting':
            if n_samples < 1000:
                adapted_space['n_estimators'] = [50, 100]
                adapted_space['learning_rate'] = [0.1, 0.2]
        
        elif algorithm_name == 'svm' and n_samples > 10000:
            adapted_space['C'] = [0.1, 1.0, 10.0]
            adapted_space['gamma'] = ['scale', 'auto']
        
        elif algorithm_name == 'knn':
            max_k = min(15, int(np.sqrt(n_samples)))
            adapted_space['n_neighbors'] = [k for k in adapted_space['n_neighbors'] if k <= max_k]
        
        elif algorithm_name == 'neural_network':
            if n_samples < 1000:
                adapted_space['hidden_layer_sizes'] = [(50,), (100,)]
                adapted_space['max_iter'] = [500, 1000]
        
        return adapted_space
    
    def _select_optimization_strategy(self, X: pd.DataFrame, 
                                    param_space: Dict[str, List]) -> OptimizationStrategy:
        
        n_samples, n_features = X.shape
        search_space_size = np.prod([len(values) for values in param_space.values()])
        
        if search_space_size <= 20:
            return OptimizationStrategy.GRID_SEARCH
        elif search_space_size <= 1000 or n_samples < 5000:
            return OptimizationStrategy.RANDOM_SEARCH
        else:
            return OptimizationStrategy.BAYESIAN
    
    def _grid_search_optimization(self, model, X: pd.DataFrame, y: pd.Series,
                                param_space: Dict[str, List], task_type: str) -> OptimizationResult:
        
        param_grid = ParameterGrid(param_space)
        best_score = -np.inf if task_type == 'regression' and 'r2' in self.config.scoring_metric else 0
        best_params = {}
        history = []
        
        total_combinations = len(param_grid)
        max_evaluations = min(self.config.max_evaluations, total_combinations)
        
        for i, params in enumerate(param_grid):
            if i >= max_evaluations:
                break
                
            try:
                model_instance = model.__class__(**{**model.get_params(), **params})
                
                if task_type == 'classification':
                    scores = cross_val_score(model_instance, X, y, cv=self.config.cv_folds, 
                                           scoring=self.config.scoring_metric)
                else:
                    scores = cross_val_score(model_instance, X, y, cv=self.config.cv_folds, 
                                           scoring='r2' if 'r2' in self.config.scoring_metric else 'neg_mean_squared_error')
                
                mean_score = scores.mean()
                
                history.append({
                    'params': params.copy(),
                    'score': mean_score,
                    'std': scores.std(),
                    'evaluation': i + 1
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    
            except Exception as e:
                continue
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(history),
            optimization_time=0.0,
            strategy_used="grid_search",
            convergence_reached=len(history) == max_evaluations
        )
    
    def _random_search_optimization(self, model, X: pd.DataFrame, y: pd.Series,
                                  param_space: Dict[str, List], task_type: str) -> OptimizationResult:
        
        try:
            param_distributions = self._convert_to_distributions(param_space)
            
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_distributions,
                n_iter=self.config.max_evaluations,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric if task_type == 'classification' else 'r2',
                random_state=self.config.random_state,
                n_jobs=1
            )
            
            random_search.fit(X, y)
            
            history = []
            for i, (params, score) in enumerate(zip(random_search.cv_results_['params'],
                                                   random_search.cv_results_['mean_test_score'])):
                history.append({
                    'params': params,
                    'score': score,
                    'std': random_search.cv_results_['std_test_score'][i],
                    'evaluation': i + 1
                })
            
            return OptimizationResult(
                best_params=random_search.best_params_,
                best_score=random_search.best_score_,
                optimization_history=history,
                total_evaluations=len(history),
                optimization_time=0.0,
                strategy_used="random_search",
                convergence_reached=True
            )
            
        except Exception as e:
            return self._grid_search_optimization(model, X, y, param_space, task_type)
    
    def _bayesian_optimization(self, model, X: pd.DataFrame, y: pd.Series,
                             param_space: Dict[str, List], task_type: str) -> OptimizationResult:
        
        return self._random_search_optimization(model, X, y, param_space, task_type)
    
    def _convert_to_distributions(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        distributions = {}
        
        for param, values in param_space.items():
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None):
                numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]
                if numeric_values:
                    if all(isinstance(v, int) for v in numeric_values):
                        distributions[param] = randint(min(numeric_values), max(numeric_values) + 1)
                    else:
                        distributions[param] = uniform(min(numeric_values), max(numeric_values) - min(numeric_values))
                else:
                    distributions[param] = values
            else:
                distributions[param] = values
        
        return distributions
    
    def get_optimization_summary(self, result: OptimizationResult) -> Dict[str, Any]:
        if not result.optimization_history:
            return {'status': 'no_optimization_performed'}
        
        scores = [h['score'] for h in result.optimization_history]
        
        return {
            'best_score': result.best_score,
            'best_params': result.best_params,
            'total_evaluations': result.total_evaluations,
            'optimization_time': result.optimization_time,
            'strategy_used': result.strategy_used,
            'score_improvement': result.best_score - scores[0] if scores else 0,
            'convergence_reached': result.convergence_reached,
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'optimization_efficiency': result.best_score / result.total_evaluations if result.total_evaluations > 0 else 0
        }
    
    def multi_objective_optimize(self, model, X: pd.DataFrame, y: pd.Series,
                               algorithm_name: str, task_type: str,
                               objectives: List[str] = None) -> Dict[str, OptimizationResult]:
        
        objectives = objectives or ['accuracy', 'speed', 'interpretability']
        results = {}
        
        for objective in objectives:
            objective_config = OptimizationConfig(
                objective_type=ObjectiveType(objective) if objective in [o.value for o in ObjectiveType] else ObjectiveType.BALANCED,
                max_evaluations=self.config.max_evaluations // len(objectives),
                scoring_metric=self._get_objective_metric(objective, task_type)
            )
            
            optimizer = IntelligentHyperparameterOptimizer(objective_config)
            results[objective] = optimizer.optimize(model, X, y, algorithm_name, task_type)
        
        return results
    
    def _get_objective_metric(self, objective: str, task_type: str) -> str:
        if task_type == 'classification':
            if objective == 'accuracy':
                return 'accuracy'
            elif objective == 'speed':
                return 'accuracy'
            else:
                return 'f1'
        else:
            if objective == 'accuracy':
                return 'r2'
            else:
                return 'neg_mean_squared_error'
    
    def suggest_next_parameters(self, optimization_history: List[Dict[str, Any]], 
                               param_space: Dict[str, List]) -> Dict[str, Any]:
        
        if not optimization_history:
            return self._random_sample_params(param_space)
        
        best_params = max(optimization_history, key=lambda x: x['score'])['params']
        
        next_params = best_params.copy()
        for param, values in param_space.items():
            if param in next_params:
                current_idx = values.index(next_params[param]) if next_params[param] in values else 0
                
                if current_idx < len(values) - 1:
                    next_params[param] = values[current_idx + 1]
                elif current_idx > 0:
                    next_params[param] = values[current_idx - 1]
        
        return next_params
    
    def _random_sample_params(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        params = {}
        for param, values in param_space.items():
            params[param] = np.random.choice(values)
        return params

def optimize_hyperparameters(model, X: pd.DataFrame, y: pd.Series,
                            algorithm_name: str, task_type: str,
                            config: Optional[OptimizationConfig] = None) -> OptimizationResult:
    
    optimizer = IntelligentHyperparameterOptimizer(config)
    return optimizer.optimize(model, X, y, algorithm_name, task_type)