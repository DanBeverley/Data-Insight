import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
import json
from abc import ABC, abstractmethod

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SCIKIT_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIKIT_OPTIMIZE_AVAILABLE = False
    logging.warning("scikit-optimize not available, using grid search fallback")

@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Tuple[Dict[str, Any], float]]
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]

class BaseOptimizer(ABC):
    """Base class for hyperparameter optimization strategies"""
    
    @abstractmethod
    def optimize(self, objective_func: Callable, param_space: Dict[str, Any], 
                n_calls: int, random_state: int) -> OptimizationResult:
        pass

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian processes"""
    
    def optimize(self, objective_func: Callable, param_space: Dict[str, Any], 
                n_calls: int, random_state: int) -> OptimizationResult:
        
        if not SCIKIT_OPTIMIZE_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        # Convert parameter space to skopt format
        dimensions = []
        param_names = []
        
        for param_name, param_range in param_space.items():
            param_names.append(param_name)
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                low, high = param_range
                if isinstance(low, int) and isinstance(high, int):
                    dimensions.append(Integer(low, high, name=param_name))
                elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    dimensions.append(Real(low, high, name=param_name))
                else:
                    dimensions.append(Categorical(param_range, name=param_name))
            elif isinstance(param_range, (list, tuple)):
                dimensions.append(Categorical(param_range, name=param_name))
            else:
                raise ValueError(f"Invalid parameter range for {param_name}: {param_range}")
        
        # Define objective wrapper
        @use_named_args(dimensions)
        def objective(**params):
            return objective_func(params)
        
        start_time = time.time()
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=random_state,
            n_initial_points=min(10, n_calls // 2),
            acq_func='EI'  # Expected Improvement
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun  # Convert back from minimization
        
        # Build optimization history
        history = []
        for i, (params_values, score) in enumerate(zip(result.x_iters, result.func_vals)):
            params_dict = dict(zip(param_names, params_values))
            history.append((params_dict, -score))
        
        convergence_info = {
            'converged': len(result.func_vals) >= n_calls,
            'best_iteration': np.argmin(result.func_vals),
            'improvement_iterations': self._count_improvements(result.func_vals)
        }
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(result.func_vals),
            optimization_time=optimization_time,
            convergence_info=convergence_info
        )
    
    def _count_improvements(self, scores: List[float]) -> int:
        """Count number of iterations that improved the best score"""
        improvements = 0
        best_so_far = float('inf')
        
        for score in scores:
            if score < best_so_far:
                improvements += 1
                best_so_far = score
                
        return improvements

class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimization fallback"""
    
    def optimize(self, objective_func: Callable, param_space: Dict[str, Any], 
                n_calls: int, random_state: int) -> OptimizationResult:
        
        np.random.seed(random_state)
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_space, n_calls)
        
        start_time = time.time()
        history = []
        best_score = float('-inf')
        best_params = None
        
        for params in param_combinations:
            score = objective_func(params)
            history.append((params, score))
            
            if score > best_score:
                best_score = score
                best_params = params
        
        optimization_time = time.time() - start_time
        
        convergence_info = {
            'converged': True,
            'best_iteration': np.argmax([score for _, score in history]),
            'improvement_iterations': len([s for _, s in history if s > float('-inf')])
        }
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            total_evaluations=len(history),
            optimization_time=optimization_time,
            convergence_info=convergence_info
        )
    
    def _generate_param_combinations(self, param_space: Dict[str, Any], max_combinations: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search"""
        
        # Convert all ranges to discrete values
        discrete_space = {}
        for param_name, param_range in param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                low, high = param_range
                if isinstance(low, int) and isinstance(high, int):
                    # Integer range
                    n_values = min(10, high - low + 1)
                    discrete_space[param_name] = np.linspace(low, high, n_values, dtype=int).tolist()
                elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
                    # Float range
                    discrete_space[param_name] = np.linspace(low, high, 10).tolist()
                else:
                    discrete_space[param_name] = list(param_range)
            elif isinstance(param_range, (list, tuple)):
                discrete_space[param_name] = list(param_range)
            else:
                discrete_space[param_name] = [param_range]
        
        # Generate combinations
        from itertools import product
        
        param_names = list(discrete_space.keys())
        param_values = [discrete_space[name] for name in param_names]
        
        combinations = list(product(*param_values))
        
        # Limit number of combinations
        if len(combinations) > max_combinations:
            np.random.shuffle(combinations)
            combinations = combinations[:max_combinations]
        
        # Convert to list of dictionaries
        param_combinations = []
        for combination in combinations:
            params = dict(zip(param_names, combination))
            param_combinations.append(params)
        
        return param_combinations

class HyperparameterOptimizer:
    """Intelligent hyperparameter optimization system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_strategy = self._get_optimization_strategy()
        self.optimization_cache: Dict[str, OptimizationResult] = {}
        
    def _get_optimization_strategy(self) -> BaseOptimizer:
        """Get optimization strategy based on availability"""
        strategy = self.config.get('optimization_strategy', 'bayesian')
        
        if strategy == 'bayesian' and SCIKIT_OPTIMIZE_AVAILABLE:
            return BayesianOptimizer()
        else:
            return GridSearchOptimizer()
    
    def optimize_algorithm(self, algorithm, X: pd.DataFrame, y: pd.Series,
                         param_space: Dict[str, Any], 
                         optimization_config: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize hyperparameters for a given algorithm"""
        
        optimization_config = optimization_config or {}
        
        # Configuration
        n_calls = optimization_config.get('n_calls', 50)
        cv_folds = optimization_config.get('cv_folds', 5)
        scoring_metric = optimization_config.get('scoring_metric', 'auto')
        random_state = optimization_config.get('random_state', 42)
        early_stopping_patience = optimization_config.get('early_stopping_patience', 10)
        
        # Determine scoring metric
        if scoring_metric == 'auto':
            if hasattr(y, 'nunique') and y.nunique() <= 20:  # Classification
                scoring_metric = 'f1_weighted' if y.nunique() > 2 else 'f1'
            else:  # Regression
                scoring_metric = 'r2'
        
        # Create cross-validation strategy
        if scoring_metric in ['accuracy', 'f1', 'f1_weighted', 'roc_auc']:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            task_type = 'classification'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            task_type = 'regression'
        
        # Define objective function
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # Create algorithm instance with parameters
                algorithm_instance = algorithm.config.model_class(**{**algorithm.config.default_params, **params})
                
                # Evaluate with cross-validation
                scores = cross_val_score(algorithm_instance, X, y, cv=cv, 
                                       scoring=self._get_sklearn_scorer(scoring_metric))
                
                mean_score = np.mean(scores)
                
                # Handle different scoring metrics (some need to be negated for minimization)
                if scoring_metric in ['neg_mean_squared_error', 'neg_mean_absolute_error']:
                    return mean_score  # Already negative
                else:
                    return mean_score  # Positive scores
                    
            except Exception as e:
                logging.debug(f"Evaluation failed for params {params}: {e}")
                return float('-inf')  # Return very bad score for invalid parameters
        
        # Convert objective for minimization (Bayesian opt minimizes)
        def minimization_objective(params: Dict[str, Any]) -> float:
            score = objective_function(params)
            return -score  # Negate for minimization
        
        # Cache key for optimization results
        cache_key = self._generate_cache_key(algorithm.config.name, param_space, X.shape, scoring_metric)
        
        # Check cache first
        if cache_key in self.optimization_cache:
            logging.info(f"Using cached optimization result for {algorithm.config.name}")
            return self.optimization_cache[cache_key]
        
        # Run optimization
        logging.info(f"Starting hyperparameter optimization for {algorithm.config.name}")
        result = self.optimization_strategy.optimize(
            minimization_objective, param_space, n_calls, random_state
        )
        
        # Apply early stopping check
        if early_stopping_patience > 0:
            result = self._apply_early_stopping(result, early_stopping_patience)
        
        # Cache result
        self.optimization_cache[cache_key] = result
        
        logging.info(f"Optimization complete for {algorithm.config.name}. "
                    f"Best score: {result.best_score:.4f} "
                    f"({result.total_evaluations} evaluations, {result.optimization_time:.2f}s)")
        
        return result
    
    def _get_sklearn_scorer(self, scoring_metric: str) -> str:
        """Map scoring metric to sklearn scorer name"""
        scorer_mapping = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'f1_weighted': 'f1_weighted',
            'roc_auc': 'roc_auc',
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        return scorer_mapping.get(scoring_metric, scoring_metric)
    
    def _generate_cache_key(self, algorithm_name: str, param_space: Dict[str, Any], 
                          data_shape: Tuple[int, int], scoring_metric: str) -> str:
        """Generate cache key for optimization results"""
        import hashlib
        
        key_data = {
            'algorithm': algorithm_name,
            'param_space': param_space,
            'data_shape': data_shape,
            'scoring_metric': scoring_metric
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _apply_early_stopping(self, result: OptimizationResult, patience: int) -> OptimizationResult:
        """Apply early stopping to optimization results"""
        
        if len(result.optimization_history) <= patience:
            return result
        
        # Find the best score and when it was achieved
        scores = [score for _, score in result.optimization_history]
        best_score_idx = np.argmax(scores)
        
        # Check if we should stop early
        if len(scores) - best_score_idx > patience:
            # Truncate history to early stopping point
            early_stop_idx = best_score_idx + patience
            truncated_history = result.optimization_history[:early_stop_idx]
            
            # Update result
            result.optimization_history = truncated_history
            result.total_evaluations = len(truncated_history)
            result.convergence_info['early_stopped'] = True
            result.convergence_info['early_stop_iteration'] = early_stop_idx
        
        return result
    
    def optimize_multiple_algorithms(self, algorithms: List, X: pd.DataFrame, y: pd.Series,
                                   algorithm_param_spaces: Dict[str, Dict[str, Any]],
                                   optimization_config: Optional[Dict[str, Any]] = None) -> Dict[str, OptimizationResult]:
        """Optimize hyperparameters for multiple algorithms"""
        
        results = {}
        total_algorithms = len(algorithms)
        
        for i, algorithm in enumerate(algorithms):
            algorithm_name = algorithm.config.name
            
            logging.info(f"Optimizing algorithm {i+1}/{total_algorithms}: {algorithm_name}")
            
            if algorithm_name in algorithm_param_spaces:
                param_space = algorithm_param_spaces[algorithm_name]
                
                try:
                    result = self.optimize_algorithm(algorithm, X, y, param_space, optimization_config)
                    results[algorithm_name] = result
                    
                except Exception as e:
                    logging.error(f"Optimization failed for {algorithm_name}: {e}")
                    # Create a default result
                    results[algorithm_name] = OptimizationResult(
                        best_params=algorithm.config.default_params,
                        best_score=0.0,
                        optimization_history=[],
                        total_evaluations=0,
                        optimization_time=0.0,
                        convergence_info={'converged': False, 'error': str(e)}
                    )
            else:
                logging.warning(f"No parameter space defined for {algorithm_name}")
                # Use default parameters
                results[algorithm_name] = OptimizationResult(
                    best_params=algorithm.config.default_params,
                    best_score=0.0,
                    optimization_history=[],
                    total_evaluations=0,
                    optimization_time=0.0,
                    convergence_info={'converged': False, 'no_param_space': True}
                )
        
        return results
    
    def get_optimization_summary(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """Generate summary of optimization results"""
        
        if not results:
            return {'error': 'No optimization results available'}
        
        summary = {
            'total_algorithms': len(results),
            'total_evaluations': sum(r.total_evaluations for r in results.values()),
            'total_optimization_time': sum(r.optimization_time for r in results.values()),
            'best_algorithm': None,
            'best_score': float('-inf'),
            'algorithm_rankings': [],
            'convergence_summary': {
                'converged': 0,
                'early_stopped': 0,
                'failed': 0
            }
        }
        
        # Find best algorithm and create rankings
        algorithm_scores = []
        for algorithm_name, result in results.items():
            algorithm_scores.append((algorithm_name, result.best_score))
            
            # Update convergence summary
            if result.convergence_info.get('converged', False):
                summary['convergence_summary']['converged'] += 1
            elif result.convergence_info.get('early_stopped', False):
                summary['convergence_summary']['early_stopped'] += 1
            else:
                summary['convergence_summary']['failed'] += 1
        
        # Sort by score (descending)
        algorithm_scores.sort(key=lambda x: x[1], reverse=True)
        
        summary['best_algorithm'] = algorithm_scores[0][0]
        summary['best_score'] = algorithm_scores[0][1]
        summary['algorithm_rankings'] = algorithm_scores
        
        return summary
    
    def clear_cache(self):
        """Clear optimization cache"""
        self.optimization_cache.clear()
        logging.info("Optimization cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about optimization cache"""
        return {
            'cache_size': len(self.optimization_cache),
            'cached_algorithms': list(set(key.split('_')[0] for key in self.optimization_cache.keys())),
            'memory_usage_estimate_mb': len(str(self.optimization_cache)) / (1024 * 1024)
        }