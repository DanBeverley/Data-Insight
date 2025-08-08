import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import json
import joblib
import os
from datetime import datetime

from .dataset_analyzer import DatasetAnalyzer, DatasetCharacteristics
from .algorithm_portfolio import AlgorithmPortfolioManager, SklearnAlgorithm
from .hyperparameter_optimizer import HyperparameterOptimizer, OptimizationResult
from .performance_validator import PerformanceValidator, ModelPerformance, ComparisonResult

@dataclass
class ModelSelectionResult:
    best_model: Any
    best_algorithm_name: str
    best_hyperparameters: Dict[str, Any]
    best_score: float
    dataset_characteristics: DatasetCharacteristics
    optimization_results: Dict[str, OptimizationResult]
    performance_comparison: ComparisonResult
    model_performances: List[ModelPerformance]
    selection_time: float
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelArtifact:
    model: Any
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    dataset_characteristics: DatasetCharacteristics
    feature_names: List[str]
    target_name: str
    creation_timestamp: str
    model_id: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntelligentModelSelector:
    """Production-grade intelligent model selection system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.dataset_analyzer = DatasetAnalyzer(self.config.get('dataset_analyzer', {}))
        self.portfolio_manager = AlgorithmPortfolioManager(self.config.get('portfolio_manager', {}))
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config.get('hyperparameter_optimizer', {}))
        self.performance_validator = PerformanceValidator(self.config.get('performance_validator', {}))
        
        # Model registry
        self.model_registry: Dict[str, ModelArtifact] = {}
        self.selection_history: List[ModelSelectionResult] = []
        
        # Configuration
        self.max_algorithms_to_try = self.config.get('max_algorithms_to_try', 5)
        self.optimization_budget = self.config.get('optimization_budget', 50)  # Max evaluations per algorithm
        self.enable_caching = self.config.get('enable_caching', True)
        self.model_storage_path = self.config.get('model_storage_path', './models')
        
        # Create storage directory
        if self.model_storage_path and not os.path.exists(self.model_storage_path):
            os.makedirs(self.model_storage_path)
    
    def select_best_model(self, X: pd.DataFrame, y: pd.Series,
                         constraints: Optional[Dict[str, Any]] = None,
                         validation_config: Optional[Dict[str, Any]] = None,
                         optimization_config: Optional[Dict[str, Any]] = None) -> ModelSelectionResult:
        """Main entry point for intelligent model selection"""
        
        start_time = time.time()
        constraints = constraints or {}
        validation_config = validation_config or {}
        optimization_config = optimization_config or {}
        
        logging.info("Starting intelligent model selection...")
        
        # Step 1: Analyze dataset characteristics
        logging.info("Step 1: Analyzing dataset characteristics...")
        dataset_characteristics = self.dataset_analyzer.analyze_dataset(X, y, constraints)
        
        logging.info(f"Dataset analysis complete: {dataset_characteristics.n_samples} samples, "
                    f"{dataset_characteristics.n_features} features, "
                    f"complexity={dataset_characteristics.complexity_score:.3f}")
        
        # Step 2: Select candidate algorithms
        logging.info("Step 2: Selecting candidate algorithms...")
        selected_algorithms = self.portfolio_manager.select_algorithms(dataset_characteristics, constraints)
        selected_algorithms = selected_algorithms[:self.max_algorithms_to_try]
        
        logging.info(f"Selected algorithms: {selected_algorithms}")
        
        # Step 3: Hyperparameter optimization for each algorithm
        logging.info("Step 3: Optimizing hyperparameters...")
        optimization_results = {}
        optimized_models = []
        
        for algorithm_name in selected_algorithms:
            try:
                logging.info(f"Optimizing {algorithm_name}...")
                
                # Create algorithm instance
                algorithm = self.portfolio_manager.create_algorithm(algorithm_name)
                algorithm_config = self.portfolio_manager.get_algorithm_info(algorithm_name)
                
                # Set optimization budget based on algorithm complexity
                budget = self._get_optimization_budget(algorithm_config)
                optimization_config['n_calls'] = budget
                
                # Optimize hyperparameters
                optimization_result = self.hyperparameter_optimizer.optimize_algorithm(
                    algorithm, X, y, algorithm_config.param_ranges, optimization_config
                )
                
                optimization_results[algorithm_name] = optimization_result
                
                # Create optimized model instance
                optimized_model = algorithm_config.model_class(
                    **{**algorithm_config.default_params, **optimization_result.best_params}
                )
                optimized_models.append((optimized_model, algorithm_name, optimization_result.best_params))
                
                # Update portfolio manager performance history
                self.portfolio_manager.update_performance_history(algorithm_name, optimization_result.best_score)
                
            except Exception as e:
                logging.error(f"Optimization failed for {algorithm_name}: {e}")
                # Create model with default parameters as fallback
                algorithm_config = self.portfolio_manager.get_algorithm_info(algorithm_name)
                default_model = algorithm_config.model_class(**algorithm_config.default_params)
                optimized_models.append((default_model, algorithm_name, algorithm_config.default_params))
                
                # Create dummy optimization result
                optimization_results[algorithm_name] = OptimizationResult(
                    best_params=algorithm_config.default_params,
                    best_score=0.0,
                    optimization_history=[],
                    total_evaluations=0,
                    optimization_time=0.0,
                    convergence_info={'converged': False, 'error': str(e)}
                )
        
        # Step 4: Performance validation and comparison
        logging.info("Step 4: Validating and comparing models...")
        model_performances = self.performance_validator.validate_multiple_models(
            optimized_models, X, y, validation_config
        )
        
        if not model_performances:
            raise RuntimeError("No models could be successfully validated")
        
        # Compare models
        comparison_result = self.performance_validator.compare_models(model_performances)
        
        # Step 5: Select best model
        best_performance = max(model_performances, key=lambda p: p.validation_metrics.primary_score)
        best_model_info = next((m for m in optimized_models if m[1] == best_performance.algorithm_name), None)
        
        if best_model_info is None:
            raise RuntimeError("Could not find best model info")
        
        best_model, best_algorithm_name, best_hyperparameters = best_model_info
        
        # Train best model on full dataset
        logging.info(f"Training final model: {best_algorithm_name}")
        best_model.fit(X, y)
        
        selection_time = time.time() - start_time
        
        # Generate final recommendation
        recommendation = self._generate_final_recommendation(
            dataset_characteristics, comparison_result, best_performance, selection_time
        )
        
        # Create result
        result = ModelSelectionResult(
            best_model=best_model,
            best_algorithm_name=best_algorithm_name,
            best_hyperparameters=best_hyperparameters,
            best_score=best_performance.validation_metrics.primary_score,
            dataset_characteristics=dataset_characteristics,
            optimization_results=optimization_results,
            performance_comparison=comparison_result,
            model_performances=model_performances,
            selection_time=selection_time,
            recommendation=recommendation,
            metadata={
                'feature_names': X.columns.tolist(),
                'target_name': y.name or 'target',
                'selection_timestamp': datetime.now().isoformat(),
                'config_used': self.config.copy()
            }
        )
        
        # Store in history
        self.selection_history.append(result)
        
        logging.info(f"Model selection complete! Best model: {best_algorithm_name} "
                    f"(score: {best_performance.validation_metrics.primary_score:.4f}, "
                    f"time: {selection_time:.2f}s)")
        
        return result
    
    def save_model(self, result: ModelSelectionResult, model_id: Optional[str] = None) -> str:
        """Save model to disk and register in model registry"""
        
        if model_id is None:
            model_id = f"{result.best_algorithm_name}_{int(time.time())}"
        
        # Create model artifact
        artifact = ModelArtifact(
            model=result.best_model,
            algorithm_name=result.best_algorithm_name,
            hyperparameters=result.best_hyperparameters,
            performance_metrics=result.model_performances[0].validation_metrics.all_metrics,
            dataset_characteristics=result.dataset_characteristics,
            feature_names=result.metadata['feature_names'],
            target_name=result.metadata['target_name'],
            creation_timestamp=datetime.now().isoformat(),
            model_id=model_id,
            metadata=result.metadata
        )
        
        # Save to disk
        if self.model_storage_path:
            model_path = os.path.join(self.model_storage_path, f"{model_id}.pkl")
            metadata_path = os.path.join(self.model_storage_path, f"{model_id}_metadata.json")
            
            # Save model
            joblib.dump(result.best_model, model_path)
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'algorithm_name': result.best_algorithm_name,
                'hyperparameters': result.best_hyperparameters,
                'performance_metrics': artifact.performance_metrics,
                'dataset_characteristics': result.dataset_characteristics.__dict__,
                'feature_names': artifact.feature_names,
                'target_name': artifact.target_name,
                'creation_timestamp': artifact.creation_timestamp,
                'version': artifact.version
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logging.info(f"Model saved: {model_path}")
        
        # Register in memory
        self.model_registry[model_id] = artifact
        
        return model_id
    
    def load_model(self, model_id: str) -> Optional[ModelArtifact]:
        """Load model from disk or registry"""
        
        # Check memory registry first
        if model_id in self.model_registry:
            return self.model_registry[model_id]
        
        # Try to load from disk
        if self.model_storage_path:
            model_path = os.path.join(self.model_storage_path, f"{model_id}.pkl")
            metadata_path = os.path.join(self.model_storage_path, f"{model_id}_metadata.json")
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                try:
                    # Load model
                    model = joblib.load(model_path)
                    
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Reconstruct dataset characteristics
                    dataset_chars_dict = metadata['dataset_characteristics']
                    dataset_characteristics = DatasetCharacteristics(**dataset_chars_dict)
                    
                    # Create artifact
                    artifact = ModelArtifact(
                        model=model,
                        algorithm_name=metadata['algorithm_name'],
                        hyperparameters=metadata['hyperparameters'],
                        performance_metrics=metadata['performance_metrics'],
                        dataset_characteristics=dataset_characteristics,
                        feature_names=metadata['feature_names'],
                        target_name=metadata['target_name'],
                        creation_timestamp=metadata['creation_timestamp'],
                        model_id=model_id,
                        version=metadata.get('version', '1.0')
                    )
                    
                    # Cache in registry
                    self.model_registry[model_id] = artifact
                    
                    return artifact
                    
                except Exception as e:
                    logging.error(f"Failed to load model {model_id}: {e}")
        
        return None
    
    def get_model_recommendations(self, X: pd.DataFrame, 
                                constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get model recommendations without full optimization (fast preview)"""
        
        constraints = constraints or {}
        
        # Quick dataset analysis
        y_dummy = pd.Series([0] * len(X))  # Dummy target for analysis
        dataset_characteristics = self.dataset_analyzer.analyze_dataset(X, y_dummy, constraints)
        
        # Get algorithm recommendations
        recommended_algorithms = self.portfolio_manager.select_algorithms(dataset_characteristics, constraints)
        
        # Get expected performance
        performance_expectations = self.dataset_analyzer.get_performance_expectations(dataset_characteristics)
        
        return {
            'dataset_characteristics': dataset_characteristics.__dict__,
            'recommended_algorithms': recommended_algorithms,
            'performance_expectations': performance_expectations,
            'estimated_training_time_minutes': performance_expectations['training_time_range'],
            'complexity_assessment': self._assess_task_complexity(dataset_characteristics),
            'resource_requirements': self._estimate_resource_requirements(dataset_characteristics)
        }
    
    def _get_optimization_budget(self, algorithm_config) -> int:
        """Get optimization budget based on algorithm complexity"""
        base_budget = self.optimization_budget
        
        if algorithm_config.complexity_tier == 'fast':
            return max(10, base_budget // 3)
        elif algorithm_config.complexity_tier == 'balanced':
            return base_budget // 2
        else:  # complex
            return base_budget
    
    def _generate_final_recommendation(self, dataset_characteristics: DatasetCharacteristics,
                                     comparison_result: ComparisonResult,
                                     best_performance: ModelPerformance,
                                     selection_time: float) -> str:
        """Generate final recommendation text"""
        
        algorithm_name = best_performance.algorithm_name
        score = best_performance.validation_metrics.primary_score
        metric = best_performance.validation_metrics.primary_metric
        
        complexity_desc = self._get_complexity_description(dataset_characteristics.complexity_score)
        
        recommendation = (
            f"Recommended model: {algorithm_name} "
            f"({metric}: {score:.4f})\n\n"
            f"Dataset complexity: {complexity_desc}\n"
            f"Selection completed in {selection_time:.1f} seconds\n\n"
            f"{comparison_result.recommendation}\n\n"
        )
        
        # Add practical considerations
        if best_performance.training_time > 60:  # More than 1 minute
            recommendation += "Note: This model requires significant training time. "
        
        if best_performance.model_size_mb > 100:  # Large model
            recommendation += "Note: This model has a large memory footprint. "
        
        if dataset_characteristics.imbalance_ratio < 0.3:
            recommendation += "Note: Dataset is imbalanced - consider additional sampling techniques. "
        
        return recommendation.strip()
    
    def _assess_task_complexity(self, characteristics: DatasetCharacteristics) -> str:
        """Assess task complexity level"""
        score = characteristics.complexity_score
        
        if score < 0.3:
            return "Low complexity - simple patterns, well-separated data"
        elif score < 0.6:
            return "Medium complexity - moderate patterns, some noise"
        else:
            return "High complexity - complex patterns, noisy data, challenging"
    
    def _get_complexity_description(self, complexity_score: float) -> str:
        """Get human-readable complexity description"""
        if complexity_score < 0.3:
            return "Low (simple patterns)"
        elif complexity_score < 0.6:
            return "Medium (moderate complexity)"
        else:
            return "High (complex patterns)"
    
    def _estimate_resource_requirements(self, characteristics: DatasetCharacteristics) -> Dict[str, str]:
        """Estimate resource requirements"""
        
        memory_req = characteristics.memory_requirement_mb
        n_samples = characteristics.n_samples
        
        if memory_req < 100:
            memory_desc = "Low (< 100MB)"
        elif memory_req < 1000:
            memory_desc = "Medium (100MB - 1GB)"
        else:
            memory_desc = "High (> 1GB)"
        
        if n_samples < 1000:
            compute_desc = "Low (fast training)"
        elif n_samples < 10000:
            compute_desc = "Medium (moderate training time)"
        else:
            compute_desc = "High (long training time)"
        
        return {
            'memory': memory_desc,
            'compute': compute_desc,
            'storage': f"~{memory_req:.1f}MB for model storage"
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        validation_summary = self.performance_validator.get_validation_summary()
        optimization_cache_info = self.hyperparameter_optimizer.get_cache_info()
        
        return {
            'model_selection_runs': len(self.selection_history),
            'models_in_registry': len(self.model_registry),
            'algorithm_portfolio': portfolio_summary,
            'validation_history': validation_summary,
            'optimization_cache': optimization_cache_info,
            'configuration': self.config,
            'storage_path': self.model_storage_path,
            'recent_selections': [
                {
                    'algorithm': result.best_algorithm_name,
                    'score': result.best_score,
                    'timestamp': result.metadata.get('selection_timestamp', 'unknown')
                } for result in self.selection_history[-5:]  # Last 5 selections
            ]
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.hyperparameter_optimizer.clear_cache()
        logging.info("All caches cleared")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = []
        
        for model_id, artifact in self.model_registry.items():
            models.append({
                'model_id': model_id,
                'algorithm_name': artifact.algorithm_name,
                'creation_timestamp': artifact.creation_timestamp,
                'performance_metrics': artifact.performance_metrics,
                'dataset_size': f"{artifact.dataset_characteristics.n_samples}x{artifact.dataset_characteristics.n_features}",
                'version': artifact.version
            })
        
        return sorted(models, key=lambda x: x['creation_timestamp'], reverse=True)