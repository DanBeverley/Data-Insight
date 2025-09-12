import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from ..core.project_definition import Objective, Domain, RiskLevel

class BenchmarkCategory(Enum):
    INDUSTRY_STANDARD = "industry_standard"
    BEST_IN_CLASS = "best_in_class"
    REGULATORY_MINIMUM = "regulatory_minimum"
    HISTORICAL_BASELINE = "historical_baseline"

@dataclass
class BenchmarkThreshold:
    value: float
    category: BenchmarkCategory
    source: str
    confidence: float = 0.95
    
@dataclass
class BenchmarkResult:
    objective: Objective
    achieved_score: float
    benchmark_score: float
    category: BenchmarkCategory
    performance_ratio: float
    meets_benchmark: bool
    confidence_interval: Tuple[float, float]
    contextual_factors: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ObjectiveBenchmarker:
    
    def __init__(self):
        self.domain_benchmarks = self._initialize_domain_benchmarks()
        self.objective_benchmarks = self._initialize_objective_benchmarks()
        self.historical_cache = {}
        
    def benchmark_objective_performance(self, 
                                      objective: Objective,
                                      domain: Domain,
                                      achieved_metrics: Dict[str, Any],
                                      model_metadata: Dict[str, Any],
                                      dataset_characteristics: Optional[Dict] = None) -> BenchmarkResult:
        
        benchmark_thresholds = self._get_benchmark_thresholds(objective, domain)
        achieved_score = self._extract_objective_score(objective, achieved_metrics, model_metadata)
        
        best_benchmark = self._select_appropriate_benchmark(
            benchmark_thresholds, dataset_characteristics, domain
        )
        
        performance_ratio = achieved_score / best_benchmark.value if best_benchmark.value > 0 else 0
        meets_benchmark = achieved_score >= best_benchmark.value
        
        confidence_interval = self._calculate_confidence_interval(
            achieved_score, achieved_metrics, best_benchmark.confidence
        )
        
        contextual_factors = self._analyze_contextual_factors(
            objective, domain, dataset_characteristics, model_metadata
        )
        
        recommendations = self._generate_recommendations(
            objective, achieved_score, best_benchmark, contextual_factors
        )
        
        return BenchmarkResult(
            objective=objective,
            achieved_score=achieved_score,
            benchmark_score=best_benchmark.value,
            category=best_benchmark.category,
            performance_ratio=performance_ratio,
            meets_benchmark=meets_benchmark,
            confidence_interval=confidence_interval,
            contextual_factors=contextual_factors,
            recommendations=recommendations
        )
    
    def benchmark_multi_objective(self,
                                 objectives: List[Objective],
                                 domain: Domain,
                                 achieved_metrics: Dict[str, Any],
                                 model_metadata: Dict[str, Any],
                                 objective_weights: Optional[Dict[Objective, float]] = None) -> Dict[Objective, BenchmarkResult]:
        
        results = {}
        weights = objective_weights or {obj: 1.0/len(objectives) for obj in objectives}
        
        for objective in objectives:
            result = self.benchmark_objective_performance(
                objective, domain, achieved_metrics, model_metadata
            )
            results[objective] = result
        
        return results
    
    def _initialize_domain_benchmarks(self) -> Dict[Domain, Dict[Objective, List[BenchmarkThreshold]]]:
        return {
            Domain.FINANCE: {
                Objective.ACCURACY: [
                    BenchmarkThreshold(0.85, BenchmarkCategory.REGULATORY_MINIMUM, "Basel III", 0.95),
                    BenchmarkThreshold(0.92, BenchmarkCategory.INDUSTRY_STANDARD, "Financial ML Survey 2024", 0.90),
                    BenchmarkThreshold(0.96, BenchmarkCategory.BEST_IN_CLASS, "Top Tier Banks", 0.85)
                ],
                Objective.INTERPRETABILITY: [
                    BenchmarkThreshold(0.90, BenchmarkCategory.REGULATORY_MINIMUM, "Model Risk Management", 0.98),
                    BenchmarkThreshold(0.95, BenchmarkCategory.INDUSTRY_STANDARD, "Fed Guidelines", 0.95)
                ],
                Objective.SPEED: [
                    BenchmarkThreshold(100, BenchmarkCategory.INDUSTRY_STANDARD, "Real-time Trading", 0.90),
                    BenchmarkThreshold(50, BenchmarkCategory.BEST_IN_CLASS, "HFT Standards", 0.85)
                ],
                Objective.FAIRNESS: [
                    BenchmarkThreshold(0.85, BenchmarkCategory.REGULATORY_MINIMUM, "Fair Lending Act", 0.95)
                ]
            },
            Domain.HEALTHCARE: {
                Objective.ACCURACY: [
                    BenchmarkThreshold(0.88, BenchmarkCategory.REGULATORY_MINIMUM, "FDA Guidelines", 0.98),
                    BenchmarkThreshold(0.94, BenchmarkCategory.INDUSTRY_STANDARD, "Medical AI Research", 0.92),
                    BenchmarkThreshold(0.97, BenchmarkCategory.BEST_IN_CLASS, "Top Medical Centers", 0.88)
                ],
                Objective.INTERPRETABILITY: [
                    BenchmarkThreshold(0.95, BenchmarkCategory.REGULATORY_MINIMUM, "Clinical Decision Support", 0.98)
                ]
            },
            Domain.RETAIL: {
                Objective.ACCURACY: [
                    BenchmarkThreshold(0.75, BenchmarkCategory.INDUSTRY_STANDARD, "E-commerce Benchmarks", 0.90),
                    BenchmarkThreshold(0.85, BenchmarkCategory.BEST_IN_CLASS, "Amazon/Google Standards", 0.85)
                ],
                Objective.SPEED: [
                    BenchmarkThreshold(200, BenchmarkCategory.INDUSTRY_STANDARD, "E-commerce Response Time", 0.90),
                    BenchmarkThreshold(100, BenchmarkCategory.BEST_IN_CLASS, "Real-time Personalization", 0.85)
                ]
            }
        }
    
    def _initialize_objective_benchmarks(self) -> Dict[Objective, Dict[str, BenchmarkThreshold]]:
        return {
            Objective.ACCURACY: {
                "baseline": BenchmarkThreshold(0.70, BenchmarkCategory.HISTORICAL_BASELINE, "General ML", 0.85),
                "good": BenchmarkThreshold(0.85, BenchmarkCategory.INDUSTRY_STANDARD, "Production Systems", 0.90),
                "excellent": BenchmarkThreshold(0.95, BenchmarkCategory.BEST_IN_CLASS, "SOTA Research", 0.80)
            },
            Objective.SPEED: {
                "baseline": BenchmarkThreshold(1000, BenchmarkCategory.HISTORICAL_BASELINE, "Batch Processing", 0.90),
                "good": BenchmarkThreshold(200, BenchmarkCategory.INDUSTRY_STANDARD, "Web Services", 0.85),
                "excellent": BenchmarkThreshold(50, BenchmarkCategory.BEST_IN_CLASS, "Real-time Systems", 0.80)
            },
            Objective.INTERPRETABILITY: {
                "baseline": BenchmarkThreshold(0.60, BenchmarkCategory.HISTORICAL_BASELINE, "Basic Explanation", 0.85),
                "good": BenchmarkThreshold(0.80, BenchmarkCategory.INDUSTRY_STANDARD, "Business Understanding", 0.90),
                "excellent": BenchmarkThreshold(0.95, BenchmarkCategory.BEST_IN_CLASS, "Full Transparency", 0.95)
            }
        }
    
    def _get_benchmark_thresholds(self, objective: Objective, domain: Domain) -> List[BenchmarkThreshold]:
        domain_specific = self.domain_benchmarks.get(domain, {}).get(objective, [])
        general_benchmarks = list(self.objective_benchmarks.get(objective, {}).values())
        
        return domain_specific + general_benchmarks
    
    def _extract_objective_score(self, objective: Objective, metrics: Dict, metadata: Dict) -> float:
        score_extractors = {
            Objective.ACCURACY: lambda: max(
                metrics.get('accuracy', 0),
                metrics.get('f1_score', 0),
                metrics.get('best_score', 0),
                metrics.get('validation_metrics', {}).get('primary_score', 0),
                metadata.get('best_score', 0)
            ),
            Objective.SPEED: lambda: 1000.0 / max(
                metrics.get('inference_latency_ms', 1000),
                metrics.get('prediction_time', 1000),
                metadata.get('performance_validation', {}).get('prediction_time', 1000),
                1  # Avoid division by zero
            ),
            Objective.INTERPRETABILITY: lambda: metrics.get('interpretability_score', 
                1.0 if any(alg in metadata.get('best_algorithm', '').lower() 
                          for alg in ['linear', 'tree', 'logistic']) else 0.5
            ),
            Objective.FAIRNESS: lambda: metrics.get('fairness_score', 0.8),
            Objective.COMPLIANCE: lambda: metrics.get('compliance_score', 0.9),
            Objective.ROBUSTNESS: lambda: metrics.get('stability_score', 0.8),
            Objective.COST_EFFICIENCY: lambda: 1.0 / max(metrics.get('training_time', 1), 1),
            Objective.SCALABILITY: lambda: metrics.get('scalability_score', 0.7),
            Objective.INNOVATION: lambda: metrics.get('innovation_score', 0.6)
        }
        
        extractor = score_extractors.get(objective, lambda: 0.8)
        return float(extractor())
    
    def _select_appropriate_benchmark(self, 
                                    thresholds: List[BenchmarkThreshold],
                                    dataset_chars: Optional[Dict],
                                    domain: Domain) -> BenchmarkThreshold:
        
        if not thresholds:
            return BenchmarkThreshold(0.8, BenchmarkCategory.HISTORICAL_BASELINE, "Default", 0.70)
        
        domain_priority = [BenchmarkCategory.REGULATORY_MINIMUM, BenchmarkCategory.INDUSTRY_STANDARD]
        
        if domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            for category in domain_priority:
                for threshold in thresholds:
                    if threshold.category == category:
                        return threshold
        
        industry_thresholds = [t for t in thresholds if t.category == BenchmarkCategory.INDUSTRY_STANDARD]
        if industry_thresholds:
            return max(industry_thresholds, key=lambda x: x.confidence)
        
        return max(thresholds, key=lambda x: (x.confidence, x.value))
    
    def _calculate_confidence_interval(self, score: float, metrics: Dict, benchmark_confidence: float) -> Tuple[float, float]:
        variance = metrics.get('cross_validation_std', 0.05) ** 2
        std_error = np.sqrt(variance)
        
        z_score = 1.96 if benchmark_confidence >= 0.95 else 1.645
        margin = z_score * std_error
        
        return (max(0, score - margin), min(1, score + margin))
    
    def _analyze_contextual_factors(self, 
                                  objective: Objective,
                                  domain: Domain,
                                  dataset_chars: Optional[Dict],
                                  model_metadata: Dict) -> Dict[str, Any]:
        
        factors = {
            'domain_complexity': self._assess_domain_complexity(domain),
            'dataset_quality': self._assess_dataset_quality(dataset_chars),
            'model_complexity': self._assess_model_complexity(model_metadata),
            'resource_constraints': self._assess_resource_usage(model_metadata)
        }
        
        return factors
    
    def _assess_domain_complexity(self, domain: Domain) -> Dict[str, Any]:
        complexity_map = {
            Domain.FINANCE: {'score': 0.9, 'factors': ['regulatory', 'high_stakes', 'interpretability_critical']},
            Domain.HEALTHCARE: {'score': 0.95, 'factors': ['life_critical', 'regulatory', 'privacy_sensitive']},
            Domain.RETAIL: {'score': 0.6, 'factors': ['volume_driven', 'speed_critical']},
            Domain.GENERAL: {'score': 0.5, 'factors': ['standard_requirements']}
        }
        
        return complexity_map.get(domain, complexity_map[Domain.GENERAL])
    
    def _assess_dataset_quality(self, dataset_chars: Optional[Dict]) -> Dict[str, Any]:
        if not dataset_chars:
            return {'score': 0.7, 'factors': ['unknown_quality']}
        
        quality_score = 0.8
        factors = []
        
        if dataset_chars.get('missing_ratio', 0) > 0.1:
            quality_score -= 0.1
            factors.append('high_missing_values')
            
        if dataset_chars.get('complexity_score', 0.5) > 0.8:
            quality_score += 0.1
            factors.append('high_complexity')
            
        return {'score': max(0, quality_score), 'factors': factors}
    
    def _assess_model_complexity(self, model_metadata: Dict) -> Dict[str, Any]:
        algorithm = model_metadata.get('best_algorithm', '').lower()
        
        complexity_scores = {
            'linear': 0.2, 'logistic': 0.2, 'tree': 0.4, 'forest': 0.7,
            'gradient': 0.8, 'neural': 0.9, 'ensemble': 0.8
        }
        
        score = 0.5
        for alg, complexity in complexity_scores.items():
            if alg in algorithm:
                score = complexity
                break
                
        return {'score': score, 'algorithm': algorithm}
    
    def _assess_resource_usage(self, model_metadata: Dict) -> Dict[str, Any]:
        training_time = model_metadata.get('performance_validation', {}).get('training_time', 0)
        memory_usage = model_metadata.get('performance_validation', {}).get('memory_usage_mb', 0)
        
        resource_score = 1.0
        if training_time > 3600:  # 1 hour
            resource_score -= 0.2
        if memory_usage > 8000:  # 8GB
            resource_score -= 0.2
            
        return {'score': max(0, resource_score), 'training_time': training_time, 'memory_mb': memory_usage}
    
    def _generate_recommendations(self, 
                                objective: Objective,
                                achieved_score: float,
                                benchmark: BenchmarkThreshold,
                                context: Dict) -> List[str]:
        
        recommendations = []
        performance_gap = benchmark.value - achieved_score
        
        if performance_gap > 0.1:
            recommendations.extend(self._get_improvement_recommendations(objective, performance_gap, context))
        elif performance_gap > 0:
            recommendations.append(f"Close to {benchmark.category.value} benchmark. Minor optimizations needed.")
        else:
            recommendations.append(f"Exceeds {benchmark.category.value} benchmark by {(achieved_score/benchmark.value - 1)*100:.1f}%")
            
        return recommendations
    
    def _get_improvement_recommendations(self, 
                                       objective: Objective,
                                       gap: float,
                                       context: Dict) -> List[str]:
        
        base_recommendations = {
            Objective.ACCURACY: [
                "Consider ensemble methods for improved accuracy",
                "Increase feature engineering sophistication",
                "Tune hyperparameters more aggressively"
            ],
            Objective.SPEED: [
                "Optimize model architecture for inference speed",
                "Consider model compression techniques",
                "Implement caching strategies"
            ],
            Objective.INTERPRETABILITY: [
                "Switch to inherently interpretable models",
                "Enhance explainability analysis depth",
                "Provide more detailed feature importance"
            ]
        }
        
        recommendations = base_recommendations.get(objective, ["Review objective-specific optimization strategies"])
        
        if gap > 0.2:
            recommendations.insert(0, f"Significant improvement needed ({gap:.2f} gap). Consider fundamental approach changes.")
            
        return recommendations[:3]