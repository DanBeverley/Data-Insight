import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from ..core.project_definition import Objective, Domain

class TradeOffType(Enum):
    ACCURACY_VS_SPEED = "accuracy_vs_speed"
    ACCURACY_VS_INTERPRETABILITY = "accuracy_vs_interpretability"
    SPEED_VS_MEMORY = "speed_vs_memory"
    FAIRNESS_VS_ACCURACY = "fairness_vs_accuracy"
    ROBUSTNESS_VS_EFFICIENCY = "robustness_vs_efficiency"
    COST_VS_PERFORMANCE = "cost_vs_performance"

class TradeOffSeverity(Enum):
    MINIMAL = "minimal"      # <5% sacrifice
    MODERATE = "moderate"    # 5-15% sacrifice  
    SIGNIFICANT = "significant"  # 15-30% sacrifice
    SEVERE = "severe"        # >30% sacrifice

@dataclass
class TradeOffMetric:
    name: str
    achieved_value: float
    theoretical_max: float
    sacrifice_percent: float
    importance_weight: float = 1.0

@dataclass
class TradeOffScenario:
    scenario_name: str
    primary_objective: Objective
    sacrificed_metrics: List[TradeOffMetric]
    gained_metrics: List[TradeOffMetric]
    net_utility_score: float
    feasibility_score: float
    recommendation_strength: float

@dataclass 
class TradeOffReport:
    primary_objective: Objective
    chosen_trade_offs: List[TradeOffType]
    sacrifice_analysis: Dict[str, TradeOffMetric]
    alternative_scenarios: List[TradeOffScenario]
    overall_trade_off_efficiency: float
    decision_rationale: str
    optimization_suggestions: List[str]
    sensitivity_analysis: Dict[str, float]

class TradeOffAnalyzer:
    
    def __init__(self):
        self.trade_off_relationships = self._initialize_trade_off_relationships()
        self.objective_weights = self._initialize_objective_weights()
        self.theoretical_limits = self._initialize_theoretical_limits()
        
    def analyze_trade_offs(self,
                          primary_objective: Objective,
                          achieved_metrics: Dict[str, Any],
                          model_metadata: Dict[str, Any], 
                          domain: Domain,
                          constraints: Optional[Dict] = None) -> TradeOffReport:
        
        # Extract normalized metrics
        normalized_metrics = self._normalize_metrics(achieved_metrics, model_metadata)
        
        # Identify active trade-offs
        active_trade_offs = self._identify_active_trade_offs(primary_objective, normalized_metrics)
        
        # Analyze sacrifice patterns
        sacrifice_analysis = self._analyze_sacrifices(
            primary_objective, normalized_metrics, active_trade_offs
        )
        
        # Generate alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(
            primary_objective, normalized_metrics, domain, constraints
        )
        
        # Calculate overall efficiency
        trade_off_efficiency = self._calculate_trade_off_efficiency(
            primary_objective, normalized_metrics, sacrifice_analysis
        )
        
        # Generate decision rationale
        decision_rationale = self._generate_decision_rationale(
            primary_objective, sacrifice_analysis, alternative_scenarios
        )
        
        # Optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            sacrifice_analysis, alternative_scenarios
        )
        
        # Sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            primary_objective, normalized_metrics, sacrifice_analysis
        )
        
        return TradeOffReport(
            primary_objective=primary_objective,
            chosen_trade_offs=active_trade_offs,
            sacrifice_analysis=sacrifice_analysis,
            alternative_scenarios=alternative_scenarios,
            overall_trade_off_efficiency=trade_off_efficiency,
            decision_rationale=decision_rationale,
            optimization_suggestions=optimization_suggestions,
            sensitivity_analysis=sensitivity_analysis
        )
    
    def _initialize_trade_off_relationships(self) -> Dict[TradeOffType, Dict[str, Any]]:
        return {
            TradeOffType.ACCURACY_VS_SPEED: {
                'primary_metrics': ['accuracy', 'f1_score'],
                'sacrifice_metrics': ['prediction_time', 'training_time'],
                'correlation_strength': -0.75,
                'typical_sacrifice_range': (0.05, 0.25)
            },
            TradeOffType.ACCURACY_VS_INTERPRETABILITY: {
                'primary_metrics': ['accuracy', 'auc'],
                'sacrifice_metrics': ['interpretability_score', 'explainability_depth'],
                'correlation_strength': -0.65,
                'typical_sacrifice_range': (0.02, 0.15)
            },
            TradeOffType.SPEED_VS_MEMORY: {
                'primary_metrics': ['prediction_time', 'inference_speed'],
                'sacrifice_metrics': ['memory_usage', 'model_size'],
                'correlation_strength': -0.60,
                'typical_sacrifice_range': (0.10, 0.40)
            },
            TradeOffType.FAIRNESS_VS_ACCURACY: {
                'primary_metrics': ['fairness_score', 'demographic_parity'],
                'sacrifice_metrics': ['accuracy', 'precision'],
                'correlation_strength': -0.45,
                'typical_sacrifice_range': (0.01, 0.10)
            }
        }
    
    def _initialize_objective_weights(self) -> Dict[Objective, Dict[str, float]]:
        return {
            Objective.ACCURACY: {
                'accuracy': 0.4, 'precision': 0.3, 'recall': 0.3,
                'speed': 0.1, 'interpretability': 0.1, 'fairness': 0.1
            },
            Objective.SPEED: {
                'prediction_time': 0.5, 'training_time': 0.3,
                'accuracy': 0.2, 'memory_usage': 0.2
            },
            Objective.INTERPRETABILITY: {
                'interpretability_score': 0.6, 'explainability_depth': 0.4,
                'accuracy': 0.2, 'speed': 0.1
            },
            Objective.FAIRNESS: {
                'fairness_score': 0.5, 'demographic_parity': 0.3, 'equalized_odds': 0.2,
                'accuracy': 0.3
            }
        }
    
    def _initialize_theoretical_limits(self) -> Dict[str, Dict[str, float]]:
        return {
            'accuracy': {'min': 0.0, 'max': 1.0, 'excellent': 0.95, 'good': 0.85},
            'prediction_time': {'min': 1, 'max': 10000, 'excellent': 50, 'good': 200},
            'training_time': {'min': 1, 'max': 86400, 'excellent': 300, 'good': 1800},
            'memory_usage': {'min': 100, 'max': 64000, 'excellent': 2000, 'good': 8000},
            'interpretability_score': {'min': 0.0, 'max': 1.0, 'excellent': 0.95, 'good': 0.80},
            'fairness_score': {'min': 0.0, 'max': 1.0, 'excellent': 0.95, 'good': 0.85}
        }
    
    def _normalize_metrics(self, achieved_metrics: Dict, model_metadata: Dict) -> Dict[str, float]:
        raw_metrics = {
            'accuracy': max(
                achieved_metrics.get('accuracy', 0),
                achieved_metrics.get('f1_score', 0),
                achieved_metrics.get('best_score', 0),
                model_metadata.get('best_score', 0),
                achieved_metrics.get('validation_metrics', {}).get('primary_score', 0)
            ),
            'prediction_time': min(
                achieved_metrics.get('inference_latency_ms', 1000),
                achieved_metrics.get('prediction_time', 1000),
                model_metadata.get('performance_validation', {}).get('prediction_time', 1000)
            ),
            'training_time': max(
                achieved_metrics.get('training_time', 0),
                model_metadata.get('performance_validation', {}).get('training_time', 0)
            ),
            'memory_usage': max(
                achieved_metrics.get('memory_usage_mb', 0),
                model_metadata.get('performance_validation', {}).get('memory_usage_mb', 0)
            ),
            'interpretability_score': self._calculate_interpretability_score(model_metadata),
            'fairness_score': achieved_metrics.get('fairness_score', 0.85)
        }
        
        # Normalize to 0-1 scale where 1 is optimal
        normalized = {}
        for metric, value in raw_metrics.items():
            if metric in self.theoretical_limits:
                limits = self.theoretical_limits[metric]
                
                if metric in ['prediction_time', 'training_time', 'memory_usage']:
                    # Lower is better - inverse normalization
                    normalized[metric] = max(0, 1 - (value - limits['min']) / (limits['good'] - limits['min']))
                else:
                    # Higher is better - direct normalization  
                    normalized[metric] = min(1, (value - limits['min']) / (limits['good'] - limits['min']))
        
        return normalized
    
    def _calculate_interpretability_score(self, model_metadata: Dict) -> float:
        algorithm = model_metadata.get('best_algorithm', '').lower()
        
        interpretability_scores = {
            'linear': 0.95, 'logistic': 0.95, 'tree': 0.85, 'decision': 0.85,
            'forest': 0.60, 'random': 0.60, 'gradient': 0.40, 'xgboost': 0.40,
            'neural': 0.20, 'deep': 0.15, 'ensemble': 0.50
        }
        
        for pattern, score in interpretability_scores.items():
            if pattern in algorithm:
                return score
                
        return 0.70  # Default for unknown algorithms
    
    def _identify_active_trade_offs(self, 
                                  primary_objective: Objective,
                                  normalized_metrics: Dict[str, float]) -> List[TradeOffType]:
        
        active_trade_offs = []
        
        # Map objectives to their typical trade-offs
        objective_trade_offs = {
            Objective.ACCURACY: [TradeOffType.ACCURACY_VS_SPEED, TradeOffType.ACCURACY_VS_INTERPRETABILITY],
            Objective.SPEED: [TradeOffType.ACCURACY_VS_SPEED, TradeOffType.SPEED_VS_MEMORY],
            Objective.INTERPRETABILITY: [TradeOffType.ACCURACY_VS_INTERPRETABILITY],
            Objective.FAIRNESS: [TradeOffType.FAIRNESS_VS_ACCURACY]
        }
        
        potential_trade_offs = objective_trade_offs.get(primary_objective, [])
        
        for trade_off_type in potential_trade_offs:
            if self._is_trade_off_active(trade_off_type, normalized_metrics):
                active_trade_offs.append(trade_off_type)
        
        return active_trade_offs
    
    def _is_trade_off_active(self, trade_off_type: TradeOffType, metrics: Dict[str, float]) -> bool:
        relationship = self.trade_off_relationships[trade_off_type]
        primary_metrics = relationship['primary_metrics']
        sacrifice_metrics = relationship['sacrifice_metrics']
        
        # Check if primary objective is strong while sacrificial metrics are weak
        primary_strong = any(metrics.get(m, 0) > 0.7 for m in primary_metrics if m in metrics)
        sacrifice_weak = any(metrics.get(m, 1) < 0.6 for m in sacrifice_metrics if m in metrics)
        
        return primary_strong and sacrifice_weak
    
    def _analyze_sacrifices(self,
                          primary_objective: Objective,
                          normalized_metrics: Dict[str, float],
                          active_trade_offs: List[TradeOffType]) -> Dict[str, TradeOffMetric]:
        
        sacrifice_analysis = {}
        objective_weights = self.objective_weights.get(primary_objective, {})
        
        for metric_name, achieved_value in normalized_metrics.items():
            if metric_name not in objective_weights or objective_weights[metric_name] >= 0.3:
                continue  # Skip primary metrics
            
            # Estimate theoretical maximum for this metric
            theoretical_max = self._estimate_theoretical_max(metric_name, normalized_metrics)
            
            # Calculate sacrifice percentage
            sacrifice_percent = max(0, (theoretical_max - achieved_value) / theoretical_max * 100)
            
            if sacrifice_percent > 5:  # Only include meaningful sacrifices
                sacrifice_analysis[metric_name] = TradeOffMetric(
                    name=metric_name,
                    achieved_value=achieved_value,
                    theoretical_max=theoretical_max,
                    sacrifice_percent=sacrifice_percent,
                    importance_weight=objective_weights.get(metric_name, 0.1)
                )
        
        return sacrifice_analysis
    
    def _estimate_theoretical_max(self, metric_name: str, current_metrics: Dict[str, float]) -> float:
        if metric_name in self.theoretical_limits:
            limits = self.theoretical_limits[metric_name]
            
            # Use context-aware estimation based on other metrics
            if metric_name == 'accuracy':
                # Accuracy ceiling depends on data quality and complexity
                base_ceiling = 0.98
                interpretability_penalty = (1 - current_metrics.get('interpretability_score', 0.8)) * 0.1
                return base_ceiling - interpretability_penalty
            
            elif metric_name == 'prediction_time':
                # Speed ceiling depends on model complexity
                accuracy = current_metrics.get('accuracy', 0.8)
                complexity_factor = 1 + (accuracy - 0.8) * 2  # Higher accuracy = more complex
                return limits['excellent'] * complexity_factor
            
            else:
                return limits.get('excellent', 0.95)
        
        return 0.95  # Default theoretical max
    
    def _generate_alternative_scenarios(self,
                                      primary_objective: Objective,
                                      current_metrics: Dict[str, float],
                                      domain: Domain,
                                      constraints: Optional[Dict]) -> List[TradeOffScenario]:
        
        scenarios = []
        
        # Scenario 1: Maximize primary objective regardless of trade-offs
        aggressive_scenario = self._create_aggressive_scenario(primary_objective, current_metrics)
        scenarios.append(aggressive_scenario)
        
        # Scenario 2: Balanced approach
        balanced_scenario = self._create_balanced_scenario(primary_objective, current_metrics)
        scenarios.append(balanced_scenario)
        
        # Scenario 3: Conservative approach (minimize sacrifices)
        conservative_scenario = self._create_conservative_scenario(primary_objective, current_metrics)
        scenarios.append(conservative_scenario)
        
        # Domain-specific scenarios
        if domain == Domain.FINANCE:
            compliance_scenario = self._create_compliance_focused_scenario(current_metrics)
            scenarios.append(compliance_scenario)
        elif domain == Domain.HEALTHCARE:
            safety_scenario = self._create_safety_focused_scenario(current_metrics)
            scenarios.append(safety_scenario)
        
        return scenarios
    
    def _create_aggressive_scenario(self, primary_objective: Objective, current_metrics: Dict) -> TradeOffScenario:
        # Maximize primary objective, accept significant trade-offs
        sacrificed_metrics = []
        gained_metrics = []
        
        if primary_objective == Objective.ACCURACY:
            gained_metrics.append(TradeOffMetric("accuracy", 0.95, 0.98, 0, 1.0))
            sacrificed_metrics.append(TradeOffMetric("prediction_time", 0.4, 0.8, 50, 0.3))
            sacrificed_metrics.append(TradeOffMetric("interpretability_score", 0.3, 0.9, 67, 0.2))
            
        elif primary_objective == Objective.SPEED:
            gained_metrics.append(TradeOffMetric("prediction_time", 0.9, 0.95, 0, 1.0))
            sacrificed_metrics.append(TradeOffMetric("accuracy", 0.75, 0.90, 17, 0.4))
            
        net_utility = self._calculate_net_utility(gained_metrics, sacrificed_metrics)
        
        return TradeOffScenario(
            scenario_name="Aggressive Optimization",
            primary_objective=primary_objective,
            sacrificed_metrics=sacrificed_metrics,
            gained_metrics=gained_metrics,
            net_utility_score=net_utility,
            feasibility_score=0.75,
            recommendation_strength=0.60
        )
    
    def _create_balanced_scenario(self, primary_objective: Objective, current_metrics: Dict) -> TradeOffScenario:
        # Moderate improvements with acceptable trade-offs
        return TradeOffScenario(
            scenario_name="Balanced Approach",
            primary_objective=primary_objective,
            sacrificed_metrics=[],
            gained_metrics=[],
            net_utility_score=0.80,
            feasibility_score=0.90,
            recommendation_strength=0.85
        )
    
    def _create_conservative_scenario(self, primary_objective: Objective, current_metrics: Dict) -> TradeOffScenario:
        # Minimal trade-offs, preserve all metrics
        return TradeOffScenario(
            scenario_name="Conservative Optimization",
            primary_objective=primary_objective,
            sacrificed_metrics=[],
            gained_metrics=[],
            net_utility_score=0.70,
            feasibility_score=0.95,
            recommendation_strength=0.75
        )
    
    def _create_compliance_focused_scenario(self, current_metrics: Dict) -> TradeOffScenario:
        return TradeOffScenario(
            scenario_name="Regulatory Compliance",
            primary_objective=Objective.COMPLIANCE,
            sacrificed_metrics=[],
            gained_metrics=[],
            net_utility_score=0.85,
            feasibility_score=0.95,
            recommendation_strength=0.90
        )
    
    def _create_safety_focused_scenario(self, current_metrics: Dict) -> TradeOffScenario:
        return TradeOffScenario(
            scenario_name="Safety-First Approach",
            primary_objective=Objective.INTERPRETABILITY,
            sacrificed_metrics=[],
            gained_metrics=[],
            net_utility_score=0.80,
            feasibility_score=0.95,
            recommendation_strength=0.85
        )
    
    def _calculate_net_utility(self, gained_metrics: List[TradeOffMetric], sacrificed_metrics: List[TradeOffMetric]) -> float:
        total_gain = sum(m.importance_weight * (m.achieved_value / m.theoretical_max) for m in gained_metrics)
        total_sacrifice = sum(m.importance_weight * (m.sacrifice_percent / 100) for m in sacrificed_metrics)
        
        return max(0, total_gain - total_sacrifice)
    
    def _calculate_trade_off_efficiency(self,
                                      primary_objective: Objective,
                                      normalized_metrics: Dict[str, float],
                                      sacrifice_analysis: Dict[str, TradeOffMetric]) -> float:
        
        # Calculate primary objective achievement
        objective_weights = self.objective_weights.get(primary_objective, {})
        primary_achievement = sum(
            weight * normalized_metrics.get(metric, 0)
            for metric, weight in objective_weights.items()
            if weight >= 0.3 and metric in normalized_metrics
        )
        
        # Calculate sacrifice cost
        total_sacrifice_cost = sum(
            metric.importance_weight * (metric.sacrifice_percent / 100)
            for metric in sacrifice_analysis.values()
        )
        
        # Efficiency = Achievement / (1 + Sacrifice Cost)
        efficiency = primary_achievement / (1 + total_sacrifice_cost)
        return min(1.0, efficiency)
    
    def _generate_decision_rationale(self,
                                   primary_objective: Objective,
                                   sacrifice_analysis: Dict[str, TradeOffMetric],
                                   scenarios: List[TradeOffScenario]) -> str:
        
        total_sacrifice = sum(m.sacrifice_percent for m in sacrifice_analysis.values())
        
        if total_sacrifice < 10:
            return f"Achieved {primary_objective.value} optimization with minimal trade-offs ({total_sacrifice:.1f}% total sacrifice)."
        elif total_sacrifice < 25:
            return f"Balanced {primary_objective.value} optimization with acceptable trade-offs ({total_sacrifice:.1f}% total sacrifice)."
        else:
            return f"Aggressive {primary_objective.value} optimization with significant trade-offs ({total_sacrifice:.1f}% total sacrifice). Consider alternative approaches."
    
    def _generate_optimization_suggestions(self,
                                         sacrifice_analysis: Dict[str, TradeOffMetric],
                                         scenarios: List[TradeOffScenario]) -> List[str]:
        
        suggestions = []
        
        # Analyze worst sacrifices
        worst_sacrifices = sorted(sacrifice_analysis.values(), key=lambda x: x.sacrifice_percent, reverse=True)
        
        if worst_sacrifices:
            worst = worst_sacrifices[0]
            if worst.sacrifice_percent > 30:
                suggestions.append(f"High sacrifice in {worst.name} ({worst.sacrifice_percent:.1f}%). Consider hybrid approaches.")
        
        # Scenario-based suggestions
        best_scenario = max(scenarios, key=lambda x: x.recommendation_strength)
        if best_scenario.recommendation_strength > 0.8:
            suggestions.append(f"Consider '{best_scenario.scenario_name}' approach for better balance.")
        
        # General optimization suggestions
        if len(sacrifice_analysis) > 3:
            suggestions.append("Multiple trade-offs detected. Consider multi-objective optimization.")
        
        return suggestions[:3]
    
    def _perform_sensitivity_analysis(self,
                                    primary_objective: Objective,
                                    metrics: Dict[str, float],
                                    sacrifice_analysis: Dict[str, TradeOffMetric]) -> Dict[str, float]:
        
        # Analyze how sensitive trade-offs are to small changes
        sensitivity = {}
        
        for metric_name, metric in sacrifice_analysis.items():
            # Calculate how much primary objective would change with 10% improvement in sacrificed metric
            baseline_sacrifice = metric.sacrifice_percent
            improved_sacrifice = max(0, baseline_sacrifice - 10)
            
            sensitivity_score = (baseline_sacrifice - improved_sacrifice) / 10 if baseline_sacrifice > 0 else 0
            sensitivity[metric_name] = sensitivity_score
        
        return sensitivity