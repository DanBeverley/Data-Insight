import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .objective_benchmarker import ObjectiveBenchmarker, BenchmarkResult
from .performance_budget_manager import PerformanceBudgetManager, BudgetReport
from .trade_off_analyzer import TradeOffAnalyzer, TradeOffReport
from ..core.project_definition import ProjectDefinition, Objective, Domain

@dataclass
class ValidationSummary:
    validation_timestamp: datetime
    overall_success: bool
    primary_objective_met: bool
    budget_compliance_rate: float
    trade_off_efficiency: float
    
    benchmark_results: Dict[Objective, BenchmarkResult]
    budget_report: BudgetReport
    trade_off_report: TradeOffReport
    
    strategic_recommendations: List[str]
    technical_optimizations: List[str]
    risk_assessments: List[str]
    
    validation_confidence: float
    decision_support_score: float

class ValidationOrchestrator:
    """
    Production-grade validation orchestrator that provides comprehensive
    objective-driven validation with business context awareness.
    """
    
    def __init__(self):
        self.objective_benchmarker = ObjectiveBenchmarker()
        self.budget_manager = PerformanceBudgetManager()
        self.trade_off_analyzer = TradeOffAnalyzer()
        self.validation_history: List[ValidationSummary] = []
        
    def execute_comprehensive_validation(self,
                                       project_definition: ProjectDefinition,
                                       achieved_metrics: Dict[str, Any],
                                       model_metadata: Dict[str, Any],
                                       dataset_characteristics: Optional[Dict] = None,
                                       session_id: Optional[str] = None) -> ValidationSummary:
        """
        Execute comprehensive validation against business objectives with
        benchmarking, budget compliance, and trade-off analysis.
        """
        
        validation_start = datetime.now()
        
        # Set up performance budgets from constraints
        self.budget_manager.set_budgets_from_constraints(
            self._extract_constraints_dict(project_definition.technical_constraints)
        )
        
        # Execute objective benchmarking
        benchmark_results = self._execute_objective_benchmarking(
            project_definition, achieved_metrics, model_metadata, dataset_characteristics
        )
        
        # Execute budget compliance validation
        budget_report = self.budget_manager.validate_budget_compliance(
            achieved_metrics, model_metadata, session_id
        )
        
        # Execute trade-off analysis
        trade_off_report = self.trade_off_analyzer.analyze_trade_offs(
            project_definition.objective,
            achieved_metrics,
            model_metadata,
            project_definition.domain,
            self._extract_constraints_dict(project_definition.technical_constraints)
        )
        
        # Synthesize validation results
        validation_summary = self._synthesize_validation_results(
            validation_start,
            project_definition,
            benchmark_results,
            budget_report,
            trade_off_report
        )
        
        # Store validation history
        self.validation_history.append(validation_summary)
        
        return validation_summary
    
    def _execute_objective_benchmarking(self,
                                      project_definition: ProjectDefinition,
                                      achieved_metrics: Dict[str, Any],
                                      model_metadata: Dict[str, Any],
                                      dataset_characteristics: Optional[Dict]) -> Dict[Objective, BenchmarkResult]:
        """Execute benchmarking for primary and secondary objectives."""
        
        benchmark_results = {}
        
        # Primary objective benchmarking
        primary_result = self.objective_benchmarker.benchmark_objective_performance(
            project_definition.objective,
            project_definition.domain,
            achieved_metrics,
            model_metadata,
            dataset_characteristics
        )
        benchmark_results[project_definition.objective] = primary_result
        
        # Secondary objectives benchmarking (derived from constraints)
        secondary_objectives = self._identify_secondary_objectives(project_definition)
        
        for secondary_objective in secondary_objectives:
            try:
                secondary_result = self.objective_benchmarker.benchmark_objective_performance(
                    secondary_objective,
                    project_definition.domain,
                    achieved_metrics,
                    model_metadata,
                    dataset_characteristics
                )
                benchmark_results[secondary_objective] = secondary_result
            except Exception as e:
                # Log but don't fail entire validation for secondary objectives
                continue
        
        return benchmark_results
    
    def _identify_secondary_objectives(self, project_definition: ProjectDefinition) -> List[Objective]:
        """Identify secondary objectives based on constraints and domain."""
        
        secondary_objectives = []
        tech_constraints = project_definition.technical_constraints
        reg_constraints = project_definition.regulatory_constraints
        
        # Derive secondary objectives from regulatory constraints
        if reg_constraints.interpretability_required or len(reg_constraints.compliance_rules) > 0:
            secondary_objectives.append(Objective.INTERPRETABILITY)
            
        # Derive from technical constraints
        if tech_constraints.max_latency_ms is not None:
            secondary_objectives.append(Objective.SPEED)
            
        if len(reg_constraints.protected_attributes) > 0:
            secondary_objectives.append(Objective.FAIRNESS)
            
        if tech_constraints.min_accuracy is not None:
            secondary_objectives.append(Objective.ACCURACY)
            
        # Use project's actual secondary objectives if defined
        if hasattr(project_definition, 'secondary_objectives'):
            secondary_objectives.extend(project_definition.secondary_objectives)
        
        # Domain-specific secondary objectives
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            if Objective.COMPLIANCE not in secondary_objectives:
                secondary_objectives.append(Objective.COMPLIANCE)
        
        # Remove primary objective from secondary list
        if project_definition.objective in secondary_objectives:
            secondary_objectives.remove(project_definition.objective)
        
        return secondary_objectives
    
    def _extract_constraints_dict(self, constraints) -> Dict[str, Any]:
        """Extract constraints into dictionary format for budget manager."""
        
        return {
            'max_latency_ms': constraints.max_latency_ms,
            'max_training_hours': constraints.max_training_hours,
            'max_memory_gb': constraints.max_memory_gb,
            'max_storage_gb': constraints.max_storage_gb,
            'min_accuracy': constraints.min_accuracy,
            'min_precision': constraints.min_precision,
            'min_recall': constraints.min_recall,
            'max_cost_per_prediction': constraints.max_cost_per_prediction,
            'required_uptime': constraints.required_uptime
        }
    
    def _synthesize_validation_results(self,
                                     validation_start: datetime,
                                     project_definition: ProjectDefinition,
                                     benchmark_results: Dict[Objective, BenchmarkResult],
                                     budget_report: BudgetReport,
                                     trade_off_report: TradeOffReport) -> ValidationSummary:
        """Synthesize all validation results into comprehensive summary."""
        
        # Determine primary objective success
        primary_benchmark = benchmark_results.get(project_definition.objective)
        primary_objective_met = primary_benchmark.meets_benchmark if primary_benchmark else False
        
        # Calculate overall success
        budget_compliance_rate = budget_report.overall_compliance
        critical_violations = len([v for v in budget_report.violations if v.severity.value == 'critical'])
        
        overall_success = (
            primary_objective_met and
            budget_compliance_rate >= 0.8 and
            critical_violations == 0 and
            trade_off_report.overall_trade_off_efficiency >= 0.7
        )
        
        # Generate strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            project_definition, benchmark_results, budget_report, trade_off_report
        )
        
        # Generate technical optimizations
        technical_optimizations = self._generate_technical_optimizations(
            benchmark_results, budget_report, trade_off_report
        )
        
        # Generate risk assessments
        risk_assessments = self._generate_risk_assessments(
            project_definition, benchmark_results, budget_report, trade_off_report
        )
        
        # Calculate validation confidence
        validation_confidence = self._calculate_validation_confidence(
            benchmark_results, budget_report, trade_off_report
        )
        
        # Calculate decision support score
        decision_support_score = self._calculate_decision_support_score(
            project_definition, benchmark_results, budget_report, trade_off_report
        )
        
        return ValidationSummary(
            validation_timestamp=validation_start,
            overall_success=overall_success,
            primary_objective_met=primary_objective_met,
            budget_compliance_rate=budget_compliance_rate,
            trade_off_efficiency=trade_off_report.overall_trade_off_efficiency,
            benchmark_results=benchmark_results,
            budget_report=budget_report,
            trade_off_report=trade_off_report,
            strategic_recommendations=strategic_recommendations,
            technical_optimizations=technical_optimizations,
            risk_assessments=risk_assessments,
            validation_confidence=validation_confidence,
            decision_support_score=decision_support_score
        )
    
    def _generate_strategic_recommendations(self,
                                          project_definition: ProjectDefinition,
                                          benchmark_results: Dict[Objective, BenchmarkResult],
                                          budget_report: BudgetReport,
                                          trade_off_report: TradeOffReport) -> List[str]:
        """Generate high-level strategic recommendations for business stakeholders."""
        
        recommendations = []
        
        primary_benchmark = benchmark_results.get(project_definition.objective)
        
        if primary_benchmark and primary_benchmark.meets_benchmark:
            recommendations.append(
                f"{project_definition.objective.value.title()} objective achieved. "
                f"System ready for {project_definition.domain.value} deployment."
            )
        else:
            recommendations.append(
                f"{project_definition.objective.value.title()} objective not met. "
                f"Consider adjusting strategy or constraints."
            )
        
        # Budget-based strategic recommendations
        if budget_report.overall_compliance < 0.8:
            critical_violations = [v for v in budget_report.violations if v.severity.value == 'critical']
            if critical_violations:
                recommendations.append(
                    f"Critical resource violations detected. Business case may require revision."
                )
        
        # Trade-off strategic guidance
        if trade_off_report.overall_trade_off_efficiency < 0.6:
            recommendations.append(
                "Significant trade-offs detected. Consider multi-phase implementation strategy."
            )
        
        # Domain-specific strategic recommendations
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            compliance_result = benchmark_results.get(Objective.COMPLIANCE)
            if compliance_result and not compliance_result.meets_benchmark:
                recommendations.append(
                    "Regulatory compliance concerns identified. Legal review recommended before deployment."
                )
        
        return recommendations[:4]  # Limit to top 4 strategic recommendations
    
    def _generate_technical_optimizations(self,
                                        benchmark_results: Dict[Objective, BenchmarkResult],
                                        budget_report: BudgetReport,
                                        trade_off_report: TradeOffReport) -> List[str]:
        """Generate technical optimization recommendations for engineering teams."""
        
        optimizations = []
        
        # Benchmark-based optimizations
        for objective, result in benchmark_results.items():
            if not result.meets_benchmark and result.recommendations:
                optimizations.extend(result.recommendations[:2])
        
        # Budget violation optimizations
        for violation in budget_report.violations:
            if violation.mitigation_suggestions:
                optimizations.extend(violation.mitigation_suggestions[:1])
        
        # Trade-off optimizations
        if trade_off_report.optimization_suggestions:
            optimizations.extend(trade_off_report.optimization_suggestions)
        
        return list(dict.fromkeys(optimizations))[:5]  # Remove duplicates, limit to 5
    
    def _generate_risk_assessments(self,
                                 project_definition: ProjectDefinition,
                                 benchmark_results: Dict[Objective, BenchmarkResult],
                                 budget_report: BudgetReport,
                                 trade_off_report: TradeOffReport) -> List[str]:
        """Generate risk assessments for deployment readiness."""
        
        risks = []
        
        # Performance risks
        primary_benchmark = benchmark_results.get(project_definition.objective)
        if primary_benchmark and primary_benchmark.performance_ratio < 0.8:
            risks.append(f"HIGH: Primary objective significantly below benchmark ({primary_benchmark.performance_ratio:.2f})")
        
        # Resource risks
        critical_violations = [v for v in budget_report.violations if v.severity.value == 'critical']
        if critical_violations:
            risks.append(f"CRITICAL: {len(critical_violations)} resource constraint violations")
        
        # Trade-off risks
        severe_sacrifices = [
            metric for metric in trade_off_report.sacrifice_analysis.values()
            if metric.sacrifice_percent > 30
        ]
        if severe_sacrifices:
            risks.append(f"MEDIUM: Severe performance sacrifices in {len(severe_sacrifices)} areas")
        
        # Domain-specific risks
        if project_definition.domain == Domain.FINANCE:
            interpretability_result = benchmark_results.get(Objective.INTERPRETABILITY)
            if interpretability_result and not interpretability_result.meets_benchmark:
                risks.append("HIGH: Regulatory interpretability requirements not met")
        
        if project_definition.domain == Domain.HEALTHCARE:
            fairness_result = benchmark_results.get(Objective.FAIRNESS)
            if fairness_result and not fairness_result.meets_benchmark:
                risks.append("CRITICAL: Patient safety fairness requirements not met")
        
        return risks
    
    def _calculate_validation_confidence(self,
                                       benchmark_results: Dict[Objective, BenchmarkResult],
                                       budget_report: BudgetReport,
                                       trade_off_report: TradeOffReport) -> float:
        """Calculate overall confidence in validation results."""
        
        confidence_factors = []
        
        # Benchmark confidence
        if benchmark_results:
            avg_benchmark_confidence = sum(
                result.confidence_interval[1] - result.confidence_interval[0]
                for result in benchmark_results.values()
            ) / len(benchmark_results)
            confidence_factors.append(1 - avg_benchmark_confidence)  # Smaller interval = higher confidence
        
        # Budget validation confidence
        budget_confidence = 0.9 if budget_report.overall_compliance > 0.8 else 0.6
        confidence_factors.append(budget_confidence)
        
        # Trade-off analysis confidence
        trade_off_confidence = min(1.0, trade_off_report.overall_trade_off_efficiency + 0.2)
        confidence_factors.append(trade_off_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_decision_support_score(self,
                                        project_definition: ProjectDefinition,
                                        benchmark_results: Dict[Objective, BenchmarkResult],
                                        budget_report: BudgetReport,
                                        trade_off_report: TradeOffReport) -> float:
        """Calculate how well the validation supports business decision-making."""
        
        support_factors = []
        
        # Objective clarity (how clearly did we measure the objective)
        primary_benchmark = benchmark_results.get(project_definition.objective)
        if primary_benchmark:
            objective_clarity = 1.0 if primary_benchmark.meets_benchmark else 0.7
            support_factors.append(objective_clarity)
        
        # Constraint validation completeness
        constraint_coverage = min(1.0, len(budget_report.passed_budgets + budget_report.violations) / 4)
        support_factors.append(constraint_coverage)
        
        # Trade-off transparency
        trade_off_transparency = 1.0 if len(trade_off_report.alternative_scenarios) > 2 else 0.7
        support_factors.append(trade_off_transparency)
        
        # Risk assessment comprehensiveness
        risk_comprehensiveness = 0.9  # Assume good risk coverage for now
        support_factors.append(risk_comprehensiveness)
        
        return sum(support_factors) / len(support_factors) if support_factors else 0.5
    
    def get_validation_history_summary(self, limit: int = 10) -> Dict[str, Any]:
        """Get summary of recent validation history for trend analysis."""
        
        recent_validations = self.validation_history[-limit:]
        
        if not recent_validations:
            return {"message": "No validation history available"}
        
        success_rate = sum(v.overall_success for v in recent_validations) / len(recent_validations)
        avg_confidence = sum(v.validation_confidence for v in recent_validations) / len(recent_validations)
        avg_decision_support = sum(v.decision_support_score for v in recent_validations) / len(recent_validations)
        
        return {
            "total_validations": len(recent_validations),
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_decision_support": avg_decision_support,
            "latest_validation": recent_validations[-1].validation_timestamp.isoformat(),
            "trend_analysis": self._analyze_validation_trends(recent_validations)
        }
    
    def _analyze_validation_trends(self, validations: List[ValidationSummary]) -> Dict[str, str]:
        """Analyze trends in validation results."""
        
        if len(validations) < 3:
            return {"trend": "insufficient_data"}
        
        recent_success_rate = sum(v.overall_success for v in validations[-3:]) / 3
        earlier_success_rate = sum(v.overall_success for v in validations[-6:-3]) / 3 if len(validations) >= 6 else recent_success_rate
        
        if recent_success_rate > earlier_success_rate + 0.2:
            return {"trend": "improving", "description": "Validation success rate improving"}
        elif recent_success_rate < earlier_success_rate - 0.2:
            return {"trend": "declining", "description": "Validation success rate declining"}
        else:
            return {"trend": "stable", "description": "Validation performance stable"}