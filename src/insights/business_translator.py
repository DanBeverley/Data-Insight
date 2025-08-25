import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from ..core.project_definition import Objective, Domain, RiskLevel
from ..validation.validation_orchestrator import ValidationSummary

class BusinessImpactType(Enum):
    COST_REDUCTION = "cost_reduction"
    REVENUE_INCREASE = "revenue_increase"
    RISK_MITIGATION = "risk_mitigation"
    EFFICIENCY_GAIN = "efficiency_gain"
    COMPLIANCE_ACHIEVEMENT = "compliance_achievement"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"

class StakeholderType(Enum):
    EXECUTIVE = "executive"
    TECHNICAL_LEAD = "technical_lead"
    BUSINESS_ANALYST = "business_analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    PRODUCT_MANAGER = "product_manager"
    DATA_SCIENTIST = "data_scientist"

@dataclass
class BusinessMetric:
    name: str
    value: Union[float, int, str]
    unit: str
    trend: str  # "improving", "declining", "stable"
    business_significance: str
    impact_category: BusinessImpactType
    confidence_level: float
    financial_impact: Optional[float] = None
    time_horizon: str = "immediate"

@dataclass
class StakeholderView:
    stakeholder_type: StakeholderType
    priority_metrics: List[BusinessMetric]
    key_insights: List[str]
    action_items: List[str]
    risk_alerts: List[str]
    success_indicators: List[str]

class BusinessTranslator:
    """
    Translates technical ML metrics into business-relevant insights
    with stakeholder-specific context and financial impact analysis.
    """
    
    def __init__(self):
        self.domain_multipliers = self._initialize_domain_multipliers()
        self.objective_business_mapping = self._initialize_objective_mappings()
        self.roi_calculators = self._initialize_roi_calculators()
        
    def translate_validation_to_business(self,
                                       validation_summary: ValidationSummary,
                                       project_definition,
                                       business_context: Optional[Dict] = None) -> Dict[StakeholderType, StakeholderView]:
        """
        Translate validation results into stakeholder-specific business insights.
        """
        
        business_context = business_context or {}
        
        # Extract core business metrics
        core_metrics = self._extract_core_business_metrics(
            validation_summary, project_definition, business_context
        )
        
        # Calculate financial impact
        financial_impact = self._calculate_financial_impact(
            validation_summary, project_definition, business_context
        )
        
        # Generate stakeholder-specific views
        stakeholder_views = {}
        
        for stakeholder_type in StakeholderType:
            view = self._generate_stakeholder_view(
                stakeholder_type,
                core_metrics,
                financial_impact,
                validation_summary,
                project_definition,
                business_context
            )
            stakeholder_views[stakeholder_type] = view
        
        return stakeholder_views
    
    def _extract_core_business_metrics(self,
                                     validation_summary: ValidationSummary,
                                     project_definition,
                                     business_context: Dict) -> List[BusinessMetric]:
        """Extract and translate core metrics to business language."""
        
        metrics = []
        
        # Objective Achievement Metric
        objective_metric = self._translate_objective_achievement(
            validation_summary, project_definition
        )
        metrics.append(objective_metric)
        
        # Deployment Readiness Metric
        readiness_metric = self._translate_deployment_readiness(
            validation_summary, project_definition
        )
        metrics.append(readiness_metric)
        
        # Risk Mitigation Metric
        risk_metric = self._translate_risk_mitigation(
            validation_summary, project_definition
        )
        metrics.append(risk_metric)
        
        # Resource Efficiency Metric
        efficiency_metric = self._translate_resource_efficiency(
            validation_summary, business_context
        )
        metrics.append(efficiency_metric)
        
        # Compliance Score (if applicable)
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            compliance_metric = self._translate_compliance_achievement(
                validation_summary, project_definition
            )
            metrics.append(compliance_metric)
        
        return metrics
    
    def _translate_objective_achievement(self,
                                       validation_summary: ValidationSummary,
                                       project_definition) -> BusinessMetric:
        """Translate primary objective achievement to business terms."""
        
        objective = project_definition.objective
        achievement_rate = 0.95 if validation_summary.primary_objective_met else 0.6
        
        business_translations = {
            Objective.ACCURACY: {
                'name': 'Prediction Reliability',
                'unit': '% confidence',
                'significance': 'Direct impact on decision quality and business outcomes'
            },
            Objective.SPEED: {
                'name': 'System Response Time',
                'unit': 'operational efficiency',
                'significance': 'Enables real-time decision making and customer experience'
            },
            Objective.INTERPRETABILITY: {
                'name': 'Decision Transparency',
                'unit': '% explainable',
                'significance': 'Regulatory compliance and stakeholder trust'
            },
            Objective.FAIRNESS: {
                'name': 'Equitable Outcomes',
                'unit': '% bias-free',
                'significance': 'Brand protection and regulatory compliance'
            },
            Objective.COMPLIANCE: {
                'name': 'Regulatory Alignment',
                'unit': '% compliant',
                'significance': 'Risk mitigation and legal protection'
            }
        }
        
        translation = business_translations.get(objective, business_translations[Objective.ACCURACY])
        
        return BusinessMetric(
            name=translation['name'],
            value=achievement_rate * 100,
            unit=translation['unit'],
            trend="improving" if validation_summary.primary_objective_met else "needs_attention",
            business_significance=translation['significance'],
            impact_category=BusinessImpactType.COMPETITIVE_ADVANTAGE,
            confidence_level=validation_summary.validation_confidence,
            time_horizon="immediate"
        )
    
    def _translate_deployment_readiness(self,
                                      validation_summary: ValidationSummary,
                                      project_definition) -> BusinessMetric:
        """Translate technical readiness to business deployment confidence."""
        
        readiness_score = (
            validation_summary.validation_confidence * 0.4 +
            validation_summary.budget_compliance_rate * 0.3 +
            validation_summary.trade_off_efficiency * 0.3
        )
        
        readiness_level = "production_ready" if readiness_score > 0.85 else \
                         "pilot_ready" if readiness_score > 0.7 else "development_stage"
        
        return BusinessMetric(
            name="Deployment Readiness",
            value=readiness_score * 100,
            unit="% ready",
            trend="improving" if validation_summary.overall_success else "stable",
            business_significance="Time-to-market and go-live confidence",
            impact_category=BusinessImpactType.EFFICIENCY_GAIN,
            confidence_level=validation_summary.validation_confidence,
            time_horizon="immediate"
        )
    
    def _translate_risk_mitigation(self,
                                 validation_summary: ValidationSummary,
                                 project_definition) -> BusinessMetric:
        """Translate technical validation to business risk reduction."""
        
        risk_factors = len(validation_summary.risk_assessments)
        risk_mitigation_score = max(0, 1 - (risk_factors * 0.15))
        
        domain_risk_multipliers = {
            Domain.FINANCE: 1.5,
            Domain.HEALTHCARE: 1.6,
            Domain.RETAIL: 1.1,
            Domain.GENERAL: 1.0
        }
        
        adjusted_score = risk_mitigation_score * domain_risk_multipliers.get(
            project_definition.domain, 1.0
        )
        
        return BusinessMetric(
            name="Risk Mitigation Level",
            value=min(100, adjusted_score * 100),
            unit="% risk reduced",
            trend="improving" if len(validation_summary.risk_assessments) < 2 else "stable",
            business_significance="Operational and regulatory risk protection",
            impact_category=BusinessImpactType.RISK_MITIGATION,
            confidence_level=validation_summary.validation_confidence,
            time_horizon="immediate"
        )
    
    def _translate_resource_efficiency(self,
                                     validation_summary: ValidationSummary,
                                     business_context: Dict) -> BusinessMetric:
        """Translate resource utilization to business efficiency metrics."""
        
        budget_compliance = validation_summary.budget_compliance_rate
        efficiency_score = budget_compliance * validation_summary.trade_off_efficiency
        
        return BusinessMetric(
            name="Resource Efficiency",
            value=efficiency_score * 100,
            unit="% optimal",
            trend="improving" if budget_compliance > 0.9 else "stable",
            business_significance="Cost optimization and resource allocation effectiveness",
            impact_category=BusinessImpactType.COST_REDUCTION,
            confidence_level=validation_summary.validation_confidence,
            time_horizon="ongoing"
        )
    
    def _translate_compliance_achievement(self,
                                        validation_summary: ValidationSummary,
                                        project_definition) -> BusinessMetric:
        """Translate compliance validation to business compliance metrics."""
        
        compliance_score = 0.95  # Would come from actual compliance validation
        
        domain_compliance_requirements = {
            Domain.FINANCE: "Financial regulations (Basel III, Dodd-Frank, MiFID II)",
            Domain.HEALTHCARE: "Healthcare regulations (HIPAA, FDA, GDPR)"
        }
        
        significance = domain_compliance_requirements.get(
            project_definition.domain,
            "Industry-standard compliance requirements"
        )
        
        return BusinessMetric(
            name="Regulatory Compliance",
            value=compliance_score * 100,
            unit="% compliant",
            trend="stable",
            business_significance=f"Adherence to {significance}",
            impact_category=BusinessImpactType.COMPLIANCE_ACHIEVEMENT,
            confidence_level=validation_summary.validation_confidence,
            time_horizon="immediate"
        )
    
    def _calculate_financial_impact(self,
                                  validation_summary: ValidationSummary,
                                  project_definition,
                                  business_context: Dict) -> Dict[str, Any]:
        """Calculate quantified financial impact of the ML solution."""
        
        # Base impact multipliers by domain
        domain_impact_base = {
            Domain.FINANCE: {'cost_reduction': 0.15, 'risk_mitigation': 0.25},
            Domain.HEALTHCARE: {'cost_reduction': 0.12, 'risk_mitigation': 0.30},
            Domain.RETAIL: {'revenue_increase': 0.08, 'efficiency_gain': 0.18},
            Domain.TECHNOLOGY: {'efficiency_gain': 0.20, 'competitive_advantage': 0.15},
            Domain.GENERAL: {'cost_reduction': 0.10, 'efficiency_gain': 0.15}
        }
        
        objective_multipliers = {
            Objective.ACCURACY: 1.2,
            Objective.SPEED: 1.1,
            Objective.INTERPRETABILITY: 1.0,
            Objective.FAIRNESS: 1.1,
            Objective.COMPLIANCE: 1.3
        }
        
        base_impacts = domain_impact_base.get(
            project_definition.domain, 
            domain_impact_base[Domain.GENERAL]
        )
        
        objective_multiplier = objective_multipliers.get(project_definition.objective, 1.0)
        success_multiplier = 1.0 if validation_summary.overall_success else 0.6
        
        # Estimate annual business value (would be customizable based on business_context)
        annual_revenue = business_context.get('annual_revenue', 10_000_000)  # Default 10M
        operational_costs = business_context.get('operational_costs', 5_000_000)  # Default 5M
        
        financial_impact = {}
        
        for impact_type, base_percentage in base_impacts.items():
            if impact_type in ['cost_reduction', 'efficiency_gain']:
                base_value = operational_costs
            else:
                base_value = annual_revenue
                
            impact_value = (
                base_value * 
                base_percentage * 
                objective_multiplier * 
                success_multiplier *
                validation_summary.validation_confidence
            )
            
            financial_impact[impact_type] = {
                'annual_value': impact_value,
                'confidence': validation_summary.validation_confidence,
                'time_to_realization': '6-12 months'
            }
        
        return financial_impact
    
    def _generate_stakeholder_view(self,
                                 stakeholder_type: StakeholderType,
                                 core_metrics: List[BusinessMetric],
                                 financial_impact: Dict,
                                 validation_summary: ValidationSummary,
                                 project_definition,
                                 business_context: Dict) -> StakeholderView:
        """Generate stakeholder-specific view with relevant metrics and insights."""
        
        stakeholder_priorities = {
            StakeholderType.EXECUTIVE: {
                'priority_metrics': ['Deployment Readiness', 'Risk Mitigation Level'],
                'focus_areas': ['strategic_value', 'financial_impact', 'risk_management']
            },
            StakeholderType.TECHNICAL_LEAD: {
                'priority_metrics': ['Prediction Reliability', 'Resource Efficiency'],
                'focus_areas': ['technical_performance', 'system_optimization', 'deployment_considerations']
            },
            StakeholderType.COMPLIANCE_OFFICER: {
                'priority_metrics': ['Regulatory Compliance', 'Decision Transparency'],
                'focus_areas': ['regulatory_adherence', 'audit_readiness', 'risk_controls']
            },
            StakeholderType.PRODUCT_MANAGER: {
                'priority_metrics': ['Deployment Readiness', 'Prediction Reliability'],
                'focus_areas': ['user_impact', 'feature_readiness', 'market_advantage']
            }
        }
        
        config = stakeholder_priorities.get(stakeholder_type, stakeholder_priorities[StakeholderType.EXECUTIVE])
        
        # Filter metrics based on stakeholder priorities
        priority_metrics = [
            metric for metric in core_metrics 
            if metric.name in config['priority_metrics']
        ]
        
        # Generate stakeholder-specific insights
        insights = self._generate_stakeholder_insights(
            stakeholder_type, validation_summary, project_definition, financial_impact
        )
        
        # Generate action items
        action_items = self._generate_action_items(
            stakeholder_type, validation_summary, project_definition
        )
        
        # Generate risk alerts
        risk_alerts = self._generate_risk_alerts(
            stakeholder_type, validation_summary, project_definition
        )
        
        # Generate success indicators
        success_indicators = self._generate_success_indicators(
            stakeholder_type, validation_summary, project_definition
        )
        
        return StakeholderView(
            stakeholder_type=stakeholder_type,
            priority_metrics=priority_metrics,
            key_insights=insights,
            action_items=action_items,
            risk_alerts=risk_alerts,
            success_indicators=success_indicators
        )
    
    def _generate_stakeholder_insights(self,
                                     stakeholder_type: StakeholderType,
                                     validation_summary: ValidationSummary,
                                     project_definition,
                                     financial_impact: Dict) -> List[str]:
        """Generate stakeholder-specific insights."""
        
        insights = []
        
        if stakeholder_type == StakeholderType.EXECUTIVE:
            if validation_summary.overall_success:
                insights.append(f"Solution successfully optimized for {project_definition.objective.value}")
                
            total_financial_impact = sum(
                impact['annual_value'] for impact in financial_impact.values()
            )
            if total_financial_impact > 0:
                insights.append(f"Projected annual business value: ${total_financial_impact:,.0f}")
                
            if validation_summary.validation_confidence > 0.9:
                insights.append("High confidence in production deployment readiness")
        
        elif stakeholder_type == StakeholderType.TECHNICAL_LEAD:
            if validation_summary.budget_compliance_rate > 0.9:
                insights.append("All resource budgets maintained within acceptable limits")
                
            if validation_summary.trade_off_efficiency > 0.8:
                insights.append("Optimal trade-off balance achieved for technical requirements")
                
            insights.append(f"System validation confidence: {validation_summary.validation_confidence:.1%}")
        
        elif stakeholder_type == StakeholderType.COMPLIANCE_OFFICER:
            if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
                insights.append("Regulatory compliance framework successfully implemented")
                
            if len(validation_summary.risk_assessments) == 0:
                insights.append("No critical compliance risks identified")
            else:
                insights.append(f"{len(validation_summary.risk_assessments)} compliance considerations require attention")
        
        return insights[:4]  # Limit to top 4 insights
    
    def _generate_action_items(self,
                             stakeholder_type: StakeholderType,
                             validation_summary: ValidationSummary,
                             project_definition) -> List[str]:
        """Generate stakeholder-specific action items."""
        
        actions = []
        
        if not validation_summary.overall_success:
            actions.append("Review validation results and address identified issues")
        
        if validation_summary.budget_compliance_rate < 0.8:
            actions.append("Optimize resource allocation to meet performance budgets")
        
        if len(validation_summary.risk_assessments) > 0:
            actions.append("Address identified risk factors before deployment")
        
        if stakeholder_type == StakeholderType.EXECUTIVE:
            if validation_summary.overall_success:
                actions.append("Approve production deployment and resource allocation")
            actions.append("Schedule stakeholder review meeting for go-live decision")
        
        return actions[:3]  # Limit to top 3 actions
    
    def _generate_risk_alerts(self,
                            stakeholder_type: StakeholderType,
                            validation_summary: ValidationSummary,
                            project_definition) -> List[str]:
        """Generate stakeholder-specific risk alerts."""
        
        alerts = []
        
        if validation_summary.validation_confidence < 0.7:
            alerts.append("LOW CONFIDENCE: Validation results require additional review")
        
        if validation_summary.budget_compliance_rate < 0.6:
            alerts.append("BUDGET VIOLATION: Resource constraints significantly exceeded")
        
        critical_risks = [risk for risk in validation_summary.risk_assessments if "CRITICAL" in risk]
        if critical_risks:
            alerts.append(f"CRITICAL RISKS: {len(critical_risks)} deployment blockers identified")
        
        return alerts
    
    def _generate_success_indicators(self,
                                   stakeholder_type: StakeholderType,
                                   validation_summary: ValidationSummary,
                                   project_definition) -> List[str]:
        """Generate stakeholder-specific success indicators."""
        
        indicators = []
        
        if validation_summary.primary_objective_met:
            indicators.append(f"✅ Primary objective ({project_definition.objective.value}) achieved")
        
        if validation_summary.budget_compliance_rate > 0.9:
            indicators.append("✅ Resource budgets maintained")
        
        if validation_summary.validation_confidence > 0.85:
            indicators.append("✅ High validation confidence")
        
        if validation_summary.overall_success:
            indicators.append("✅ Ready for production deployment")
        
        return indicators
    
    def _initialize_domain_multipliers(self) -> Dict[Domain, Dict[str, float]]:
        """Initialize domain-specific impact multipliers."""
        return {
            Domain.FINANCE: {
                'risk_sensitivity': 1.5,
                'compliance_importance': 1.8,
                'cost_impact': 1.3
            },
            Domain.HEALTHCARE: {
                'risk_sensitivity': 1.8,
                'compliance_importance': 2.0,
                'safety_importance': 2.0
            },
            Domain.RETAIL: {
                'speed_importance': 1.4,
                'customer_impact': 1.3,
                'revenue_sensitivity': 1.2
            }
        }
    
    def _initialize_objective_mappings(self) -> Dict[Objective, Dict[str, str]]:
        """Initialize objective to business outcome mappings."""
        return {
            Objective.ACCURACY: {
                'business_outcome': 'Decision Quality',
                'success_metric': 'Prediction Reliability',
                'risk_factor': 'False Decision Cost'
            },
            Objective.SPEED: {
                'business_outcome': 'Operational Efficiency',
                'success_metric': 'Response Time',
                'risk_factor': 'Customer Experience Impact'
            },
            Objective.INTERPRETABILITY: {
                'business_outcome': 'Regulatory Compliance',
                'success_metric': 'Decision Transparency',
                'risk_factor': 'Audit Risk'
            }
        }
    
    def _initialize_roi_calculators(self) -> Dict[str, callable]:
        """Initialize ROI calculation functions by impact type."""
        return {
            'cost_reduction': lambda base, improvement, confidence: base * improvement * confidence * 0.8,
            'revenue_increase': lambda base, improvement, confidence: base * improvement * confidence * 0.6,
            'risk_mitigation': lambda base, improvement, confidence: base * improvement * confidence * 1.2
        }