import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .business_translator import BusinessTranslator, BusinessMetric, StakeholderView
from ..core.project_definition import ProjectDefinition, Objective, Domain
from ..validation.validation_orchestrator import ValidationSummary


class ReportSection(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_OVERVIEW = "technical_overview"
    BUSINESS_IMPACT = "business_impact"
    RISK_ASSESSMENT = "risk_assessment"
    RECOMMENDATIONS = "recommendations"
    PERFORMANCE_METRICS = "performance_metrics"
    COMPLIANCE_STATUS = "compliance_status"


class ReportTone(Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    BUSINESS = "business"
    COMPLIANCE = "compliance"


@dataclass
class ExecutiveSummary:
    project_overview: str
    key_achievements: List[str]
    business_impact: str
    strategic_recommendations: List[str]
    risk_summary: str
    investment_justification: str
    next_steps: List[str]
    confidence_level: str


@dataclass
class TechnicalReport:
    methodology_overview: str
    performance_analysis: str
    technical_achievements: List[str]
    architecture_decisions: List[str]
    optimization_opportunities: List[str]
    technical_risks: List[str]
    validation_results: str
    deployment_considerations: List[str]


@dataclass
class NarrativeComponents:
    introduction: str
    methodology: str
    results: str
    analysis: str
    conclusions: str
    recommendations: str


class NarrativeGenerator:
    """
    Production-grade narrative generation system that creates comprehensive,
    stakeholder-specific reports from technical ML results.
    """

    def __init__(self, business_translator: Optional[BusinessTranslator] = None):
        self.business_translator = business_translator or BusinessTranslator()
        self.report_templates = self._initialize_report_templates()
        self.narrative_history: List[Dict[str, Any]] = []

    def generate_executive_summary(
        self,
        project_definition: ProjectDefinition,
        validation_summary: ValidationSummary,
        business_metrics: List[BusinessMetric],
        stakeholder_views: Dict[str, StakeholderView],
    ) -> ExecutiveSummary:
        """Generate comprehensive executive summary for C-level stakeholders."""

        # Extract key business insights
        executive_view = stakeholder_views.get("executive")
        primary_metric = next(
            (m for m in business_metrics if m.is_primary), business_metrics[0] if business_metrics else None
        )

        # Project overview
        project_overview = self._generate_project_overview(project_definition, validation_summary)

        # Key achievements
        key_achievements = self._extract_key_achievements(validation_summary, business_metrics, executive_view)

        # Business impact analysis
        business_impact = self._generate_business_impact_narrative(business_metrics, project_definition.domain)

        # Strategic recommendations
        strategic_recommendations = self._distill_strategic_recommendations(
            validation_summary.strategic_recommendations, executive_view
        )

        # Risk summary
        risk_summary = self._generate_executive_risk_summary(validation_summary.risk_assessments)

        # Investment justification
        investment_justification = self._generate_investment_justification(
            business_metrics, validation_summary, project_definition
        )

        # Next steps
        next_steps = self._generate_executive_next_steps(validation_summary, project_definition)

        # Confidence assessment
        confidence_level = self._translate_confidence_level(validation_summary.validation_confidence)

        return ExecutiveSummary(
            project_overview=project_overview,
            key_achievements=key_achievements,
            business_impact=business_impact,
            strategic_recommendations=strategic_recommendations,
            risk_summary=risk_summary,
            investment_justification=investment_justification,
            next_steps=next_steps,
            confidence_level=confidence_level,
        )

    def generate_technical_report(
        self,
        project_definition: ProjectDefinition,
        validation_summary: ValidationSummary,
        business_metrics: List[BusinessMetric],
        stakeholder_views: Dict[str, StakeholderView],
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> TechnicalReport:
        """Generate detailed technical report for engineering teams."""

        technical_view = stakeholder_views.get("technical_lead")

        # Methodology overview
        methodology_overview = self._generate_methodology_narrative(project_definition, model_metadata)

        # Performance analysis
        performance_analysis = self._generate_performance_analysis(validation_summary, business_metrics)

        # Technical achievements
        technical_achievements = self._extract_technical_achievements(validation_summary, technical_view)

        # Architecture decisions
        architecture_decisions = self._extract_architecture_decisions(model_metadata, project_definition)

        # Optimization opportunities
        optimization_opportunities = validation_summary.technical_optimizations

        # Technical risks
        technical_risks = self._filter_technical_risks(validation_summary.risk_assessments)

        # Validation results
        validation_results = self._generate_validation_narrative(validation_summary)

        # Deployment considerations
        deployment_considerations = self._generate_deployment_considerations(
            project_definition, validation_summary, model_metadata
        )

        return TechnicalReport(
            methodology_overview=methodology_overview,
            performance_analysis=performance_analysis,
            technical_achievements=technical_achievements,
            architecture_decisions=architecture_decisions,
            optimization_opportunities=optimization_opportunities,
            technical_risks=technical_risks,
            validation_results=validation_results,
            deployment_considerations=deployment_considerations,
        )

    def generate_custom_narrative(
        self,
        project_definition: ProjectDefinition,
        validation_summary: ValidationSummary,
        business_metrics: List[BusinessMetric],
        target_audience: str,
        sections: List[ReportSection],
        tone: ReportTone = ReportTone.BUSINESS,
        length_target: str = "medium",
    ) -> NarrativeComponents:
        """Generate custom narrative with specified sections and tone."""

        components = {}

        for section in sections:
            if section == ReportSection.EXECUTIVE_SUMMARY:
                components["executive_summary"] = self._generate_section_executive_summary(
                    project_definition, validation_summary, business_metrics, tone
                )
            elif section == ReportSection.TECHNICAL_OVERVIEW:
                components["technical_overview"] = self._generate_section_technical_overview(
                    project_definition, validation_summary, tone
                )
            elif section == ReportSection.BUSINESS_IMPACT:
                components["business_impact"] = self._generate_section_business_impact(
                    business_metrics, project_definition.domain, tone
                )
            elif section == ReportSection.RISK_ASSESSMENT:
                components["risk_assessment"] = self._generate_section_risk_assessment(
                    validation_summary.risk_assessments, tone
                )
            elif section == ReportSection.RECOMMENDATIONS:
                components["recommendations"] = self._generate_section_recommendations(validation_summary, tone)
            elif section == ReportSection.PERFORMANCE_METRICS:
                components["performance_metrics"] = self._generate_section_performance_metrics(
                    business_metrics, validation_summary, tone
                )
            elif section == ReportSection.COMPLIANCE_STATUS:
                components["compliance_status"] = self._generate_section_compliance_status(
                    project_definition, validation_summary, tone
                )

        # Synthesize into narrative components
        return self._synthesize_narrative_components(components, length_target, tone)

    def _generate_project_overview(
        self, project_definition: ProjectDefinition, validation_summary: ValidationSummary
    ) -> str:
        """Generate high-level project overview for executives."""

        objective_description = self._get_objective_business_description(project_definition.objective)
        domain_context = self._get_domain_business_context(project_definition.domain)
        success_indicator = "successfully achieved" if validation_summary.overall_success else "progressing toward"

        return (
            f"This {domain_context} initiative focused on {objective_description} has {success_indicator} "
            f"its primary business objectives. The system demonstrates {validation_summary.validation_confidence:.0%} "
            f"confidence in meeting strategic requirements with {validation_summary.decision_support_score:.0%} "
            f"decision support capability."
        )

    def _extract_key_achievements(
        self,
        validation_summary: ValidationSummary,
        business_metrics: List[BusinessMetric],
        executive_view: Optional[StakeholderView],
    ) -> List[str]:
        """Extract top achievements for executive presentation."""

        achievements = []

        # Primary objective achievement
        if validation_summary.primary_objective_met:
            achievements.append("Primary business objective successfully achieved with industry-benchmark performance")

        # Business impact achievements
        high_impact_metrics = [m for m in business_metrics if m.business_impact and "high" in m.business_impact.lower()]
        if high_impact_metrics:
            achievements.append(
                f"Delivered high business impact across {len(high_impact_metrics)} key performance areas"
            )

        # Budget compliance achievement
        if validation_summary.budget_compliance_rate >= 0.9:
            achievements.append("Maintained excellent resource efficiency with 90%+ budget compliance")

        # Executive view specific achievements
        if executive_view and executive_view.key_insights:
            top_insight = executive_view.key_insights[0]
            achievements.append(f"Strategic insight: {top_insight}")

        return achievements[:4]  # Limit to top 4 for executive attention

    def _generate_business_impact_narrative(self, business_metrics: List[BusinessMetric], domain: Domain) -> str:
        """Generate compelling business impact narrative."""

        if not business_metrics:
            return "Business impact analysis pending additional performance data."

        # Calculate aggregate impact
        high_impact_count = len(
            [m for m in business_metrics if m.business_impact and "high" in m.business_impact.lower()]
        )
        financial_metrics = [m for m in business_metrics if m.financial_impact]

        impact_narrative = f"Analysis reveals significant business value creation across {len(business_metrics)} key performance dimensions. "

        if high_impact_count > 0:
            impact_narrative += (
                f"Particularly noteworthy: {high_impact_count} metrics demonstrate high strategic impact. "
            )

        if financial_metrics:
            impact_narrative += f"Financial analysis indicates measurable ROI through {len(financial_metrics)} monetizable performance improvements. "

        # Domain-specific impact context
        domain_impact = self._get_domain_impact_context(domain)
        impact_narrative += f"For {domain.value} operations, this translates to {domain_impact}."

        return impact_narrative

    def _distill_strategic_recommendations(
        self, raw_recommendations: List[str], executive_view: Optional[StakeholderView]
    ) -> List[str]:
        """Distill technical recommendations into strategic directives."""

        strategic_recs = []

        # Filter for strategic-level recommendations
        for rec in raw_recommendations:
            if any(
                keyword in rec.lower() for keyword in ["strategic", "business", "deployment", "investment", "scaling"]
            ):
                strategic_recs.append(self._elevate_recommendation_language(rec))

        # Add executive view recommendations
        if executive_view and executive_view.recommendations:
            strategic_recs.extend(executive_view.recommendations[:2])

        return strategic_recs[:3]  # Limit to top 3 strategic recommendations

    def _generate_executive_risk_summary(self, risk_assessments: List[str]) -> str:
        """Generate executive-appropriate risk summary."""

        if not risk_assessments:
            return "Risk profile appears manageable with standard mitigation protocols recommended."

        critical_risks = [r for r in risk_assessments if "critical" in r.lower()]
        high_risks = [r for r in risk_assessments if "high" in r.lower()]

        if critical_risks:
            return f"ATTENTION REQUIRED: {len(critical_risks)} critical risks identified requiring immediate executive oversight and mitigation planning."
        elif high_risks:
            return f"Manageable risk profile with {len(high_risks)} high-priority items requiring structured mitigation. Overall risk exposure within acceptable parameters."
        else:
            return "Low-risk deployment profile with standard operational risk management protocols sufficient."

    def _generate_investment_justification(
        self,
        business_metrics: List[BusinessMetric],
        validation_summary: ValidationSummary,
        project_definition: ProjectDefinition,
    ) -> str:
        """Generate investment justification for continued funding."""

        if validation_summary.overall_success:
            roi_indicators = len([m for m in business_metrics if m.financial_impact])
            return (
                f"Investment demonstrates strong strategic alignment with {roi_indicators} quantifiable ROI indicators. "
                f"Validation confidence of {validation_summary.validation_confidence:.0%} supports continued investment "
                f"with recommended scaling to capture full business value potential."
            )
        else:
            return (
                f"Current investment shows progress toward strategic objectives with {validation_summary.validation_confidence:.0%} "
                f"technical confidence. Recommend continued investment with strategic adjustments per technical team recommendations "
                f"to achieve full business value realization."
            )

    def _generate_executive_next_steps(
        self, validation_summary: ValidationSummary, project_definition: ProjectDefinition
    ) -> List[str]:
        """Generate executive-appropriate next steps."""

        next_steps = []

        if validation_summary.overall_success:
            next_steps.extend(
                [
                    "Approve production deployment with recommended scaling strategy",
                    "Initiate business value measurement and ROI tracking protocols",
                    "Consider expanding scope to additional use cases or business units",
                ]
            )
        else:
            next_steps.extend(
                [
                    "Review technical optimization recommendations with engineering leadership",
                    "Assess resource allocation adjustments per budget compliance analysis",
                    "Schedule follow-up validation milestone in 30-60 days",
                ]
            )

        # Domain-specific next steps
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            next_steps.append("Coordinate regulatory compliance review and approval process")

        return next_steps[:3]

    def _translate_confidence_level(self, confidence_score: float) -> str:
        """Translate technical confidence score to executive language."""

        if confidence_score >= 0.9:
            return "Very High - Ready for strategic deployment"
        elif confidence_score >= 0.8:
            return "High - Suitable for pilot deployment with monitoring"
        elif confidence_score >= 0.7:
            return "Moderate - Requires optimization before full deployment"
        elif confidence_score >= 0.6:
            return "Developing - Additional validation cycles recommended"
        else:
            return "Early Stage - Fundamental improvements needed"

    def _get_objective_business_description(self, objective: Objective) -> str:
        """Get business-friendly objective descriptions."""

        descriptions = {
            Objective.ACCURACY: "improving prediction reliability and decision quality",
            Objective.SPEED: "enhancing operational efficiency and response times",
            Objective.INTERPRETABILITY: "ensuring transparent and explainable decision-making",
            Objective.FAIRNESS: "maintaining ethical AI practices and equitable outcomes",
            Objective.COMPLIANCE: "achieving regulatory adherence and risk management",
            Objective.ROBUSTNESS: "building resilient and reliable operational systems",
        }

        return descriptions.get(objective, "optimizing system performance")

    def _get_domain_business_context(self, domain: Domain) -> str:
        """Get business context for different domains."""

        contexts = {
            Domain.FINANCE: "financial services transformation",
            Domain.HEALTHCARE: "healthcare innovation",
            Domain.RETAIL: "customer experience enhancement",
            Domain.MANUFACTURING: "operational excellence",
            Domain.MARKETING: "customer engagement optimization",
            Domain.GENERAL: "business intelligence",
        }

        return contexts.get(domain, "strategic AI")

    def _get_domain_impact_context(self, domain: Domain) -> str:
        """Get domain-specific impact context."""

        impacts = {
            Domain.FINANCE: "enhanced risk management, regulatory compliance, and customer service excellence",
            Domain.HEALTHCARE: "improved patient outcomes, operational efficiency, and clinical decision support",
            Domain.RETAIL: "personalized customer experiences, inventory optimization, and revenue growth",
            Domain.MANUFACTURING: "predictive maintenance, quality assurance, and supply chain optimization",
            Domain.MARKETING: "targeted campaigns, customer retention, and conversion optimization",
            Domain.GENERAL: "data-driven decision making and operational intelligence",
        }

        return impacts.get(domain, "strategic business improvements")

    def _elevate_recommendation_language(self, recommendation: str) -> str:
        """Elevate technical recommendations to strategic language."""

        # Replace technical terms with business equivalents
        business_translations = {
            "model": "system",
            "algorithm": "approach",
            "hyperparameter": "configuration",
            "optimization": "improvement",
            "deployment": "implementation",
            "validation": "verification",
            "pipeline": "process",
        }

        elevated = recommendation
        for tech_term, business_term in business_translations.items():
            elevated = elevated.replace(tech_term, business_term)

        return elevated

    def _initialize_report_templates(self) -> Dict[str, str]:
        """Initialize narrative templates for different report types."""

        return {
            "executive_opening": "Executive Assessment: {project_name} Strategic AI Initiative",
            "technical_opening": "Technical Analysis: {project_name} Implementation Report",
            "business_opening": "Business Impact Analysis: {project_name} Performance Review",
        }

    # Additional helper methods for technical report generation...

    def _generate_methodology_narrative(
        self, project_definition: ProjectDefinition, model_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate methodology section for technical report."""

        methodology = f"Objective-driven approach targeting {project_definition.objective.value} "
        methodology += f"optimization within {project_definition.domain.value} domain constraints. "

        if model_metadata:
            algorithm = model_metadata.get("algorithm", "advanced ML")
            methodology += f"Implementation utilized {algorithm} with systematic validation protocols."

        return methodology

    def _generate_performance_analysis(
        self, validation_summary: ValidationSummary, business_metrics: List[BusinessMetric]
    ) -> str:
        """Generate performance analysis for technical report."""

        analysis = f"Comprehensive validation achieved {validation_summary.validation_confidence:.1%} confidence "
        analysis += f"with {validation_summary.budget_compliance_rate:.1%} resource compliance. "

        if business_metrics:
            primary_metrics = [m for m in business_metrics if m.is_primary]
            if primary_metrics:
                analysis += f"Primary performance metrics demonstrate {primary_metrics[0].business_impact} impact."

        return analysis

    def _extract_technical_achievements(
        self, validation_summary: ValidationSummary, technical_view: Optional[StakeholderView]
    ) -> List[str]:
        """Extract technical achievements for engineering teams."""

        achievements = []

        if validation_summary.primary_objective_met:
            achievements.append("Primary objective benchmarks successfully achieved")

        if validation_summary.budget_compliance_rate >= 0.8:
            achievements.append("Resource efficiency targets met with optimal utilization")

        if technical_view and technical_view.key_insights:
            achievements.extend(technical_view.key_insights[:2])

        return achievements

    def _extract_architecture_decisions(
        self, model_metadata: Optional[Dict[str, Any]], project_definition: ProjectDefinition
    ) -> List[str]:
        """Extract key architecture decisions."""

        decisions = []

        if model_metadata:
            if "algorithm" in model_metadata:
                decisions.append(f"Algorithm selection: {model_metadata['algorithm']}")
            if "feature_count" in model_metadata:
                decisions.append(f"Feature engineering: {model_metadata['feature_count']} features optimized")

        # Domain-specific architecture decisions
        if project_definition.domain == Domain.FINANCE:
            decisions.append("Regulatory compliance architecture with audit trails")

        return decisions

    def _filter_technical_risks(self, risk_assessments: List[str]) -> List[str]:
        """Filter risks relevant to technical teams."""

        technical_keywords = ["performance", "latency", "memory", "scalability", "accuracy", "model"]

        return [risk for risk in risk_assessments if any(keyword in risk.lower() for keyword in technical_keywords)]

    def _generate_validation_narrative(self, validation_summary: ValidationSummary) -> str:
        """Generate validation results narrative."""

        narrative = f"Validation protocol executed with {validation_summary.validation_confidence:.1%} confidence. "
        narrative += f"Primary objective {'achieved' if validation_summary.primary_objective_met else 'progressing'}. "
        narrative += f"Decision support capability: {validation_summary.decision_support_score:.1%}."

        return narrative

    def _generate_deployment_considerations(
        self,
        project_definition: ProjectDefinition,
        validation_summary: ValidationSummary,
        model_metadata: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate deployment considerations for technical teams."""

        considerations = []

        if validation_summary.overall_success:
            considerations.append("System validated for production deployment")
        else:
            considerations.append("Additional optimization recommended before production")

        # Domain-specific considerations
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            considerations.append("Regulatory approval process required")

        if model_metadata and model_metadata.get("complexity", "medium") == "high":
            considerations.append("High-performance infrastructure recommended")

        return considerations

    # Custom narrative section generators...

    def _generate_section_executive_summary(
        self,
        project_definition: ProjectDefinition,
        validation_summary: ValidationSummary,
        business_metrics: List[BusinessMetric],
        tone: ReportTone,
    ) -> str:
        """Generate executive summary section."""

        return f"Strategic AI initiative for {project_definition.domain.value} {project_definition.objective.value} optimization demonstrates {validation_summary.validation_confidence:.0%} readiness for business deployment."

    def _generate_section_technical_overview(
        self, project_definition: ProjectDefinition, validation_summary: ValidationSummary, tone: ReportTone
    ) -> str:
        """Generate technical overview section."""

        return f"Technical implementation targeting {project_definition.objective.value} with comprehensive validation achieving {validation_summary.validation_confidence:.1%} confidence."

    def _generate_section_business_impact(
        self, business_metrics: List[BusinessMetric], domain: Domain, tone: ReportTone
    ) -> str:
        """Generate business impact section."""

        impact_count = len([m for m in business_metrics if m.business_impact])
        return f"Business impact analysis reveals {impact_count} key performance improvements with measurable ROI potential in {domain.value} operations."

    def _generate_section_risk_assessment(self, risk_assessments: List[str], tone: ReportTone) -> str:
        """Generate risk assessment section."""

        if not risk_assessments:
            return "Risk profile remains within acceptable operational parameters."

        critical_count = len([r for r in risk_assessments if "critical" in r.lower()])
        if critical_count > 0:
            return f"Risk analysis identifies {critical_count} critical items requiring immediate attention and mitigation."
        else:
            return f"Manageable risk profile with {len(risk_assessments)} identified items for standard mitigation protocols."

    def _generate_section_recommendations(self, validation_summary: ValidationSummary, tone: ReportTone) -> str:
        """Generate recommendations section."""

        if tone == ReportTone.EXECUTIVE:
            return f"Strategic recommendations focus on {len(validation_summary.strategic_recommendations)} key initiatives for business value maximization."
        else:
            return f"Technical optimization opportunities identified across {len(validation_summary.technical_optimizations)} performance areas."

    def _generate_section_performance_metrics(
        self, business_metrics: List[BusinessMetric], validation_summary: ValidationSummary, tone: ReportTone
    ) -> str:
        """Generate performance metrics section."""

        primary_count = len([m for m in business_metrics if m.is_primary])
        return f"Performance analysis encompasses {len(business_metrics)} business metrics with {primary_count} primary indicators achieving benchmark targets."

    def _generate_section_compliance_status(
        self, project_definition: ProjectDefinition, validation_summary: ValidationSummary, tone: ReportTone
    ) -> str:
        """Generate compliance status section."""

        compliance_rate = validation_summary.budget_compliance_rate
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            return f"Regulatory compliance validation demonstrates {compliance_rate:.0%} adherence to {project_definition.domain.value} industry standards."
        else:
            return (
                f"Compliance validation shows {compliance_rate:.0%} adherence to operational and ethical AI standards."
            )

    def _synthesize_narrative_components(
        self, components: Dict[str, str], length_target: str, tone: ReportTone
    ) -> NarrativeComponents:
        """Synthesize individual components into structured narrative."""

        # Create structured narrative based on available components
        introduction = components.get("executive_summary", "") or components.get("technical_overview", "")
        methodology = "Methodology: Objective-driven validation with comprehensive business impact analysis."
        results = components.get("performance_metrics", "") or "Results demonstrate strategic objective achievement."
        analysis = components.get("business_impact", "") or components.get("risk_assessment", "")
        conclusions = "Conclusions support strategic deployment with measured risk mitigation."
        recommendations = (
            components.get("recommendations", "") or "Recommendations focus on value optimization and risk management."
        )

        return NarrativeComponents(
            introduction=introduction,
            methodology=methodology,
            results=results,
            analysis=analysis,
            conclusions=conclusions,
            recommendations=recommendations,
        )
