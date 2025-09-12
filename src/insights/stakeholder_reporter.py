import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from .business_translator import BusinessTranslator, BusinessMetric, StakeholderView
from .narrative_generator import NarrativeGenerator, ExecutiveSummary, TechnicalReport, ReportSection, ReportTone
from .dashboard_engine import DashboardEngine, DashboardUpdate
from ..core.project_definition import ProjectDefinition, Objective, Domain
from ..validation.validation_orchestrator import ValidationSummary

class ReportFormat(Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    EMAIL = "email"
    POWERPOINT = "powerpoint"
    DASHBOARD_EXPORT = "dashboard_export"

class AudienceType(Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    BUSINESS = "business"
    COMPLIANCE = "compliance"
    CUSTOMER = "customer"
    INVESTOR = "investor"

class ReportFrequency(Enum):
    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"
    EVENT_DRIVEN = "event_driven"

class DeliveryChannel(Enum):
    EMAIL = "email"
    DASHBOARD = "dashboard"
    API = "api"
    FILE_EXPORT = "file_export"
    SLACK = "slack"
    WEBHOOK = "webhook"

@dataclass
class ReportTemplate:
    template_id: str
    name: str
    audience_type: AudienceType
    format: ReportFormat
    sections: List[ReportSection]
    tone: ReportTone
    styling: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReportSchedule:
    schedule_id: str
    frequency: ReportFrequency
    audience_type: AudienceType
    delivery_channels: List[DeliveryChannel]
    recipients: List[str]
    conditions: Optional[Dict[str, Any]] = None
    last_generated: Optional[datetime] = None
    next_scheduled: Optional[datetime] = None

@dataclass
class GeneratedReport:
    report_id: str
    generation_timestamp: datetime
    audience_type: AudienceType
    format: ReportFormat
    content: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    delivery_status: Dict[str, str] = field(default_factory=dict)

@dataclass
class ReportDeliveryResult:
    delivery_id: str
    report_id: str
    channel: DeliveryChannel
    recipient: str
    status: str  # "success", "failed", "pending"
    timestamp: datetime
    error_message: Optional[str] = None
    tracking_info: Optional[Dict[str, Any]] = None

class StakeholderReporter:
    """
    Production-grade stakeholder reporting system that generates and distributes
    automated reports with audience-specific content and formatting.
    """
    
    def __init__(self, 
                 business_translator: Optional[BusinessTranslator] = None,
                 narrative_generator: Optional[NarrativeGenerator] = None,
                 dashboard_engine: Optional[DashboardEngine] = None,
                 output_directory: Optional[str] = None):
        
        self.business_translator = business_translator or BusinessTranslator()
        self.narrative_generator = narrative_generator or NarrativeGenerator(self.business_translator)
        self.dashboard_engine = dashboard_engine or DashboardEngine(self.business_translator, self.narrative_generator)
        
        self.output_directory = Path(output_directory) if output_directory else Path("./reports")
        self.output_directory.mkdir(exist_ok=True)
        
        # Report management
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.active_schedules: Dict[str, ReportSchedule] = {}
        self.generated_reports: Dict[str, GeneratedReport] = {}
        self.delivery_history: List[ReportDeliveryResult] = []
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Report generation cache
        self.generation_cache: Dict[str, Any] = {}
        
    def generate_comprehensive_report(self,
                                    project_definition: ProjectDefinition,
                                    validation_summary: ValidationSummary,
                                    business_metrics: List[BusinessMetric],
                                    stakeholder_views: Dict[str, StakeholderView],
                                    audience_type: AudienceType,
                                    format: ReportFormat,
                                    custom_sections: Optional[List[ReportSection]] = None) -> GeneratedReport:
        """Generate comprehensive stakeholder report with specified format and content."""
        
        report_id = f"{audience_type.value}_{format.value}_{int(time.time())}"
        
        # Select appropriate template or create custom
        template = self._get_template_for_audience(audience_type, format)
        if custom_sections:
            template.sections = custom_sections
        
        # Generate content based on audience and format
        if format == ReportFormat.PDF or format == ReportFormat.HTML:
            content = self._generate_formatted_report(
                project_definition, validation_summary, business_metrics, 
                stakeholder_views, template
            )
        elif format == ReportFormat.JSON:
            content = self._generate_json_report(
                project_definition, validation_summary, business_metrics, stakeholder_views, audience_type
            )
        elif format == ReportFormat.EMAIL:
            content = self._generate_email_report(
                project_definition, validation_summary, business_metrics, stakeholder_views, audience_type
            )
        elif format == ReportFormat.POWERPOINT:
            content = self._generate_presentation_report(
                project_definition, validation_summary, business_metrics, stakeholder_views, audience_type
            )
        elif format == ReportFormat.DASHBOARD_EXPORT:
            content = self._generate_dashboard_export(
                project_definition, validation_summary, business_metrics, stakeholder_views
            )
        else:
            content = self._generate_markdown_report(
                project_definition, validation_summary, business_metrics, stakeholder_views, template
            )
        
        # Save to file if applicable
        file_path = None
        if format in [ReportFormat.PDF, ReportFormat.HTML, ReportFormat.MARKDOWN, ReportFormat.JSON]:
            file_path = self._save_report_to_file(content, report_id, format)
        
        # Create report metadata
        metadata = {
            'project_name': f"{project_definition.domain.value}_{project_definition.objective.value}",
            'generation_duration_ms': 0,  # Would be calculated in real implementation
            'content_sections': [section.value for section in template.sections],
            'audience_type': audience_type.value,
            'validation_confidence': validation_summary.validation_confidence,
            'business_metrics_count': len(business_metrics),
            'overall_success': validation_summary.overall_success
        }
        
        # Create generated report object
        generated_report = GeneratedReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            audience_type=audience_type,
            format=format,
            content=content,
            metadata=metadata,
            file_path=str(file_path) if file_path else None
        )
        
        # Store generated report
        self.generated_reports[report_id] = generated_report
        
        return generated_report
    
    def setup_automated_reporting(self,
                                audience_type: AudienceType,
                                frequency: ReportFrequency,
                                delivery_channels: List[DeliveryChannel],
                                recipients: List[str],
                                conditions: Optional[Dict[str, Any]] = None) -> str:
        """Setup automated report generation and delivery schedule."""
        
        schedule_id = f"schedule_{audience_type.value}_{frequency.value}_{int(time.time())}"
        
        # Calculate next scheduled time
        next_scheduled = self._calculate_next_scheduled_time(frequency)
        
        # Create report schedule
        schedule = ReportSchedule(
            schedule_id=schedule_id,
            frequency=frequency,
            audience_type=audience_type,
            delivery_channels=delivery_channels,
            recipients=recipients,
            conditions=conditions,
            next_scheduled=next_scheduled
        )
        
        # Store schedule
        self.active_schedules[schedule_id] = schedule
        
        return schedule_id
    
    def deliver_report(self,
                      report: GeneratedReport,
                      delivery_channels: List[DeliveryChannel],
                      recipients: List[str],
                      delivery_options: Optional[Dict[str, Any]] = None) -> List[ReportDeliveryResult]:
        """Deliver generated report through specified channels."""
        
        delivery_results = []
        
        for channel in delivery_channels:
            for recipient in recipients:
                delivery_id = f"delivery_{report.report_id}_{channel.value}_{int(time.time())}"
                
                try:
                    if channel == DeliveryChannel.EMAIL:
                        result = self._deliver_via_email(report, recipient, delivery_options)
                    elif channel == DeliveryChannel.DASHBOARD:
                        result = self._deliver_via_dashboard(report, recipient, delivery_options)
                    elif channel == DeliveryChannel.API:
                        result = self._deliver_via_api(report, recipient, delivery_options)
                    elif channel == DeliveryChannel.FILE_EXPORT:
                        result = self._deliver_via_file_export(report, recipient, delivery_options)
                    elif channel == DeliveryChannel.SLACK:
                        result = self._deliver_via_slack(report, recipient, delivery_options)
                    elif channel == DeliveryChannel.WEBHOOK:
                        result = self._deliver_via_webhook(report, recipient, delivery_options)
                    else:
                        result = ReportDeliveryResult(
                            delivery_id=delivery_id,
                            report_id=report.report_id,
                            channel=channel,
                            recipient=recipient,
                            status="failed",
                            timestamp=datetime.now(),
                            error_message=f"Unsupported delivery channel: {channel.value}"
                        )
                    
                    delivery_results.append(result)
                    self.delivery_history.append(result)
                    
                except Exception as e:
                    error_result = ReportDeliveryResult(
                        delivery_id=delivery_id,
                        report_id=report.report_id,
                        channel=channel,
                        recipient=recipient,
                        status="failed",
                        timestamp=datetime.now(),
                        error_message=str(e)
                    )
                    delivery_results.append(error_result)
                    self.delivery_history.append(error_result)
        
        # Update report delivery status
        for result in delivery_results:
            if report.report_id not in self.generated_reports:
                continue
            self.generated_reports[report.report_id].delivery_status[f"{result.channel.value}_{result.recipient}"] = result.status
        
        return delivery_results
    
    def generate_executive_briefing(self,
                                  project_definition: ProjectDefinition,
                                  validation_summary: ValidationSummary,
                                  business_metrics: List[BusinessMetric],
                                  stakeholder_views: Dict[str, StakeholderView]) -> GeneratedReport:
        """Generate executive briefing with high-level strategic insights."""
        
        executive_view = stakeholder_views.get('executive')
        
        # Generate executive summary
        executive_summary = self.narrative_generator.generate_executive_summary(
            project_definition, validation_summary, business_metrics, stakeholder_views
        )
        
        # Create executive briefing content
        briefing_content = self._create_executive_briefing_content(
            project_definition, validation_summary, business_metrics, executive_summary
        )
        
        # Generate report
        return self._create_briefing_report(briefing_content, AudienceType.EXECUTIVE)
    
    def generate_technical_deep_dive(self,
                                   project_definition: ProjectDefinition,
                                   validation_summary: ValidationSummary,
                                   business_metrics: List[BusinessMetric],
                                   stakeholder_views: Dict[str, StakeholderView],
                                   model_metadata: Optional[Dict[str, Any]] = None) -> GeneratedReport:
        """Generate technical deep-dive report for engineering teams."""
        
        technical_view = stakeholder_views.get('technical_lead')
        
        # Generate technical report
        technical_report = self.narrative_generator.generate_technical_report(
            project_definition, validation_summary, business_metrics, stakeholder_views, model_metadata
        )
        
        # Create technical deep-dive content
        deep_dive_content = self._create_technical_deep_dive_content(
            project_definition, validation_summary, business_metrics, technical_report, model_metadata
        )
        
        # Generate report
        return self._create_briefing_report(deep_dive_content, AudienceType.TECHNICAL)
    
    def generate_compliance_report(self,
                                 project_definition: ProjectDefinition,
                                 validation_summary: ValidationSummary,
                                 business_metrics: List[BusinessMetric],
                                 stakeholder_views: Dict[str, StakeholderView]) -> GeneratedReport:
        """Generate compliance report for regulatory requirements."""
        
        compliance_view = stakeholder_views.get('compliance_officer')
        
        # Generate compliance-specific content
        compliance_content = self._create_compliance_report_content(
            project_definition, validation_summary, business_metrics, compliance_view
        )
        
        # Generate report
        return self._create_briefing_report(compliance_content, AudienceType.COMPLIANCE)
    
    def check_scheduled_reports(self) -> List[str]:
        """Check and execute scheduled reports that are due."""
        
        current_time = datetime.now()
        executed_schedules = []
        
        for schedule_id, schedule in self.active_schedules.items():
            if schedule.next_scheduled and current_time >= schedule.next_scheduled:
                # Check conditions if specified
                if schedule.conditions and not self._check_schedule_conditions(schedule.conditions):
                    continue
                
                # Execute scheduled report generation
                # Note: In production, this would need access to current project data
                # For now, we'll mark as executed and update schedule
                schedule.last_generated = current_time
                schedule.next_scheduled = self._calculate_next_scheduled_time(schedule.frequency, current_time)
                
                executed_schedules.append(schedule_id)
        
        return executed_schedules
    
    def get_report_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get analytics on report generation and delivery performance."""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter recent reports and deliveries
        recent_reports = [
            report for report in self.generated_reports.values()
            if report.generation_timestamp >= cutoff_date
        ]
        
        recent_deliveries = [
            delivery for delivery in self.delivery_history
            if delivery.timestamp >= cutoff_date
        ]
        
        # Calculate analytics
        analytics = {
            'reporting_period_days': days_back,
            'total_reports_generated': len(recent_reports),
            'total_deliveries_attempted': len(recent_deliveries),
            'delivery_success_rate': self._calculate_delivery_success_rate(recent_deliveries),
            'reports_by_audience': self._analyze_reports_by_audience(recent_reports),
            'reports_by_format': self._analyze_reports_by_format(recent_reports),
            'deliveries_by_channel': self._analyze_deliveries_by_channel(recent_deliveries),
            'average_generation_time_ms': self._calculate_average_generation_time(recent_reports),
            'most_popular_sections': self._analyze_popular_sections(recent_reports),
            'error_analysis': self._analyze_delivery_errors(recent_deliveries),
            'schedule_performance': self._analyze_schedule_performance()
        }
        
        return analytics
    
    # Content generation methods
    
    def _generate_formatted_report(self,
                                 project_definition: ProjectDefinition,
                                 validation_summary: ValidationSummary,
                                 business_metrics: List[BusinessMetric],
                                 stakeholder_views: Dict[str, StakeholderView],
                                 template: ReportTemplate) -> str:
        """Generate formatted report content (HTML/PDF-ready)."""
        
        # Generate narrative components
        narrative = self.narrative_generator.generate_custom_narrative(
            project_definition, validation_summary, business_metrics,
            template.audience_type.value, template.sections, template.tone
        )
        
        # Create formatted report
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{template.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ border-bottom: 2px solid #007acc; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
                .alert {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{project_definition.domain.value.title()} AI Initiative Report</h1>
                <h2>Audience: {template.audience_type.value.title()}</h2>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{narrative.introduction}</p>
            </div>
            
            <div class="section">
                <h2>Key Results</h2>
                <p>{narrative.results}</p>
            </div>
            
            <div class="section">
                <h2>Analysis</h2>
                <p>{narrative.analysis}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <p>{narrative.recommendations}</p>
            </div>
        </body>
        </html>
        """
        
        return report_content
    
    def _generate_json_report(self,
                            project_definition: ProjectDefinition,
                            validation_summary: ValidationSummary,
                            business_metrics: List[BusinessMetric],
                            stakeholder_views: Dict[str, StakeholderView],
                            audience_type: AudienceType) -> Dict[str, Any]:
        """Generate structured JSON report."""
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'audience_type': audience_type.value,
                'project': {
                    'domain': project_definition.domain.value,
                    'objective': project_definition.objective.value
                }
            },
            'validation_summary': {
                'overall_success': validation_summary.overall_success,
                'primary_objective_met': validation_summary.primary_objective_met,
                'validation_confidence': validation_summary.validation_confidence,
                'decision_support_score': validation_summary.decision_support_score
            },
            'business_metrics': [
                {
                    'name': metric.metric_name,
                    'value': metric.current_value,
                    'business_impact': metric.business_impact,
                    'is_primary': metric.is_primary
                }
                for metric in business_metrics
            ],
            'strategic_recommendations': validation_summary.strategic_recommendations,
            'technical_optimizations': validation_summary.technical_optimizations,
            'risk_assessments': validation_summary.risk_assessments,
            'stakeholder_insights': {
                stakeholder_type: {
                    'key_insights': view.key_insights,
                    'recommendations': view.recommendations
                }
                for stakeholder_type, view in stakeholder_views.items()
            }
        }
    
    def _generate_email_report(self,
                             project_definition: ProjectDefinition,
                             validation_summary: ValidationSummary,
                             business_metrics: List[BusinessMetric],
                             stakeholder_views: Dict[str, StakeholderView],
                             audience_type: AudienceType) -> Dict[str, str]:
        """Generate email-formatted report."""
        
        stakeholder_view = stakeholder_views.get(audience_type.value)
        
        # Create subject line
        success_indicator = "✅ Success" if validation_summary.overall_success else "⚠️ In Progress"
        subject = f"{success_indicator}: {project_definition.domain.value.title()} AI Initiative Update"
        
        # Create email body
        body = f"""
        Hello,

        Here's your {audience_type.value} update for the {project_definition.domain.value} AI initiative:

        🎯 Primary Objective: {project_definition.objective.value.title()}
        📊 Overall Status: {'Successful' if validation_summary.overall_success else 'In Progress'}
        🔍 Validation Confidence: {validation_summary.validation_confidence:.0%}

        Key Highlights:
        """
        
        if stakeholder_view:
            for insight in stakeholder_view.key_insights[:3]:
                body += f"• {insight}\n"
        
        body += f"""
        
        Performance Summary:
        • {len(business_metrics)} business metrics tracked
        • {validation_summary.budget_compliance_rate:.0%} budget compliance rate
        • {validation_summary.trade_off_efficiency:.0%} trade-off efficiency
        
        Next Steps:
        """
        
        if stakeholder_view:
            for rec in stakeholder_view.recommendations[:2]:
                body += f"• {rec}\n"
        
        body += f"""
        
        For detailed analysis, access your stakeholder dashboard or contact the AI team.

        Best regards,
        Data Insight AI System
        """
        
        return {
            'subject': subject,
            'body': body,
            'format': 'text/plain'
        }
    
    def _generate_presentation_report(self,
                                    project_definition: ProjectDefinition,
                                    validation_summary: ValidationSummary,
                                    business_metrics: List[BusinessMetric],
                                    stakeholder_views: Dict[str, StakeholderView],
                                    audience_type: AudienceType) -> Dict[str, Any]:
        """Generate presentation-ready report structure."""
        
        slides = []
        
        # Title slide
        slides.append({
            'title': f"{project_definition.domain.value.title()} AI Initiative",
            'subtitle': f"{audience_type.value.title()} Briefing",
            'content': f"Generated: {datetime.now().strftime('%B %d, %Y')}"
        })
        
        # Executive overview slide
        slides.append({
            'title': 'Executive Overview',
            'content': [
                f"Objective: {project_definition.objective.value.title()}",
                f"Status: {'✅ Successful' if validation_summary.overall_success else '⚠️ In Progress'}",
                f"Confidence: {validation_summary.validation_confidence:.0%}",
                f"Business Metrics: {len(business_metrics)} tracked"
            ]
        })
        
        # Key insights slide
        stakeholder_view = stakeholder_views.get(audience_type.value)
        if stakeholder_view:
            slides.append({
                'title': 'Key Insights',
                'content': stakeholder_view.key_insights[:4]
            })
        
        # Performance metrics slide
        primary_metrics = [m for m in business_metrics if m.is_primary]
        slides.append({
            'title': 'Performance Metrics',
            'content': [
                f"{metric.metric_name}: {metric.current_value}"
                for metric in primary_metrics[:4]
            ]
        })
        
        # Recommendations slide
        slides.append({
            'title': 'Strategic Recommendations',
            'content': validation_summary.strategic_recommendations[:3]
        })
        
        # Risk assessment slide
        if validation_summary.risk_assessments:
            slides.append({
                'title': 'Risk Assessment',
                'content': validation_summary.risk_assessments[:3]
            })
        
        return {
            'presentation_format': 'slides',
            'slides': slides,
            'template': 'corporate',
            'audience': audience_type.value
        }
    
    def _generate_dashboard_export(self,
                                 project_definition: ProjectDefinition,
                                 validation_summary: ValidationSummary,
                                 business_metrics: List[BusinessMetric],
                                 stakeholder_views: Dict[str, StakeholderView]) -> Dict[str, Any]:
        """Generate dashboard export configuration."""
        
        return {
            'dashboard_config': {
                'project_name': f"{project_definition.domain.value}_{project_definition.objective.value}",
                'export_timestamp': datetime.now().isoformat()
            },
            'metrics_data': {
                metric.metric_name: {
                    'current_value': metric.current_value,
                    'target_value': metric.target_value,
                    'business_impact': metric.business_impact,
                    'trend': 'stable'  # Would be calculated from historical data
                }
                for metric in business_metrics
            },
            'validation_status': {
                'overall_success': validation_summary.overall_success,
                'confidence': validation_summary.validation_confidence,
                'compliance_rate': validation_summary.budget_compliance_rate
            },
            'stakeholder_views': {
                stakeholder_type: {
                    'insights': view.key_insights,
                    'recommendations': view.recommendations,
                    'focus_areas': view.focus_areas
                }
                for stakeholder_type, view in stakeholder_views.items()
            }
        }
    
    def _generate_markdown_report(self,
                                project_definition: ProjectDefinition,
                                validation_summary: ValidationSummary,
                                business_metrics: List[BusinessMetric],
                                stakeholder_views: Dict[str, StakeholderView],
                                template: ReportTemplate) -> str:
        """Generate markdown-formatted report."""
        
        narrative = self.narrative_generator.generate_custom_narrative(
            project_definition, validation_summary, business_metrics,
            template.audience_type.value, template.sections, template.tone
        )
        
        markdown_content = f"""
# {project_definition.domain.value.title()} AI Initiative Report

**Audience:** {template.audience_type.value.title()}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status:** {'✅ Successful' if validation_summary.overall_success else '⚠️ In Progress'}

## Executive Summary

{narrative.introduction}

## Key Results

{narrative.results}

## Analysis

{narrative.analysis}

## Strategic Recommendations

{narrative.recommendations}

## Performance Metrics

| Metric | Value | Impact | Primary |
|--------|-------|---------|---------|
"""
        
        for metric in business_metrics[:5]:
            markdown_content += f"| {metric.metric_name} | {metric.current_value} | {metric.business_impact or 'N/A'} | {'Yes' if metric.is_primary else 'No'} |\n"
        
        markdown_content += f"""

## Validation Summary

- **Overall Success:** {validation_summary.overall_success}
- **Primary Objective Met:** {validation_summary.primary_objective_met}
- **Validation Confidence:** {validation_summary.validation_confidence:.1%}
- **Budget Compliance:** {validation_summary.budget_compliance_rate:.1%}

## Risk Assessment

"""
        
        for risk in validation_summary.risk_assessments[:3]:
            markdown_content += f"- {risk}\n"
        
        return markdown_content
    
    # Helper methods for content creation
    
    def _create_executive_briefing_content(self,
                                         project_definition: ProjectDefinition,
                                         validation_summary: ValidationSummary,
                                         business_metrics: List[BusinessMetric],
                                         executive_summary: ExecutiveSummary) -> Dict[str, Any]:
        """Create executive briefing content structure."""
        
        return {
            'briefing_type': 'executive',
            'project_overview': executive_summary.project_overview,
            'key_achievements': executive_summary.key_achievements,
            'business_impact': executive_summary.business_impact,
            'investment_justification': executive_summary.investment_justification,
            'strategic_recommendations': executive_summary.strategic_recommendations,
            'risk_summary': executive_summary.risk_summary,
            'next_steps': executive_summary.next_steps,
            'confidence_level': executive_summary.confidence_level
        }
    
    def _create_technical_deep_dive_content(self,
                                          project_definition: ProjectDefinition,
                                          validation_summary: ValidationSummary,
                                          business_metrics: List[BusinessMetric],
                                          technical_report: TechnicalReport,
                                          model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create technical deep-dive content structure."""
        
        return {
            'briefing_type': 'technical',
            'methodology_overview': technical_report.methodology_overview,
            'performance_analysis': technical_report.performance_analysis,
            'technical_achievements': technical_report.technical_achievements,
            'architecture_decisions': technical_report.architecture_decisions,
            'optimization_opportunities': technical_report.optimization_opportunities,
            'technical_risks': technical_report.technical_risks,
            'validation_results': technical_report.validation_results,
            'deployment_considerations': technical_report.deployment_considerations,
            'model_metadata': model_metadata or {}
        }
    
    def _create_compliance_report_content(self,
                                        project_definition: ProjectDefinition,
                                        validation_summary: ValidationSummary,
                                        business_metrics: List[BusinessMetric],
                                        compliance_view: Optional[StakeholderView]) -> Dict[str, Any]:
        """Create compliance report content structure."""
        
        compliance_metrics = [m for m in business_metrics if 'compliance' in m.metric_name.lower()]
        
        return {
            'briefing_type': 'compliance',
            'domain': project_definition.domain.value,
            'compliance_score': validation_summary.budget_compliance_rate,
            'regulatory_requirements': self._get_regulatory_requirements(project_definition.domain),
            'compliance_metrics': [
                {
                    'name': metric.metric_name,
                    'value': metric.current_value,
                    'status': 'compliant' if metric.current_value >= 0.8 else 'non_compliant'
                }
                for metric in compliance_metrics
            ],
            'audit_findings': compliance_view.key_insights if compliance_view else [],
            'remediation_actions': compliance_view.recommendations if compliance_view else [],
            'certification_status': 'pending_review'
        }
    
    def _create_briefing_report(self, content: Dict[str, Any], audience_type: AudienceType) -> GeneratedReport:
        """Create briefing report from content structure."""
        
        report_id = f"briefing_{audience_type.value}_{int(time.time())}"
        
        return GeneratedReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            audience_type=audience_type,
            format=ReportFormat.JSON,
            content=content,
            metadata={
                'briefing_type': content.get('briefing_type', 'general'),
                'content_sections': list(content.keys()),
                'generation_source': 'stakeholder_reporter'
            }
        )
    
    # Template and schedule management
    
    def _initialize_default_templates(self):
        """Initialize default report templates for different audiences."""
        
        templates = {
            'executive_summary': ReportTemplate(
                template_id='executive_summary',
                name='Executive Summary Report',
                audience_type=AudienceType.EXECUTIVE,
                format=ReportFormat.PDF,
                sections=[ReportSection.EXECUTIVE_SUMMARY, ReportSection.BUSINESS_IMPACT, ReportSection.RECOMMENDATIONS],
                tone=ReportTone.EXECUTIVE
            ),
            'technical_deep_dive': ReportTemplate(
                template_id='technical_deep_dive',
                name='Technical Deep Dive Report',
                audience_type=AudienceType.TECHNICAL,
                format=ReportFormat.HTML,
                sections=[ReportSection.TECHNICAL_OVERVIEW, ReportSection.PERFORMANCE_METRICS, ReportSection.RECOMMENDATIONS],
                tone=ReportTone.TECHNICAL
            ),
            'business_review': ReportTemplate(
                template_id='business_review',
                name='Business Review Report',
                audience_type=AudienceType.BUSINESS,
                format=ReportFormat.MARKDOWN,
                sections=[ReportSection.BUSINESS_IMPACT, ReportSection.PERFORMANCE_METRICS, ReportSection.RECOMMENDATIONS],
                tone=ReportTone.BUSINESS
            ),
            'compliance_audit': ReportTemplate(
                template_id='compliance_audit',
                name='Compliance Audit Report',
                audience_type=AudienceType.COMPLIANCE,
                format=ReportFormat.PDF,
                sections=[ReportSection.COMPLIANCE_STATUS, ReportSection.RISK_ASSESSMENT, ReportSection.RECOMMENDATIONS],
                tone=ReportTone.COMPLIANCE
            )
        }
        
        self.report_templates.update(templates)
    
    def _get_template_for_audience(self, audience_type: AudienceType, format: ReportFormat) -> ReportTemplate:
        """Get appropriate template for audience type and format."""
        
        # Find matching template
        for template in self.report_templates.values():
            if template.audience_type == audience_type and template.format == format:
                return template
        
        # Return default template if no match
        return ReportTemplate(
            template_id=f'default_{audience_type.value}',
            name=f'Default {audience_type.value.title()} Report',
            audience_type=audience_type,
            format=format,
            sections=[ReportSection.EXECUTIVE_SUMMARY, ReportSection.PERFORMANCE_METRICS, ReportSection.RECOMMENDATIONS],
            tone=ReportTone.BUSINESS
        )
    
    def _save_report_to_file(self, content: Union[str, Dict[str, Any]], report_id: str, format: ReportFormat) -> Optional[Path]:
        """Save report content to file."""
        
        file_extension = {
            ReportFormat.PDF: '.pdf',
            ReportFormat.HTML: '.html',
            ReportFormat.MARKDOWN: '.md',
            ReportFormat.JSON: '.json'
        }.get(format, '.txt')
        
        file_path = self.output_directory / f"{report_id}{file_extension}"
        
        try:
            if isinstance(content, dict):
                with open(file_path, 'w') as f:
                    json.dump(content, f, indent=2, default=str)
            else:
                with open(file_path, 'w') as f:
                    f.write(content)
            
            return file_path
        except Exception as e:
            print(f"Error saving report to file: {e}")
            return None
    
    # Delivery methods (stubs for production implementation)
    
    def _deliver_via_email(self, report: GeneratedReport, recipient: str, options: Optional[Dict[str, Any]]) -> ReportDeliveryResult:
        """Deliver report via email."""
        # Production implementation would integrate with email service
        return ReportDeliveryResult(
            delivery_id=f"email_{int(time.time())}",
            report_id=report.report_id,
            channel=DeliveryChannel.EMAIL,
            recipient=recipient,
            status="success",
            timestamp=datetime.now(),
            tracking_info={"method": "smtp", "message_id": f"msg_{int(time.time())}"}
        )
    
    def _deliver_via_dashboard(self, report: GeneratedReport, recipient: str, options: Optional[Dict[str, Any]]) -> ReportDeliveryResult:
        """Deliver report via dashboard."""
        return ReportDeliveryResult(
            delivery_id=f"dashboard_{int(time.time())}",
            report_id=report.report_id,
            channel=DeliveryChannel.DASHBOARD,
            recipient=recipient,
            status="success",
            timestamp=datetime.now(),
            tracking_info={"dashboard_url": f"/dashboard/reports/{report.report_id}"}
        )
    
    def _deliver_via_api(self, report: GeneratedReport, recipient: str, options: Optional[Dict[str, Any]]) -> ReportDeliveryResult:
        """Deliver report via API endpoint."""
        return ReportDeliveryResult(
            delivery_id=f"api_{int(time.time())}",
            report_id=report.report_id,
            channel=DeliveryChannel.API,
            recipient=recipient,
            status="success",
            timestamp=datetime.now(),
            tracking_info={"endpoint": f"/api/reports/{report.report_id}", "method": "GET"}
        )
    
    def _deliver_via_file_export(self, report: GeneratedReport, recipient: str, options: Optional[Dict[str, Any]]) -> ReportDeliveryResult:
        """Deliver report via file export."""
        return ReportDeliveryResult(
            delivery_id=f"file_{int(time.time())}",
            report_id=report.report_id,
            channel=DeliveryChannel.FILE_EXPORT,
            recipient=recipient,
            status="success",
            timestamp=datetime.now(),
            tracking_info={"file_path": report.file_path}
        )
    
    def _deliver_via_slack(self, report: GeneratedReport, recipient: str, options: Optional[Dict[str, Any]]) -> ReportDeliveryResult:
        """Deliver report via Slack."""
        return ReportDeliveryResult(
            delivery_id=f"slack_{int(time.time())}",
            report_id=report.report_id,
            channel=DeliveryChannel.SLACK,
            recipient=recipient,
            status="success",
            timestamp=datetime.now(),
            tracking_info={"channel": recipient, "message_ts": str(int(time.time()))}
        )
    
    def _deliver_via_webhook(self, report: GeneratedReport, recipient: str, options: Optional[Dict[str, Any]]) -> ReportDeliveryResult:
        """Deliver report via webhook."""
        return ReportDeliveryResult(
            delivery_id=f"webhook_{int(time.time())}",
            report_id=report.report_id,
            channel=DeliveryChannel.WEBHOOK,
            recipient=recipient,
            status="success",
            timestamp=datetime.now(),
            tracking_info={"webhook_url": recipient, "response_code": 200}
        )
    
    # Scheduling and analytics helpers
    
    def _calculate_next_scheduled_time(self, frequency: ReportFrequency, from_time: Optional[datetime] = None) -> datetime:
        """Calculate next scheduled report time."""
        
        base_time = from_time or datetime.now()
        
        if frequency == ReportFrequency.DAILY:
            return base_time + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            return base_time + timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            return base_time + timedelta(days=30)
        elif frequency == ReportFrequency.QUARTERLY:
            return base_time + timedelta(days=90)
        else:
            return base_time + timedelta(hours=1)  # Default fallback
    
    def _check_schedule_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check if schedule conditions are met."""
        # Production implementation would check actual system conditions
        return True
    
    def _get_regulatory_requirements(self, domain: Domain) -> List[str]:
        """Get regulatory requirements for domain."""
        
        requirements = {
            Domain.FINANCE: ['Basel III', 'MiFID II', 'GDPR', 'SOX'],
            Domain.HEALTHCARE: ['HIPAA', 'FDA 21 CFR Part 820', 'GDPR', 'IEC 62304'],
        }
        
        return requirements.get(domain, ['GDPR', 'General Data Protection'])
    
    # Analytics calculation methods
    
    def _calculate_delivery_success_rate(self, deliveries: List[ReportDeliveryResult]) -> float:
        """Calculate delivery success rate."""
        if not deliveries:
            return 0.0
        
        successful_deliveries = len([d for d in deliveries if d.status == "success"])
        return successful_deliveries / len(deliveries)
    
    def _analyze_reports_by_audience(self, reports: List[GeneratedReport]) -> Dict[str, int]:
        """Analyze report generation by audience type."""
        
        audience_counts = {}
        for report in reports:
            audience = report.audience_type.value
            audience_counts[audience] = audience_counts.get(audience, 0) + 1
        
        return audience_counts
    
    def _analyze_reports_by_format(self, reports: List[GeneratedReport]) -> Dict[str, int]:
        """Analyze report generation by format."""
        
        format_counts = {}
        for report in reports:
            format_type = report.format.value
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
        
        return format_counts
    
    def _analyze_deliveries_by_channel(self, deliveries: List[ReportDeliveryResult]) -> Dict[str, int]:
        """Analyze deliveries by channel."""
        
        channel_counts = {}
        for delivery in deliveries:
            channel = delivery.channel.value
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        return channel_counts
    
    def _calculate_average_generation_time(self, reports: List[GeneratedReport]) -> float:
        """Calculate average report generation time."""
        
        total_time = sum(report.metadata.get('generation_duration_ms', 0) for report in reports)
        return total_time / len(reports) if reports else 0.0
    
    def _analyze_popular_sections(self, reports: List[GeneratedReport]) -> Dict[str, int]:
        """Analyze most popular report sections."""
        
        section_counts = {}
        for report in reports:
            sections = report.metadata.get('content_sections', [])
            for section in sections:
                section_counts[section] = section_counts.get(section, 0) + 1
        
        return section_counts
    
    def _analyze_delivery_errors(self, deliveries: List[ReportDeliveryResult]) -> Dict[str, int]:
        """Analyze delivery errors."""
        
        error_counts = {}
        for delivery in deliveries:
            if delivery.status == "failed" and delivery.error_message:
                error_type = delivery.error_message.split(':')[0] if ':' in delivery.error_message else 'unknown'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts
    
    def _analyze_schedule_performance(self) -> Dict[str, Any]:
        """Analyze scheduled report performance."""
        
        total_schedules = len(self.active_schedules)
        executed_schedules = len([s for s in self.active_schedules.values() if s.last_generated])
        
        return {
            'total_active_schedules': total_schedules,
            'executed_schedules': executed_schedules,
            'execution_rate': executed_schedules / total_schedules if total_schedules > 0 else 0.0,
            'schedule_types': {
                freq.value: len([s for s in self.active_schedules.values() if s.frequency == freq])
                for freq in ReportFrequency
            }
        }