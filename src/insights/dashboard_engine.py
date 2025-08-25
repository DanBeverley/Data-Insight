import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .business_translator import BusinessTranslator, BusinessMetric, StakeholderView
from .narrative_generator import NarrativeGenerator, ExecutiveSummary, TechnicalReport
from ..core.project_definition import ProjectDefinition, Objective, Domain
from ..validation.validation_orchestrator import ValidationSummary

class MetricTrend(Enum):
    IMPROVING = "improving"
    DECLINING = "declining" 
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"

class VisualizationType(Enum):
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    TREND_LINE = "trend_line"
    KPI_CARD = "kpi_card"

@dataclass
class DashboardUpdate:
    update_id: str
    timestamp: datetime
    metric_updates: Dict[str, Any]
    trend_changes: Dict[str, MetricTrend]
    alerts: List['DashboardAlert']
    narrative_summary: str
    stakeholder_notifications: Dict[str, List[str]]

@dataclass 
class DashboardAlert:
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    threshold_value: Optional[float]
    current_value: float
    timestamp: datetime
    auto_resolve: bool = False
    resolution_criteria: Optional[str] = None

@dataclass
class VisualizationConfig:
    chart_type: VisualizationType
    title: str
    data_source: str
    refresh_interval_seconds: int
    styling: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    stakeholder_visibility: List[str] = field(default_factory=list)

@dataclass
class DashboardLayout:
    layout_id: str
    stakeholder_type: str
    widgets: List['DashboardWidget']
    refresh_policy: Dict[str, int]
    alert_preferences: Dict[str, Any]

@dataclass
class DashboardWidget:
    widget_id: str
    title: str
    visualization: VisualizationConfig
    position: Dict[str, int]  # x, y, width, height
    priority: int
    data_binding: Dict[str, str]

class DashboardEngine:
    """
    Production-grade dashboard engine that provides real-time business intelligence
    with stakeholder-specific views and automated insight delivery.
    """
    
    def __init__(self, 
                 business_translator: Optional[BusinessTranslator] = None,
                 narrative_generator: Optional[NarrativeGenerator] = None):
        self.business_translator = business_translator or BusinessTranslator()
        self.narrative_generator = narrative_generator or NarrativeGenerator(self.business_translator)
        
        self.dashboard_layouts: Dict[str, DashboardLayout] = {}
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.active_alerts: Dict[str, DashboardAlert] = {}
        self.dashboard_state: Dict[str, Any] = {}
        self.update_history: List[DashboardUpdate] = []
        
        # Initialize default layouts
        self._initialize_default_layouts()
        
        # Set up automatic refresh schedules
        self.refresh_schedules: Dict[str, datetime] = {}
        
    def create_real_time_dashboard(self,
                                 project_definition: ProjectDefinition,
                                 validation_summary: ValidationSummary,
                                 business_metrics: List[BusinessMetric],
                                 stakeholder_views: Dict[str, StakeholderView],
                                 layout_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive real-time dashboard with stakeholder-specific views."""
        
        update_id = f"dashboard_init_{int(time.time())}"
        
        # Generate executive dashboard
        executive_dashboard = self._create_executive_dashboard(
            project_definition, validation_summary, business_metrics, stakeholder_views.get('executive')
        )
        
        # Generate technical dashboard
        technical_dashboard = self._create_technical_dashboard(
            project_definition, validation_summary, business_metrics, stakeholder_views.get('technical_lead')
        )
        
        # Generate business dashboard
        business_dashboard = self._create_business_dashboard(
            project_definition, validation_summary, business_metrics, stakeholder_views.get('product_manager')
        )
        
        # Generate compliance dashboard if needed
        compliance_dashboard = None
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            compliance_dashboard = self._create_compliance_dashboard(
                project_definition, validation_summary, business_metrics, stakeholder_views.get('compliance_officer')
            )
        
        # Set up real-time monitoring
        monitoring_config = self._setup_real_time_monitoring(project_definition, business_metrics)
        
        # Initialize alert system
        alert_system = self._initialize_alert_system(business_metrics, validation_summary)
        
        dashboard_config = {
            'dashboard_id': update_id,
            'created_at': datetime.now().isoformat(),
            'project_context': {
                'name': f"{project_definition.domain.value}_{project_definition.objective.value}",
                'domain': project_definition.domain.value,
                'objective': project_definition.objective.value
            },
            'stakeholder_dashboards': {
                'executive': executive_dashboard,
                'technical': technical_dashboard,
                'business': business_dashboard
            },
            'monitoring': monitoring_config,
            'alerts': alert_system,
            'refresh_intervals': self._get_refresh_intervals(),
            'data_sources': self._map_data_sources(business_metrics)
        }
        
        if compliance_dashboard:
            dashboard_config['stakeholder_dashboards']['compliance'] = compliance_dashboard
        
        # Store dashboard state
        self.dashboard_state[update_id] = dashboard_config
        
        return dashboard_config
    
    def update_dashboard_metrics(self,
                               dashboard_id: str,
                               new_metrics: Dict[str, Any],
                               validation_update: Optional[ValidationSummary] = None) -> DashboardUpdate:
        """Update dashboard with new metrics and generate alerts if needed."""
        
        update_id = f"update_{dashboard_id}_{int(time.time())}"
        timestamp = datetime.now()
        
        # Update metric history
        trend_changes = {}
        for metric_name, value in new_metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            
            self.metric_history[metric_name].append((timestamp, value))
            
            # Calculate trends
            trend_changes[metric_name] = self._calculate_metric_trend(metric_name)
        
        # Check for alerts
        alerts = self._check_metric_alerts(new_metrics, trend_changes)
        
        # Generate narrative summary
        narrative_summary = self._generate_update_narrative(new_metrics, trend_changes, alerts)
        
        # Generate stakeholder notifications
        stakeholder_notifications = self._generate_stakeholder_notifications(
            new_metrics, trend_changes, alerts, dashboard_id
        )
        
        # Create update object
        dashboard_update = DashboardUpdate(
            update_id=update_id,
            timestamp=timestamp,
            metric_updates=new_metrics,
            trend_changes=trend_changes,
            alerts=alerts,
            narrative_summary=narrative_summary,
            stakeholder_notifications=stakeholder_notifications
        )
        
        # Store update
        self.update_history.append(dashboard_update)
        
        # Update dashboard state
        if dashboard_id in self.dashboard_state:
            self.dashboard_state[dashboard_id]['last_update'] = timestamp.isoformat()
            self.dashboard_state[dashboard_id]['current_metrics'] = new_metrics
        
        return dashboard_update
    
    def get_stakeholder_view(self, 
                           dashboard_id: str, 
                           stakeholder_type: str,
                           time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get stakeholder-specific dashboard view with filtered data."""
        
        if dashboard_id not in self.dashboard_state:
            return {"error": "Dashboard not found"}
        
        dashboard_config = self.dashboard_state[dashboard_id]
        stakeholder_dashboard = dashboard_config.get('stakeholder_dashboards', {}).get(stakeholder_type)
        
        if not stakeholder_dashboard:
            return {"error": f"No dashboard configured for stakeholder type: {stakeholder_type}"}
        
        # Apply time range filtering
        filtered_metrics = self._filter_metrics_by_time_range(dashboard_id, time_range)
        
        # Get recent alerts for this stakeholder
        stakeholder_alerts = self._filter_alerts_by_stakeholder(stakeholder_type)
        
        # Generate real-time insights
        real_time_insights = self._generate_real_time_insights(filtered_metrics, stakeholder_type)
        
        return {
            'dashboard_layout': stakeholder_dashboard,
            'current_metrics': filtered_metrics,
            'recent_alerts': stakeholder_alerts,
            'real_time_insights': real_time_insights,
            'last_updated': dashboard_config.get('last_update'),
            'next_refresh': self._get_next_refresh_time(stakeholder_type)
        }
    
    def create_custom_visualization(self,
                                  dashboard_id: str,
                                  metric_name: str,
                                  visualization_type: VisualizationType,
                                  stakeholder_types: List[str],
                                  custom_config: Optional[Dict[str, Any]] = None) -> VisualizationConfig:
        """Create custom visualization for specific metrics."""
        
        config = VisualizationConfig(
            chart_type=visualization_type,
            title=f"{metric_name} - {visualization_type.value.replace('_', ' ').title()}",
            data_source=f"metric_{metric_name}",
            refresh_interval_seconds=self._get_optimal_refresh_interval(metric_name),
            stakeholder_visibility=stakeholder_types
        )
        
        # Apply custom styling
        if custom_config:
            config.styling.update(custom_config.get('styling', {}))
            config.filters.update(custom_config.get('filters', {}))
        
        # Add visualization type specific defaults
        if visualization_type == VisualizationType.GAUGE:
            config.styling.update({
                'min_value': 0,
                'max_value': 100,
                'threshold_zones': [
                    {'min': 0, 'max': 30, 'color': 'red'},
                    {'min': 30, 'max': 70, 'color': 'yellow'},
                    {'min': 70, 'max': 100, 'color': 'green'}
                ]
            })
        elif visualization_type == VisualizationType.KPI_CARD:
            config.styling.update({
                'show_trend': True,
                'show_target': True,
                'highlight_changes': True
            })
        
        return config
    
    def export_dashboard_config(self, dashboard_id: str, include_data: bool = False) -> Dict[str, Any]:
        """Export dashboard configuration for backup or sharing."""
        
        if dashboard_id not in self.dashboard_state:
            return {"error": "Dashboard not found"}
        
        config = self.dashboard_state[dashboard_id].copy()
        
        export_data = {
            'dashboard_config': config,
            'layouts': {k: v for k, v in self.dashboard_layouts.items() if k.startswith(dashboard_id)},
            'export_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if include_data:
            export_data.update({
                'metric_history': {
                    k: [(ts.isoformat(), val) for ts, val in v] 
                    for k, v in self.metric_history.items()
                },
                'alert_history': [
                    {
                        'alert_id': alert.alert_id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.active_alerts.values()
                ],
                'update_history': [
                    {
                        'update_id': update.update_id,
                        'timestamp': update.timestamp.isoformat(),
                        'metrics_count': len(update.metric_updates),
                        'alerts_count': len(update.alerts)
                    }
                    for update in self.update_history[-10:]  # Last 10 updates
                ]
            })
        
        return export_data
    
    def _create_executive_dashboard(self, 
                                  project_definition: ProjectDefinition,
                                  validation_summary: ValidationSummary,
                                  business_metrics: List[BusinessMetric],
                                  executive_view: Optional[StakeholderView]) -> Dict[str, Any]:
        """Create executive-focused dashboard layout."""
        
        widgets = []
        
        # Primary KPI card
        widgets.append(self._create_kpi_widget(
            "primary_objective_kpi",
            f"{project_definition.objective.value.title()} Performance",
            "primary_objective_score",
            position={'x': 0, 'y': 0, 'width': 4, 'height': 2}
        ))
        
        # ROI gauge
        widgets.append(self._create_gauge_widget(
            "roi_gauge",
            "Business Value Score",
            "roi_indicator",
            position={'x': 4, 'y': 0, 'width': 3, 'height': 2}
        ))
        
        # Success rate trend
        widgets.append(self._create_trend_widget(
            "success_trend",
            "Validation Success Rate",
            "validation_success_rate",
            position={'x': 7, 'y': 0, 'width': 5, 'height': 2}
        ))
        
        # Strategic insights
        strategic_insights = []
        if executive_view:
            strategic_insights = executive_view.key_insights[:3]
        
        widgets.append({
            'widget_id': 'strategic_insights',
            'title': 'Strategic Insights',
            'type': 'insight_panel',
            'content': strategic_insights,
            'position': {'x': 0, 'y': 2, 'width': 6, 'height': 3}
        })
        
        # Risk overview
        widgets.append({
            'widget_id': 'risk_overview',
            'title': 'Risk Assessment',
            'type': 'risk_panel',
            'content': validation_summary.risk_assessments[:3],
            'position': {'x': 6, 'y': 2, 'width': 6, 'height': 3}
        })
        
        return {
            'layout_type': 'executive',
            'refresh_interval': 300,  # 5 minutes
            'widgets': widgets,
            'alert_preferences': {
                'critical_only': True,
                'auto_escalate': True,
                'notification_methods': ['email', 'dashboard']
            }
        }
    
    def _create_technical_dashboard(self,
                                  project_definition: ProjectDefinition,
                                  validation_summary: ValidationSummary,
                                  business_metrics: List[BusinessMetric],
                                  technical_view: Optional[StakeholderView]) -> Dict[str, Any]:
        """Create technical team-focused dashboard layout."""
        
        widgets = []
        
        # Performance metrics chart
        widgets.append(self._create_line_chart_widget(
            "performance_metrics",
            "Model Performance Metrics",
            ["accuracy", "precision", "recall", "f1_score"],
            position={'x': 0, 'y': 0, 'width': 8, 'height': 3}
        ))
        
        # Resource utilization gauge
        widgets.append(self._create_gauge_widget(
            "resource_utilization",
            "Resource Utilization",
            "resource_usage_percent",
            position={'x': 8, 'y': 0, 'width': 4, 'height': 3}
        ))
        
        # Budget compliance
        widgets.append(self._create_bar_chart_widget(
            "budget_compliance",
            "Budget Compliance",
            "budget_metrics",
            position={'x': 0, 'y': 3, 'width': 6, 'height': 3}
        ))
        
        # Technical optimizations
        technical_optimizations = validation_summary.technical_optimizations[:5]
        widgets.append({
            'widget_id': 'technical_optimizations',
            'title': 'Optimization Opportunities',
            'type': 'list_panel',
            'content': technical_optimizations,
            'position': {'x': 6, 'y': 3, 'width': 6, 'height': 3}
        })
        
        return {
            'layout_type': 'technical',
            'refresh_interval': 60,  # 1 minute
            'widgets': widgets,
            'alert_preferences': {
                'all_severities': True,
                'technical_focus': True,
                'notification_methods': ['dashboard', 'slack']
            }
        }
    
    def _create_business_dashboard(self,
                                 project_definition: ProjectDefinition,
                                 validation_summary: ValidationSummary,
                                 business_metrics: List[BusinessMetric],
                                 business_view: Optional[StakeholderView]) -> Dict[str, Any]:
        """Create business stakeholder-focused dashboard layout."""
        
        widgets = []
        
        # Business impact overview
        impact_metrics = [m for m in business_metrics if m.business_impact]
        widgets.append(self._create_pie_chart_widget(
            "business_impact",
            "Business Impact Distribution",
            "impact_categories",
            position={'x': 0, 'y': 0, 'width': 6, 'height': 3}
        ))
        
        # Financial metrics
        financial_metrics = [m for m in business_metrics if m.financial_impact]
        if financial_metrics:
            widgets.append(self._create_kpi_widget(
                "financial_impact",
                "Financial Impact",
                "roi_calculation",
                position={'x': 6, 'y': 0, 'width': 6, 'height': 3}
            ))
        
        # Customer impact (domain-specific)
        if project_definition.domain == Domain.RETAIL:
            widgets.append(self._create_line_chart_widget(
                "customer_metrics",
                "Customer Experience Metrics",
                ["satisfaction_score", "conversion_rate", "retention_rate"],
                position={'x': 0, 'y': 3, 'width': 8, 'height': 3}
            ))
        
        return {
            'layout_type': 'business',
            'refresh_interval': 180,  # 3 minutes
            'widgets': widgets,
            'alert_preferences': {
                'business_impact_focus': True,
                'escalation_rules': True,
                'notification_methods': ['email', 'dashboard']
            }
        }
    
    def _create_compliance_dashboard(self,
                                   project_definition: ProjectDefinition,
                                   validation_summary: ValidationSummary,
                                   business_metrics: List[BusinessMetric],
                                   compliance_view: Optional[StakeholderView]) -> Dict[str, Any]:
        """Create compliance-focused dashboard for regulated domains."""
        
        widgets = []
        
        # Compliance score gauge
        widgets.append(self._create_gauge_widget(
            "compliance_score",
            "Overall Compliance Score",
            "compliance_rating",
            position={'x': 0, 'y': 0, 'width': 4, 'height': 2}
        ))
        
        # Regulatory requirements checklist
        widgets.append({
            'widget_id': 'regulatory_checklist',
            'title': f'{project_definition.domain.value.title()} Regulatory Compliance',
            'type': 'checklist_panel',
            'content': self._get_domain_compliance_requirements(project_definition.domain),
            'position': {'x': 4, 'y': 0, 'width': 8, 'height': 4}
        })
        
        # Audit trail
        widgets.append({
            'widget_id': 'audit_trail',
            'title': 'Audit Trail',
            'type': 'timeline_panel',
            'content': 'compliance_audit_events',
            'position': {'x': 0, 'y': 2, 'width': 4, 'height': 4}
        })
        
        return {
            'layout_type': 'compliance',
            'refresh_interval': 120,  # 2 minutes
            'widgets': widgets,
            'alert_preferences': {
                'compliance_critical': True,
                'audit_notifications': True,
                'notification_methods': ['email', 'dashboard', 'audit_log']
            }
        }
    
    # Widget creation helpers
    
    def _create_kpi_widget(self, widget_id: str, title: str, metric: str, position: Dict[str, int]) -> Dict[str, Any]:
        """Create KPI card widget."""
        return {
            'widget_id': widget_id,
            'title': title,
            'type': 'kpi_card',
            'data_source': metric,
            'position': position,
            'styling': {
                'show_trend': True,
                'show_target': True,
                'highlight_changes': True
            }
        }
    
    def _create_gauge_widget(self, widget_id: str, title: str, metric: str, position: Dict[str, int]) -> Dict[str, Any]:
        """Create gauge widget."""
        return {
            'widget_id': widget_id,
            'title': title,
            'type': 'gauge',
            'data_source': metric,
            'position': position,
            'styling': {
                'min_value': 0,
                'max_value': 100,
                'threshold_zones': [
                    {'min': 0, 'max': 30, 'color': 'red'},
                    {'min': 30, 'max': 70, 'color': 'yellow'},
                    {'min': 70, 'max': 100, 'color': 'green'}
                ]
            }
        }
    
    def _create_line_chart_widget(self, widget_id: str, title: str, metrics: List[str], position: Dict[str, int]) -> Dict[str, Any]:
        """Create line chart widget."""
        return {
            'widget_id': widget_id,
            'title': title,
            'type': 'line_chart',
            'data_sources': metrics,
            'position': position,
            'styling': {
                'show_legend': True,
                'show_grid': True,
                'time_window': '24h'
            }
        }
    
    def _create_bar_chart_widget(self, widget_id: str, title: str, metric: str, position: Dict[str, int]) -> Dict[str, Any]:
        """Create bar chart widget."""
        return {
            'widget_id': widget_id,
            'title': title,
            'type': 'bar_chart',
            'data_source': metric,
            'position': position,
            'styling': {
                'orientation': 'vertical',
                'show_values': True
            }
        }
    
    def _create_pie_chart_widget(self, widget_id: str, title: str, metric: str, position: Dict[str, int]) -> Dict[str, Any]:
        """Create pie chart widget."""
        return {
            'widget_id': widget_id,
            'title': title,
            'type': 'pie_chart',
            'data_source': metric,
            'position': position,
            'styling': {
                'show_legend': True,
                'show_percentages': True
            }
        }
    
    def _create_trend_widget(self, widget_id: str, title: str, metric: str, position: Dict[str, int]) -> Dict[str, Any]:
        """Create trend line widget."""
        return {
            'widget_id': widget_id,
            'title': title,
            'type': 'trend_line',
            'data_source': metric,
            'position': position,
            'styling': {
                'time_window': '7d',
                'show_trend_direction': True,
                'highlight_anomalies': True
            }
        }
    
    # Helper methods
    
    def _setup_real_time_monitoring(self, project_definition: ProjectDefinition, business_metrics: List[BusinessMetric]) -> Dict[str, Any]:
        """Setup real-time monitoring configuration."""
        return {
            'enabled': True,
            'polling_interval_seconds': 30,
            'metrics_monitored': [m.metric_name for m in business_metrics],
            'alert_thresholds': self._calculate_alert_thresholds(business_metrics),
            'data_retention_days': 90
        }
    
    def _initialize_alert_system(self, business_metrics: List[BusinessMetric], validation_summary: ValidationSummary) -> Dict[str, Any]:
        """Initialize alert system configuration."""
        return {
            'enabled': True,
            'alert_channels': ['dashboard', 'email'],
            'severity_escalation': {
                'info': 0,
                'warning': 300,  # 5 minutes
                'critical': 60   # 1 minute
            },
            'auto_resolution': True,
            'alert_templates': self._create_alert_templates()
        }
    
    def _calculate_metric_trend(self, metric_name: str) -> MetricTrend:
        """Calculate trend for a specific metric."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 3:
            return MetricTrend.INSUFFICIENT_DATA
        
        recent_values = [val for _, val in self.metric_history[metric_name][-5:]]
        
        # Simple trend calculation
        if len(recent_values) >= 3:
            first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
            second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
            
            change_percent = abs(second_half - first_half) / first_half if first_half != 0 else 0
            
            if change_percent > 0.1:  # 10% change threshold
                return MetricTrend.IMPROVING if second_half > first_half else MetricTrend.DECLINING
            elif change_percent > 0.05:  # 5% volatility threshold
                return MetricTrend.VOLATILE
            else:
                return MetricTrend.STABLE
        
        return MetricTrend.INSUFFICIENT_DATA
    
    def _check_metric_alerts(self, new_metrics: Dict[str, Any], trend_changes: Dict[str, MetricTrend]) -> List[DashboardAlert]:
        """Check for alert conditions in new metrics."""
        alerts = []
        
        for metric_name, value in new_metrics.items():
            # Check threshold alerts
            if isinstance(value, (int, float)):
                if value < 0.5:  # Generic low performance alert
                    alerts.append(DashboardAlert(
                        alert_id=f"low_{metric_name}_{int(time.time())}",
                        severity=AlertSeverity.WARNING,
                        title=f"Low {metric_name.replace('_', ' ').title()}",
                        message=f"{metric_name} has dropped to {value:.2f}",
                        metric_name=metric_name,
                        threshold_value=0.5,
                        current_value=value,
                        timestamp=datetime.now()
                    ))
            
            # Check trend alerts
            if trend_changes.get(metric_name) == MetricTrend.DECLINING:
                alerts.append(DashboardAlert(
                    alert_id=f"declining_{metric_name}_{int(time.time())}",
                    severity=AlertSeverity.INFO,
                    title=f"Declining Trend in {metric_name.replace('_', ' ').title()}",
                    message=f"{metric_name} showing declining trend",
                    metric_name=metric_name,
                    threshold_value=None,
                    current_value=value if isinstance(value, (int, float)) else 0,
                    timestamp=datetime.now()
                ))
        
        return alerts
    
    def _generate_update_narrative(self, new_metrics: Dict[str, Any], trend_changes: Dict[str, MetricTrend], alerts: List[DashboardAlert]) -> str:
        """Generate narrative summary of dashboard update."""
        
        narrative = f"Dashboard updated with {len(new_metrics)} metrics. "
        
        improving_count = len([t for t in trend_changes.values() if t == MetricTrend.IMPROVING])
        declining_count = len([t for t in trend_changes.values() if t == MetricTrend.DECLINING])
        
        if improving_count > declining_count:
            narrative += f"Overall positive trends with {improving_count} improving metrics. "
        elif declining_count > improving_count:
            narrative += f"Performance concerns with {declining_count} declining metrics. "
        else:
            narrative += "Stable performance indicators across tracked metrics. "
        
        if alerts:
            critical_alerts = len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
            if critical_alerts > 0:
                narrative += f"ATTENTION: {critical_alerts} critical alerts require immediate attention."
            else:
                narrative += f"{len(alerts)} alerts generated for monitoring."
        
        return narrative
    
    def _generate_stakeholder_notifications(self, new_metrics: Dict[str, Any], 
                                          trend_changes: Dict[str, MetricTrend], 
                                          alerts: List[DashboardAlert],
                                          dashboard_id: str) -> Dict[str, List[str]]:
        """Generate stakeholder-specific notifications."""
        
        notifications = {
            'executive': [],
            'technical': [],
            'business': [],
            'compliance': []
        }
        
        # Executive notifications (critical only)
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            notifications['executive'].extend([
                f"Critical Alert: {alert.title}" for alert in critical_alerts[:2]
            ])
        
        # Technical notifications (all alerts)
        notifications['technical'].extend([
            f"{alert.severity.value.upper()}: {alert.title}" for alert in alerts
        ])
        
        # Business notifications (business-relevant metrics)
        business_metrics = [k for k in new_metrics.keys() if 'roi' in k.lower() or 'revenue' in k.lower() or 'customer' in k.lower()]
        if business_metrics:
            notifications['business'].append(f"Business metrics updated: {', '.join(business_metrics)}")
        
        # Compliance notifications (compliance-related alerts)
        compliance_alerts = [a for a in alerts if 'compliance' in a.message.lower() or 'regulatory' in a.message.lower()]
        if compliance_alerts:
            notifications['compliance'].extend([
                f"Compliance Alert: {alert.title}" for alert in compliance_alerts
            ])
        
        return notifications
    
    def _initialize_default_layouts(self):
        """Initialize default dashboard layouts for different stakeholders."""
        
        default_layouts = {
            'executive': self._create_default_executive_layout(),
            'technical': self._create_default_technical_layout(),
            'business': self._create_default_business_layout(),
            'compliance': self._create_default_compliance_layout()
        }
        
        self.dashboard_layouts.update(default_layouts)
    
    def _create_default_executive_layout(self) -> DashboardLayout:
        """Create default executive dashboard layout."""
        return DashboardLayout(
            layout_id='default_executive',
            stakeholder_type='executive',
            widgets=[],
            refresh_policy={'default': 300},
            alert_preferences={'critical_only': True}
        )
    
    def _create_default_technical_layout(self) -> DashboardLayout:
        """Create default technical dashboard layout."""
        return DashboardLayout(
            layout_id='default_technical',
            stakeholder_type='technical',
            widgets=[],
            refresh_policy={'default': 60},
            alert_preferences={'all_severities': True}
        )
    
    def _create_default_business_layout(self) -> DashboardLayout:
        """Create default business dashboard layout."""
        return DashboardLayout(
            layout_id='default_business',
            stakeholder_type='business',
            widgets=[],
            refresh_policy={'default': 180},
            alert_preferences={'business_focus': True}
        )
    
    def _create_default_compliance_layout(self) -> DashboardLayout:
        """Create default compliance dashboard layout."""
        return DashboardLayout(
            layout_id='default_compliance',
            stakeholder_type='compliance',
            widgets=[],
            refresh_policy={'default': 120},
            alert_preferences={'compliance_critical': True}
        )
    
    def _get_refresh_intervals(self) -> Dict[str, int]:
        """Get refresh intervals for different dashboard types."""
        return {
            'executive': 300,    # 5 minutes
            'technical': 60,     # 1 minute
            'business': 180,     # 3 minutes
            'compliance': 120    # 2 minutes
        }
    
    def _map_data_sources(self, business_metrics: List[BusinessMetric]) -> Dict[str, str]:
        """Map business metrics to data source identifiers."""
        return {
            metric.metric_name: f"business_metric_{metric.metric_name}"
            for metric in business_metrics
        }
    
    def _calculate_alert_thresholds(self, business_metrics: List[BusinessMetric]) -> Dict[str, Dict[str, float]]:
        """Calculate alert thresholds for business metrics."""
        thresholds = {}
        
        for metric in business_metrics:
            thresholds[metric.metric_name] = {
                'critical_low': 0.3,
                'warning_low': 0.5,
                'warning_high': 1.2,
                'critical_high': 1.5
            }
        
        return thresholds
    
    def _create_alert_templates(self) -> Dict[str, str]:
        """Create alert message templates."""
        return {
            'performance_degradation': "{metric_name} has degraded to {current_value:.2f} (threshold: {threshold})",
            'threshold_breach': "{metric_name} breached {severity} threshold: {current_value:.2f}",
            'trend_alert': "{metric_name} showing {trend} trend over last {time_period}"
        }
    
    def _get_domain_compliance_requirements(self, domain: Domain) -> List[Dict[str, Any]]:
        """Get domain-specific compliance requirements."""
        
        requirements = {
            Domain.FINANCE: [
                {'requirement': 'Model Interpretability', 'status': 'pending', 'regulation': 'Basel III'},
                {'requirement': 'Bias Testing', 'status': 'pending', 'regulation': 'Fair Credit Reporting Act'},
                {'requirement': 'Data Governance', 'status': 'pending', 'regulation': 'SOX'},
                {'requirement': 'Audit Trail', 'status': 'pending', 'regulation': 'MiFID II'}
            ],
            Domain.HEALTHCARE: [
                {'requirement': 'Data Privacy', 'status': 'pending', 'regulation': 'HIPAA'},
                {'requirement': 'Clinical Validation', 'status': 'pending', 'regulation': 'FDA 21 CFR Part 820'},
                {'requirement': 'Patient Safety', 'status': 'pending', 'regulation': 'IEC 62304'},
                {'requirement': 'Quality Management', 'status': 'pending', 'regulation': 'ISO 13485'}
            ]
        }
        
        return requirements.get(domain, [])
    
    def _filter_metrics_by_time_range(self, dashboard_id: str, time_range: Optional[Tuple[datetime, datetime]]) -> Dict[str, Any]:
        """Filter metrics by specified time range."""
        
        if not time_range:
            # Return latest metrics
            return self.dashboard_state.get(dashboard_id, {}).get('current_metrics', {})
        
        start_time, end_time = time_range
        filtered_metrics = {}
        
        for metric_name, history in self.metric_history.items():
            filtered_history = [
                (timestamp, value) for timestamp, value in history
                if start_time <= timestamp <= end_time
            ]
            if filtered_history:
                filtered_metrics[metric_name] = filtered_history[-1][1]  # Latest value in range
        
        return filtered_metrics
    
    def _filter_alerts_by_stakeholder(self, stakeholder_type: str) -> List[DashboardAlert]:
        """Filter alerts relevant to specific stakeholder type."""
        
        # Stakeholder-specific alert filtering logic
        if stakeholder_type == 'executive':
            return [alert for alert in self.active_alerts.values() if alert.severity == AlertSeverity.CRITICAL]
        elif stakeholder_type == 'technical':
            return list(self.active_alerts.values())
        elif stakeholder_type == 'compliance':
            return [
                alert for alert in self.active_alerts.values() 
                if 'compliance' in alert.message.lower() or 'regulatory' in alert.message.lower()
            ]
        else:
            return [alert for alert in self.active_alerts.values() if alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]]
    
    def _generate_real_time_insights(self, filtered_metrics: Dict[str, Any], stakeholder_type: str) -> List[str]:
        """Generate real-time insights for stakeholder."""
        
        insights = []
        
        if stakeholder_type == 'executive':
            insights.append("Strategic performance indicators within expected ranges")
            if any(v < 0.7 for v in filtered_metrics.values() if isinstance(v, (int, float))):
                insights.append("Some performance metrics below optimal levels - review recommended")
        
        elif stakeholder_type == 'technical':
            insights.append(f"Monitoring {len(filtered_metrics)} performance metrics")
            avg_performance = sum(v for v in filtered_metrics.values() if isinstance(v, (int, float))) / max(1, len(filtered_metrics))
            insights.append(f"Average performance score: {avg_performance:.2f}")
        
        return insights
    
    def _get_next_refresh_time(self, stakeholder_type: str) -> str:
        """Get next scheduled refresh time for stakeholder dashboard."""
        
        intervals = self._get_refresh_intervals()
        interval = intervals.get(stakeholder_type, 300)
        next_refresh = datetime.now() + timedelta(seconds=interval)
        return next_refresh.isoformat()
    
    def _get_optimal_refresh_interval(self, metric_name: str) -> int:
        """Get optimal refresh interval for specific metric."""
        
        # Dynamic refresh intervals based on metric volatility
        if metric_name in self.metric_history:
            volatility = self._calculate_metric_volatility(metric_name)
            if volatility > 0.2:
                return 30  # High volatility - refresh every 30 seconds
            elif volatility > 0.1:
                return 60  # Medium volatility - refresh every minute
            else:
                return 300  # Low volatility - refresh every 5 minutes
        
        return 120  # Default 2 minute refresh
    
    def _calculate_metric_volatility(self, metric_name: str) -> float:
        """Calculate volatility for a metric based on historical data."""
        
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 5:
            return 0.0
        
        values = [val for _, val in self.metric_history[metric_name][-10:]]
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((val - mean) ** 2 for val in values) / len(values)
        std_dev = variance ** 0.5
        
        # Return coefficient of variation as volatility measure
        return std_dev / mean if mean != 0 else 0.0