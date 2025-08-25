import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .business_translator import BusinessTranslator, BusinessMetric, StakeholderView
from .narrative_generator import NarrativeGenerator, ExecutiveSummary, TechnicalReport, ReportSection, ReportTone
from .dashboard_engine import DashboardEngine, DashboardUpdate, MetricTrend, DashboardAlert
from .stakeholder_reporter import StakeholderReporter, ReportFormat, AudienceType, ReportFrequency, DeliveryChannel, GeneratedReport
from ..core.project_definition import ProjectDefinition, Objective, Domain
from ..validation.validation_orchestrator import ValidationSummary

class InsightPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class InsightTrigger(Enum):
    VALIDATION_COMPLETE = "validation_complete"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    SCHEDULE_BASED = "schedule_based"
    STAKEHOLDER_REQUEST = "stakeholder_request"
    ANOMALY_DETECTED = "anomaly_detected"
    COMPLIANCE_ALERT = "compliance_alert"

class CommunicationMode(Enum):
    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    SCHEDULED = "scheduled"
    REAL_TIME = "real_time"

@dataclass
class InsightSession:
    session_id: str
    project_definition: ProjectDefinition
    validation_summary: ValidationSummary
    business_metrics: List[BusinessMetric]
    stakeholder_views: Dict[str, StakeholderView]
    created_at: datetime
    last_updated: datetime
    active_dashboards: List[str] = field(default_factory=list)
    generated_reports: List[str] = field(default_factory=list)
    communication_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class InsightAction:
    action_id: str
    trigger: InsightTrigger
    priority: InsightPriority
    target_stakeholders: List[AudienceType]
    action_type: str  # "report", "dashboard_update", "alert", "notification"
    content: Dict[str, Any]
    scheduled_time: Optional[datetime] = None
    execution_status: str = "pending"  # "pending", "executed", "failed"
    execution_result: Optional[Dict[str, Any]] = None

class InsightOrchestrator:
    """
    Production-grade insight orchestrator that coordinates all business intelligence
    components to deliver comprehensive, stakeholder-specific insights.
    """
    
    def __init__(self,
                 business_translator: Optional[BusinessTranslator] = None,
                 narrative_generator: Optional[NarrativeGenerator] = None,
                 dashboard_engine: Optional[DashboardEngine] = None,
                 stakeholder_reporter: Optional[StakeholderReporter] = None):
        
        # Initialize core components
        self.business_translator = business_translator or BusinessTranslator()
        self.narrative_generator = narrative_generator or NarrativeGenerator(self.business_translator)
        self.dashboard_engine = dashboard_engine or DashboardEngine(self.business_translator, self.narrative_generator)
        self.stakeholder_reporter = stakeholder_reporter or StakeholderReporter(
            self.business_translator, self.narrative_generator, self.dashboard_engine
        )
        
        # Session and action management
        self.active_sessions: Dict[str, InsightSession] = {}
        self.pending_actions: Dict[str, InsightAction] = {}
        self.action_history: List[InsightAction] = []
        
        # Communication orchestration state
        self.communication_preferences: Dict[str, Dict[str, Any]] = {}
        self.insight_triggers: Dict[InsightTrigger, List[callable]] = {}
        self.stakeholder_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default triggers and preferences
        self._initialize_default_triggers()
        self._initialize_default_preferences()
        
    def orchestrate_comprehensive_insights(self,
                                         project_definition: ProjectDefinition,
                                         validation_summary: ValidationSummary,
                                         model_metadata: Optional[Dict[str, Any]] = None,
                                         stakeholder_requests: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Orchestrate comprehensive insight generation and delivery across all stakeholders.
        Returns session_id for tracking.
        """
        
        session_id = f"insight_session_{int(time.time())}"
        
        # Step 1: Translate technical results to business metrics
        business_metrics = self.business_translator.translate_validation_results(
            project_definition, validation_summary, model_metadata
        )
        
        # Step 2: Generate stakeholder-specific views
        stakeholder_views = self._generate_all_stakeholder_views(
            project_definition, validation_summary, business_metrics, stakeholder_requests
        )
        
        # Step 3: Create insight session
        insight_session = InsightSession(
            session_id=session_id,
            project_definition=project_definition,
            validation_summary=validation_summary,
            business_metrics=business_metrics,
            stakeholder_views=stakeholder_views,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Step 4: Create real-time dashboards for all stakeholders
        dashboard_config = self.dashboard_engine.create_real_time_dashboard(
            project_definition, validation_summary, business_metrics, stakeholder_views
        )
        insight_session.active_dashboards.append(dashboard_config['dashboard_id'])
        
        # Step 5: Generate initial reports for key stakeholders
        initial_reports = self._generate_initial_reports(
            project_definition, validation_summary, business_metrics, stakeholder_views
        )
        insight_session.generated_reports.extend([report.report_id for report in initial_reports])
        
        # Step 6: Set up proactive monitoring and communication
        self._setup_proactive_monitoring(session_id, project_definition, validation_summary)
        
        # Step 7: Execute immediate high-priority actions
        immediate_actions = self._identify_immediate_actions(
            project_definition, validation_summary, business_metrics, stakeholder_views
        )
        
        for action in immediate_actions:
            self._execute_insight_action(action, insight_session)
        
        # Store session
        self.active_sessions[session_id] = insight_session
        
        # Log orchestration event
        self._log_communication_event(session_id, "comprehensive_insights_initiated", {
            'stakeholder_count': len(stakeholder_views),
            'business_metrics_count': len(business_metrics),
            'dashboard_created': True,
            'initial_reports_count': len(initial_reports)
        })
        
        return session_id
    
    def update_insights_with_new_data(self,
                                    session_id: str,
                                    new_validation_summary: Optional[ValidationSummary] = None,
                                    new_metrics: Optional[Dict[str, Any]] = None,
                                    model_updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update existing insight session with new data and trigger appropriate communications."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Update session data
        if new_validation_summary:
            session.validation_summary = new_validation_summary
        
        session.last_updated = datetime.now()
        
        # Update business metrics if new data available
        if new_metrics or new_validation_summary:
            updated_business_metrics = self.business_translator.translate_validation_results(
                session.project_definition, 
                new_validation_summary or session.validation_summary,
                model_updates
            )
            session.business_metrics = updated_business_metrics
        
        # Update stakeholder views
        session.stakeholder_views = self._generate_all_stakeholder_views(
            session.project_definition, session.validation_summary, session.business_metrics
        )
        
        # Update dashboards with new metrics
        dashboard_updates = []
        for dashboard_id in session.active_dashboards:
            if new_metrics:
                dashboard_update = self.dashboard_engine.update_dashboard_metrics(
                    dashboard_id, new_metrics, new_validation_summary
                )
                dashboard_updates.append(dashboard_update)
        
        # Check for triggered actions
        triggered_actions = self._check_update_triggers(session, new_metrics, model_updates)
        
        # Execute triggered actions
        execution_results = []
        for action in triggered_actions:
            result = self._execute_insight_action(action, session)
            execution_results.append(result)
        
        # Log update event
        self._log_communication_event(session_id, "insights_updated", {
            'new_validation_summary': new_validation_summary is not None,
            'new_metrics_count': len(new_metrics) if new_metrics else 0,
            'dashboard_updates': len(dashboard_updates),
            'triggered_actions': len(triggered_actions)
        })
        
        return {
            'session_id': session_id,
            'update_timestamp': session.last_updated.isoformat(),
            'dashboard_updates': [update.update_id for update in dashboard_updates],
            'triggered_actions': [action.action_id for action in triggered_actions],
            'execution_results': execution_results
        }
    
    def request_stakeholder_specific_insight(self,
                                           session_id: str,
                                           stakeholder_type: AudienceType,
                                           insight_request: Dict[str, Any],
                                           delivery_preferences: Optional[Dict[str, Any]] = None) -> GeneratedReport:
        """Generate and deliver specific insight requested by stakeholder."""
        
        if session_id not in self.active_sessions:
            raise ValueError("Session not found")
        
        session = self.active_sessions[session_id]
        
        # Determine request type and parameters
        request_type = insight_request.get('type', 'general_report')
        format_preference = insight_request.get('format', ReportFormat.PDF)
        sections_requested = insight_request.get('sections', [])
        urgency = insight_request.get('urgency', 'normal')
        
        # Generate requested insight
        if request_type == 'executive_briefing':
            generated_report = self.stakeholder_reporter.generate_executive_briefing(
                session.project_definition, session.validation_summary,
                session.business_metrics, session.stakeholder_views
            )
        elif request_type == 'technical_deep_dive':
            generated_report = self.stakeholder_reporter.generate_technical_deep_dive(
                session.project_definition, session.validation_summary,
                session.business_metrics, session.stakeholder_views
            )
        elif request_type == 'compliance_report':
            generated_report = self.stakeholder_reporter.generate_compliance_report(
                session.project_definition, session.validation_summary,
                session.business_metrics, session.stakeholder_views
            )
        else:
            # Generate custom report
            custom_sections = [ReportSection(s) for s in sections_requested] if sections_requested else None
            generated_report = self.stakeholder_reporter.generate_comprehensive_report(
                session.project_definition, session.validation_summary,
                session.business_metrics, session.stakeholder_views,
                stakeholder_type, format_preference, custom_sections
            )
        
        # Deliver according to preferences
        if delivery_preferences:
            delivery_channels = [DeliveryChannel(ch) for ch in delivery_preferences.get('channels', ['dashboard'])]
            recipients = delivery_preferences.get('recipients', [stakeholder_type.value])
            
            delivery_results = self.stakeholder_reporter.deliver_report(
                generated_report, delivery_channels, recipients, delivery_preferences
            )
        
        # Update session
        session.generated_reports.append(generated_report.report_id)
        session.last_updated = datetime.now()
        
        # Log stakeholder request
        self._log_communication_event(session_id, "stakeholder_request_fulfilled", {
            'stakeholder_type': stakeholder_type.value,
            'request_type': request_type,
            'report_id': generated_report.report_id,
            'urgency': urgency
        })
        
        return generated_report
    
    def setup_intelligent_monitoring(self,
                                   session_id: str,
                                   monitoring_config: Dict[str, Any]) -> Dict[str, str]:
        """Setup intelligent monitoring with automated insight triggers."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Configure performance monitoring
        performance_thresholds = monitoring_config.get('performance_thresholds', {})
        self._setup_performance_monitoring(session_id, performance_thresholds)
        
        # Configure anomaly detection
        anomaly_config = monitoring_config.get('anomaly_detection', {})
        self._setup_anomaly_detection(session_id, anomaly_config)
        
        # Configure stakeholder notifications
        notification_config = monitoring_config.get('notifications', {})
        self._setup_stakeholder_notifications(session_id, notification_config)
        
        # Configure automated reporting schedules
        reporting_schedules = monitoring_config.get('reporting_schedules', {})
        self._setup_automated_reporting(session_id, reporting_schedules)
        
        return {
            'session_id': session_id,
            'monitoring_status': 'active',
            'performance_monitoring': 'enabled' if performance_thresholds else 'disabled',
            'anomaly_detection': 'enabled' if anomaly_config else 'disabled',
            'stakeholder_notifications': 'configured',
            'automated_reporting': 'scheduled' if reporting_schedules else 'disabled'
        }
    
    def get_insight_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of insight session."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Calculate session metrics
        session_duration = (datetime.now() - session.created_at).total_seconds()
        communication_events = len(session.communication_log)
        
        # Get dashboard status
        dashboard_status = {}
        for dashboard_id in session.active_dashboards:
            dashboard_status[dashboard_id] = 'active'  # Would check actual status in production
        
        # Get report status
        report_status = {
            report_id: self.stakeholder_reporter.generated_reports.get(report_id, {}).get('status', 'unknown')
            for report_id in session.generated_reports
        }
        
        # Calculate stakeholder engagement
        stakeholder_engagement = self._calculate_stakeholder_engagement(session)
        
        # Get pending actions
        pending_actions = [
            {
                'action_id': action.action_id,
                'trigger': action.trigger.value,
                'priority': action.priority.value,
                'target_stakeholders': [st.value for st in action.target_stakeholders],
                'scheduled_time': action.scheduled_time.isoformat() if action.scheduled_time else None
            }
            for action in self.pending_actions.values()
            if any(session_id in action.content.get('session_ids', [session_id]))
        ]
        
        return {
            'session_id': session_id,
            'project_context': {
                'domain': session.project_definition.domain.value,
                'objective': session.project_definition.objective.value
            },
            'session_metrics': {
                'created_at': session.created_at.isoformat(),
                'last_updated': session.last_updated.isoformat(),
                'duration_hours': round(session_duration / 3600, 2),
                'communication_events': communication_events
            },
            'validation_status': {
                'overall_success': session.validation_summary.overall_success,
                'primary_objective_met': session.validation_summary.primary_objective_met,
                'validation_confidence': session.validation_summary.validation_confidence,
                'decision_support_score': session.validation_summary.decision_support_score
            },
            'business_intelligence': {
                'business_metrics_tracked': len(session.business_metrics),
                'stakeholder_views_generated': len(session.stakeholder_views),
                'primary_metrics': len([m for m in session.business_metrics if m.is_primary]),
                'high_impact_metrics': len([m for m in session.business_metrics if m.business_impact and 'high' in m.business_impact.lower()])
            },
            'dashboard_status': dashboard_status,
            'report_status': report_status,
            'stakeholder_engagement': stakeholder_engagement,
            'pending_actions': pending_actions,
            'communication_summary': self._get_communication_summary(session)
        }
    
    def execute_cross_stakeholder_communication(self,
                                              session_id: str,
                                              communication_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordinated communication across multiple stakeholders."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Parse communication plan
        target_stakeholders = [AudienceType(st) for st in communication_plan.get('stakeholders', [])]
        message_type = communication_plan.get('message_type', 'update')
        priority = InsightPriority(communication_plan.get('priority', 'medium'))
        coordination_strategy = communication_plan.get('coordination', 'sequential')  # sequential, parallel, cascading
        
        # Generate stakeholder-specific content
        communication_content = {}
        for stakeholder_type in target_stakeholders:
            content = self._generate_stakeholder_communication(
                session, stakeholder_type, message_type, priority
            )
            communication_content[stakeholder_type.value] = content
        
        # Execute communication based on strategy
        execution_results = []
        
        if coordination_strategy == 'parallel':
            # Execute all communications simultaneously
            for stakeholder_type in target_stakeholders:
                result = self._execute_stakeholder_communication(
                    session, stakeholder_type, communication_content[stakeholder_type.value]
                )
                execution_results.append(result)
        
        elif coordination_strategy == 'cascading':
            # Executive first, then others based on hierarchy
            stakeholder_hierarchy = [AudienceType.EXECUTIVE, AudienceType.BUSINESS, AudienceType.TECHNICAL, AudienceType.COMPLIANCE]
            for stakeholder_type in stakeholder_hierarchy:
                if stakeholder_type in target_stakeholders:
                    result = self._execute_stakeholder_communication(
                        session, stakeholder_type, communication_content[stakeholder_type.value]
                    )
                    execution_results.append(result)
                    # Add delay for cascading effect in production
        
        else:  # sequential
            for stakeholder_type in target_stakeholders:
                result = self._execute_stakeholder_communication(
                    session, stakeholder_type, communication_content[stakeholder_type.value]
                )
                execution_results.append(result)
        
        # Log cross-stakeholder communication
        self._log_communication_event(session_id, "cross_stakeholder_communication", {
            'stakeholder_count': len(target_stakeholders),
            'message_type': message_type,
            'priority': priority.value,
            'coordination_strategy': coordination_strategy,
            'execution_results': len(execution_results)
        })
        
        return {
            'session_id': session_id,
            'communication_id': f"cross_comm_{int(time.time())}",
            'target_stakeholders': [st.value for st in target_stakeholders],
            'execution_strategy': coordination_strategy,
            'execution_results': execution_results,
            'completion_timestamp': datetime.now().isoformat()
        }
    
    # Private helper methods
    
    def _generate_all_stakeholder_views(self,
                                      project_definition: ProjectDefinition,
                                      validation_summary: ValidationSummary,
                                      business_metrics: List[BusinessMetric],
                                      specific_requests: Optional[Dict[str, List[str]]] = None) -> Dict[str, StakeholderView]:
        """Generate views for all relevant stakeholders."""
        
        stakeholder_views = {}
        
        # Always generate core stakeholder views
        core_stakeholders = [AudienceType.EXECUTIVE, AudienceType.TECHNICAL, AudienceType.BUSINESS]
        
        # Add compliance if needed for regulated domains
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            core_stakeholders.append(AudienceType.COMPLIANCE)
        
        for stakeholder_type in core_stakeholders:
            view = self.business_translator.generate_stakeholder_view(
                stakeholder_type.value, business_metrics, project_definition, validation_summary
            )
            
            # Apply specific requests if provided
            if specific_requests and stakeholder_type.value in specific_requests:
                view = self._customize_stakeholder_view(view, specific_requests[stakeholder_type.value])
            
            stakeholder_views[stakeholder_type.value] = view
        
        return stakeholder_views
    
    def _customize_stakeholder_view(self, view: StakeholderView, requests: List[str]) -> StakeholderView:
        """Customize stakeholder view based on specific requests."""
        
        # Add requested focus areas
        for request in requests:
            if request not in view.focus_areas:
                view.focus_areas.append(request)
        
        # Add request-specific insights
        view.key_insights.extend([f"Requested analysis: {req}" for req in requests[:2]])
        
        return view
    
    def _generate_initial_reports(self,
                                project_definition: ProjectDefinition,
                                validation_summary: ValidationSummary,
                                business_metrics: List[BusinessMetric],
                                stakeholder_views: Dict[str, StakeholderView]) -> List[GeneratedReport]:
        """Generate initial reports for key stakeholders."""
        
        initial_reports = []
        
        # Executive briefing
        if 'executive' in stakeholder_views:
            executive_report = self.stakeholder_reporter.generate_executive_briefing(
                project_definition, validation_summary, business_metrics, stakeholder_views
            )
            initial_reports.append(executive_report)
        
        # Technical deep dive if technical stakeholder exists
        if 'technical_lead' in stakeholder_views:
            technical_report = self.stakeholder_reporter.generate_technical_deep_dive(
                project_definition, validation_summary, business_metrics, stakeholder_views
            )
            initial_reports.append(technical_report)
        
        # Compliance report for regulated domains
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE] and 'compliance_officer' in stakeholder_views:
            compliance_report = self.stakeholder_reporter.generate_compliance_report(
                project_definition, validation_summary, business_metrics, stakeholder_views
            )
            initial_reports.append(compliance_report)
        
        return initial_reports
    
    def _setup_proactive_monitoring(self,
                                  session_id: str,
                                  project_definition: ProjectDefinition,
                                  validation_summary: ValidationSummary):
        """Setup proactive monitoring and alerting."""
        
        # Set up performance threshold monitoring
        if not validation_summary.overall_success:
            self._create_insight_action(
                InsightTrigger.PERFORMANCE_THRESHOLD,
                InsightPriority.HIGH,
                [AudienceType.TECHNICAL, AudienceType.EXECUTIVE],
                "alert",
                {
                    'session_id': session_id,
                    'message': 'Primary objective not met - requires attention',
                    'threshold': 'validation_success'
                }
            )
        
        # Set up compliance monitoring for regulated domains
        if project_definition.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            self._create_insight_action(
                InsightTrigger.COMPLIANCE_ALERT,
                InsightPriority.CRITICAL,
                [AudienceType.COMPLIANCE, AudienceType.EXECUTIVE],
                "compliance_check",
                {
                    'session_id': session_id,
                    'domain': project_definition.domain.value,
                    'scheduled_time': datetime.now() + timedelta(hours=24)
                }
            )
    
    def _identify_immediate_actions(self,
                                  project_definition: ProjectDefinition,
                                  validation_summary: ValidationSummary,
                                  business_metrics: List[BusinessMetric],
                                  stakeholder_views: Dict[str, StakeholderView]) -> List[InsightAction]:
        """Identify actions that should be executed immediately."""
        
        immediate_actions = []
        
        # Critical validation failures
        if not validation_summary.overall_success:
            immediate_actions.append(self._create_insight_action(
                InsightTrigger.VALIDATION_COMPLETE,
                InsightPriority.CRITICAL,
                [AudienceType.EXECUTIVE, AudienceType.TECHNICAL],
                "alert",
                {
                    'message': 'Validation failed - immediate review required',
                    'validation_confidence': validation_summary.validation_confidence,
                    'failed_objectives': 'primary_objective' if not validation_summary.primary_objective_met else 'secondary_objectives'
                }
            ))
        
        # Low confidence scores
        if validation_summary.validation_confidence < 0.7:
            immediate_actions.append(self._create_insight_action(
                InsightTrigger.PERFORMANCE_THRESHOLD,
                InsightPriority.HIGH,
                [AudienceType.TECHNICAL],
                "notification",
                {
                    'message': f'Low validation confidence: {validation_summary.validation_confidence:.1%}',
                    'recommended_actions': validation_summary.technical_optimizations
                }
            ))
        
        # High-impact metrics below threshold
        critical_metrics = [m for m in business_metrics if m.is_primary and m.current_value < 0.6]
        if critical_metrics:
            immediate_actions.append(self._create_insight_action(
                InsightTrigger.PERFORMANCE_THRESHOLD,
                InsightPriority.HIGH,
                [AudienceType.BUSINESS, AudienceType.TECHNICAL],
                "performance_alert",
                {
                    'metrics': [{'name': m.metric_name, 'value': m.current_value} for m in critical_metrics],
                    'impact_assessment': 'high'
                }
            ))
        
        return immediate_actions
    
    def _create_insight_action(self,
                             trigger: InsightTrigger,
                             priority: InsightPriority,
                             target_stakeholders: List[AudienceType],
                             action_type: str,
                             content: Dict[str, Any],
                             scheduled_time: Optional[datetime] = None) -> InsightAction:
        """Create insight action for execution."""
        
        action_id = f"action_{trigger.value}_{int(time.time())}"
        
        action = InsightAction(
            action_id=action_id,
            trigger=trigger,
            priority=priority,
            target_stakeholders=target_stakeholders,
            action_type=action_type,
            content=content,
            scheduled_time=scheduled_time
        )
        
        # Store for tracking
        self.pending_actions[action_id] = action
        
        return action
    
    def _execute_insight_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute specific insight action."""
        
        execution_result = {
            'action_id': action.action_id,
            'execution_timestamp': datetime.now().isoformat(),
            'status': 'success',
            'results': {}
        }
        
        try:
            if action.action_type == "alert":
                result = self._execute_alert_action(action, session)
            elif action.action_type == "notification":
                result = self._execute_notification_action(action, session)
            elif action.action_type == "report":
                result = self._execute_report_action(action, session)
            elif action.action_type == "dashboard_update":
                result = self._execute_dashboard_update_action(action, session)
            elif action.action_type == "compliance_check":
                result = self._execute_compliance_check_action(action, session)
            elif action.action_type == "performance_alert":
                result = self._execute_performance_alert_action(action, session)
            else:
                result = {'message': f'Unknown action type: {action.action_type}'}
            
            execution_result['results'] = result
            action.execution_status = 'executed'
            action.execution_result = result
            
        except Exception as e:
            execution_result['status'] = 'failed'
            execution_result['error'] = str(e)
            action.execution_status = 'failed'
        
        # Remove from pending and add to history
        if action.action_id in self.pending_actions:
            del self.pending_actions[action.action_id]
        self.action_history.append(action)
        
        return execution_result
    
    def _execute_alert_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute alert action."""
        
        alert_message = action.content.get('message', 'System alert')
        
        # Create dashboard alerts for target stakeholders
        for stakeholder in action.target_stakeholders:
            dashboard_id = session.active_dashboards[0] if session.active_dashboards else None
            if dashboard_id:
                # In production, would create actual dashboard alert
                pass
        
        return {
            'alert_type': 'system_alert',
            'message': alert_message,
            'stakeholders_notified': [st.value for st in action.target_stakeholders],
            'priority': action.priority.value
        }
    
    def _execute_notification_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute notification action."""
        
        notification_message = action.content.get('message', 'System notification')
        
        return {
            'notification_type': 'stakeholder_notification',
            'message': notification_message,
            'delivery_channels': ['dashboard', 'email'],
            'stakeholders': [st.value for st in action.target_stakeholders]
        }
    
    def _execute_report_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute report generation action."""
        
        report_type = action.content.get('report_type', 'summary')
        
        # Generate appropriate report
        if report_type == 'executive_briefing':
            report = self.stakeholder_reporter.generate_executive_briefing(
                session.project_definition, session.validation_summary,
                session.business_metrics, session.stakeholder_views
            )
        else:
            # Generate general report
            report = self.stakeholder_reporter.generate_comprehensive_report(
                session.project_definition, session.validation_summary,
                session.business_metrics, session.stakeholder_views,
                action.target_stakeholders[0], ReportFormat.PDF
            )
        
        return {
            'report_generated': True,
            'report_id': report.report_id,
            'report_type': report_type,
            'stakeholders': [st.value for st in action.target_stakeholders]
        }
    
    def _execute_dashboard_update_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute dashboard update action."""
        
        new_metrics = action.content.get('metrics', {})
        
        update_results = []
        for dashboard_id in session.active_dashboards:
            if new_metrics:
                dashboard_update = self.dashboard_engine.update_dashboard_metrics(
                    dashboard_id, new_metrics
                )
                update_results.append(dashboard_update.update_id)
        
        return {
            'dashboards_updated': len(update_results),
            'update_ids': update_results,
            'metrics_updated': list(new_metrics.keys()) if new_metrics else []
        }
    
    def _execute_compliance_check_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute compliance check action."""
        
        domain = action.content.get('domain', session.project_definition.domain.value)
        
        # Generate compliance report
        compliance_report = self.stakeholder_reporter.generate_compliance_report(
            session.project_definition, session.validation_summary,
            session.business_metrics, session.stakeholder_views
        )
        
        return {
            'compliance_check_completed': True,
            'domain': domain,
            'report_id': compliance_report.report_id,
            'compliance_status': 'pending_review'
        }
    
    def _execute_performance_alert_action(self, action: InsightAction, session: InsightSession) -> Dict[str, Any]:
        """Execute performance alert action."""
        
        metrics = action.content.get('metrics', [])
        impact_level = action.content.get('impact_assessment', 'medium')
        
        return {
            'performance_alert_sent': True,
            'affected_metrics': [m['name'] for m in metrics],
            'impact_level': impact_level,
            'stakeholders_notified': [st.value for st in action.target_stakeholders]
        }
    
    def _check_update_triggers(self,
                             session: InsightSession,
                             new_metrics: Optional[Dict[str, Any]],
                             model_updates: Optional[Dict[str, Any]]) -> List[InsightAction]:
        """Check for triggered actions based on updates."""
        
        triggered_actions = []
        
        # Check metric threshold triggers
        if new_metrics:
            for metric_name, value in new_metrics.items():
                if isinstance(value, (int, float)):
                    # Low performance trigger
                    if value < 0.5:
                        triggered_actions.append(self._create_insight_action(
                            InsightTrigger.PERFORMANCE_THRESHOLD,
                            InsightPriority.HIGH,
                            [AudienceType.TECHNICAL],
                            "notification",
                            {
                                'session_id': session.session_id,
                                'metric_name': metric_name,
                                'current_value': value,
                                'threshold': 0.5
                            }
                        ))
                    
                    # Critical performance trigger
                    if value < 0.3:
                        triggered_actions.append(self._create_insight_action(
                            InsightTrigger.PERFORMANCE_THRESHOLD,
                            InsightPriority.CRITICAL,
                            [AudienceType.EXECUTIVE, AudienceType.TECHNICAL],
                            "alert",
                            {
                                'session_id': session.session_id,
                                'metric_name': metric_name,
                                'current_value': value,
                                'severity': 'critical'
                            }
                        ))
        
        # Check validation status triggers
        if not session.validation_summary.overall_success:
            triggered_actions.append(self._create_insight_action(
                InsightTrigger.VALIDATION_COMPLETE,
                InsightPriority.HIGH,
                [AudienceType.TECHNICAL, AudienceType.BUSINESS],
                "report",
                {
                    'session_id': session.session_id,
                    'report_type': 'failure_analysis'
                }
            ))
        
        return triggered_actions
    
    def _generate_stakeholder_communication(self,
                                          session: InsightSession,
                                          stakeholder_type: AudienceType,
                                          message_type: str,
                                          priority: InsightPriority) -> Dict[str, Any]:
        """Generate stakeholder-specific communication content."""
        
        stakeholder_view = session.stakeholder_views.get(stakeholder_type.value)
        
        if message_type == 'update':
            return {
                'type': 'status_update',
                'subject': f"{session.project_definition.domain.value.title()} AI Initiative Update",
                'key_points': stakeholder_view.key_insights[:3] if stakeholder_view else [],
                'recommendations': stakeholder_view.recommendations[:2] if stakeholder_view else [],
                'priority': priority.value,
                'dashboard_link': f"/dashboard/{session.active_dashboards[0]}" if session.active_dashboards else None
            }
        elif message_type == 'alert':
            return {
                'type': 'alert',
                'subject': f"ALERT: {session.project_definition.domain.value.title()} AI Initiative",
                'alert_message': f"Priority {priority.value} alert requires your attention",
                'action_required': True,
                'priority': priority.value
            }
        else:
            return {
                'type': 'general',
                'subject': f"{session.project_definition.domain.value.title()} AI Initiative Communication",
                'message': f"General communication for {stakeholder_type.value}",
                'priority': priority.value
            }
    
    def _execute_stakeholder_communication(self,
                                         session: InsightSession,
                                         stakeholder_type: AudienceType,
                                         content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute communication to specific stakeholder."""
        
        # In production, would integrate with actual communication systems
        communication_result = {
            'stakeholder': stakeholder_type.value,
            'communication_type': content.get('type', 'general'),
            'delivery_status': 'delivered',
            'timestamp': datetime.now().isoformat(),
            'channels_used': ['dashboard', 'email']  # Based on stakeholder preferences
        }
        
        # Log communication
        session.communication_log.append({
            'timestamp': datetime.now().isoformat(),
            'stakeholder': stakeholder_type.value,
            'communication_type': content.get('type'),
            'priority': content.get('priority'),
            'delivery_status': 'delivered'
        })
        
        return communication_result
    
    def _initialize_default_triggers(self):
        """Initialize default insight triggers."""
        
        self.insight_triggers = {
            InsightTrigger.VALIDATION_COMPLETE: [],
            InsightTrigger.PERFORMANCE_THRESHOLD: [],
            InsightTrigger.SCHEDULE_BASED: [],
            InsightTrigger.STAKEHOLDER_REQUEST: [],
            InsightTrigger.ANOMALY_DETECTED: [],
            InsightTrigger.COMPLIANCE_ALERT: []
        }
    
    def _initialize_default_preferences(self):
        """Initialize default communication preferences."""
        
        self.communication_preferences = {
            'executive': {
                'frequency': 'daily',
                'channels': ['email', 'dashboard'],
                'priority_threshold': 'high',
                'format_preference': 'summary'
            },
            'technical': {
                'frequency': 'real_time',
                'channels': ['dashboard', 'slack'],
                'priority_threshold': 'medium',
                'format_preference': 'detailed'
            },
            'business': {
                'frequency': 'weekly',
                'channels': ['email', 'dashboard'],
                'priority_threshold': 'medium',
                'format_preference': 'business_focused'
            },
            'compliance': {
                'frequency': 'monthly',
                'channels': ['email', 'audit_log'],
                'priority_threshold': 'high',
                'format_preference': 'compliance_report'
            }
        }
    
    def _log_communication_event(self, session_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log communication event for tracking and analytics."""
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id].communication_log.append({
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'event_data': event_data
            })
    
    def _calculate_stakeholder_engagement(self, session: InsightSession) -> Dict[str, Any]:
        """Calculate stakeholder engagement metrics."""
        
        engagement_metrics = {}
        
        for stakeholder_type in session.stakeholder_views.keys():
            stakeholder_events = [
                event for event in session.communication_log
                if event.get('stakeholder') == stakeholder_type
            ]
            
            engagement_metrics[stakeholder_type] = {
                'communication_events': len(stakeholder_events),
                'last_interaction': stakeholder_events[-1]['timestamp'] if stakeholder_events else None,
                'engagement_level': 'high' if len(stakeholder_events) > 5 else 'medium' if len(stakeholder_events) > 2 else 'low'
            }
        
        return engagement_metrics
    
    def _get_communication_summary(self, session: InsightSession) -> Dict[str, Any]:
        """Get summary of communication activities."""
        
        return {
            'total_communications': len(session.communication_log),
            'communication_types': list(set(event.get('event_type', 'unknown') for event in session.communication_log)),
            'stakeholder_interactions': len(set(event.get('stakeholder') for event in session.communication_log if event.get('stakeholder'))),
            'last_communication': session.communication_log[-1]['timestamp'] if session.communication_log else None
        }
    
    # Additional helper methods for monitoring setup
    
    def _setup_performance_monitoring(self, session_id: str, thresholds: Dict[str, float]):
        """Setup performance threshold monitoring."""
        # Production implementation would set up actual monitoring
        pass
    
    def _setup_anomaly_detection(self, session_id: str, config: Dict[str, Any]):
        """Setup anomaly detection monitoring."""
        # Production implementation would configure anomaly detection
        pass
    
    def _setup_stakeholder_notifications(self, session_id: str, config: Dict[str, Any]):
        """Setup stakeholder notification preferences."""
        # Production implementation would configure notification systems
        pass
    
    def _setup_automated_reporting(self, session_id: str, schedules: Dict[str, Any]):
        """Setup automated reporting schedules."""
        
        for stakeholder, schedule_config in schedules.items():
            frequency = ReportFrequency(schedule_config.get('frequency', 'weekly'))
            channels = [DeliveryChannel(ch) for ch in schedule_config.get('channels', ['email'])]
            recipients = schedule_config.get('recipients', [stakeholder])
            
            self.stakeholder_reporter.setup_automated_reporting(
                AudienceType(stakeholder), frequency, channels, recipients
            )