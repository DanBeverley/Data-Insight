"""Production-grade Strategic Control Layer - Project Definition System"""

import time
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

class Objective(Enum):
    """Strategic business objectives for AI initiatives."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    INTERPRETABILITY = "interpretability"
    FAIRNESS = "fairness"
    COMPLIANCE = "compliance"
    ROBUSTNESS = "robustness"
    COST_EFFICIENCY = "cost_efficiency"
    SCALABILITY = "scalability"
    INNOVATION = "innovation"
    
    def get_description(self) -> str:
        """Get human-readable description of objective."""
        descriptions = {
            self.ACCURACY: "Maximize prediction accuracy and decision quality",
            self.SPEED: "Optimize for operational efficiency and response times",
            self.INTERPRETABILITY: "Ensure transparent and explainable AI decisions",
            self.FAIRNESS: "Maintain ethical AI practices and equitable outcomes",
            self.COMPLIANCE: "Achieve regulatory adherence and risk management",
            self.ROBUSTNESS: "Build resilient and reliable systems",
            self.COST_EFFICIENCY: "Optimize resource utilization and operational costs",
            self.SCALABILITY: "Enable growth and expansion capabilities",
            self.INNOVATION: "Drive competitive advantage through AI innovation"
        }
        return descriptions.get(self, "Strategic AI objective")
    
    def get_success_metrics(self) -> List[str]:
        """Get key success metrics for this objective."""
        metrics = {
            self.ACCURACY: ["prediction_accuracy", "precision", "recall", "f1_score"],
            self.SPEED: ["inference_latency", "throughput", "training_time", "deployment_speed"],
            self.INTERPRETABILITY: ["explanation_quality", "feature_importance_clarity", "decision_transparency"],
            self.FAIRNESS: ["demographic_parity", "equalized_odds", "bias_metrics"],
            self.COMPLIANCE: ["regulatory_adherence", "audit_completeness", "policy_compliance"],
            self.ROBUSTNESS: ["adversarial_resilience", "data_drift_tolerance", "system_stability"],
            self.COST_EFFICIENCY: ["cost_per_prediction", "resource_utilization", "roi_metrics"],
            self.SCALABILITY: ["concurrent_users", "data_volume_capacity", "geographic_reach"],
            self.INNOVATION: ["competitive_advantage", "novel_capabilities", "market_differentiation"]
        }
        return metrics.get(self, ["general_performance"])

class Domain(Enum):
    """Business domains with specific regulatory and operational requirements."""
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    MARKETING = "marketing"
    EDUCATION = "education"
    GOVERNMENT = "government"
    ENERGY = "energy"
    TRANSPORTATION = "transportation"
    GENERAL = "general"
    
    def get_regulatory_requirements(self) -> List[str]:
        """Get regulatory requirements for this domain."""
        requirements = {
            self.FINANCE: ["Basel III", "MiFID II", "PCI DSS", "SOX", "GDPR", "Fair Credit Reporting Act"],
            self.HEALTHCARE: ["HIPAA", "FDA 21 CFR Part 820", "GDPR", "IEC 62304", "ISO 13485", "HITECH"],
            self.RETAIL: ["GDPR", "CCPA", "PCI DSS", "FTC Guidelines"],
            self.MANUFACTURING: ["ISO 9001", "ISO 27001", "GDPR", "Industry 4.0 Standards"],
            self.MARKETING: ["GDPR", "CCPA", "CAN-SPAM Act", "FTC Truth in Advertising"],
            self.EDUCATION: ["FERPA", "COPPA", "GDPR", "Section 508"],
            self.GOVERNMENT: ["FISMA", "FedRAMP", "Section 508", "Privacy Act"],
            self.ENERGY: ["NERC CIP", "ISO 50001", "Environmental Regulations"],
            self.TRANSPORTATION: ["DOT Regulations", "Safety Standards", "Environmental Compliance"]
        }
        return requirements.get(self, ["GDPR", "General Data Protection"])
    
    def get_risk_profile(self) -> str:
        """Get typical risk profile for this domain."""
        risk_profiles = {
            self.FINANCE: "high",
            self.HEALTHCARE: "critical",
            self.GOVERNMENT: "critical",
            self.EDUCATION: "high",
            self.ENERGY: "high",
            self.TRANSPORTATION: "high",
            self.RETAIL: "medium",
            self.MANUFACTURING: "medium",
            self.MARKETING: "low"
        }
        return risk_profiles.get(self, "medium")
    
    def get_typical_constraints(self) -> Dict[str, Any]:
        """Get typical constraints for this domain."""
        constraints = {
            self.FINANCE: {
                "max_latency_ms": 100,
                "interpretability_required": True,
                "audit_trail_required": True,
                "data_retention_years": 7
            },
            self.HEALTHCARE: {
                "max_latency_ms": 500,
                "interpretability_required": True,
                "privacy_level": "maximum",
                "clinical_validation_required": True
            },
            self.RETAIL: {
                "max_latency_ms": 200,
                "scalability_required": True,
                "real_time_processing": True
            }
        }
        return constraints.get(self, {})

class RiskLevel(Enum):
    """Risk levels for AI projects with associated governance requirements."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def get_governance_requirements(self) -> List[str]:
        """Get governance requirements for this risk level."""
        governance = {
            self.LOW: ["Basic documentation", "Standard testing"],
            self.MEDIUM: ["Comprehensive documentation", "Extended testing", "Risk assessment"],
            self.HIGH: ["Full documentation", "Extensive testing", "Risk assessment", "Executive approval", "Compliance review"],
            self.CRITICAL: ["Complete documentation", "Comprehensive testing", "Full risk assessment", "Board approval", "Regulatory review", "Continuous monitoring"]
        }
        return governance.get(self, ["Standard requirements"])
    
    def get_approval_authority(self) -> str:
        """Get required approval authority for this risk level."""
        authorities = {
            self.LOW: "Team Lead",
            self.MEDIUM: "Department Head",
            self.HIGH: "Executive Committee",
            self.CRITICAL: "Board of Directors"
        }
        return authorities.get(self, "Team Lead")

class Priority(Enum):
    """Project priority levels affecting resource allocation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def get_resource_allocation(self) -> Dict[str, float]:
        """Get typical resource allocation percentages."""
        allocations = {
            self.LOW: {"budget_percentage": 5, "team_percentage": 10, "infrastructure_percentage": 5},
            self.MEDIUM: {"budget_percentage": 15, "team_percentage": 25, "infrastructure_percentage": 15},
            self.HIGH: {"budget_percentage": 35, "team_percentage": 50, "infrastructure_percentage": 40},
            self.CRITICAL: {"budget_percentage": 45, "team_percentage": 75, "infrastructure_percentage": 60}
        }
        return allocations.get(self, {"budget_percentage": 10, "team_percentage": 20, "infrastructure_percentage": 10})

class ProjectPhase(Enum):
    """Project lifecycle phases with specific deliverables."""
    IDEATION = "ideation"
    PLANNING = "planning"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    
    def get_key_deliverables(self) -> List[str]:
        """Get key deliverables for this phase."""
        deliverables = {
            self.IDEATION: ["Business case", "Initial requirements", "Feasibility study"],
            self.PLANNING: ["Project charter", "Resource plan", "Risk assessment", "Timeline"],
            self.DEVELOPMENT: ["Model development", "Feature engineering", "Testing framework"],
            self.VALIDATION: ["Performance validation", "Business validation", "Compliance validation"],
            self.DEPLOYMENT: ["Production deployment", "Monitoring setup", "Documentation"],
            self.MONITORING: ["Performance monitoring", "Business impact measurement", "Maintenance"],
            self.OPTIMIZATION: ["Performance optimization", "Cost optimization", "Feature enhancement"]
        }
        return deliverables.get(self, ["Standard deliverables"])

@dataclass
class BusinessContext:
    """Business context and strategic alignment information."""
    business_unit: str
    strategic_initiative: Optional[str] = None
    success_criteria: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    budget_range: Optional[str] = None
    timeline_months: Optional[int] = None
    expected_roi: Optional[float] = None
    competitive_advantage: Optional[str] = None
    market_opportunity: Optional[str] = None
    
    def validate_context(self) -> List[str]:
        """Validate business context completeness."""
        issues = []
        if not self.business_unit:
            issues.append("Business unit must be specified")
        if not self.success_criteria:
            issues.append("Success criteria must be defined")
        if not self.stakeholders:
            issues.append("Key stakeholders must be identified")
        if self.timeline_months and self.timeline_months <= 0:
            issues.append("Timeline must be positive")
        return issues

@dataclass
class TechnicalConstraints:
    """Technical constraints and requirements."""
    max_latency_ms: Optional[int] = None
    max_training_hours: Optional[float] = None
    max_memory_gb: Optional[float] = None
    max_storage_gb: Optional[float] = None
    min_accuracy: Optional[float] = None
    min_precision: Optional[float] = None
    min_recall: Optional[float] = None
    max_cost_per_prediction: Optional[float] = None
    required_uptime: Optional[float] = None  # Percentage
    scalability_requirements: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.scalability_requirements is None:
            self.scalability_requirements = {}
    
    def validate_constraints(self) -> List[str]:
        """Validate technical constraints."""
        issues = []
        if self.min_accuracy and (self.min_accuracy < 0 or self.min_accuracy > 1):
            issues.append("Minimum accuracy must be between 0 and 1")
        if self.required_uptime and (self.required_uptime < 0 or self.required_uptime > 100):
            issues.append("Required uptime must be between 0 and 100%")
        if self.max_latency_ms and self.max_latency_ms <= 0:
            issues.append("Maximum latency must be positive")
        return issues

@dataclass
class RegulatoryConstraints:
    """Regulatory and compliance constraints."""
    protected_attributes: List[str] = field(default_factory=list)
    interpretability_required: bool = False
    compliance_rules: List[str] = field(default_factory=list)
    data_retention_requirements: Optional[Dict[str, int]] = None  # years
    geographic_restrictions: List[str] = field(default_factory=list)
    audit_trail_required: bool = False
    privacy_level: str = "standard"  # standard, high, maximum
    encryption_required: bool = False
    anonymization_required: bool = False
    
    def __post_init__(self):
        if self.data_retention_requirements is None:
            self.data_retention_requirements = {}
    
    def validate_regulatory_compliance(self, domain: Domain) -> List[str]:
        """Validate regulatory compliance for domain."""
        issues = []
        domain_requirements = domain.get_regulatory_requirements()
        
        # Check if domain-specific requirements are addressed
        if domain == Domain.FINANCE and not self.audit_trail_required:
            issues.append("Financial domain requires audit trail")
        if domain == Domain.HEALTHCARE and self.privacy_level == "standard":
            issues.append("Healthcare domain requires high or maximum privacy level")
        
        return issues

@dataclass
class ProjectDefinition:
    """Comprehensive project definition with strategic, technical, and regulatory aspects."""
    
    # Core project identity
    project_id: str
    name: str
    description: str
    
    # Strategic elements
    objective: Objective
    domain: Domain
    priority: Priority
    risk_level: RiskLevel
    business_context: BusinessContext
    
    # Constraints
    technical_constraints: TechnicalConstraints
    regulatory_constraints: RegulatoryConstraints
    
    # Project management
    current_phase: ProjectPhase = ProjectPhase.IDEATION
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    approvals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Secondary objectives (up to 2)
    secondary_objectives: List[Objective] = field(default_factory=list)
    
    def __post_init__(self):
        # Auto-set risk level based on domain if not explicitly set
        if hasattr(self, '_risk_level_auto_set'):
            return  # Avoid infinite recursion
        
        domain_risk = self.domain.get_risk_profile()
        if domain_risk == "critical" and self.risk_level.value != "critical":
            object.__setattr__(self, 'risk_level', RiskLevel.CRITICAL)
        elif domain_risk == "high" and self.risk_level.value in ["low", "medium"]:
            object.__setattr__(self, 'risk_level', RiskLevel.HIGH)
        
        object.__setattr__(self, '_risk_level_auto_set', True)
        
        # Limit secondary objectives
        if len(self.secondary_objectives) > 2:
            object.__setattr__(self, 'secondary_objectives', self.secondary_objectives[:2])
    
    def validate_project_definition(self) -> Dict[str, List[str]]:
        """Comprehensive validation of project definition."""
        validation_results = {
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Basic validation
        if not self.name or len(self.name.strip()) < 3:
            validation_results["errors"].append("Project name must be at least 3 characters")
        if not self.description or len(self.description.strip()) < 10:
            validation_results["errors"].append("Project description must be at least 10 characters")
        
        # Business context validation
        context_issues = self.business_context.validate_context()
        validation_results["errors"].extend(context_issues)
        
        # Technical constraints validation
        tech_issues = self.technical_constraints.validate_constraints()
        validation_results["errors"].extend(tech_issues)
        
        # Regulatory validation
        reg_issues = self.regulatory_constraints.validate_regulatory_compliance(self.domain)
        validation_results["warnings"].extend(reg_issues)
        
        # Strategic alignment validation
        if self.objective in self.secondary_objectives:
            validation_results["errors"].append("Primary objective cannot be a secondary objective")
        
        # Risk and priority alignment
        if self.risk_level == RiskLevel.CRITICAL and self.priority == Priority.LOW:
            validation_results["warnings"].append("Critical risk projects typically require high priority")
        
        # Domain-specific recommendations
        domain_constraints = self.domain.get_typical_constraints()
        if domain_constraints:
            for constraint, value in domain_constraints.items():
                if constraint == "interpretability_required" and value and not self.regulatory_constraints.interpretability_required:
                    validation_results["recommendations"].append(f"{self.domain.value.title()} domain typically requires interpretability")
        
        return validation_results
    
    def get_governance_requirements(self) -> List[str]:
        """Get governance requirements based on risk level."""
        return self.risk_level.get_governance_requirements()
    
    def get_required_approval_authority(self) -> str:
        """Get required approval authority."""
        return self.risk_level.get_approval_authority()
    
    def get_all_objectives(self) -> List[Objective]:
        """Get primary and secondary objectives."""
        return [self.objective] + self.secondary_objectives
    
    def get_success_metrics(self) -> Dict[str, List[str]]:
        """Get all success metrics for all objectives."""
        metrics = {"primary": self.objective.get_success_metrics()}
        for i, secondary in enumerate(self.secondary_objectives):
            metrics[f"secondary_{i+1}"] = secondary.get_success_metrics()
        return metrics
    
    def estimate_project_duration(self) -> Dict[str, int]:
        """Estimate project duration in months by phase."""
        base_durations = {
            "ideation": 1,
            "planning": 1,
            "development": 4,
            "validation": 2,
            "deployment": 1,
            "monitoring": 12  # Ongoing
        }
        
        # Adjust based on complexity factors
        complexity_factor = 1.0
        
        # Risk level adjustment
        if self.risk_level == RiskLevel.CRITICAL:
            complexity_factor *= 1.5
        elif self.risk_level == RiskLevel.HIGH:
            complexity_factor *= 1.3
        
        # Domain complexity adjustment
        if self.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            complexity_factor *= 1.4
        
        # Regulatory complexity
        if len(self.regulatory_constraints.compliance_rules) > 3:
            complexity_factor *= 1.2
        
        adjusted_durations = {}
        for phase, duration in base_durations.items():
            if phase == "monitoring":  # Monitoring is ongoing
                adjusted_durations[phase] = duration
            else:
                adjusted_durations[phase] = max(1, int(duration * complexity_factor))
        
        return adjusted_durations
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Get estimated resource requirements."""
        base_resources = self.priority.get_resource_allocation()
        
        # Adjust based on domain and risk
        if self.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            for key in base_resources:
                base_resources[key] *= 1.3
        
        if self.risk_level == RiskLevel.CRITICAL:
            for key in base_resources:
                base_resources[key] *= 1.2
        
        return {
            "team_size_estimate": max(3, int(base_resources["team_percentage"] / 10)),
            "budget_allocation_percentage": min(50, base_resources["budget_percentage"]),
            "infrastructure_percentage": base_resources["infrastructure_percentage"],
            "specialized_roles_needed": self._get_specialized_roles()
        }
    
    def _get_specialized_roles(self) -> List[str]:
        """Get specialized roles needed based on project characteristics."""
        roles = ["Data Scientist", "ML Engineer"]
        
        if self.regulatory_constraints.interpretability_required:
            roles.append("AI Ethics Specialist")
        
        if self.domain in [Domain.FINANCE, Domain.HEALTHCARE]:
            roles.append("Compliance Officer")
            roles.append("Domain Expert")
        
        if self.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            roles.append("Security Specialist")
            roles.append("Risk Manager")
        
        if self.regulatory_constraints.audit_trail_required:
            roles.append("Audit Specialist")
        
        return roles
    
    def update_phase(self, new_phase: ProjectPhase, updated_by: Optional[str] = None) -> bool:
        """Update project phase with validation."""
        # Phase progression validation
        phase_order = list(ProjectPhase)
        current_index = phase_order.index(self.current_phase)
        new_index = phase_order.index(new_phase)
        
        # Allow moving forward or staying in same phase
        if new_index >= current_index:
            object.__setattr__(self, 'current_phase', new_phase)
            object.__setattr__(self, 'updated_at', datetime.now())
            if updated_by:
                if not hasattr(self, 'phase_history'):
                    object.__setattr__(self, 'phase_history', [])
                self.phase_history.append({
                    'phase': new_phase.value,
                    'timestamp': datetime.now().isoformat(),
                    'updated_by': updated_by
                })
            return True
        return False
    
    def add_approval(self, authority: str, approved_by: str, notes: Optional[str] = None) -> bool:
        """Add approval record."""
        required_authority = self.get_required_approval_authority()
        
        self.approvals[authority] = {
            'approved_by': approved_by,
            'timestamp': datetime.now().isoformat(),
            'notes': notes,
            'authority_level': authority
        }
        
        return authority == required_authority
    
    def is_approved(self) -> bool:
        """Check if project has required approvals."""
        required_authority = self.get_required_approval_authority()
        return required_authority in self.approvals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'objective': self.objective.value,
            'domain': self.domain.value,
            'priority': self.priority.value,
            'risk_level': self.risk_level.value,
            'current_phase': self.current_phase.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'secondary_objectives': [obj.value for obj in self.secondary_objectives],
            'business_context': {
                'business_unit': self.business_context.business_unit,
                'strategic_initiative': self.business_context.strategic_initiative,
                'success_criteria': self.business_context.success_criteria,
                'stakeholders': self.business_context.stakeholders,
                'timeline_months': self.business_context.timeline_months
            },
            'governance_requirements': self.get_governance_requirements(),
            'approval_status': self.is_approved()
        }