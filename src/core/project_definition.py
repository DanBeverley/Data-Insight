from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

class Objective(Enum):
    ACCURACY = "maximize_accuracy"
    SPEED = "minimize_latency"
    INTERPRETABILITY = "maximize_interpretability"
    FAIRNESS = "balance_accuracy_fairness"
    COST = "minimize_cost"
    ROBUSTNESS = "maximize_robustness"
    COMPLIANCE = "regulatory_compliance"

class Domain(Enum):
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"
    GENERAL = "general"

class RiskLevel(Enum):
    LOW = "low_risk"
    MEDIUM = "medium_risk"
    HIGH = "high_risk"

@dataclass
class Constraints:
    max_latency_ms: Optional[int] = None
    max_training_hours: Optional[int] = None
    max_memory_gb: Optional[float] = None
    min_accuracy: Optional[float] = None
    protected_attributes: List[str] = field(default_factory=list)
    interpretability_required: bool = False
    compliance_rules: List[str] = field(default_factory=list)
    
@dataclass
class ProjectDefinition:
    objective: Objective
    domain: Domain = Domain.GENERAL
    risk_level: RiskLevel = RiskLevel.MEDIUM
    constraints: Constraints = field(default_factory=Constraints)
    context: Dict[str, Any] = field(default_factory=dict)