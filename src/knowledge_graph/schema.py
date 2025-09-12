"""
Knowledge Graph Schema Definition

Defines the nodes, relationships, and structure for the DataInsight AI
knowledge graph that captures relationships between data, features, 
models, and business outcomes.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


class NodeType(Enum):
    """Core node types in the knowledge graph"""
    DATASET = "Dataset"
    COLUMN = "Column"
    FEATURE = "Feature"
    MODEL = "Model"
    PROJECT = "Project"
    BUSINESS_OBJECTIVE = "BusinessObjective"
    USER = "User"
    EXECUTION = "Execution"
    PIPELINE = "Pipeline"
    DOMAIN = "Domain"


class RelationshipType(Enum):
    """Relationship types between nodes"""
    HAS_COLUMN = "HAS_COLUMN"
    GENERATES_FEATURE = "GENERATES_FEATURE"
    DERIVED_FROM = "DERIVED_FROM"
    TRAINED_ON = "TRAINED_ON"
    APPLIES_TO = "APPLIES_TO"
    ACHIEVES = "ACHIEVES"
    EVALUATED_BY = "EVALUATED_BY"
    BELONGS_TO = "BELONGS_TO"
    USES_PIPELINE = "USES_PIPELINE"
    EXECUTED_BY = "EXECUTED_BY"
    HAS_EXECUTION = "HAS_EXECUTION"
    SIMILAR_TO = "SIMILAR_TO"
    INFLUENCED_BY = "INFLUENCED_BY"


@dataclass
class NodeProperties:
    """Base properties for all nodes"""
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class DatasetNode(NodeProperties):
    """Dataset node properties"""
    shape: tuple  # (rows, columns)
    file_hash: str
    data_types: Dict[str, str]
    missing_percentage: float
    domain: Optional[str] = None
    source: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class ColumnNode(NodeProperties):
    """Column node properties"""
    data_type: str
    semantic_type: str
    cardinality: float
    missing_percentage: float
    uniqueness: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    distribution_type: Optional[str] = None


@dataclass
class FeatureNode(NodeProperties):
    """Feature node properties"""
    data_type: str
    generation_method: str
    importance_score: Optional[float] = None
    stability_score: Optional[float] = None
    computational_cost: Optional[float] = None
    interpretation: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None


@dataclass
class ModelNode(NodeProperties):
    """Model node properties"""
    algorithm: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    trust_score: float
    interpretability_score: float
    training_time: float
    model_size_mb: Optional[float] = None
    prediction_latency_ms: Optional[float] = None


@dataclass 
class ProjectNode(NodeProperties):
    """Project node properties"""
    objective: str
    domain: str
    business_goal: str
    constraints: Dict[str, Any]
    success_criteria: Dict[str, float]
    status: str  # 'active', 'completed', 'failed'
    priority: Optional[str] = None


@dataclass
class BusinessObjectiveNode(NodeProperties):
    """Business objective node properties"""
    category: str  # 'prediction', 'optimization', 'analysis'
    target_metric: str
    success_threshold: float
    stakeholders: List[str]
    business_value: Optional[str] = None
    urgency: Optional[str] = None


@dataclass
class UserNode(NodeProperties):
    """User node properties"""
    role: str
    expertise_level: str  # 'beginner', 'intermediate', 'expert'
    department: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionNode(NodeProperties):
    """Execution/run node properties"""
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'running', 'completed', 'failed'
    stage_results: Dict[str, Any]
    resource_usage: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


@dataclass
class PipelineNode(NodeProperties):
    """Pipeline configuration node properties"""
    version: str
    configuration: Dict[str, Any]
    stages: List[str]
    validation_rules: Optional[Dict[str, Any]] = None


@dataclass
class Relationship:
    """Relationship between nodes"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    created_at: datetime
    weight: Optional[float] = None


class GraphSchema:
    """Knowledge graph schema manager"""
    
    def __init__(self):
        self.node_types = {
            NodeType.DATASET: DatasetNode,
            NodeType.COLUMN: ColumnNode,
            NodeType.FEATURE: FeatureNode,
            NodeType.MODEL: ModelNode,
            NodeType.PROJECT: ProjectNode,
            NodeType.BUSINESS_OBJECTIVE: BusinessObjectiveNode,
            NodeType.USER: UserNode,
            NodeType.EXECUTION: ExecutionNode,
            NodeType.PIPELINE: PipelineNode
        }
    
    def get_node_schema(self, node_type: NodeType) -> type:
        """Get the schema class for a node type"""
        return self.node_types.get(node_type)
    
    def validate_node(self, node_type: NodeType, properties: Dict[str, Any]) -> bool:
        """Validate node properties against schema"""
        schema_class = self.get_node_schema(node_type)
        if not schema_class:
            return False
        
        try:
            required_fields = schema_class.__annotations__.keys()
            for field in required_fields:
                if field not in properties and not hasattr(schema_class, field):
                    return False
            return True
        except Exception:
            return False
    
    def get_relationship_constraints(self) -> Dict[RelationshipType, Dict[str, List[NodeType]]]:
        """Define valid source->target node types for each relationship"""
        return {
            RelationshipType.HAS_COLUMN: {
                'source': [NodeType.DATASET],
                'target': [NodeType.COLUMN]
            },
            RelationshipType.GENERATES_FEATURE: {
                'source': [NodeType.COLUMN],
                'target': [NodeType.FEATURE]
            },
            RelationshipType.DERIVED_FROM: {
                'source': [NodeType.FEATURE],
                'target': [NodeType.COLUMN, NodeType.FEATURE]
            },
            RelationshipType.TRAINED_ON: {
                'source': [NodeType.MODEL],
                'target': [NodeType.FEATURE, NodeType.DATASET]
            },
            RelationshipType.APPLIES_TO: {
                'source': [NodeType.MODEL, NodeType.PIPELINE],
                'target': [NodeType.PROJECT]
            },
            RelationshipType.ACHIEVES: {
                'source': [NodeType.PROJECT],
                'target': [NodeType.BUSINESS_OBJECTIVE]
            },
            RelationshipType.EVALUATED_BY: {
                'source': [NodeType.PROJECT, NodeType.MODEL],
                'target': [NodeType.USER]
            },
            RelationshipType.EXECUTED_BY: {
                'source': [NodeType.EXECUTION],
                'target': [NodeType.USER]
            },
            RelationshipType.HAS_EXECUTION: {
                'source': [NodeType.PROJECT],
                'target': [NodeType.EXECUTION]
            },
            RelationshipType.USES_PIPELINE: {
                'source': [NodeType.EXECUTION],
                'target': [NodeType.PIPELINE]
            }
        }
    
    def generate_cypher_create_constraints(self) -> List[str]:
        """Generate Cypher statements for creating graph constraints"""
        constraints = []
        
        # Unique constraints for node IDs
        for node_type in NodeType:
            constraints.append(
                f"CREATE CONSTRAINT {node_type.value.lower()}_id_unique "
                f"IF NOT EXISTS FOR (n:{node_type.value}) "
                f"REQUIRE n.id IS UNIQUE"
            )
        
        return constraints
    
    def generate_cypher_create_indexes(self) -> List[str]:
        """Generate Cypher statements for creating performance indexes"""
        indexes = []
        
        # Common search indexes
        indexes.extend([
            "CREATE INDEX dataset_domain IF NOT EXISTS FOR (d:Dataset) ON d.domain",
            "CREATE INDEX column_semantic_type IF NOT EXISTS FOR (c:Column) ON c.semantic_type", 
            "CREATE INDEX feature_importance IF NOT EXISTS FOR (f:Feature) ON f.importance_score",
            "CREATE INDEX model_algorithm IF NOT EXISTS FOR (m:Model) ON m.algorithm",
            "CREATE INDEX project_status IF NOT EXISTS FOR (p:Project) ON p.status",
            "CREATE INDEX execution_created_at IF NOT EXISTS FOR (e:Execution) ON e.created_at"
        ])
        
        return indexes