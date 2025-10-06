"""
Knowledge Graph Service

Provides abstraction layer for graph database operations, supporting both
Neo4j and PostgreSQL with Age extension for graph queries.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
import uuid

from .schema import (
    GraphSchema, NodeType, RelationshipType, 
    NodeProperties, Relationship
)

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SessionDataStorage:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.correlations: List[Dict[str, Any]] = []
        self.datasets: List[Dict[str, Any]] = []
        self.models: List[Dict[str, Any]] = []
        self.executions: List[Dict[str, Any]] = []

    def add_session(self, session_id: str, data: Dict[str, Any]):
        self.sessions[session_id] = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        if 'dataset_info' in data:
            self.datasets.append(data['dataset_info'])

        if 'correlations' in data:
            self.correlations.extend(data['correlations'])

        if 'model_results' in data:
            self.models.extend(data['model_results'])

    def add_execution(self, execution_data: Dict[str, Any]):
        """Store execution data from adaptive learning"""
        self.executions.append(execution_data)
        # Keep only recent executions
        self.executions = self.executions[-500:]

    def get_all_data(self) -> Dict[str, Any]:
        return {
            'sessions': len(self.sessions),
            'datasets': self.datasets,
            'correlations': self.correlations,
            'models': self.models,
            'executions': self.executions[-10:]  # Recent executions
        }

class GraphDatabaseInterface(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass
    
    @abstractmethod
    def create_node(self, node_type: NodeType, properties: Dict[str, Any]) -> str:
        """Create a node and return its ID"""
        pass
    
    @abstractmethod
    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between nodes"""
        pass
    
    @abstractmethod
    def find_node(self, node_type: NodeType, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a node by filters"""
        pass
    
    @abstractmethod
    def find_nodes(self, node_type: NodeType, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find multiple nodes by filters"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a raw query"""
        pass
    
    @abstractmethod
    def get_node_relationships(self, node_id: str, relationship_types: List[RelationshipType] = None) -> List[Dict[str, Any]]:
        """Get all relationships for a node"""
        pass


class Neo4jGraphDatabase(GraphDatabaseInterface):
    """Neo4j implementation of graph database interface"""
    
    def __init__(self, uri: str, username: str, password: str):
        if not HAS_NEO4J:
            raise ImportError("neo4j package not installed. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
    
    def connect(self) -> bool:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j database")
    
    def create_node(self, node_type: NodeType, properties: Dict[str, Any]) -> str:
        """Create a node in Neo4j"""
        if 'id' not in properties:
            properties['id'] = str(uuid.uuid4())
        
        # Convert datetime objects to strings
        processed_props = {}
        for key, value in properties.items():
            if isinstance(value, datetime):
                processed_props[key] = value.isoformat()
            elif isinstance(value, dict):
                processed_props[key] = json.dumps(value)
            elif isinstance(value, list):
                processed_props[key] = json.dumps(value)
            else:
                processed_props[key] = value
        
        query = f"""
        CREATE (n:{node_type.value} $props)
        RETURN n.id as id
        """
        
        with self.driver.session() as session:
            result = session.run(query, props=processed_props)
            return result.single()['id']
    
    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship in Neo4j"""
        try:
            processed_props = {}
            for key, value in relationship.properties.items():
                if isinstance(value, datetime):
                    processed_props[key] = value.isoformat()
                elif isinstance(value, dict):
                    processed_props[key] = json.dumps(value)
                else:
                    processed_props[key] = value
            
            processed_props['created_at'] = relationship.created_at.isoformat()
            if relationship.weight is not None:
                processed_props['weight'] = relationship.weight
            
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[r:{relationship.relationship_type.value} $props]->(b)
            RETURN r
            """
            
            with self.driver.session() as session:
                session.run(query, 
                           source_id=relationship.source_id,
                           target_id=relationship.target_id,
                           props=processed_props)
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def find_node(self, node_type: NodeType, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single node in Neo4j"""
        where_clauses = []
        params = {}
        
        for key, value in filters.items():
            param_name = f"param_{key}"
            where_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value
        
        where_clause = " AND ".join(where_clauses) if where_clauses else ""
        query = f"""
        MATCH (n:{node_type.value})
        {f'WHERE {where_clause}' if where_clause else ''}
        RETURN n
        LIMIT 1
        """
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            record = result.single()
            return dict(record['n']) if record else None
    
    def find_nodes(self, node_type: NodeType, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find multiple nodes in Neo4j"""
        where_clauses = []
        params = {}
        
        for key, value in filters.items():
            param_name = f"param_{key}"
            where_clauses.append(f"n.{key} = ${param_name}")
            params[param_name] = value
        
        where_clause = " AND ".join(where_clauses) if where_clauses else ""
        query = f"""
        MATCH (n:{node_type.value})
        {f'WHERE {where_clause}' if where_clause else ''}
        RETURN n
        """
        
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record['n']) for record in result]
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute raw Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def get_node_relationships(self, node_id: str, relationship_types: List[RelationshipType] = None) -> List[Dict[str, Any]]:
        """Get all relationships for a node"""
        rel_filter = ""
        if relationship_types:
            rel_names = "|".join([rt.value for rt in relationship_types])
            rel_filter = f":{rel_names}"
        
        query = f"""
        MATCH (n {{id: $node_id}})-[r{rel_filter}]-(other)
        RETURN r, other
        """
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            relationships = []
            for record in result:
                rel_data = dict(record['r'])
                other_node = dict(record['other'])
                relationships.append({
                    'relationship': rel_data,
                    'connected_node': other_node
                })
            return relationships


class PostgresGraphDatabase(GraphDatabaseInterface):
    """PostgreSQL with Age extension implementation"""
    
    def __init__(self, connection_string: str):
        if not HAS_POSTGRES:
            raise ImportError("psycopg2 package not installed. Install with: pip install psycopg2-binary")
        
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self) -> bool:
        """Establish connection to PostgreSQL"""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            
            # Load Age extension
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS age;")
                cursor.execute("LOAD 'age';")
                cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
            
            self.connection.commit()
            logger.info("Connected to PostgreSQL with Age extension")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from PostgreSQL")
    
    def create_node(self, node_type: NodeType, properties: Dict[str, Any]) -> str:
        """Create a node in PostgreSQL/Age"""
        if 'id' not in properties:
            properties['id'] = str(uuid.uuid4())
        
        # Convert properties to JSON-compatible format
        processed_props = {}
        for key, value in properties.items():
            if isinstance(value, datetime):
                processed_props[key] = value.isoformat()
            else:
                processed_props[key] = value
        
        query = f"""
        SELECT * FROM cypher('knowledge_graph', $$
            CREATE (n:{node_type.value} %s)
            RETURN n.id
        $$) AS (id agtype);
        """
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query % json.dumps(processed_props))
            result = cursor.fetchone()
            self.connection.commit()
            return json.loads(result['id'])
    
    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship in PostgreSQL/Age"""
        try:
            processed_props = relationship.properties.copy()
            processed_props['created_at'] = relationship.created_at.isoformat()
            if relationship.weight is not None:
                processed_props['weight'] = relationship.weight
            
            query = f"""
            SELECT * FROM cypher('knowledge_graph', $$
                MATCH (a {{id: "{relationship.source_id}"}}), (b {{id: "{relationship.target_id}"}})
                CREATE (a)-[r:{relationship.relationship_type.value} %s]->(b)
                RETURN r
            $$) AS (r agtype);
            """
            
            with self.connection.cursor() as cursor:
                cursor.execute(query % json.dumps(processed_props))
                self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def find_node(self, node_type: NodeType, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single node in PostgreSQL/Age"""
        # This is a simplified implementation
        # Full implementation would require proper Age query construction
        return None
    
    def find_nodes(self, node_type: NodeType, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find multiple nodes in PostgreSQL/Age"""
        return []
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute raw query in PostgreSQL/Age"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, parameters or {})
            return [dict(row) for row in cursor.fetchall()]
    
    def get_node_relationships(self, node_id: str, relationship_types: List[RelationshipType] = None) -> List[Dict[str, Any]]:
        """Get all relationships for a node"""
        return []


class KnowledgeGraphService:
    """Main service class for knowledge graph operations"""
    
    def __init__(self, database: GraphDatabaseInterface):
        self.database = database
        self.schema = GraphSchema()
        self.connected = False
    
    def initialize(self) -> bool:
        """Initialize the knowledge graph service"""
        if not self.database.connect():
            return False
        
        self.connected = True
        
        # Create constraints and indexes if using Neo4j
        if isinstance(self.database, Neo4jGraphDatabase):
            try:
                constraints = self.schema.generate_cypher_create_constraints()
                indexes = self.schema.generate_cypher_create_indexes()
                
                for constraint in constraints:
                    try:
                        self.database.execute_query(constraint)
                    except Exception as e:
                        logger.debug(f"Constraint may already exist: {e}")
                
                for index in indexes:
                    try:
                        self.database.execute_query(index)
                    except Exception as e:
                        logger.debug(f"Index may already exist: {e}")
                        
                logger.info("Graph schema constraints and indexes created")
            except Exception as e:
                logger.warning(f"Failed to create some constraints/indexes: {e}")
        
        return True
    
    def shutdown(self):
        """Shutdown the knowledge graph service"""
        if self.connected:
            self.database.disconnect()
            self.connected = False
    
    def create_dataset_node(self, dataset_characteristics: Dict[str, Any]) -> str:
        """Create a dataset node from dataset characteristics"""
        node_properties = {
            'id': str(uuid.uuid4()),
            'name': dataset_characteristics.get('name', 'Unknown Dataset'),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'shape': dataset_characteristics.get('shape', (0, 0)),
            'file_hash': dataset_characteristics.get('file_hash', ''),
            'data_types': dataset_characteristics.get('data_types', {}),
            'missing_percentage': dataset_characteristics.get('overall_missing_percentage', 0.0),
            'domain': dataset_characteristics.get('domain'),
            'source': dataset_characteristics.get('source'),
            'quality_score': dataset_characteristics.get('quality_score'),
            'metadata': dataset_characteristics.get('metadata', {})
        }
        
        return self.database.create_node(NodeType.DATASET, node_properties)
    
    def create_model_node(self, model_info: Dict[str, Any], performance_metrics: Dict[str, float]) -> str:
        """Create a model node"""
        node_properties = {
            'id': str(uuid.uuid4()),
            'name': model_info.get('name', f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'algorithm': model_info.get('algorithm', 'Unknown'),
            'hyperparameters': model_info.get('hyperparameters', {}),
            'performance_metrics': performance_metrics,
            'trust_score': model_info.get('trust_score', 0.0),
            'interpretability_score': model_info.get('interpretability_score', 0.0),
            'training_time': model_info.get('training_time', 0.0),
            'model_size_mb': model_info.get('model_size_mb'),
            'prediction_latency_ms': model_info.get('prediction_latency_ms'),
            'metadata': model_info.get('metadata', {})
        }
        
        return self.database.create_node(NodeType.MODEL, node_properties)
    
    def create_project_node(self, project_definition: Dict[str, Any]) -> str:
        """Create a project node from project definition"""
        node_properties = {
            'id': str(uuid.uuid4()),
            'name': project_definition.get('name', 'Unnamed Project'),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'objective': project_definition.get('objective', ''),
            'domain': project_definition.get('domain', ''),
            'business_goal': project_definition.get('business_goal', ''),
            'constraints': project_definition.get('constraints', {}),
            'success_criteria': project_definition.get('success_criteria', {}),
            'status': 'active',
            'priority': project_definition.get('priority'),
            'metadata': project_definition.get('metadata', {})
        }
        
        return self.database.create_node(NodeType.PROJECT, node_properties)
    
    def create_execution_node(self, execution_info: Dict[str, Any]) -> str:
        """Create an execution node"""
        node_properties = {
            'id': execution_info.get('execution_id', str(uuid.uuid4())),
            'name': f"Execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': execution_info.get('start_time', datetime.now()),
            'updated_at': datetime.now(),
            'start_time': execution_info.get('start_time', datetime.now()),
            'end_time': execution_info.get('end_time'),
            'status': execution_info.get('status', 'running'),
            'stage_results': execution_info.get('stage_results', {}),
            'resource_usage': execution_info.get('resource_usage'),
            'error_message': execution_info.get('error_message'),
            'metadata': execution_info.get('metadata', {})
        }
        
        return self.database.create_node(NodeType.EXECUTION, node_properties)
    
    def link_execution_to_project(self, execution_id: str, project_id: str) -> bool:
        """Create relationship between execution and project"""
        relationship = Relationship(
            source_id=project_id,
            target_id=execution_id,
            relationship_type=RelationshipType.HAS_EXECUTION,
            properties={},
            created_at=datetime.now()
        )
        
        return self.database.create_relationship(relationship)
    
    def link_model_to_project(self, model_id: str, project_id: str) -> bool:
        """Create relationship between model and project"""
        relationship = Relationship(
            source_id=model_id,
            target_id=project_id,
            relationship_type=RelationshipType.APPLIES_TO,
            properties={},
            created_at=datetime.now()
        )
        
        return self.database.create_relationship(relationship)
    
    def query_similar_projects(self, project_characteristics: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar projects based on characteristics"""
        # This would implement a sophisticated similarity search
        # For now, we'll do a simple domain-based search
        
        if isinstance(self.database, Neo4jGraphDatabase):
            query = """
            MATCH (p:Project)
            WHERE p.domain = $domain AND p.status = 'completed'
            WITH p, p.constraints as constraints
            RETURN p, 
                   size([key IN keys(constraints) WHERE constraints[key] = $constraints[key]]) as similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            parameters = {
                'domain': project_characteristics.get('domain'),
                'constraints': project_characteristics.get('constraints', {}),
                'limit': limit
            }
            
            return self.database.execute_query(query, parameters)
        
        return []
    
    def get_feature_importance_ranking(self, business_objective: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top features for a specific business objective"""
        if isinstance(self.database, Neo4jGraphDatabase):
            query = """
            MATCH (f:Feature)-[:USED_IN]->(m:Model)-[:APPLIES_TO]->(p:Project)-[:ACHIEVES]->(o:BusinessObjective)
            WHERE o.name CONTAINS $objective
            WITH f, avg(f.importance_score) as avg_importance, count(*) as usage_count
            RETURN f.name, f.generation_method, avg_importance, usage_count
            ORDER BY avg_importance DESC, usage_count DESC
            LIMIT $limit
            """
            
            parameters = {
                'objective': business_objective,
                'limit': limit
            }
            
            return self.database.execute_query(query, parameters)
        
        return []
    
    def record_pipeline_execution(self, execution_data: Dict[str, Any]) -> Dict[str, str]:
        """Record a complete pipeline execution to the knowledge graph"""
        node_ids = {}
        
        try:
            # Create dataset node
            if 'dataset_characteristics' in execution_data:
                dataset_id = self.create_dataset_node(execution_data['dataset_characteristics'])
                node_ids['dataset'] = dataset_id
            
            # Create project node
            if 'project_definition' in execution_data:
                project_id = self.create_project_node(execution_data['project_definition'])
                node_ids['project'] = project_id
            
            # Create execution node
            execution_id = self.create_execution_node(execution_data)
            node_ids['execution'] = execution_id
            
            # Create model node if available
            if 'model_info' in execution_data and 'performance_metrics' in execution_data:
                model_id = self.create_model_node(
                    execution_data['model_info'], 
                    execution_data['performance_metrics']
                )
                node_ids['model'] = model_id
                
                # Link model to project
                if 'project' in node_ids:
                    self.link_model_to_project(model_id, node_ids['project'])
            
            # Link execution to project
            if 'project' in node_ids:
                self.link_execution_to_project(execution_id, node_ids['project'])
            
            logger.info(f"Successfully recorded pipeline execution to knowledge graph: {node_ids}")
            
        except Exception as e:
            logger.error(f"Failed to record pipeline execution: {e}")
            
        return node_ids