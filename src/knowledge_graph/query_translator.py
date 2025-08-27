"""
Natural Language to Graph Query Translation

Translates natural language queries into formal graph database queries
using LLM capabilities with structured prompts.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..llm.interface import LLMInterface
from .schema import GraphSchema, NodeType, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from query translation and execution"""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    natural_language_response: str
    execution_time: float
    error_message: Optional[str] = None


class NaturalLanguageQueryTranslator:
    """Translates natural language to graph database queries"""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.schema = GraphSchema()
        self._build_schema_description()
    
    def _build_schema_description(self) -> str:
        """Build a comprehensive schema description for the LLM"""
        schema_desc = """
DataInsight AI Knowledge Graph Schema:

NODES:
- Dataset: Represents uploaded datasets with properties like shape, data_types, domain, quality_score
- Column: Individual columns from datasets with semantic_type, cardinality, missing_percentage
- Feature: Generated or engineered features with importance_score, generation_method
- Model: Trained models with algorithm, performance_metrics, trust_score, interpretability_score
- Project: Data science projects with objective, domain, business_goal, constraints
- BusinessObjective: High-level business goals with category, target_metric, success_threshold
- User: Users of the system with role, department, expertise_level
- Execution: Pipeline runs with status, stage_results, resource_usage
- Pipeline: Pipeline configurations with version, configuration, stages

RELATIONSHIPS:
- HAS_COLUMN: Dataset → Column (dataset contains columns)
- GENERATES_FEATURE: Column → Feature (column generates features)
- DERIVED_FROM: Feature → Column/Feature (feature derived from source)
- TRAINED_ON: Model → Feature/Dataset (model trained on features/data)
- APPLIES_TO: Model/Pipeline → Project (model applies to project)
- ACHIEVES: Project → BusinessObjective (project achieves objective)
- EVALUATED_BY: Project/Model → User (evaluated by user)
- HAS_EXECUTION: Project → Execution (project has execution runs)
- USES_PIPELINE: Execution → Pipeline (execution uses pipeline)

EXAMPLE QUERIES:
1. "What features are most predictive for churn?"
   → MATCH (f:Feature)-[:USED_IN]->(m:Model)-[:APPLIES_TO]->(p:Project)-[:ACHIEVES]->(o:BusinessObjective) 
      WHERE o.name CONTAINS 'churn' 
      RETURN f.name, f.importance_score ORDER BY f.importance_score DESC LIMIT 10

2. "Which models perform best on financial data?"
   → MATCH (m:Model)-[:APPLIES_TO]->(p:Project) 
      WHERE p.domain = 'finance' 
      RETURN m.algorithm, m.performance_metrics ORDER BY m.trust_score DESC LIMIT 5

3. "Show me projects similar to customer segmentation"
   → MATCH (p:Project)-[:ACHIEVES]->(o:BusinessObjective) 
      WHERE p.business_goal CONTAINS 'segmentation' OR o.category = 'clustering'
      RETURN p.name, p.objective, p.domain, p.business_goal
"""
        
        self.schema_description = schema_desc
        return schema_desc
    
    def translate_query(self, natural_language_query: str, database_type: str = "neo4j") -> str:
        """Translate natural language to graph database query"""
        
        if database_type.lower() == "neo4j":
            return self._translate_to_cypher(natural_language_query)
        elif database_type.lower() == "postgres":
            return self._translate_to_age_sql(natural_language_query)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    def _translate_to_cypher(self, natural_language_query: str) -> str:
        """Translate to Cypher query for Neo4j"""
        
        system_prompt = f"""You are an expert Neo4j Cypher query translator. 
Your task is to translate natural language questions into valid Cypher queries based on the provided schema.

IMPORTANT RULES:
1. Only use node types and relationships that exist in the schema
2. Use proper Cypher syntax with correct property names
3. Always include RETURN clause with relevant data
4. Add appropriate LIMIT clauses (default 10) unless specified
5. Use WHERE clauses for filtering based on the question context
6. Return only the Cypher query, no explanations

{self.schema_description}

Translate this question to a Cypher query:"""

        user_prompt = f"Question: {natural_language_query}"
        
        try:
            response = self.llm.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent query generation
            )
            
            # Extract just the Cypher query from the response
            cypher_query = self._extract_cypher_from_response(response)
            return cypher_query
            
        except Exception as e:
            logger.error(f"Error translating query to Cypher: {e}")
            return ""
    
    def _translate_to_age_sql(self, natural_language_query: str) -> str:
        """Translate to SQL query with Age extension for PostgreSQL"""
        
        system_prompt = f"""You are an expert PostgreSQL Age extension query translator.
Your task is to translate natural language questions into valid SQL queries using Age graph functions.

IMPORTANT RULES:
1. Use SELECT * FROM cypher('knowledge_graph', $$ ... $$) AS (result agtype);
2. Inside the cypher function, use Cypher-like syntax
3. Always include RETURN clause with relevant data
4. Add appropriate LIMIT clauses (default 10) unless specified
5. Return only the SQL query, no explanations

{self.schema_description}

Translate this question to an Age SQL query:"""

        user_prompt = f"Question: {natural_language_query}"
        
        try:
            response = self.llm.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Extract just the SQL query from the response
            sql_query = self._extract_sql_from_response(response)
            return sql_query
            
        except Exception as e:
            logger.error(f"Error translating query to Age SQL: {e}")
            return ""
    
    def _extract_cypher_from_response(self, response: str) -> str:
        """Extract Cypher query from LLM response"""
        # Remove common prefixes and clean up
        response = response.strip()
        
        # Look for MATCH, CREATE, or other Cypher keywords at the start
        cypher_keywords = ['MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'WITH', 'RETURN']
        
        for keyword in cypher_keywords:
            if keyword in response.upper():
                start_idx = response.upper().find(keyword)
                query = response[start_idx:].strip()
                
                # Remove markdown code blocks if present
                if query.startswith('```'):
                    lines = query.split('\n')[1:]  # Skip first line with ```
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]  # Remove last line with ```
                    query = '\n'.join(lines).strip()
                
                return query
        
        return response.strip()
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        response = response.strip()
        
        # Look for SELECT statement
        if 'SELECT' in response.upper():
            start_idx = response.upper().find('SELECT')
            query = response[start_idx:].strip()
            
            # Remove markdown code blocks if present
            if query.startswith('```'):
                lines = query.split('\n')[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                query = '\n'.join(lines).strip()
            
            return query
        
        return response.strip()
    
    def synthesize_natural_response(self, query: str, results: List[Dict[str, Any]], 
                                  original_question: str) -> str:
        """Convert query results back to natural language"""
        
        if not results:
            return f"I couldn't find any results for: {original_question}"
        
        results_summary = self._summarize_results(results)
        
        system_prompt = """You are a data science expert who explains technical results in clear, business-friendly language.
Convert the technical query results into a natural, conversational response.

GUIDELINES:
1. Start with a direct answer to the user's question
2. Highlight key insights and patterns
3. Use business-friendly terminology
4. Keep it concise but informative
5. Include specific numbers/metrics when relevant
6. End with actionable insights if appropriate"""

        user_prompt = f"""
Original Question: {original_question}
Query Executed: {query}
Results Summary: {results_summary}
Number of Results: {len(results)}

Please provide a natural language explanation of these results:"""

        try:
            response = self.llm.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=400,
                temperature=0.3
            )
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing natural response: {e}")
            return f"Found {len(results)} results for your query about {original_question}."
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Create a structured summary of query results"""
        if not results:
            return "No results found."
        
        # Take first few results for summary
        summary_results = results[:5]
        
        summary_parts = []
        for i, result in enumerate(summary_results, 1):
            result_summary = f"Result {i}: "
            key_values = []
            
            for key, value in result.items():
                if isinstance(value, (str, int, float)):
                    key_values.append(f"{key}={value}")
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    nested_values = [f"{k}={v}" for k, v in value.items() if isinstance(v, (str, int, float))]
                    if nested_values:
                        key_values.append(f"{key}={{{'_'.join(nested_values[:3])}}}")
            
            result_summary += ", ".join(key_values[:3])  # Limit to 3 key-value pairs
            summary_parts.append(result_summary)
        
        return "; ".join(summary_parts)


class QueryPatternMatcher:
    """Matches common query patterns to improve translation accuracy"""
    
    def __init__(self):
        self.patterns = {
            'feature_importance': {
                'keywords': ['important', 'predictive', 'significant', 'top features', 'best features'],
                'template': """MATCH (f:Feature)-[:TRAINED_ON]->(m:Model)-[:APPLIES_TO]->(p:Project)
                             WHERE {domain_filter}
                             RETURN f.name, f.importance_score, f.generation_method, m.algorithm
                             ORDER BY f.importance_score DESC LIMIT {limit}"""
            },
            'model_performance': {
                'keywords': ['best model', 'top model', 'performance', 'accuracy', 'best performing'],
                'template': """MATCH (m:Model)-[:APPLIES_TO]->(p:Project)
                             WHERE {domain_filter}
                             RETURN m.algorithm, m.performance_metrics, m.trust_score, p.name
                             ORDER BY m.trust_score DESC LIMIT {limit}"""
            },
            'similar_projects': {
                'keywords': ['similar', 'like', 'comparable', 'related projects'],
                'template': """MATCH (p:Project)-[:ACHIEVES]->(o:BusinessObjective)
                             WHERE {similarity_filter}
                             RETURN p.name, p.domain, p.business_goal, p.objective
                             LIMIT {limit}"""
            },
            'data_quality': {
                'keywords': ['quality', 'issues', 'problems', 'missing data', 'data quality'],
                'template': """MATCH (d:Dataset)-[:HAS_COLUMN]->(c:Column)
                             WHERE {quality_filter}
                             RETURN d.name, d.quality_score, c.name, c.missing_percentage
                             ORDER BY d.quality_score ASC LIMIT {limit}"""
            }
        }
    
    def match_pattern(self, query: str) -> Optional[Dict[str, Any]]:
        """Match query against known patterns"""
        query_lower = query.lower()
        
        for pattern_name, pattern_info in self.patterns.items():
            if any(keyword in query_lower for keyword in pattern_info['keywords']):
                return {
                    'pattern': pattern_name,
                    'template': pattern_info['template'],
                    'confidence': 0.8
                }
        
        return None