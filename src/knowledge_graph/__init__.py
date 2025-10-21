"""
Knowledge Graph Module for DataInsight AI

This module provides graph-based knowledge representation and querying
capabilities for the DataInsight AI platform, enabling relationship-aware
data science intelligence.
"""

from .schema import GraphSchema, NodeType, RelationshipType
from .service import KnowledgeGraphService

__all__ = ["GraphSchema", "NodeType", "RelationshipType", "KnowledgeGraphService"]
