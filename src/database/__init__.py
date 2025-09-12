"""Database module for production-grade PostgreSQL integration"""

from .connection import DatabaseManager, get_database_manager
from .models import Base, DatasetCharacteristics, ProjectConfig, PipelineExecution, LearningPattern
from .migrations import run_migrations

__all__ = [
    'DatabaseManager',
    'get_database_manager', 
    'Base',
    'DatasetCharacteristics',
    'ProjectConfig', 
    'PipelineExecution',
    'LearningPattern',
    'run_migrations'
]