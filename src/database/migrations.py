"""Database migration utilities for schema management"""

import logging
from typing import Optional
from pathlib import Path

try:
    from alembic.config import Config
    from alembic import command
    from sqlalchemy import inspect

    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False

from .models import Base
from .connection import DatabaseManager


def run_migrations(database_manager: DatabaseManager, alembic_config_path: Optional[str] = None) -> bool:
    """Run database migrations using Alembic"""

    if not ALEMBIC_AVAILABLE:
        logging.warning("Alembic not available, falling back to direct table creation")
        return create_tables_direct(database_manager)

    try:
        config_path = alembic_config_path or "alembic.ini"
        if not Path(config_path).exists():
            logging.info("Alembic config not found, creating tables directly")
            return create_tables_direct(database_manager)

        alembic_cfg = Config(config_path)
        alembic_cfg.set_main_option("sqlalchemy.url", database_manager.config.database_url)

        command.upgrade(alembic_cfg, "head")
        logging.info("Database migrations completed successfully")
        return True

    except Exception as e:
        logging.error(f"Migration failed: {e}")
        logging.info("Attempting direct table creation as fallback")
        return create_tables_direct(database_manager)


def create_tables_direct(database_manager: DatabaseManager) -> bool:
    """Create database tables directly using SQLAlchemy"""

    try:
        Base.metadata.create_all(bind=database_manager.engine)
        logging.info("Database tables created successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to create tables: {e}")
        return False


def check_database_schema(database_manager: DatabaseManager) -> dict:
    """Check current database schema status"""

    try:
        inspector = inspect(database_manager.engine)
        existing_tables = inspector.get_table_names()

        expected_tables = [
            "dataset_characteristics",
            "project_configs",
            "pipeline_executions",
            "learning_patterns",
            "execution_feedback",
            "system_metrics",
            "model_registry",
        ]

        missing_tables = [table for table in expected_tables if table not in existing_tables]
        extra_tables = [table for table in existing_tables if table not in expected_tables]

        return {
            "status": "complete" if not missing_tables else "incomplete",
            "existing_tables": existing_tables,
            "missing_tables": missing_tables,
            "extra_tables": extra_tables,
            "total_expected": len(expected_tables),
            "total_existing": len(existing_tables),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def initialize_database_schema(database_manager: DatabaseManager) -> bool:
    """Initialize database schema with error handling and validation"""

    logger = logging.getLogger(__name__)

    try:
        logger.info("Checking database connectivity...")
        health_check = database_manager.test_connection()
        if health_check["status"] != "healthy":
            logger.error(f"Database connection failed: {health_check.get('error', 'Unknown error')}")
            return False

        logger.info("Checking existing database schema...")
        schema_status = check_database_schema(database_manager)

        if schema_status["status"] == "complete":
            logger.info("Database schema is up to date")
            return True

        if schema_status["status"] == "error":
            logger.error(f"Schema check failed: {schema_status.get('error')}")
            return False

        missing_tables = schema_status.get("missing_tables", [])
        if missing_tables:
            logger.info(f"Missing tables detected: {missing_tables}")
            logger.info("Running database initialization...")

            success = run_migrations(database_manager)
            if success:
                # Verify schema after migration
                final_status = check_database_schema(database_manager)
                if final_status["status"] == "complete":
                    logger.info("Database initialization completed successfully")
                    return True
                else:
                    logger.error("Schema verification failed after migration")
                    return False
            else:
                logger.error("Database initialization failed")
                return False

        return True

    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False
