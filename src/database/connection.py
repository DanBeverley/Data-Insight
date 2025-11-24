"""Production-grade database connection management with PostgreSQL support"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager, asynccontextmanager
from urllib.parse import urlparse

try:
    from sqlalchemy import create_engine, event, pool
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.engine import Engine

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class DatabaseConfig:
    """Database configuration with environment-based settings"""

    def __init__(self):
        self.database_url = self._get_database_url()
        self.echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
        self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))
        self.connect_timeout = int(os.getenv("DATABASE_CONNECT_TIMEOUT", "10"))

    def _get_database_url(self) -> str:
        """Get database URL from environment with intelligent defaults"""

        if db_url := os.getenv("DATABASE_URL"):
            return db_url

        if os.getenv("POSTGRES_HOST"):
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            database = os.getenv("POSTGRES_DB", "data_insight")
            username = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "")

            return f"postgresql://{username}:{password}@{host}:{port}/{database}"

        return "sqlite:///data_insight_meta.db"

    def is_postgresql(self) -> bool:
        """Check if configured for PostgreSQL"""
        return self.database_url.startswith("postgresql")

    def is_sqlite(self) -> bool:
        """Check if configured for SQLite"""
        return self.database_url.startswith("sqlite")


class DatabaseManager:
    """Production-grade database manager with connection pooling and health monitoring"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for database operations")

        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.logger = logging.getLogger(__name__)

        self._initialize_engine()
        self._setup_event_listeners()

    def _initialize_engine(self):
        """Initialize database engine with production settings"""

        engine_kwargs = {"echo": self.config.echo, "future": True}

        if self.config.is_postgresql():
            engine_kwargs.update(
                {
                    "poolclass": QueuePool,
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "pool_timeout": self.config.pool_timeout,
                    "pool_recycle": self.config.pool_recycle,
                    "connect_args": {
                        "connect_timeout": self.config.connect_timeout,
                        "application_name": "DataInsight_AI",
                    },
                }
            )
        elif self.config.is_sqlite():
            engine_kwargs.update({"connect_args": {"check_same_thread": False, "timeout": 30}})

        self.engine = create_engine(self.config.database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        self.logger.info(f"Database engine initialized: {self._mask_password(self.config.database_url)}")

    def _setup_event_listeners(self):
        """Setup database event listeners for monitoring and optimization"""

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if self.config.is_sqlite():
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=memory")
                cursor.close()

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            if self.config.is_postgresql():
                cursor = dbapi_connection.cursor()
                cursor.execute("SET statement_timeout = '300s'")
                cursor.execute("SET lock_timeout = '10s'")
                cursor.close()

    @contextmanager
    def get_session(self):
        """Get database session with proper error handling and cleanup"""

        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> Dict[str, Any]:
        """Test database connectivity and return status"""

        try:
            with self.get_session() as session:
                if self.config.is_postgresql():
                    from sqlalchemy import text

                    result = session.execute(text("SELECT version()")).scalar()
                    version = result.split()[0] if result else "Unknown"
                else:
                    from sqlalchemy import text

                    result = session.execute(text("SELECT sqlite_version()")).scalar()
                    version = f"SQLite {result}" if result else "Unknown"

                return {
                    "status": "healthy",
                    "database_type": "PostgreSQL" if self.config.is_postgresql() else "SQLite",
                    "version": version,
                    "url": self._mask_password(self.config.database_url),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "url": self._mask_password(self.config.database_url)}

    def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status for monitoring"""

        if not self.engine or not hasattr(self.engine.pool, "status"):
            return {"status": "no_pool_info"}

        pool = self.engine.pool
        return {
            "pool_size": getattr(pool, "size", lambda: 0)(),
            "checked_in": getattr(pool, "checkedin", lambda: 0)(),
            "checked_out": getattr(pool, "checkedout", lambda: 0)(),
            "overflow": getattr(pool, "overflow", lambda: 0)(),
            "invalid": getattr(pool, "invalid", lambda: 0)(),
        }

    def execute_health_check(self) -> bool:
        """Execute comprehensive health check"""

        try:
            connection_test = self.test_connection()
            if connection_test["status"] != "healthy":
                return False

            with self.get_session() as session:
                if self.config.is_postgresql():
                    session.execute("SELECT 1").scalar()
                else:
                    session.execute("SELECT 1").scalar()

            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """Close database connections and cleanup"""

        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connections closed")

    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging"""

        try:
            parsed = urlparse(url)
            if parsed.password:
                masked_netloc = parsed.netloc.replace(parsed.password, "***")
                return url.replace(parsed.netloc, masked_netloc)
            return url
        except:
            return url.replace("://", "://***@").split("@")[-1]


_database_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Get global database manager instance (singleton pattern)"""

    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager(config)
    return _database_manager


def initialize_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Initialize database with configuration"""

    global _database_manager
    _database_manager = DatabaseManager(config)
    return _database_manager


def close_database():
    """Close global database manager"""

    global _database_manager
    if _database_manager:
        _database_manager.close()
        _database_manager = None
