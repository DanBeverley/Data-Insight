"""Production database configuration management"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseSettings:
    """Database configuration settings"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "data_insight"
    username: str = "postgres"
    password: str = ""
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Performance settings
    echo_sql: bool = False
    statement_timeout: int = 300
    lock_timeout: int = 10
    
    # Backup and maintenance
    backup_retention_days: int = 30
    auto_vacuum: bool = True
    enable_monitoring: bool = True
    
    # Feature flags
    enable_optimized_queries: bool = True
    enable_batch_operations: bool = True
    enable_connection_pooling: bool = True


class DatabaseConfigManager:
    """Manages database configuration from multiple sources"""
    
    def __init__(self):
        self.settings = DatabaseSettings()
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment and config files"""
        
        # Load from environment variables
        self._load_from_environment()
        
        # Load from config file if exists
        config_file = Path("config/database.yml")
        if config_file.exists():
            self._load_from_file(config_file)
    
    def _load_from_environment(self):
        """Load database settings from environment variables"""
        
        env_mappings = {
            'POSTGRES_HOST': ('host', str),
            'POSTGRES_PORT': ('port', int),
            'POSTGRES_DB': ('database', str),
            'POSTGRES_USER': ('username', str),
            'POSTGRES_PASSWORD': ('password', str),
            'DATABASE_POOL_SIZE': ('pool_size', int),
            'DATABASE_MAX_OVERFLOW': ('max_overflow', int),
            'DATABASE_POOL_TIMEOUT': ('pool_timeout', int),
            'DATABASE_ECHO': ('echo_sql', bool),
            'DATABASE_STATEMENT_TIMEOUT': ('statement_timeout', int),
            'DATABASE_ENABLE_MONITORING': ('enable_monitoring', bool),
            'DATABASE_OPTIMIZED_QUERIES': ('enable_optimized_queries', bool),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if attr_type == bool:
                    parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    parsed_value = attr_type(value)
                setattr(self.settings, attr_name, parsed_value)
    
    def _load_from_file(self, config_file: Path):
        """Load settings from YAML configuration file"""
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            db_config = config_data.get('database', {})
            for key, value in db_config.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
        except ImportError:
            pass  # YAML not available, skip file loading
        except Exception as e:
            print(f"Failed to load database config from file: {e}")
    
    def get_database_url(self, for_alembic: bool = False) -> str:
        """Generate database URL for SQLAlchemy or Alembic"""
        
        # Check for direct DATABASE_URL override
        if db_url := os.getenv('DATABASE_URL'):
            return db_url
        
        # For development, allow SQLite fallback
        if self.settings.host == 'localhost' and not os.getenv('FORCE_POSTGRES'):
            sqlite_path = os.getenv('SQLITE_PATH', 'data_insight_meta.db')
            return f"sqlite:///{sqlite_path}"
        
        # Construct PostgreSQL URL
        password_part = f":{self.settings.password}" if self.settings.password else ""
        return (
            f"postgresql://{self.settings.username}{password_part}"
            f"@{self.settings.host}:{self.settings.port}/{self.settings.database}"
        )
    
    def get_connection_config(self) -> Dict[str, Any]:
        """Get connection configuration dictionary"""
        
        return {
            'database_url': self.get_database_url(),
            'echo': self.settings.echo_sql,
            'pool_size': self.settings.pool_size,
            'max_overflow': self.settings.max_overflow,
            'pool_timeout': self.settings.pool_timeout,
            'pool_recycle': self.settings.pool_recycle,
            'connect_args': {
                'connect_timeout': 10,
                'application_name': 'DataInsight_AI',
                'options': f'-c statement_timeout={self.settings.statement_timeout}s'
            } if 'postgresql' in self.get_database_url() else {'timeout': 30}
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and health check configuration"""
        
        return {
            'enable_monitoring': self.settings.enable_monitoring,
            'health_check_interval': 60,
            'connection_pool_monitoring': True,
            'slow_query_threshold': 5.0,
            'alert_on_connection_exhaustion': True
        }
    
    def get_maintenance_config(self) -> Dict[str, Any]:
        """Get database maintenance configuration"""
        
        return {
            'backup_retention_days': self.settings.backup_retention_days,
            'auto_vacuum': self.settings.auto_vacuum,
            'cleanup_interval_hours': 24,
            'index_maintenance': True,
            'statistics_update': True
        }
    
    def is_production_ready(self) -> bool:
        """Check if configuration is suitable for production"""
        
        checks = [
            self.settings.host != 'localhost',
            bool(self.settings.password),
            self.settings.pool_size >= 5,
            self.settings.enable_monitoring,
        ]
        
        return all(checks)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current database configuration"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required settings
        if not self.settings.username:
            validation_results['errors'].append("Database username not configured")
            validation_results['valid'] = False
        
        if 'postgresql' in self.get_database_url() and not self.settings.password:
            validation_results['warnings'].append("No database password configured")
        
        # Performance checks
        if self.settings.pool_size < 5:
            validation_results['warnings'].append("Small connection pool size may limit performance")
        
        if self.settings.statement_timeout > 600:
            validation_results['warnings'].append("High statement timeout may mask slow queries")
        
        # Security checks
        if self.settings.echo_sql:
            validation_results['warnings'].append("SQL echo is enabled - may log sensitive data")
        
        return validation_results


# Global configuration instance
_db_config_manager: Optional[DatabaseConfigManager] = None


def get_database_config() -> DatabaseConfigManager:
    """Get global database configuration manager"""
    global _db_config_manager
    if _db_config_manager is None:
        _db_config_manager = DatabaseConfigManager()
    return _db_config_manager


def initialize_database_config(force_reload: bool = False) -> DatabaseConfigManager:
    """Initialize or reload database configuration"""
    global _db_config_manager
    if force_reload or _db_config_manager is None:
        _db_config_manager = DatabaseConfigManager()
    return _db_config_manager