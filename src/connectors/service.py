from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
import hashlib
import base64
import os

from .base import DatabaseConnector, ConnectionConfig, ConnectionResult, TableInfo
from .postgresql import PostgreSQLConnector
from .mysql import MySQLConnector
from .sqlite import SQLiteConnector

logger = logging.getLogger(__name__)


class ConnectionManager:
    CONNECTOR_MAP = {
        "postgresql": PostgreSQLConnector,
        "postgres": PostgreSQLConnector,
        "mysql": MySQLConnector,
        "sqlite": SQLiteConnector,
    }

    def __init__(self, db_path: str = "data/databases/connections.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_connections: Dict[str, DatabaseConnector] = {}
        self._encryption_key = os.getenv("CONNECTION_ENCRYPTION_KEY", "default_key_change_me")
        self._init_database()

    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_connections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    db_type TEXT NOT NULL,
                    host TEXT,
                    port INTEGER,
                    database_name TEXT NOT NULL,
                    username TEXT,
                    password_hash TEXT,
                    file_path TEXT,
                    options TEXT,
                    created_at TEXT,
                    last_used TEXT
                )
            """
            )
            conn.commit()

    def _encrypt(self, value: str) -> str:
        key = hashlib.sha256(self._encryption_key.encode()).digest()
        encoded = base64.b64encode(value.encode()).decode()
        return encoded

    def _decrypt(self, value: str) -> str:
        return base64.b64decode(value.encode()).decode()

    def get_connector(self, db_type: str) -> type:
        connector_class = self.CONNECTOR_MAP.get(db_type.lower())
        if not connector_class:
            raise ValueError(f"Unsupported database type: {db_type}")
        return connector_class

    def test_connection(self, config: ConnectionConfig) -> ConnectionResult:
        connector_class = self.get_connector(config.db_type)
        connector = connector_class(config)
        return connector.test_connection()

    def connect(self, connection_id: str, config: ConnectionConfig) -> ConnectionResult:
        connector_class = self.get_connector(config.db_type)
        connector = connector_class(config)
        result = connector.connect()

        if result.success:
            self._active_connections[connection_id] = connector

        return result

    def disconnect(self, connection_id: str) -> bool:
        connector = self._active_connections.pop(connection_id, None)
        if connector:
            connector.disconnect()
            return True
        return False

    def get_connection(self, connection_id: str) -> Optional[DatabaseConnector]:
        return self._active_connections.get(connection_id)

    def list_tables(self, connection_id: str) -> List[TableInfo]:
        connector = self.get_connection(connection_id)
        if not connector:
            return []
        return connector.list_tables()

    def get_table_schema(self, connection_id: str, table_name: str) -> List[Dict]:
        connector = self.get_connection(connection_id)
        if not connector:
            return []
        return connector.get_table_schema(table_name)

    def execute_query(self, connection_id: str, query: str):
        connector = self.get_connection(connection_id)
        if not connector:
            raise ConnectionError("Connection not found")
        return connector.execute_query(query)

    def load_table(self, connection_id: str, table_name: str, limit: Optional[int] = None):
        connector = self.get_connection(connection_id)
        if not connector:
            raise ConnectionError("Connection not found")
        return connector.load_table(table_name, limit)

    def save_connection(self, connection_id: str, name: str, config: ConnectionConfig) -> bool:
        with self._get_db() as conn:
            password_hash = self._encrypt(config.password) if config.password else None
            conn.execute(
                """
                INSERT OR REPLACE INTO saved_connections 
                (id, name, db_type, host, port, database_name, username, password_hash, file_path, options, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    connection_id,
                    name,
                    config.db_type,
                    config.host,
                    config.port,
                    config.database,
                    config.username,
                    password_hash,
                    config.file_path,
                    json.dumps(config.options) if config.options else None,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return True

    def get_saved_connections(self) -> List[Dict[str, Any]]:
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT id, name, db_type, host, port, database_name, created_at, last_used FROM saved_connections"
            )
            return [dict(row) for row in cursor.fetchall()]

    def load_saved_connection(self, connection_id: str) -> Optional[ConnectionConfig]:
        with self._get_db() as conn:
            cursor = conn.execute("SELECT * FROM saved_connections WHERE id = ?", (connection_id,))
            row = cursor.fetchone()
            if not row:
                return None

            conn.execute(
                "UPDATE saved_connections SET last_used = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), connection_id),
            )
            conn.commit()

            return ConnectionConfig(
                db_type=row["db_type"],
                host=row["host"],
                port=row["port"],
                database=row["database_name"],
                username=row["username"],
                password=self._decrypt(row["password_hash"]) if row["password_hash"] else None,
                file_path=row["file_path"],
                options=json.loads(row["options"]) if row["options"] else {},
            )

    def delete_saved_connection(self, connection_id: str) -> bool:
        with self._get_db() as conn:
            cursor = conn.execute("DELETE FROM saved_connections WHERE id = ?", (connection_id,))
            conn.commit()
            return cursor.rowcount > 0


_manager_instance: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ConnectionManager()
    return _manager_instance
