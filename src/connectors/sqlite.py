from typing import Dict, List, Optional
import pandas as pd
import sqlite3
import logging
from pathlib import Path

from .base import DatabaseConnector, ConnectionConfig, ConnectionResult, TableInfo

logger = logging.getLogger(__name__)


class SQLiteConnector(DatabaseConnector):
    def connect(self) -> ConnectionResult:
        try:
            db_path = self.config.file_path or self.config.database
            if not Path(db_path).exists():
                return ConnectionResult(
                    success=False, message="Database file not found", error=f"File does not exist: {db_path}"
                )

            self._connection = sqlite3.connect(db_path)
            tables = self.list_tables()
            return ConnectionResult(success=True, message=f"Connected to SQLite: {db_path}", tables=tables)
        except Exception as e:
            return ConnectionResult(success=False, message="Connection failed", error=str(e))

    def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    def test_connection(self) -> ConnectionResult:
        result = self.connect()
        if result.success:
            self.disconnect()
        return result

    def list_tables(self) -> List[TableInfo]:
        if not self._connection:
            return []

        cursor = self._connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")

        tables = []
        for row in cursor.fetchall():
            table_name = row[0]
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            tables.append(TableInfo(name=table_name, row_count=count))

        cursor.close()
        return tables

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        if not self._connection:
            return []

        cursor = self._connection.cursor()
        cursor.execute(f'PRAGMA table_info("{table_name}")')

        columns = []
        for row in cursor.fetchall():
            columns.append({"name": row[1], "type": row[2], "nullable": row[3] == 0})

        cursor.close()
        return columns

    def execute_query(self, query: str) -> pd.DataFrame:
        if not self._connection:
            raise ConnectionError("Not connected")
        return pd.read_sql_query(query, self._connection)

    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        query = f'SELECT * FROM "{table_name}"'
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query)
