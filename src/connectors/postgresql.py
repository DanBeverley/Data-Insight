from typing import Dict, List, Optional
import pandas as pd
import logging

from .base import DatabaseConnector, ConnectionConfig, ConnectionResult, TableInfo

logger = logging.getLogger(__name__)


class PostgreSQLConnector(DatabaseConnector):
    def connect(self) -> ConnectionResult:
        try:
            import psycopg2

            self._connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port or 5432,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                **self.config.options,
            )
            tables = self.list_tables()
            return ConnectionResult(
                success=True, message=f"Connected to PostgreSQL: {self.config.database}", tables=tables
            )
        except ImportError:
            return ConnectionResult(
                success=False, message="psycopg2 not installed", error="pip install psycopg2-binary"
            )
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
        cursor.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """
        )

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
        cursor.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """,
            (table_name,),
        )

        columns = []
        for row in cursor.fetchall():
            columns.append({"name": row[0], "type": row[1], "nullable": row[2] == "YES"})

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
