"""Scalable DataFrame abstraction using DuckDB"""

import duckdb
import pandas as pd
import os
from typing import Union, List, Dict, Any, Optional


class ScalableDataFrame:
    """
    A wrapper around DuckDB to handle larger-than-memory datasets.
    Provides a pandas-like interface for common operations but executes lazily via SQL.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.conn = duckdb.connect(database=":memory:")
        self._table_name = "dataset"
        self._load_data()

    def _load_data(self):
        """Load data into DuckDB"""
        if not self.filepath:
            raise ValueError("Dataset path is required but was not provided")

        try:
            self.conn.execute(
                f"CREATE TABLE {self._table_name} AS SELECT * FROM read_csv_auto('{self.filepath}', ignore_errors=True)"
            )
        except Exception as e:
            if self.filepath and self.filepath.endswith((".xls", ".xlsx")):
                df = pd.read_excel(self.filepath)
                self.conn.execute(f"CREATE TABLE {self._table_name} AS SELECT * FROM df")
            else:
                raise e

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return top n rows as pandas DataFrame"""
        return self.conn.execute(f"SELECT * FROM {self._table_name} LIMIT {n}").df()

    def query(self, sql_query: str) -> pd.DataFrame:
        """Execute raw SQL query"""
        # Replace 'df' or 'dataset' in query with actual table name if needed
        # But we encourage using the table name directly or just 'SELECT * FROM dataset'
        return self.conn.execute(sql_query).df()

    def get_summary(self) -> Dict[str, Any]:
        """Get basic summary stats efficiently"""
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()[0]
        columns_info = self.conn.execute(f"DESCRIBE {self._table_name}").fetchall()
        columns = [col[0] for col in columns_info]

        return {"n_rows": row_count, "n_columns": len(columns), "columns": columns}

    def to_pandas(self) -> pd.DataFrame:
        """Convert entire dataset to pandas (WARNING: Check size first)"""
        return self.conn.execute(f"SELECT * FROM {self._table_name}").df()
