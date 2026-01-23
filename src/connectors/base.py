from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class TableInfo:
    name: str
    row_count: Optional[int] = None
    columns: List[Dict[str, str]] = None


@dataclass
class ConnectionConfig:
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    file_path: Optional[str] = None
    options: Dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class ConnectionResult:
    success: bool
    message: str
    tables: List[TableInfo] = None
    error: Optional[str] = None


class DatabaseConnector(ABC):
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connection = None

    @abstractmethod
    def connect(self) -> ConnectionResult:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def test_connection(self) -> ConnectionResult:
        pass

    @abstractmethod
    def list_tables(self) -> List[TableInfo]:
        pass

    @abstractmethod
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        pass

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
