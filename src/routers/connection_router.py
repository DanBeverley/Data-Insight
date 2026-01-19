import logging
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.connectors.base import ConnectionConfig
from src.connectors.service import get_connection_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/connections", tags=["connections"])


class TestConnectionRequest(BaseModel):
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    file_path: Optional[str] = None


class ConnectRequest(BaseModel):
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    file_path: Optional[str] = None
    save_connection: bool = False
    connection_name: Optional[str] = None


class QueryRequest(BaseModel):
    query: str


class LoadTableRequest(BaseModel):
    table_name: str
    limit: Optional[int] = 10000
    session_id: str


@router.post("/test")
async def test_connection(request: TestConnectionRequest):
    manager = get_connection_manager()

    config = ConnectionConfig(
        db_type=request.db_type,
        host=request.host,
        port=request.port,
        database=request.database,
        username=request.username,
        password=request.password,
        file_path=request.file_path,
    )

    result = manager.test_connection(config)

    return {
        "success": result.success,
        "message": result.message,
        "error": result.error,
        "tables": [{"name": t.name, "row_count": t.row_count} for t in (result.tables or [])],
    }


@router.post("/connect")
async def connect_database(request: ConnectRequest):
    manager = get_connection_manager()
    connection_id = str(uuid.uuid4())

    config = ConnectionConfig(
        db_type=request.db_type,
        host=request.host,
        port=request.port,
        database=request.database,
        username=request.username,
        password=request.password,
        file_path=request.file_path,
    )

    result = manager.connect(connection_id, config)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or result.message)

    if request.save_connection:
        name = request.connection_name or f"{request.db_type}:{request.database}"
        manager.save_connection(connection_id, name, config)

    return {
        "connection_id": connection_id,
        "success": True,
        "message": result.message,
        "tables": [{"name": t.name, "row_count": t.row_count} for t in (result.tables or [])],
    }


@router.get("/saved")
async def get_saved_connections():
    manager = get_connection_manager()
    return {"connections": manager.get_saved_connections()}


@router.post("/saved/{connection_id}/connect")
async def connect_saved(connection_id: str):
    manager = get_connection_manager()

    config = manager.load_saved_connection(connection_id)
    if not config:
        raise HTTPException(status_code=404, detail="Connection not found")

    result = manager.connect(connection_id, config)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or result.message)

    return {
        "connection_id": connection_id,
        "success": True,
        "tables": [{"name": t.name, "row_count": t.row_count} for t in (result.tables or [])],
    }


@router.delete("/saved/{connection_id}")
async def delete_saved_connection(connection_id: str):
    manager = get_connection_manager()
    success = manager.delete_saved_connection(connection_id)

    if not success:
        raise HTTPException(status_code=404, detail="Connection not found")

    return {"message": "Connection deleted"}


@router.get("/{connection_id}/tables")
async def get_tables(connection_id: str):
    manager = get_connection_manager()
    tables = manager.list_tables(connection_id)

    if not tables:
        raise HTTPException(status_code=404, detail="Connection not found or no tables")

    return {"tables": [{"name": t.name, "row_count": t.row_count} for t in tables]}


@router.get("/{connection_id}/schema/{table_name}")
async def get_table_schema(connection_id: str, table_name: str):
    manager = get_connection_manager()
    schema = manager.get_table_schema(connection_id, table_name)

    if not schema:
        raise HTTPException(status_code=404, detail="Table not found")

    return {"table": table_name, "columns": schema}


@router.post("/{connection_id}/query")
async def execute_query(connection_id: str, request: QueryRequest):
    manager = get_connection_manager()

    try:
        df = manager.execute_query(connection_id, request.query)
        return {
            "success": True,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(100).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{connection_id}/load")
async def load_table_to_session(connection_id: str, request: LoadTableRequest):
    manager = get_connection_manager()

    try:
        df = manager.load_table(connection_id, request.table_name, request.limit)

        # Register with DatasetRegistry so agent can see it via list_datasets()
        from data_scientist_chatbot.app.utils.dataset_registry import DatasetRegistry
        from pathlib import Path
        import tempfile

        # Save CSV to temp location first
        temp_dir = Path(tempfile.gettempdir()) / "db_exports"
        temp_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = f"{request.table_name}.csv"
        temp_csv_path = temp_dir / csv_filename
        df.to_csv(temp_csv_path, index=False)

        # Register with DatasetRegistry (copies to data/datasets/{session_id}/)
        registry = DatasetRegistry(request.session_id)
        registry.register(filename=csv_filename, source_path=str(temp_csv_path), rows=len(df), columns=len(df.columns))

        # Clean up temp file
        temp_csv_path.unlink(missing_ok=True)

        # Clear stale artifacts/insights from previous analysis
        from src.api_utils.session_management import session_data_manager, clear_transient_agent_state

        clear_transient_agent_state(request.session_id, f"database table {request.table_name} loaded")

        session_data = session_data_manager.get_session(request.session_id)
        if not session_data:
            session_data = session_data_manager.create_session(request.session_id)

        # Store in datasets dict (supports multiple datasets)
        if "datasets" not in session_data:
            session_data["datasets"] = {}
        session_data["datasets"][csv_filename] = df

        # Also set as current dataframe for backward compat
        session_data["dataframe"] = df
        session_data["filename"] = csv_filename
        session_data["source"] = "database"
        session_data_manager.set_session(request.session_id, session_data)

        # Sync to sandbox immediately with unique filename
        try:
            from data_scientist_chatbot.app.tools import refresh_sandbox_data

            refresh_sandbox_data(request.session_id, df, csv_filename)
        except Exception as sandbox_err:
            import logging

            logging.warning(f"Sandbox sync deferred (sandbox may not be active yet): {sandbox_err}")

        return {
            "success": True,
            "rows": len(df),
            "columns": list(df.columns),
            "session_id": request.session_id,
            "message": f"Loaded {len(df)} rows from {request.table_name}. Use list_datasets() to see it.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{connection_id}")
async def disconnect(connection_id: str):
    manager = get_connection_manager()
    success = manager.disconnect(connection_id)

    return {"success": success, "message": "Disconnected" if success else "Connection not found"}
