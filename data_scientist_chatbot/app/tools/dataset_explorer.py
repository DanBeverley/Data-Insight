"""
Intelligent dataset exploration tools for LLM agents.
These tools let agents explore complex datasets naturally without rigid rules.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)


def inspect_dataset(session_id: str) -> str:
    """
    Get overview of uploaded dataset structure.
    Returns manifest as JSON string for the LLM to understand what's available.
    """
    try:
        import builtins

        session_store = getattr(builtins, "_session_store", {})

        if session_id not in session_store:
            return json.dumps({"error": "No dataset loaded for this session"})

        manifest = session_store[session_id].get("dataset_manifest")
        if not manifest:
            return json.dumps({"error": "No dataset manifest found"})

        summary = {
            "total_files": manifest["total_files"],
            "total_size_mb": manifest["total_size_mb"],
            "file_types": manifest.get("file_types_summary", {}),
            "folders": list(manifest["structure"].keys()),
            "sample_files": manifest["files"][:20] if len(manifest["files"]) > 20 else manifest["files"],
        }

        return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def list_files(session_id: str, folder: Optional[str] = None, extension: Optional[str] = None) -> str:
    """
    List files in the dataset. Can filter by folder or extension.
    Returns file listing as JSON.
    """
    try:
        import builtins

        session_store = getattr(builtins, "_session_store", {})

        manifest = session_store.get(session_id, {}).get("dataset_manifest")
        if not manifest:
            return json.dumps({"error": "No dataset loaded"})

        files = manifest["files"]

        if folder:
            files = [f for f in files if f["parent_folder"] == folder]

        if extension:
            ext = extension if extension.startswith(".") else f".{extension}"
            files = [f for f in files if f["extension"] == ext]

        return json.dumps(
            {
                "count": len(files),
                "files": [
                    {"path": f["path"], "name": f["name"], "size_kb": round(f["size_bytes"] / 1024, 2)} for f in files
                ],
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


def load_file(session_id: str, file_path: str) -> str:
    """
    Intelligently load any file from the uploaded dataset.
    Tries pandas first, falls back to text. Returns data info or preview.
    """
    try:
        import builtins

        session_store = getattr(builtins, "_session_store", {})

        manifest = session_store.get(session_id, {}).get("dataset_manifest")
        if not manifest:
            return json.dumps({"error": "No dataset loaded"})

        root_path = Path(manifest["root_path"])
        full_path = root_path / file_path

        if not full_path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})

        try:
            df = pd.read_csv(full_path)
            session_store[session_id]["dataframe"] = df
            session_store[session_id]["dataset_path"] = str(full_path)

            return json.dumps(
                {
                    "success": True,
                    "type": "dataframe",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "preview": df.head(5).to_dict(),
                    "loaded_to_session": True,
                },
                indent=2,
            )

        except Exception as pd_error:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1000)

            return json.dumps(
                {
                    "success": True,
                    "type": "text",
                    "preview": content,
                    "note": f"Could not load as dataframe: {str(pd_error)}",
                },
                indent=2,
            )

    except Exception as e:
        return json.dumps({"error": str(e)})


def combine_files(session_id: str, file_pattern: str) -> str:
    """
    Combine multiple files matching a pattern into single dataframe.
    Example: combine_files("stocks/*.csv") to combine all stock CSVs.
    """
    try:
        import builtins
        from glob import glob

        session_store = getattr(builtins, "_session_store", {})
        manifest = session_store.get(session_id, {}).get("dataset_manifest")

        if not manifest:
            return json.dumps({"error": "No dataset loaded"})

        root_path = Path(manifest["root_path"])
        matching_files = list(root_path.glob(file_pattern))

        if not matching_files:
            return json.dumps({"error": f"No files match pattern: {file_pattern}"})

        dfs = []
        failed = []

        for file_path in matching_files[:100]:
            try:
                df = pd.read_csv(file_path)
                df["source_file"] = file_path.name
                dfs.append(df)
            except Exception as e:
                failed.append({"file": file_path.name, "error": str(e)})

        if not dfs:
            return json.dumps({"error": "No files could be loaded", "failed": failed})

        combined_df = pd.concat(dfs, ignore_index=True)
        session_store[session_id]["dataframe"] = combined_df

        return json.dumps(
            {
                "success": True,
                "files_combined": len(dfs),
                "files_failed": len(failed),
                "combined_shape": combined_df.shape,
                "columns": list(combined_df.columns),
                "loaded_to_session": True,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})
