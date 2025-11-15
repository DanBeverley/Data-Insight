"""Dataset manifest generator for intelligent file handling"""

import os
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_manifest(root_path: Path, session_id: str) -> Dict[str, Any]:
    """
    Create a simple manifest describing the dataset structure.
    No rigid rules - just describe what's there, let the LLM decide what to do.
    """
    manifest = {
        "session_id": session_id,
        "root_path": str(root_path),
        "structure": {},
        "files": [],
        "total_files": 0,
        "total_size_mb": 0,
    }

    total_size = 0

    for item_path in root_path.rglob("*"):
        if item_path.is_file():
            relative_path = item_path.relative_to(root_path)
            file_size = item_path.stat().st_size
            total_size += file_size

            file_info = {
                "path": str(relative_path),
                "name": item_path.name,
                "extension": item_path.suffix,
                "size_bytes": file_size,
                "parent_folder": str(relative_path.parent) if relative_path.parent != Path(".") else "root",
            }

            manifest["files"].append(file_info)

            folder_key = str(relative_path.parent) if relative_path.parent != Path(".") else "root"
            if folder_key not in manifest["structure"]:
                manifest["structure"][folder_key] = []
            manifest["structure"][folder_key].append(item_path.name)

    manifest["total_files"] = len(manifest["files"])
    manifest["total_size_mb"] = round(total_size / (1024 * 1024), 2)

    file_types = {}
    for f in manifest["files"]:
        ext = f["extension"] or "no_extension"
        file_types[ext] = file_types.get(ext, 0) + 1
    manifest["file_types_summary"] = file_types

    return manifest


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract ZIP file to destination, return extraction path"""
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    logger.info(f"Extracted ZIP to {extract_to}")
    return extract_to


def handle_file_upload(file_path: Path, filename: str, session_id: str) -> Dict[str, Any]:
    """
    Intelligently handle any uploaded file.
    Returns manifest describing what was uploaded.
    """
    result = {
        "success": False,
        "type": "unknown",
        "manifest": None,
        "data_path": None,
    }

    try:
        if filename.endswith(".zip"):
            extract_path = Path(tempfile.gettempdir()) / "datainsight" / session_id / "extracted"
            extracted_root = extract_zip(file_path, extract_path)

            manifest = create_manifest(extracted_root, session_id)
            result["success"] = True
            result["type"] = "archive"
            result["manifest"] = manifest
            result["data_path"] = str(extracted_root)

        else:
            single_file_dir = Path(tempfile.gettempdir()) / "datainsight" / session_id / "single_file"
            single_file_dir.mkdir(parents=True, exist_ok=True)

            dest_path = single_file_dir / filename
            if file_path != dest_path:
                import shutil

                shutil.copy2(file_path, dest_path)

            manifest = create_manifest(single_file_dir, session_id)
            result["success"] = True
            result["type"] = "single_file"
            result["manifest"] = manifest
            result["data_path"] = str(single_file_dir)

        logger.info(f"Processed upload: {filename} -> {result['type']}")
        return result

    except Exception as e:
        logger.error(f"File upload handling failed: {e}")
        result["error"] = str(e)
        return result
