"""Dataset manifest generator for intelligent file handling"""

import os
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

import logging
import sys

# Add src to path to allow imports if running from different context
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.data_ingestion.hierarchical_scanner import HierarchicalDatasetScanner

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
            extract_path = Path(tempfile.gettempdir()) / "Quorvix" / session_id / "extracted"
            extracted_root = extract_zip(file_path, extract_path)

            # Use HierarchicalDatasetScanner for richer manifest
            try:
                scanner = HierarchicalDatasetScanner()
                index = scanner.scan_folder(extracted_root, max_depth=3, include_schema=True)

                # Convert index to manifest format expected by frontend/agent
                manifest = {
                    "session_id": session_id,
                    "root_path": str(extracted_root),
                    "structure": {name: info.file_count for name, info in index.folders.items()},
                    "files": [],  # Populated below if needed, or we rely on the index
                    "total_files": index.total_files,
                    "total_size_mb": index.total_size_mb,
                    "hierarchy_index": scanner.to_dict(index),  # Embed full index
                    "type": "hierarchical",
                }

                # Populate flat file list for backward compatibility
                for folder_name, folder_info in index.folders.items():
                    for file_name in folder_info.sample_files:  # Note: this is just samples
                        manifest["files"].append(
                            {
                                "path": f"{folder_name}/{file_name}",
                                "name": file_name,
                                "parent_folder": folder_name,
                                "size_bytes": 0,  # We don't have individual size in this view easily
                            }
                        )

            except Exception as scan_error:
                logger.error(f"Hierarchical scan failed, falling back to basic: {scan_error}")
                manifest = create_manifest(extracted_root, session_id)

            result["success"] = True
            result["type"] = "archive"
            result["manifest"] = manifest
            result["data_path"] = str(extracted_root)

        else:
            single_file_dir = Path(tempfile.gettempdir()) / "Quorvix" / session_id / "single_file"
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
