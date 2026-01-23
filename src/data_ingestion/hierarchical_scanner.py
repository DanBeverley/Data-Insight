"""
Hierarchical dataset scanner for folder-based datasets

Scans folder structures containing multiple CSV/Excel/Parquet files
and provides indexing, metadata, and smart selection capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FolderInfo:
    """Information about a folder in hierarchical dataset"""

    name: str
    path: str
    file_count: int
    total_size_mb: float
    file_types: Dict[str, int]
    sample_files: List[str]
    schema: Optional[Dict[str, Any]] = None


@dataclass
class DatasetIndex:
    """Complete index of hierarchical dataset"""

    root_path: str
    total_files: int
    total_size_mb: float
    folders: Dict[str, FolderInfo]
    metadata_files: List[str]
    scanned_at: str
    supports_date_filtering: bool = False
    date_column: Optional[str] = None


class HierarchicalDatasetScanner:
    """Scans and indexes hierarchical folder-based datasets"""

    SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}
    MAX_SAMPLE_FILES = 100
    SCHEMA_SAMPLE_ROWS = 10

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def scan_folder(self, root_path: Path, max_depth: int = 3, include_schema: bool = True) -> DatasetIndex:
        """
        Scan folder structure and create comprehensive index

        Args:
            root_path: Root folder to scan
            max_depth: Maximum folder depth to scan
            include_schema: Whether to sample schemas from files

        Returns:
            DatasetIndex with complete folder structure
        """
        root_path = Path(root_path)

        if not root_path.exists():
            raise FileNotFoundError(f"Path does not exist: {root_path}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_path}")

        self.logger.info(f"Scanning hierarchical dataset at: {root_path}")

        folders = {}
        metadata_files = []
        total_files = 0
        total_size = 0

        for folder_path in self._discover_folders(root_path, max_depth):
            folder_info = self._scan_single_folder(folder_path, root_path, include_schema)

            if folder_info.file_count > 0:
                folders[folder_info.name] = folder_info
                total_files += folder_info.file_count
                total_size += folder_info.total_size_mb

        metadata_files = self._find_metadata_files(root_path)

        date_support, date_col = self._detect_date_column_support(folders)

        index = DatasetIndex(
            root_path=str(root_path),
            total_files=total_files,
            total_size_mb=round(total_size, 2),
            folders=folders,
            metadata_files=metadata_files,
            scanned_at=datetime.now().isoformat(),
            supports_date_filtering=date_support,
            date_column=date_col,
        )

        self.logger.info(f"Scan complete: {total_files} files in {len(folders)} folders " f"({total_size:.2f} MB)")

        return index

    def _discover_folders(self, root_path: Path, max_depth: int) -> List[Path]:
        """Recursively discover all folders up to max_depth"""
        folders = []

        def _recurse(current_path: Path, depth: int):
            if depth > max_depth:
                return

            try:
                for item in current_path.iterdir():
                    if item.is_dir():
                        has_data_files = any(
                            f.suffix.lower() in self.SUPPORTED_EXTENSIONS for f in item.iterdir() if f.is_file()
                        )

                        if has_data_files:
                            folders.append(item)

                        _recurse(item, depth + 1)
            except PermissionError:
                self.logger.warning(f"Permission denied: {current_path}")

        _recurse(root_path, 0)
        return folders

    def _scan_single_folder(self, folder_path: Path, root_path: Path, include_schema: bool) -> FolderInfo:
        """Scan a single folder for data files"""
        data_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS]

        file_types = {}
        total_size = 0

        for file in data_files:
            ext = file.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            total_size += file.stat().st_size

        sample_files = [f.name for f in data_files[: self.MAX_SAMPLE_FILES]]

        schema = None
        if include_schema and data_files:
            schema = self._extract_schema(data_files[0])

        relative_name = folder_path.relative_to(root_path).as_posix()

        return FolderInfo(
            name=relative_name,
            path=str(folder_path),
            file_count=len(data_files),
            total_size_mb=round(total_size / 1_000_000, 2),
            file_types=file_types,
            sample_files=sample_files,
            schema=schema,
        )

    def _extract_schema(self, file_path: Path) -> Dict[str, Any]:
        """Extract schema from sample file"""
        try:
            ext = file_path.suffix.lower()

            if ext == ".csv":
                df = pd.read_csv(file_path, nrows=self.SCHEMA_SAMPLE_ROWS)
            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
                df = df.head(self.SCHEMA_SAMPLE_ROWS)
            elif ext in {".xlsx", ".xls"}:
                df = pd.read_excel(file_path, nrows=self.SCHEMA_SAMPLE_ROWS)
            else:
                return None

            return {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "sample_values": {col: df[col].dropna().head(3).tolist() for col in df.columns},
            }

        except Exception as e:
            self.logger.warning(f"Failed to extract schema from {file_path}: {e}")
            return None

    def _find_metadata_files(self, root_path: Path) -> List[str]:
        """Find metadata/description files in root"""
        metadata_patterns = ["*meta*.csv", "*info*.csv", "*readme*", "*description*"]
        metadata_files = []

        for pattern in metadata_patterns:
            for file in root_path.glob(pattern):
                if file.is_file():
                    metadata_files.append(file.name)

        return metadata_files

    def _detect_date_column_support(self, folders: Dict[str, FolderInfo]) -> Tuple[bool, Optional[str]]:
        """Detect if dataset supports date-based filtering"""
        date_columns = ["date", "Date", "datetime", "timestamp", "time"]

        for folder_info in folders.values():
            if folder_info.schema:
                schema_cols = folder_info.schema.get("columns", [])
                for date_col in date_columns:
                    if date_col in schema_cols:
                        return True, date_col

        return False, None

    def get_selection_suggestions(
        self, index: DatasetIndex, max_files: int = 1000, max_size_mb: int = 500
    ) -> Dict[str, Any]:
        """Generate smart selection suggestions based on constraints"""
        suggestions = {"strategy": None, "folders": [], "estimated_files": 0, "estimated_size_mb": 0, "warnings": []}

        if index.total_files <= max_files and index.total_size_mb <= max_size_mb:
            suggestions["strategy"] = "load_all"
            suggestions["folders"] = list(index.folders.keys())
            suggestions["estimated_files"] = index.total_files
            suggestions["estimated_size_mb"] = index.total_size_mb
            return suggestions

        sorted_folders = sorted(index.folders.items(), key=lambda x: x[1].file_count)

        selected_folders = []
        selected_files = 0
        selected_size = 0

        for folder_name, folder_info in sorted_folders:
            if selected_files + folder_info.file_count <= max_files:
                if selected_size + folder_info.total_size_mb <= max_size_mb:
                    selected_folders.append(folder_name)
                    selected_files += folder_info.file_count
                    selected_size += folder_info.total_size_mb

        if selected_files > 0:
            suggestions["strategy"] = "partial_folders"
            suggestions["folders"] = selected_folders
            suggestions["estimated_files"] = selected_files
            suggestions["estimated_size_mb"] = round(selected_size, 2)
            suggestions["warnings"].append(
                f"Loading {selected_files}/{index.total_files} files "
                f"({len(selected_folders)}/{len(index.folders)} folders)"
            )
        else:
            suggestions["strategy"] = "sampling"
            suggestions["warnings"].append("Dataset too large - recommend sampling within folders")

        return suggestions

    def to_dict(self, index: DatasetIndex) -> Dict[str, Any]:
        """Convert DatasetIndex to dictionary for API response"""
        return {
            "root_path": index.root_path,
            "total_files": index.total_files,
            "total_size_mb": index.total_size_mb,
            "folders": {name: asdict(folder_info) for name, folder_info in index.folders.items()},
            "metadata_files": index.metadata_files,
            "scanned_at": index.scanned_at,
            "supports_date_filtering": index.supports_date_filtering,
            "date_column": index.date_column,
        }
