"""
Hierarchical dataset loader with progress tracking and memory management

Loads datasets from folder structures with support for:
- Progressive loading with real-time progress
- Multiple loading strategies (concatenate, separate, lazy)
- Memory-efficient chunked loading
- Date filtering and sampling
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class LoadingProgress:
    """Progress information during dataset loading"""

    status: str
    current_file: str
    loaded_files: int
    total_files: int
    progress_percent: float
    current_size_mb: float
    estimated_total_mb: float
    error: Optional[str] = None


@dataclass
class LoadingResult:
    """Result of dataset loading operation"""

    success: bool
    dataframe: Optional[pd.DataFrame] = None
    shape: Optional[tuple] = None
    size_mb: Optional[float] = None
    files_loaded: int = 0
    files_failed: int = 0
    loading_time_seconds: float = 0
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HierarchicalDatasetLoader:
    """Loads hierarchical datasets with memory and performance optimization"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_dataset(
        self,
        root_path: Path,
        selection: Dict[str, Any],
        progress_callback: Optional[Callable[[LoadingProgress], None]] = None,
    ) -> Generator[LoadingProgress, None, LoadingResult]:
        """
        Load dataset with specified selection criteria

        Args:
            root_path: Root folder path
            selection: Selection configuration with filters
            progress_callback: Optional callback for progress updates

        Yields:
            LoadingProgress objects during loading

        Returns:
            LoadingResult with loaded dataframe
        """
        start_time = datetime.now()

        try:
            files_to_load = self._get_files_from_selection(root_path, selection)

            if not files_to_load:
                yield LoadingProgress(
                    status="error",
                    current_file="",
                    loaded_files=0,
                    total_files=0,
                    progress_percent=0,
                    current_size_mb=0,
                    estimated_total_mb=0,
                    error="No files match selection criteria",
                )
                return LoadingResult(success=False, error="No files match selection criteria")

            strategy = selection.get("strategy", "concatenate")

            self.logger.info(f"Loading {len(files_to_load)} files using '{strategy}' strategy")

            if strategy == "concatenate":
                result = yield from self._load_concatenated(files_to_load, selection, progress_callback)
            elif strategy == "separate":
                result = yield from self._load_separate(files_to_load, selection, progress_callback)
            elif strategy == "sample":
                result = yield from self._load_sampled(files_to_load, selection, progress_callback)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            result.loading_time_seconds = (datetime.now() - start_time).total_seconds()
            return result

        except Exception as e:
            self.logger.error(f"Loading failed: {e}", exc_info=True)
            yield LoadingProgress(
                status="error",
                current_file="",
                loaded_files=0,
                total_files=len(files_to_load) if "files_to_load" in locals() else 0,
                progress_percent=0,
                current_size_mb=0,
                estimated_total_mb=0,
                error=str(e),
            )
            return LoadingResult(success=False, error=str(e))

    def _get_files_from_selection(self, root_path: Path, selection: Dict[str, Any]) -> List[Path]:
        """Get list of files matching selection criteria"""
        root_path = Path(root_path)
        files = []

        selected_folders = selection.get("folders", [])
        file_pattern = selection.get("file_pattern", "*")
        max_files = selection.get("max_files", None)

        for folder_name in selected_folders:
            folder_path = root_path / folder_name

            if not folder_path.exists():
                continue

            for ext in [".csv", ".parquet", ".xlsx", ".xls"]:
                pattern_files = list(folder_path.glob(f"{file_pattern}{ext}"))
                files.extend(pattern_files)

        if max_files and len(files) > max_files:
            files = files[:max_files]

        return sorted(files)

    def _load_concatenated(
        self, files: List[Path], selection: Dict[str, Any], progress_callback: Optional[Callable]
    ) -> Generator[LoadingProgress, None, LoadingResult]:
        """Load all files and concatenate into single DataFrame"""
        dataframes = []
        files_loaded = 0
        files_failed = 0
        total_size = 0

        date_filter = selection.get("date_range")
        date_column = selection.get("date_column", "Date")

        for i, file_path in enumerate(files):
            try:
                df = self._load_single_file(file_path)

                if df is None:
                    files_failed += 1
                    continue

                df["_source_file"] = file_path.stem
                df["_source_folder"] = file_path.parent.name

                if date_filter and date_column in df.columns:
                    df = self._apply_date_filter(df, date_column, date_filter)

                file_size = file_path.stat().st_size / 1_000_000
                total_size += file_size

                dataframes.append(df)
                files_loaded += 1

                progress = LoadingProgress(
                    status="loading",
                    current_file=file_path.name,
                    loaded_files=files_loaded,
                    total_files=len(files),
                    progress_percent=round((i + 1) / len(files) * 100, 1),
                    current_size_mb=round(total_size, 2),
                    estimated_total_mb=round(total_size / (i + 1) * len(files), 2),
                )

                yield progress

                if progress_callback:
                    progress_callback(progress)

            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
                files_failed += 1

        if not dataframes:
            return LoadingResult(success=False, error="All files failed to load", files_failed=files_failed)

        self.logger.info(f"Concatenating {len(dataframes)} DataFrames...")

        final_df = pd.concat(dataframes, ignore_index=True)

        df_size_mb = final_df.memory_usage(deep=True).sum() / 1_000_000

        return LoadingResult(
            success=True,
            dataframe=final_df,
            shape=final_df.shape,
            size_mb=round(df_size_mb, 2),
            files_loaded=files_loaded,
            files_failed=files_failed,
            metadata={
                "strategy": "concatenate",
                "source_files": [f.stem for f in files[:100]],
                "total_source_files": len(files),
                "has_source_tracking": True,
            },
        )

    def _load_separate(
        self, files: List[Path], selection: Dict[str, Any], progress_callback: Optional[Callable]
    ) -> Generator[LoadingProgress, None, LoadingResult]:
        """Load files separately (returns dict of DataFrames)"""
        dataframes_dict = {}
        files_loaded = 0
        files_failed = 0
        total_size = 0

        for i, file_path in enumerate(files):
            try:
                df = self._load_single_file(file_path)

                if df is None:
                    files_failed += 1
                    continue

                key = file_path.stem
                dataframes_dict[key] = df

                file_size = file_path.stat().st_size / 1_000_000
                total_size += file_size
                files_loaded += 1

                progress = LoadingProgress(
                    status="loading",
                    current_file=file_path.name,
                    loaded_files=files_loaded,
                    total_files=len(files),
                    progress_percent=round((i + 1) / len(files) * 100, 1),
                    current_size_mb=round(total_size, 2),
                    estimated_total_mb=round(total_size / (i + 1) * len(files), 2),
                )

                yield progress

                if progress_callback:
                    progress_callback(progress)

            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
                files_failed += 1

        total_rows = sum(len(df) for df in dataframes_dict.values())

        return LoadingResult(
            success=True,
            dataframe=dataframes_dict,
            shape=(total_rows, "varies"),
            size_mb=round(total_size, 2),
            files_loaded=files_loaded,
            files_failed=files_failed,
            metadata={
                "strategy": "separate",
                "ticker_count": len(dataframes_dict),
                "tickers": list(dataframes_dict.keys())[:100],
            },
        )

    def _load_sampled(
        self, files: List[Path], selection: Dict[str, Any], progress_callback: Optional[Callable]
    ) -> Generator[LoadingProgress, None, LoadingResult]:
        """Load with sampling (sample rows from each file)"""
        sample_size = selection.get("sample_size", 1000)

        dataframes = []
        files_loaded = 0
        files_failed = 0

        for i, file_path in enumerate(files):
            try:
                df = self._load_single_file(file_path, sample_size=sample_size)

                if df is None:
                    files_failed += 1
                    continue

                df["_source_file"] = file_path.stem
                dataframes.append(df)
                files_loaded += 1

                progress = LoadingProgress(
                    status="sampling",
                    current_file=file_path.name,
                    loaded_files=files_loaded,
                    total_files=len(files),
                    progress_percent=round((i + 1) / len(files) * 100, 1),
                    current_size_mb=0,
                    estimated_total_mb=0,
                )

                yield progress

                if progress_callback:
                    progress_callback(progress)

            except Exception as e:
                self.logger.warning(f"Failed to sample {file_path}: {e}")
                files_failed += 1

        if not dataframes:
            return LoadingResult(success=False, error="All files failed to load", files_failed=files_failed)

        final_df = pd.concat(dataframes, ignore_index=True)

        return LoadingResult(
            success=True,
            dataframe=final_df,
            shape=final_df.shape,
            size_mb=round(final_df.memory_usage(deep=True).sum() / 1_000_000, 2),
            files_loaded=files_loaded,
            files_failed=files_failed,
            metadata={"strategy": "sample", "sample_size_per_file": sample_size, "is_sample": True},
        )

    def _load_single_file(self, file_path: Path, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load a single file with appropriate reader"""
        ext = file_path.suffix.lower()

        try:
            if ext == ".csv":
                if sample_size:
                    return pd.read_csv(file_path, nrows=sample_size)
                return pd.read_csv(file_path)

            elif ext == ".parquet":
                df = pd.read_parquet(file_path)
                if sample_size:
                    return df.head(sample_size)
                return df

            elif ext in {".xlsx", ".xls"}:
                if sample_size:
                    return pd.read_excel(file_path, nrows=sample_size)
                return pd.read_excel(file_path)

            else:
                self.logger.warning(f"Unsupported file type: {ext}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def _apply_date_filter(self, df: pd.DataFrame, date_column: str, date_range: List[str]) -> pd.DataFrame:
        """Apply date range filter to DataFrame"""
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])

                mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
                return df[mask]

            return df

        except Exception as e:
            self.logger.warning(f"Date filtering failed: {e}")
            return df
