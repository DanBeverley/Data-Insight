"""Dataset Registry - Per-session multi-dataset tracking with lazy loading support."""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_BASE_DIR = Path("data/datasets")


@dataclass
class DatasetInfo:
    """Metadata for a registered dataset."""

    id: str
    filename: str
    file_path: str
    uploaded_at: str
    profiled: bool = False
    rag_indexed: bool = False
    rows: int = 0
    columns: int = 0
    file_type: str = ""
    size_bytes: int = 0


class DatasetRegistry:
    """Per-session registry for managing multiple datasets."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.storage_dir = DATASETS_BASE_DIR / session_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.storage_dir / "registry.json"
        self._datasets: Dict[str, DatasetInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                self._datasets = {k: DatasetInfo(**v) for k, v in data.get("datasets", {}).items()}
            except Exception as e:
                logger.warning(f"[REGISTRY] Failed to load: {e}")
                self._datasets = {}

    def _save(self) -> None:
        """Persist registry to disk."""
        data = {"datasets": {k: asdict(v) for k, v in self._datasets.items()}}
        self.registry_path.write_text(json.dumps(data, indent=2))

    def register(self, filename: str, source_path: str, rows: int = 0, columns: int = 0) -> DatasetInfo:
        """Register a new dataset, copying file to session storage."""
        import hashlib

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_path = self.storage_dir / filename
        if source != dest_path:
            shutil.copy2(source, dest_path)

        dataset_id = f"ds_{hashlib.md5(filename.encode()).hexdigest()[:8]}"

        info = DatasetInfo(
            id=dataset_id,
            filename=filename,
            file_path=str(dest_path),
            uploaded_at=datetime.now().isoformat(),
            rows=rows,
            columns=columns,
            file_type=source.suffix.lower(),
            size_bytes=dest_path.stat().st_size,
        )

        self._datasets[filename] = info
        self._save()
        logger.info(f"[REGISTRY] Registered: {filename} ({rows}x{columns})")
        return info

    def get(self, filename: str) -> Optional[DatasetInfo]:
        """Get dataset info by filename."""
        return self._datasets.get(filename)

    def list_all(self) -> List[DatasetInfo]:
        """List all registered datasets."""
        return list(self._datasets.values())

    def mark_profiled(self, filename: str) -> None:
        """Mark a dataset as profiled."""
        if filename in self._datasets:
            self._datasets[filename].profiled = True
            self._save()

    def mark_rag_indexed(self, filename: str) -> None:
        """Mark a dataset as indexed in RAG."""
        if filename in self._datasets:
            self._datasets[filename].rag_indexed = True
            self._save()

    def get_file_path(self, filename: str) -> Optional[str]:
        """Get the file path for a dataset."""
        info = self._datasets.get(filename)
        return info.file_path if info else None

    def load_dataframe(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a dataset as DataFrame based on file type."""
        info = self._datasets.get(filename)
        if not info:
            return None

        path = Path(info.file_path)
        if not path.exists():
            return None

        suffix = info.file_type.lower()

        try:
            if suffix == ".csv":
                return pd.read_csv(path)
            elif suffix == ".tsv":
                return pd.read_csv(path, sep="\t")
            elif suffix in [".xlsx", ".xls"]:
                return pd.read_excel(path)
            elif suffix == ".json":
                return pd.read_json(path)
            elif suffix == ".parquet":
                return pd.read_parquet(path)
            else:
                logger.warning(f"[REGISTRY] Unsupported format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"[REGISTRY] Failed to load {filename}: {e}")
            return None

    def delete(self, filename: str) -> bool:
        """Remove a dataset from registry and disk."""
        if filename not in self._datasets:
            return False

        info = self._datasets[filename]
        try:
            Path(info.file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"[REGISTRY] Failed to delete file: {e}")

        del self._datasets[filename]
        self._save()
        return True

    def count(self) -> int:
        """Get total number of registered datasets."""
        return len(self._datasets)
