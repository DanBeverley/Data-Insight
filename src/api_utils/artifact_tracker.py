from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
import json
import os


class ArtifactCategory(Enum):
    VISUALIZATION = "visualization"
    DATASET = "dataset"
    MODEL = "model"
    REPORT = "report"
    ARCHIVE = "archive"
    METADATA = "metadata"
    OTHER = "other"


class ArtifactTracker:
    STORAGE_FILE = Path(__file__).parent.parent.parent / "artifact_storage.json"

    CATEGORY_MAPPING = {
        '.png': ArtifactCategory.VISUALIZATION,
        '.jpg': ArtifactCategory.VISUALIZATION,
        '.jpeg': ArtifactCategory.VISUALIZATION,
        '.svg': ArtifactCategory.VISUALIZATION,
        '.pdf': ArtifactCategory.REPORT,
        '.csv': ArtifactCategory.DATASET,
        '.parquet': ArtifactCategory.DATASET,
        '.xlsx': ArtifactCategory.DATASET,
        '.pkl': ArtifactCategory.MODEL,
        '.joblib': ArtifactCategory.MODEL,
        '.h5': ArtifactCategory.MODEL,
        '.pt': ArtifactCategory.MODEL,
        '.pth': ArtifactCategory.MODEL,
        '.onnx': ArtifactCategory.MODEL,
        '.json': ArtifactCategory.METADATA,
        '.yaml': ArtifactCategory.METADATA,
        '.yml': ArtifactCategory.METADATA,
        '.zip': ArtifactCategory.ARCHIVE,
        '.tar.gz': ArtifactCategory.ARCHIVE,
        '.tar': ArtifactCategory.ARCHIVE,
        '.html': ArtifactCategory.REPORT,
    }

    CATEGORY_ICONS = {
        ArtifactCategory.VISUALIZATION: "📊",
        ArtifactCategory.DATASET: "📁",
        ArtifactCategory.MODEL: "🤖",
        ArtifactCategory.REPORT: "📄",
        ArtifactCategory.ARCHIVE: "📦",
        ArtifactCategory.METADATA: "🏷️",
        ArtifactCategory.OTHER: "📎",
    }

    CATEGORY_LABELS = {
        ArtifactCategory.VISUALIZATION: "Visualizations",
        ArtifactCategory.DATASET: "Datasets",
        ArtifactCategory.MODEL: "Models",
        ArtifactCategory.REPORT: "Reports",
        ArtifactCategory.ARCHIVE: "Archives",
        ArtifactCategory.METADATA: "Metadata",
        ArtifactCategory.OTHER: "Other Files",
    }

    def __init__(self):
        self.session_artifacts: Dict[str, Dict[str, Any]] = {}
        self._load_from_file()

    def _load_from_file(self):
        if self.STORAGE_FILE.exists():
            try:
                with open(self.STORAGE_FILE, 'r') as f:
                    self.session_artifacts = json.load(f)
            except Exception as e:
                print(f"[ArtifactTracker] ERROR loading: {e}")
                self.session_artifacts = {}
        else:
            self.session_artifacts = {}

    def _save_to_file(self):
        try:
            self.STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.STORAGE_FILE, 'w') as f:
                json.dump(self.session_artifacts, f, indent=2)
        except Exception as e:
            print(f"[ArtifactTracker] ERROR saving: {e}")

    def categorize_file(self, filename: str, description: Optional[str] = None) -> ArtifactCategory:
        filename_lower = filename.lower()

        for ext, category in self.CATEGORY_MAPPING.items():
            if filename_lower.endswith(ext):
                if category == ArtifactCategory.METADATA and description:
                    if 'model' in description.lower() or 'pipeline' in description.lower():
                        return ArtifactCategory.MODEL
                return category

        if description:
            description_lower = description.lower()
            if 'trained model' in description_lower or 'model file' in description_lower or 'machine learning model' in description_lower:
                return ArtifactCategory.MODEL
            elif 'plot' in description_lower or 'chart' in description_lower or 'visualization' in description_lower or 'graph' in description_lower:
                return ArtifactCategory.VISUALIZATION
            elif 'dataset' in description_lower or 'data file' in description_lower:
                return ArtifactCategory.DATASET
            elif 'report' in description_lower or 'analysis' in description_lower:
                return ArtifactCategory.REPORT

        return ArtifactCategory.OTHER

    def add_artifact(
        self,
        session_id: str,
        filename: str,
        file_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if session_id not in self.session_artifacts:
            self.session_artifacts[session_id] = {
                'artifacts': [],
                'created_at': datetime.now().isoformat()
            }

        category = self.categorize_file(filename, description)
        file_hash = self._generate_file_hash(filename, file_path)

        existing = next(
            (a for a in self.session_artifacts[session_id]['artifacts']
             if a['file_hash'] == file_hash),
            None
        )

        if existing:
            existing['last_updated'] = datetime.now().isoformat()
            return existing

        artifact = {
            'artifact_id': f"{session_id}_{len(self.session_artifacts[session_id]['artifacts'])}",
            'filename': filename,
            'file_path': file_path,
            'category': category.value,
            'category_label': self.CATEGORY_LABELS[category],
            'icon': self.CATEGORY_ICONS[category],
            'description': description or self._generate_description(filename, category),
            'size': self._get_file_size(file_path),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'metadata': metadata or {},
            'file_hash': file_hash
        }

        self.session_artifacts[session_id]['artifacts'].append(artifact)
        self._save_to_file()
        print(f"[ArtifactTracker] Added: {filename}")

        return artifact

    def get_session_artifacts(
        self,
        session_id: str,
        category: Optional[ArtifactCategory] = None
    ) -> Dict[str, Any]:
        self._load_from_file()

        if session_id not in self.session_artifacts:
            return {'artifacts': [], 'categories': {}, 'total_count': 0}

        artifacts = self.session_artifacts[session_id]['artifacts']

        if category:
            artifacts = [a for a in artifacts if a['category'] == category.value]

        categorized = {}
        for cat in ArtifactCategory:
            cat_artifacts = [a for a in artifacts if a['category'] == cat.value]
            if cat_artifacts:
                categorized[cat.value] = {
                    'label': self.CATEGORY_LABELS[cat],
                    'icon': self.CATEGORY_ICONS[cat],
                    'count': len(cat_artifacts),
                    'artifacts': sorted(cat_artifacts, key=lambda x: x['created_at'], reverse=True)
                }

        return {
            'artifacts': sorted(artifacts, key=lambda x: x['created_at'], reverse=True),
            'categories': categorized,
            'total_count': len(artifacts),
            'session_created': self.session_artifacts[session_id]['created_at']
        }

    def get_new_artifacts(self, session_id: str, since: str) -> List[Dict[str, Any]]:
        self._load_from_file()
        if session_id not in self.session_artifacts:
            return []

        try:
            since_clean = since.replace('Z', '+00:00') if since.endswith('Z') else since
            since_dt = datetime.fromisoformat(since_clean)

            if since_dt.tzinfo is None:
                from datetime import timezone
                since_dt = since_dt.replace(tzinfo=timezone.utc)

            new_artifacts = []
            for a in self.session_artifacts[session_id]['artifacts']:
                artifact_dt = datetime.fromisoformat(a['created_at'])
                if artifact_dt.tzinfo is None:
                    from datetime import timezone
                    artifact_dt = artifact_dt.replace(tzinfo=timezone.utc)
                if artifact_dt > since_dt:
                    new_artifacts.append(a)

            return sorted(new_artifacts, key=lambda x: x['created_at'], reverse=True)
        except (ValueError, KeyError):
            return []

    def remove_artifact(self, session_id: str, artifact_id: str) -> bool:
        if session_id not in self.session_artifacts:
            return False

        artifacts = self.session_artifacts[session_id]['artifacts']
        initial_len = len(artifacts)

        self.session_artifacts[session_id]['artifacts'] = [
            a for a in artifacts if a['artifact_id'] != artifact_id
        ]

        removed = len(self.session_artifacts[session_id]['artifacts']) < initial_len
        if removed:
            self._save_to_file()
        return removed

    def clear_session_artifacts(self, session_id: str) -> bool:
        if session_id in self.session_artifacts:
            del self.session_artifacts[session_id]
            self._save_to_file()
            return True
        return False

    def _generate_file_hash(self, filename: str, file_path: str) -> str:
        content = f"{filename}:{file_path}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_file_size(self, file_path: str) -> str:
        try:
            if file_path.startswith('/static/'):
                relative_path = file_path[1:]
            else:
                relative_path = file_path

            path = Path(relative_path)
            if path.exists():
                size_bytes = path.stat().st_size
                return self._format_file_size(size_bytes)
        except:
            pass
        return "Unknown"

    def _format_file_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _generate_description(self, filename: str, category: ArtifactCategory) -> str:
        if category == ArtifactCategory.VISUALIZATION:
            name_without_ext = filename.rsplit('.', 1)[0]
            name_parts = name_without_ext.replace('_', ' ').replace('-', ' ')
            if name_parts.startswith('plot '):
                return "Generated visualization"
            return name_parts.title()

        descriptions = {
            ArtifactCategory.DATASET: "Processed dataset",
            ArtifactCategory.MODEL: "Trained model",
            ArtifactCategory.REPORT: "Analysis report",
            ArtifactCategory.ARCHIVE: "Archived files",
            ArtifactCategory.METADATA: "Metadata file",
            ArtifactCategory.OTHER: "Generated file"
        }
        return descriptions.get(category, "File")


_instance = None

def get_artifact_tracker():
    global _instance
    if _instance is None:
        _instance = ArtifactTracker()
    return _instance

artifact_tracker = get_artifact_tracker()
