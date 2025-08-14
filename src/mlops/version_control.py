import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import joblib

@dataclass
class DataVersion:
    hash_id: str
    timestamp: str
    shape: Tuple[int, int]
    schema: Dict[str, str]
    checksum: str
    file_path: str
    size_mb: float

@dataclass
class ModelVersion:
    version_id: str
    algorithm: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: str
    file_path: str
    data_version: str
    training_time: float

@dataclass
class PipelineVersion:
    pipeline_id: str
    stages: List[str]
    config: Dict[str, Any]
    timestamp: str
    code_hash: str
    dependencies: Dict[str, str]
    file_path: str

class VersionManager:
    def __init__(self, base_path: str = "versions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.data_path = self.base_path / "data"
        self.models_path = self.base_path / "models"
        self.pipelines_path = self.base_path / "pipelines"
        
        for path in [self.data_path, self.models_path, self.pipelines_path]:
            path.mkdir(exist_ok=True)
        
        self.metadata_file = self.base_path / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'data_versions': {},
                'model_versions': {},
                'pipeline_versions': {}
            }
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_hash(self, data: Any) -> str:
        if isinstance(data, pd.DataFrame):
            content = data.to_string().encode()
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True).encode()
        else:
            content = str(data).encode()
        
        return hashlib.sha256(content).hexdigest()[:16]
    
    def version_data(self, df: pd.DataFrame, name: str = "dataset") -> DataVersion:
        hash_id = self._compute_hash(df)
        timestamp = datetime.now().isoformat()
        
        existing_version = self.metadata['data_versions'].get(hash_id)
        if existing_version:
            return DataVersion(**existing_version)
        
        file_name = f"{name}_{hash_id}_{int(time.time())}.parquet"
        file_path = self.data_path / file_name
        
        df.to_parquet(file_path)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        schema = {col: str(df[col].dtype) for col in df.columns}
        checksum = self._compute_hash(df.values.tobytes())
        
        version = DataVersion(
            hash_id=hash_id,
            timestamp=timestamp,
            shape=df.shape,
            schema=schema,
            checksum=checksum,
            file_path=str(file_path),
            size_mb=round(size_mb, 2)
        )
        
        self.metadata['data_versions'][hash_id] = asdict(version)
        self._save_metadata()
        
        return version
    
    def version_model(self, model: Any, algorithm: str, parameters: Dict[str, Any],
                     performance_metrics: Dict[str, float], data_version: str,
                     training_time: float = 0.0, name: str = "model") -> ModelVersion:
        
        version_id = f"{algorithm}_{self._compute_hash(parameters)}_{int(time.time())}"
        timestamp = datetime.now().isoformat()
        
        file_name = f"{name}_{version_id}.joblib"
        file_path = self.models_path / file_name
        
        joblib.dump(model, file_path)
        
        version = ModelVersion(
            version_id=version_id,
            algorithm=algorithm,
            parameters=parameters,
            performance_metrics=performance_metrics,
            timestamp=timestamp,
            file_path=str(file_path),
            data_version=data_version,
            training_time=training_time
        )
        
        self.metadata['model_versions'][version_id] = asdict(version)
        self._save_metadata()
        
        return version
    
    def version_pipeline(self, stages: List[str], config: Dict[str, Any],
                        dependencies: Optional[Dict[str, str]] = None,
                        name: str = "pipeline") -> PipelineVersion:
        
        pipeline_content = {
            'stages': stages,
            'config': config,
            'dependencies': dependencies or {}
        }
        
        pipeline_id = f"{name}_{self._compute_hash(pipeline_content)}_{int(time.time())}"
        timestamp = datetime.now().isoformat()
        code_hash = self._compute_hash(stages)
        
        file_name = f"{pipeline_id}.json"
        file_path = self.pipelines_path / file_name
        
        with open(file_path, 'w') as f:
            json.dump(pipeline_content, f, indent=2)
        
        version = PipelineVersion(
            pipeline_id=pipeline_id,
            stages=stages,
            config=config,
            timestamp=timestamp,
            code_hash=code_hash,
            dependencies=dependencies or {},
            file_path=str(file_path)
        )
        
        self.metadata['pipeline_versions'][pipeline_id] = asdict(version)
        self._save_metadata()
        
        return version
    
    def load_data(self, hash_id: str) -> pd.DataFrame:
        version_info = self.metadata['data_versions'].get(hash_id)
        if not version_info:
            raise ValueError(f"Data version {hash_id} not found")
        
        return pd.read_parquet(version_info['file_path'])
    
    def load_model(self, version_id: str) -> Any:
        version_info = self.metadata['model_versions'].get(version_id)
        if not version_info:
            raise ValueError(f"Model version {version_id} not found")
        
        return joblib.load(version_info['file_path'])
    
    def load_pipeline_config(self, pipeline_id: str) -> Dict[str, Any]:
        version_info = self.metadata['pipeline_versions'].get(pipeline_id)
        if not version_info:
            raise ValueError(f"Pipeline version {pipeline_id} not found")
        
        with open(version_info['file_path'], 'r') as f:
            return json.load(f)
    
    def list_versions(self, version_type: str = "all") -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        
        if version_type in ["all", "data"]:
            results['data'] = list(self.metadata['data_versions'].values())
        
        if version_type in ["all", "models"]:
            results['models'] = list(self.metadata['model_versions'].values())
        
        if version_type in ["all", "pipelines"]:
            results['pipelines'] = list(self.metadata['pipeline_versions'].values())
        
        return results
    
    def get_latest_version(self, version_type: str, name_filter: str = "") -> Optional[Dict[str, Any]]:
        versions_key = f"{version_type}_versions"
        if versions_key not in self.metadata:
            return None
        
        versions = self.metadata[versions_key].values()
        if name_filter:
            versions = [v for v in versions if name_filter in v.get('file_path', '')]
        
        if not versions:
            return None
        
        return max(versions, key=lambda x: x['timestamp'])
    
    def compare_models(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        model1 = self.metadata['model_versions'].get(version_id1)
        model2 = self.metadata['model_versions'].get(version_id2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        comparison = {
            'algorithm_comparison': {
                'model1': model1['algorithm'],
                'model2': model2['algorithm'],
                'same_algorithm': model1['algorithm'] == model2['algorithm']
            },
            'performance_comparison': {},
            'parameter_changes': {},
            'training_time': {
                'model1': model1['training_time'],
                'model2': model2['training_time'],
                'improvement': model2['training_time'] - model1['training_time']
            }
        }
        
        for metric in model1['performance_metrics']:
            if metric in model2['performance_metrics']:
                val1, val2 = model1['performance_metrics'][metric], model2['performance_metrics'][metric]
                comparison['performance_comparison'][metric] = {
                    'model1': val1,
                    'model2': val2,
                    'improvement': val2 - val1,
                    'improvement_pct': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
        
        for param in model1['parameters']:
            if param in model2['parameters']:
                if model1['parameters'][param] != model2['parameters'][param]:
                    comparison['parameter_changes'][param] = {
                        'from': model1['parameters'][param],
                        'to': model2['parameters'][param]
                    }
        
        return comparison
    
    def cleanup_old_versions(self, keep_latest: int = 5):
        for version_type in ['data_versions', 'model_versions', 'pipeline_versions']:
            versions = list(self.metadata[version_type].items())
            versions.sort(key=lambda x: x[1]['timestamp'], reverse=True)
            
            to_remove = versions[keep_latest:]
            
            for version_id, version_info in to_remove:
                file_path = Path(version_info['file_path'])
                if file_path.exists():
                    file_path.unlink()
                del self.metadata[version_type][version_id]
        
        self._save_metadata()
    
    def get_lineage(self, model_version_id: str) -> Dict[str, Any]:
        model_info = self.metadata['model_versions'].get(model_version_id)
        if not model_info:
            return {}
        
        data_version_id = model_info['data_version']
        data_info = self.metadata['data_versions'].get(data_version_id)
        
        lineage = {
            'model': model_info,
            'data': data_info,
            'dependencies': {
                'data_version': data_version_id,
                'model_version': model_version_id
            },
            'trace': [
                f"Data ingested: {data_info['timestamp'] if data_info else 'Unknown'}",
                f"Model trained: {model_info['timestamp']}",
                f"Training time: {model_info['training_time']}s"
            ]
        }
        
        return lineage