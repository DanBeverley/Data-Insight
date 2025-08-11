import json
import os
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLBACK = "rollback"

@dataclass
class DeploymentConfig:
    environment: Environment
    model_version: str
    pipeline_version: str
    config_overrides: Dict[str, Any]
    health_check_endpoint: str
    rollback_threshold: float
    max_replicas: int
    min_replicas: int

@dataclass
class Deployment:
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    timestamp: str
    health_score: float
    error_message: str
    metrics: Dict[str, Any]

class DeploymentManager:
    def __init__(self, base_path: str = "deployments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.environments = {
            Environment.DEVELOPMENT: self.base_path / "dev",
            Environment.STAGING: self.base_path / "staging", 
            Environment.PRODUCTION: self.base_path / "prod"
        }
        
        for env_path in self.environments.values():
            env_path.mkdir(exist_ok=True)
        
        self.deployments_file = self.base_path / "deployments.json"
        self._load_deployments()
    
    def _load_deployments(self):
        if self.deployments_file.exists():
            with open(self.deployments_file, 'r') as f:
                data = json.load(f)
                self.deployments = {k: Deployment(**v) for k, v in data.items()}
        else:
            self.deployments = {}
    
    def _save_deployments(self):
        data = {k: asdict(v) for k, v in self.deployments.items()}
        with open(self.deployments_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def deploy(self, config: DeploymentConfig, 
               model_artifacts: Dict[str, str],
               pipeline_artifacts: Dict[str, str]) -> str:
        
        deployment_id = f"{config.environment.value}_{int(time.time())}"
        timestamp = datetime.now().isoformat()
        
        deployment = Deployment(
            deployment_id=deployment_id,
            config=config,
            status=DeploymentStatus.DEPLOYING,
            timestamp=timestamp,
            health_score=0.0,
            error_message="",
            metrics={}
        )
        
        try:
            env_path = self.environments[config.environment]
            deployment_path = env_path / deployment_id
            deployment_path.mkdir(exist_ok=True)
            
            self._copy_artifacts(deployment_path, model_artifacts, pipeline_artifacts)
            self._create_deployment_manifest(deployment_path, config)
            
            if config.environment == Environment.PRODUCTION:
                old_deployment = self.get_active_deployment(config.environment)
                if old_deployment:
                    self._blue_green_deploy(deployment_path, old_deployment)
                else:
                    self._direct_deploy(deployment_path)
            else:
                self._direct_deploy(deployment_path)
            
            health_score = self._run_health_checks(config)
            
            if health_score > 0.8:
                deployment.status = DeploymentStatus.ACTIVE
                deployment.health_score = health_score
            else:
                deployment.status = DeploymentStatus.FAILED
                deployment.error_message = f"Health check failed: {health_score}"
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            logging.error(f"Deployment {deployment_id} failed: {e}")
        
        self.deployments[deployment_id] = deployment
        self._save_deployments()
        
        return deployment_id
    
    def _copy_artifacts(self, deployment_path: Path, 
                       model_artifacts: Dict[str, str],
                       pipeline_artifacts: Dict[str, str]):
        
        models_dir = deployment_path / "models"
        pipelines_dir = deployment_path / "pipelines"
        models_dir.mkdir(exist_ok=True)
        pipelines_dir.mkdir(exist_ok=True)
        
        for name, path in model_artifacts.items():
            if os.path.exists(path):
                shutil.copy2(path, models_dir / f"{name}.joblib")
        
        for name, path in pipeline_artifacts.items():
            if os.path.exists(path):
                shutil.copy2(path, pipelines_dir / f"{name}.json")
    
    def _create_deployment_manifest(self, deployment_path: Path, config: DeploymentConfig):
        manifest = {
            'environment': config.environment.value,
            'model_version': config.model_version,
            'pipeline_version': config.pipeline_version,
            'config': config.config_overrides,
            'health_check': config.health_check_endpoint,
            'scaling': {
                'min_replicas': config.min_replicas,
                'max_replicas': config.max_replicas
            },
            'rollback_threshold': config.rollback_threshold,
            'deployment_time': datetime.now().isoformat()
        }
        
        with open(deployment_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _blue_green_deploy(self, new_deployment_path: Path, old_deployment: Deployment):
        try:
            logging.info("Starting blue-green deployment")
            
            self._deploy_to_staging_slot(new_deployment_path)
            self._validate_staging_deployment(new_deployment_path)
            self._switch_traffic(new_deployment_path)
            
            logging.info("Blue-green deployment completed successfully")
            
        except Exception as e:
            logging.error(f"Blue-green deployment failed: {e}")
            self._rollback_deployment(old_deployment)
            raise
    
    def _direct_deploy(self, deployment_path: Path):
        logging.info(f"Direct deployment to {deployment_path}")
        
        with open(deployment_path / "deploy.lock", 'w') as f:
            f.write(str(os.getpid()))
    
    def _deploy_to_staging_slot(self, deployment_path: Path):
        staging_path = deployment_path.parent / f"{deployment_path.name}_staging"
        if staging_path.exists():
            shutil.rmtree(staging_path)
        shutil.copytree(deployment_path, staging_path)
    
    def _validate_staging_deployment(self, deployment_path: Path):
        manifest_path = deployment_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Deployment manifest not found")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if not (deployment_path / "models").exists():
            raise ValueError("Model artifacts not found")
        
        if not (deployment_path / "pipelines").exists():
            raise ValueError("Pipeline artifacts not found")
    
    def _switch_traffic(self, deployment_path: Path):
        logging.info("Switching traffic to new deployment")
        
        active_symlink = deployment_path.parent / "active"
        if active_symlink.exists():
            active_symlink.unlink()
        
        active_symlink.symlink_to(deployment_path.name)
    
    def _run_health_checks(self, config: DeploymentConfig) -> float:
        checks_passed = 0
        total_checks = 3
        
        if self._check_artifacts_exist(config):
            checks_passed += 1
        
        if self._check_configuration_valid(config):
            checks_passed += 1
        
        if self._check_model_loadable(config):
            checks_passed += 1
        
        return checks_passed / total_checks
    
    def _check_artifacts_exist(self, config: DeploymentConfig) -> bool:
        try:
            env_path = self.environments[config.environment]
            deployment_dirs = [d for d in env_path.iterdir() if d.is_dir()]
            
            if not deployment_dirs:
                return False
            
            latest_deployment = max(deployment_dirs, key=lambda p: p.stat().st_mtime)
            
            return (latest_deployment / "models").exists() and \
                   (latest_deployment / "pipelines").exists()
        except:
            return False
    
    def _check_configuration_valid(self, config: DeploymentConfig) -> bool:
        try:
            required_fields = ['environment', 'model_version', 'pipeline_version']
            return all(hasattr(config, field) for field in required_fields)
        except:
            return False
    
    def _check_model_loadable(self, config: DeploymentConfig) -> bool:
        try:
            env_path = self.environments[config.environment]
            deployment_dirs = [d for d in env_path.iterdir() if d.is_dir()]
            
            if not deployment_dirs:
                return False
            
            latest_deployment = max(deployment_dirs, key=lambda p: p.stat().st_mtime)
            models_path = latest_deployment / "models"
            
            model_files = list(models_path.glob("*.joblib"))
            return len(model_files) > 0
        except:
            return False
    
    def rollback(self, deployment_id: str) -> bool:
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
        
        try:
            env = deployment.config.environment
            deployments_for_env = [
                d for d in self.deployments.values() 
                if d.config.environment == env and d.status == DeploymentStatus.ACTIVE
            ]
            
            deployments_for_env.sort(key=lambda x: x.timestamp, reverse=True)
            
            if len(deployments_for_env) < 2:
                logging.warning("No previous deployment to rollback to")
                return False
            
            previous_deployment = deployments_for_env[1]
            return self._rollback_deployment(previous_deployment)
            
        except Exception as e:
            logging.error(f"Rollback failed: {e}")
            return False
    
    def _rollback_deployment(self, target_deployment: Deployment) -> bool:
        try:
            env_path = self.environments[target_deployment.config.environment]
            target_path = env_path / target_deployment.deployment_id
            
            if not target_path.exists():
                return False
            
            active_symlink = env_path / "active"
            if active_symlink.exists():
                active_symlink.unlink()
            
            active_symlink.symlink_to(target_deployment.deployment_id)
            
            target_deployment.status = DeploymentStatus.ACTIVE
            self._save_deployments()
            
            logging.info(f"Rolled back to deployment {target_deployment.deployment_id}")
            return True
            
        except Exception as e:
            logging.error(f"Rollback failed: {e}")
            return False
    
    def get_active_deployment(self, environment: Environment) -> Optional[Deployment]:
        active_deployments = [
            d for d in self.deployments.values()
            if d.config.environment == environment and d.status == DeploymentStatus.ACTIVE
        ]
        
        if not active_deployments:
            return None
        
        return max(active_deployments, key=lambda x: x.timestamp)
    
    def list_deployments(self, environment: Optional[Environment] = None) -> List[Deployment]:
        deployments = list(self.deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.config.environment == environment]
        
        return sorted(deployments, key=lambda x: x.timestamp, reverse=True)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        deployment = self.deployments.get(deployment_id)
        return deployment.status if deployment else None
    
    def monitor_deployments(self) -> Dict[str, Any]:
        monitoring_data = {
            'total_deployments': len(self.deployments),
            'by_environment': {},
            'by_status': {},
            'health_summary': []
        }
        
        for env in Environment:
            env_deployments = [d for d in self.deployments.values() 
                             if d.config.environment == env]
            monitoring_data['by_environment'][env.value] = len(env_deployments)
        
        for status in DeploymentStatus:
            status_deployments = [d for d in self.deployments.values() 
                                if d.status == status]
            monitoring_data['by_status'][status.value] = len(status_deployments)
        
        active_deployments = [d for d in self.deployments.values() 
                            if d.status == DeploymentStatus.ACTIVE]
        
        for deployment in active_deployments:
            monitoring_data['health_summary'].append({
                'deployment_id': deployment.deployment_id,
                'environment': deployment.config.environment.value,
                'health_score': deployment.health_score,
                'timestamp': deployment.timestamp
            })
        
        return monitoring_data
    
    def cleanup_old_deployments(self, keep_count: int = 5):
        for environment in Environment:
            env_deployments = [
                d for d in self.deployments.values()
                if d.config.environment == environment
            ]
            
            env_deployments.sort(key=lambda x: x.timestamp, reverse=True)
            to_remove = env_deployments[keep_count:]
            
            for deployment in to_remove:
                deployment_path = self.environments[environment] / deployment.deployment_id
                if deployment_path.exists():
                    shutil.rmtree(deployment_path)
                
                del self.deployments[deployment.deployment_id]
        
        self._save_deployments()

def create_deployment_config(environment: str, model_version: str, 
                           pipeline_version: str, **kwargs) -> DeploymentConfig:
    
    env_enum = Environment(environment.lower())
    
    defaults = {
        'config_overrides': {},
        'health_check_endpoint': '/health',
        'rollback_threshold': 0.8,
        'max_replicas': 10,
        'min_replicas': 1
    }
    
    defaults.update(kwargs)
    
    return DeploymentConfig(
        environment=env_enum,
        model_version=model_version,
        pipeline_version=pipeline_version,
        **defaults
    )