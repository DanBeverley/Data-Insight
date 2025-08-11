from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from .version_control import VersionManager
from .deployment_manager import DeploymentManager, DeploymentConfig, Environment
from .monitoring import PerformanceMonitor, RetrainingTrigger, MetricType
from .auto_scaler import PredictiveScaler, ScalingPolicy, ScalingMetrics

class MLOpsOrchestrator:
    def __init__(self, base_path: str = "mlops"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.version_manager = VersionManager(str(self.base_path / "versions"))
        self.deployment_manager = DeploymentManager(str(self.base_path / "deployments"))
        self.monitor = PerformanceMonitor(str(self.base_path / "monitoring"))
        self.scaler = PredictiveScaler(str(self.base_path / "scaling"))
        
        self.retraining_trigger = RetrainingTrigger(self.monitor)
        self.setup_callbacks()
    
    def setup_callbacks(self):
        def retraining_callback(deployment_id: str, trigger_info: Dict[str, Any]):
            logging.info(f"Retraining triggered for {deployment_id}: {trigger_info['reason']}")
            self.trigger_retraining_pipeline(deployment_id, trigger_info)
        
        self.retraining_trigger.add_retraining_callback(retraining_callback)
    
    def deploy_model_pipeline(self, model, pipeline_config: Dict[str, Any], 
                            data_df, algorithm: str, performance_metrics: Dict[str, float],
                            environment: str = "staging") -> str:
        
        data_version = self.version_manager.version_data(data_df, "training_data")
        
        model_version = self.version_manager.version_model(
            model=model,
            algorithm=algorithm,
            parameters=model.get_params() if hasattr(model, 'get_params') else {},
            performance_metrics=performance_metrics,
            data_version=data_version.hash_id,
            training_time=0.0
        )
        
        pipeline_version = self.version_manager.version_pipeline(
            stages=pipeline_config.get('stages', []),
            config=pipeline_config.get('config', {}),
            dependencies=pipeline_config.get('dependencies', {})
        )
        
        deployment_config = DeploymentConfig(
            environment=Environment(environment),
            model_version=model_version.version_id,
            pipeline_version=pipeline_version.pipeline_id,
            config_overrides={},
            health_check_endpoint='/health',
            rollback_threshold=0.8,
            max_replicas=5,
            min_replicas=1
        )
        
        model_artifacts = {"main_model": model_version.file_path}
        pipeline_artifacts = {"main_pipeline": pipeline_version.file_path}
        
        deployment_id = self.deployment_manager.deploy(
            deployment_config, model_artifacts, pipeline_artifacts
        )
        
        scaling_policy = ScalingPolicy(
            min_replicas=1,
            max_replicas=5,
            target_cpu_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0
        )
        
        self.scaler.start_auto_scaling({deployment_id: scaling_policy})
        
        return deployment_id
    
    def monitor_deployment(self, deployment_id: str, 
                          prediction_latency: float = 0.0,
                          accuracy: float = 0.0,
                          error_rate: float = 0.0):
        
        if prediction_latency > 0:
            self.monitor.record_metric(deployment_id, MetricType.LATENCY, prediction_latency)
        
        if accuracy > 0:
            self.monitor.record_metric(deployment_id, MetricType.ACCURACY, accuracy)
        
        if error_rate >= 0:
            self.monitor.record_metric(deployment_id, MetricType.ERROR_RATE, error_rate)
        
        scaling_metrics = ScalingMetrics(
            timestamp=datetime.now().isoformat(),
            request_rate=100.0,
            avg_latency=prediction_latency,
            cpu_usage=50.0,
            memory_usage=500.0,
            active_replicas=1,
            queue_length=0
        )
        
        self.scaler.record_metrics(deployment_id, scaling_metrics)
    
    def get_deployment_health(self, deployment_id: str) -> Dict[str, Any]:
        dashboard_data = self.monitor.get_dashboard_data(deployment_id)
        scaling_recommendations = self.scaler.get_scaling_recommendations(
            deployment_id, ScalingPolicy()
        )
        
        deployment_status = self.deployment_manager.get_deployment_status(deployment_id)
        
        return {
            'deployment_id': deployment_id,
            'deployment_status': deployment_status.value if deployment_status else 'unknown',
            'health_status': dashboard_data.get('health_status', 'unknown'),
            'monitoring': dashboard_data,
            'scaling': scaling_recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        return self.deployment_manager.rollback(deployment_id)
    
    def trigger_retraining_pipeline(self, deployment_id: str, trigger_info: Dict[str, Any]):
        deployment = self.deployment_manager.deployments.get(deployment_id)
        if not deployment:
            logging.error(f"Deployment {deployment_id} not found for retraining")
            return
        
        model_version_id = deployment.config.model_version
        lineage = self.version_manager.get_lineage(model_version_id)
        
        retraining_metadata = {
            'original_deployment': deployment_id,
            'trigger_reason': trigger_info.get('reason', 'unknown'),
            'trigger_metric': trigger_info.get('trigger_metric', 'unknown'),
            'lineage': lineage,
            'timestamp': datetime.now().isoformat()
        }
        
        logging.info(f"Retraining pipeline triggered: {retraining_metadata}")
    
    def promote_to_production(self, staging_deployment_id: str) -> Optional[str]:
        staging_deployment = self.deployment_manager.deployments.get(staging_deployment_id)
        if not staging_deployment:
            logging.error(f"Staging deployment {staging_deployment_id} not found")
            return None
        
        if staging_deployment.config.environment != Environment.STAGING:
            logging.error(f"Deployment {staging_deployment_id} is not in staging")
            return None
        
        health_data = self.get_deployment_health(staging_deployment_id)
        if health_data['health_status'] != 'healthy':
            logging.error(f"Deployment {staging_deployment_id} is not healthy for promotion")
            return None
        
        prod_config = DeploymentConfig(
            environment=Environment.PRODUCTION,
            model_version=staging_deployment.config.model_version,
            pipeline_version=staging_deployment.config.pipeline_version,
            config_overrides={},
            health_check_endpoint='/health',
            rollback_threshold=0.85,
            max_replicas=10,
            min_replicas=2
        )
        
        model_artifacts = {"main_model": "placeholder_path"}
        pipeline_artifacts = {"main_pipeline": "placeholder_path"}
        
        prod_deployment_id = self.deployment_manager.deploy(
            prod_config, model_artifacts, pipeline_artifacts
        )
        
        return prod_deployment_id
    
    def cleanup_resources(self, keep_versions: int = 5, keep_deployments: int = 3):
        self.version_manager.cleanup_old_versions(keep_versions)
        self.deployment_manager.cleanup_old_deployments(keep_deployments)
    
    def get_system_overview(self) -> Dict[str, Any]:
        monitoring_data = self.deployment_manager.monitor_deployments()
        versions_data = self.version_manager.list_versions()
        
        active_deployments = [
            d for d in self.deployment_manager.deployments.values()
            if d.status.value == 'active'
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'deployments': monitoring_data,
            'versions': {
                'data_versions': len(versions_data.get('data', [])),
                'model_versions': len(versions_data.get('models', [])),
                'pipeline_versions': len(versions_data.get('pipelines', []))
            },
            'active_deployments': len(active_deployments),
            'environments': {
                env.value: len([d for d in active_deployments if d.config.environment == env])
                for env in Environment
            }
        }