import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingEvent(Enum):
    TRAFFIC_SPIKE = "traffic_spike"
    HIGH_LATENCY = "high_latency"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    SCHEDULED = "scheduled"

@dataclass
class ScalingMetrics:
    timestamp: str
    request_rate: float
    avg_latency: float
    cpu_usage: float
    memory_usage: float
    active_replicas: int
    queue_length: int

@dataclass
class ScalingAction:
    action_id: str
    deployment_id: str
    direction: ScalingDirection
    trigger_event: ScalingEvent
    current_replicas: int
    target_replicas: int
    timestamp: str
    reason: str
    cost_impact: float

@dataclass
class ScalingPolicy:
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown_minutes: int = 5
    scale_down_cooldown_minutes: int = 10
    request_rate_threshold: int = 100
    latency_threshold_ms: float = 500.0

class PredictiveScaler:
    def __init__(self, base_path: str = "scaling"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.scaling_history = []
        self.predictions = {}
        
        self.scaling_file = self.base_path / "scaling_history.json"
        self.predictions_file = self.base_path / "predictions.json"
        
        self.scaling_active = False
        self.scaler_thread = None
        
        self._load_history()
    
    def _load_history(self):
        if self.scaling_file.exists():
            with open(self.scaling_file, 'r') as f:
                data = json.load(f)
                self.scaling_history = [ScalingAction(**action) for action in data]
        
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                self.predictions = json.load(f)
    
    def _save_history(self):
        with open(self.scaling_file, 'w') as f:
            json.dump([asdict(action) for action in self.scaling_history], f, indent=2)
        
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
    
    def record_metrics(self, deployment_id: str, metrics: ScalingMetrics):
        key = f"{deployment_id}_metrics"
        self.metrics_history[key].append(metrics)
    
    def predict_demand(self, deployment_id: str, hours_ahead: int = 1) -> Dict[str, float]:
        key = f"{deployment_id}_metrics"
        historical_metrics = list(self.metrics_history[key])
        
        if len(historical_metrics) < 10:
            return {'request_rate': 0, 'confidence': 0.0}
        
        recent_metrics = historical_metrics[-24:] if len(historical_metrics) >= 24 else historical_metrics
        
        timestamps = []
        request_rates = []
        
        for metric in recent_metrics:
            try:
                ts = datetime.fromisoformat(metric.timestamp)
                timestamps.append(ts.timestamp())
                request_rates.append(metric.request_rate)
            except:
                continue
        
        if len(request_rates) < 5:
            return {'request_rate': np.mean(request_rates), 'confidence': 0.5}
        
        seasonal_prediction = self._seasonal_forecast(request_rates, hours_ahead)
        trend_prediction = self._trend_forecast(timestamps, request_rates, hours_ahead)
        
        final_prediction = (seasonal_prediction + trend_prediction) / 2
        confidence = min(0.9, len(request_rates) / 24)
        
        return {
            'request_rate': max(0, final_prediction),
            'confidence': confidence,
            'seasonal_component': seasonal_prediction,
            'trend_component': trend_prediction
        }
    
    def _seasonal_forecast(self, values: List[float], hours_ahead: int) -> float:
        if len(values) < 12:
            return np.mean(values)
        
        hourly_averages = {}
        current_hour = datetime.now().hour
        
        for i, value in enumerate(values[-24:]):
            hour = (current_hour - (24 - i - 1)) % 24
            if hour not in hourly_averages:
                hourly_averages[hour] = []
            hourly_averages[hour].append(value)
        
        target_hour = (current_hour + hours_ahead) % 24
        
        if target_hour in hourly_averages:
            return np.mean(hourly_averages[target_hour])
        else:
            return np.mean(values)
    
    def _trend_forecast(self, timestamps: List[float], values: List[float], hours_ahead: int) -> float:
        if len(values) < 3:
            return np.mean(values)
        
        x = np.array(timestamps)
        y = np.array(values)
        
        try:
            coefficients = np.polyfit(x, y, 1)
            future_timestamp = timestamps[-1] + (hours_ahead * 3600)
            prediction = np.polyval(coefficients, future_timestamp)
            return max(0, prediction)
        except:
            return np.mean(values)
    
    def calculate_optimal_replicas(self, deployment_id: str, policy: ScalingPolicy) -> int:
        key = f"{deployment_id}_metrics"
        recent_metrics = list(self.metrics_history[key])[-5:]
        
        if not recent_metrics:
            return policy.min_replicas
        
        latest_metrics = recent_metrics[-1]
        prediction = self.predict_demand(deployment_id, 1)
        
        predicted_request_rate = prediction['request_rate']
        current_replicas = latest_metrics.active_replicas
        
        cpu_based_replicas = max(1, int(latest_metrics.cpu_usage / policy.target_cpu_utilization))
        memory_based_replicas = max(1, int(latest_metrics.memory_usage / policy.target_memory_utilization))
        
        request_based_replicas = max(1, int(predicted_request_rate / policy.request_rate_threshold))
        
        recommended_replicas = max(cpu_based_replicas, memory_based_replicas, request_based_replicas)
        
        return min(policy.max_replicas, max(policy.min_replicas, recommended_replicas))
    
    def should_scale(self, deployment_id: str, policy: ScalingPolicy) -> Tuple[bool, ScalingDirection, ScalingEvent]:
        key = f"{deployment_id}_metrics"
        recent_metrics = list(self.metrics_history[key])[-3:]
        
        if len(recent_metrics) < 2:
            return False, ScalingDirection.STABLE, None
        
        latest_metrics = recent_metrics[-1]
        current_replicas = latest_metrics.active_replicas
        
        if self._is_in_cooldown(deployment_id, policy):
            return False, ScalingDirection.STABLE, None
        
        optimal_replicas = self.calculate_optimal_replicas(deployment_id, policy)
        
        if optimal_replicas > current_replicas:
            trigger_event = self._identify_scale_up_trigger(latest_metrics, policy)
            return True, ScalingDirection.UP, trigger_event
        
        elif optimal_replicas < current_replicas:
            if self._confirm_scale_down(recent_metrics, policy):
                return True, ScalingDirection.DOWN, ScalingEvent.HIGH_CPU
        
        return False, ScalingDirection.STABLE, None
    
    def _is_in_cooldown(self, deployment_id: str, policy: ScalingPolicy) -> bool:
        recent_actions = [
            action for action in self.scaling_history
            if action.deployment_id == deployment_id
        ]
        
        if not recent_actions:
            return False
        
        last_action = max(recent_actions, key=lambda x: x.timestamp)
        last_action_time = datetime.fromisoformat(last_action.timestamp)
        
        if last_action.direction == ScalingDirection.UP:
            cooldown_minutes = policy.scale_up_cooldown_minutes
        else:
            cooldown_minutes = policy.scale_down_cooldown_minutes
        
        return datetime.now() - last_action_time < timedelta(minutes=cooldown_minutes)
    
    def _identify_scale_up_trigger(self, metrics: ScalingMetrics, policy: ScalingPolicy) -> ScalingEvent:
        if metrics.cpu_usage > policy.scale_up_threshold:
            return ScalingEvent.HIGH_CPU
        elif metrics.memory_usage > policy.scale_up_threshold:
            return ScalingEvent.HIGH_MEMORY
        elif metrics.avg_latency > policy.latency_threshold_ms:
            return ScalingEvent.HIGH_LATENCY
        elif metrics.request_rate > policy.request_rate_threshold:
            return ScalingEvent.TRAFFIC_SPIKE
        else:
            return ScalingEvent.SCHEDULED
    
    def _confirm_scale_down(self, recent_metrics: List[ScalingMetrics], policy: ScalingPolicy) -> bool:
        if len(recent_metrics) < 3:
            return False
        
        for metrics in recent_metrics:
            if (metrics.cpu_usage > policy.scale_down_threshold or 
                metrics.memory_usage > policy.scale_down_threshold or
                metrics.avg_latency > policy.latency_threshold_ms):
                return False
        
        return True
    
    def execute_scaling_action(self, deployment_id: str, target_replicas: int, 
                             direction: ScalingDirection, trigger_event: ScalingEvent,
                             current_replicas: int, reason: str = "") -> str:
        
        action_id = f"{deployment_id}_scale_{int(time.time())}"
        
        cost_impact = self._calculate_cost_impact(current_replicas, target_replicas)
        
        action = ScalingAction(
            action_id=action_id,
            deployment_id=deployment_id,
            direction=direction,
            trigger_event=trigger_event,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            timestamp=datetime.now().isoformat(),
            reason=reason or f"{trigger_event.value} triggered scaling {direction.value}",
            cost_impact=cost_impact
        )
        
        self.scaling_history.append(action)
        self._save_history()
        
        return action_id
    
    def _calculate_cost_impact(self, current_replicas: int, target_replicas: int) -> float:
        replica_cost_per_hour = 0.10
        hours_impact = 1.0
        
        replica_change = target_replicas - current_replicas
        return replica_change * replica_cost_per_hour * hours_impact
    
    def start_auto_scaling(self, deployment_configs: Dict[str, ScalingPolicy], 
                          check_interval_seconds: int = 60):
        
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.deployment_configs = deployment_configs
        
        self.scaler_thread = threading.Thread(
            target=self._auto_scaling_loop, 
            args=(check_interval_seconds,)
        )
        self.scaler_thread.daemon = True
        self.scaler_thread.start()
    
    def stop_auto_scaling(self):
        self.scaling_active = False
        if self.scaler_thread:
            self.scaler_thread.join()
    
    def _auto_scaling_loop(self, interval_seconds: int):
        while self.scaling_active:
            try:
                for deployment_id, policy in self.deployment_configs.items():
                    should_scale, direction, trigger_event = self.should_scale(deployment_id, policy)
                    
                    if should_scale:
                        current_replicas = self._get_current_replicas(deployment_id)
                        target_replicas = self.calculate_optimal_replicas(deployment_id, policy)
                        
                        self.execute_scaling_action(
                            deployment_id, target_replicas, direction, 
                            trigger_event, current_replicas
                        )
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Auto-scaling error: {e}")
                time.sleep(interval_seconds)
    
    def _get_current_replicas(self, deployment_id: str) -> int:
        key = f"{deployment_id}_metrics"
        recent_metrics = list(self.metrics_history[key])
        
        if recent_metrics:
            return recent_metrics[-1].active_replicas
        return 1
    
    def get_scaling_recommendations(self, deployment_id: str, policy: ScalingPolicy) -> Dict[str, Any]:
        key = f"{deployment_id}_metrics"
        recent_metrics = list(self.metrics_history[key])
        
        if not recent_metrics:
            return {'error': 'No metrics available'}
        
        latest_metrics = recent_metrics[-1]
        optimal_replicas = self.calculate_optimal_replicas(deployment_id, policy)
        prediction = self.predict_demand(deployment_id, 1)
        
        should_scale, direction, trigger_event = self.should_scale(deployment_id, policy)
        
        recommendation = {
            'deployment_id': deployment_id,
            'current_replicas': latest_metrics.active_replicas,
            'recommended_replicas': optimal_replicas,
            'should_scale': should_scale,
            'scaling_direction': direction.value if should_scale else 'stable',
            'trigger_reason': trigger_event.value if trigger_event else None,
            'demand_prediction': prediction,
            'current_metrics': {
                'cpu_usage': latest_metrics.cpu_usage,
                'memory_usage': latest_metrics.memory_usage,
                'request_rate': latest_metrics.request_rate,
                'avg_latency': latest_metrics.avg_latency
            },
            'cost_impact': self._calculate_cost_impact(
                latest_metrics.active_replicas, optimal_replicas
            ),
            'confidence': prediction.get('confidence', 0.5)
        }
        
        return recommendation
    
    def get_scaling_history(self, deployment_id: str, hours_back: int = 24) -> List[ScalingAction]:
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_history = [
            action for action in self.scaling_history
            if (action.deployment_id == deployment_id and 
                datetime.fromisoformat(action.timestamp) > cutoff_time)
        ]
        
        return sorted(filtered_history, key=lambda x: x.timestamp, reverse=True)

def create_scaling_policy(min_replicas: int = 1, max_replicas: int = 10, **kwargs) -> ScalingPolicy:
    return ScalingPolicy(min_replicas=min_replicas, max_replicas=max_replicas, **kwargs)