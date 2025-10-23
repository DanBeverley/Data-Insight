import time
import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"


@dataclass
class MetricPoint:
    timestamp: str
    value: float
    metadata: Dict[str, Any]


@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    timestamp: str
    threshold_value: float
    current_value: float
    deployment_id: str


@dataclass
class PerformanceThresholds:
    accuracy_min: float = 0.8
    latency_max_ms: float = 100
    error_rate_max: float = 0.05
    memory_usage_max_mb: float = 1000
    cpu_usage_max_pct: float = 80


class PerformanceMonitor:
    def __init__(self, base_path: str = "monitoring"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        self.metrics_file = self.base_path / "metrics.json"
        self.alerts_file = self.base_path / "alerts.json"

        self.metrics = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.thresholds = PerformanceThresholds()

        self.alert_callbacks = []
        self.monitoring_active = False
        self.monitor_thread = None

        self._load_data()

    def _load_data(self):
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                data = json.load(f)
                for key, points in data.items():
                    self.metrics[key] = deque([MetricPoint(**p) for p in points], maxlen=1000)

        if self.alerts_file.exists():
            with open(self.alerts_file, "r") as f:
                data = json.load(f)
                self.alerts = [Alert(**a) for a in data]

    def _save_data(self):
        metrics_data = {key: [asdict(point) for point in points] for key, points in self.metrics.items()}
        with open(self.metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)

        alerts_data = [asdict(alert) for alert in self.alerts]
        with open(self.alerts_file, "w") as f:
            json.dump(alerts_data, f, indent=2, default=str)

    def record_metric(
        self, deployment_id: str, metric_type: MetricType, value: float, metadata: Optional[Dict[str, Any]] = None
    ):
        key = f"{deployment_id}_{metric_type.value}"
        timestamp = datetime.now().isoformat()

        point = MetricPoint(timestamp=timestamp, value=value, metadata=metadata or {})

        self.metrics[key].append(point)
        self._check_thresholds(deployment_id, metric_type, value)

    def _check_thresholds(self, deployment_id: str, metric_type: MetricType, value: float):
        alert_triggered = False
        severity = AlertSeverity.LOW
        threshold_value = 0.0

        if metric_type == MetricType.ACCURACY and value < self.thresholds.accuracy_min:
            alert_triggered = True
            severity = AlertSeverity.CRITICAL if value < 0.6 else AlertSeverity.HIGH
            threshold_value = self.thresholds.accuracy_min

        elif metric_type == MetricType.LATENCY and value > self.thresholds.latency_max_ms:
            alert_triggered = True
            severity = AlertSeverity.HIGH if value > 500 else AlertSeverity.MEDIUM
            threshold_value = self.thresholds.latency_max_ms

        elif metric_type == MetricType.ERROR_RATE and value > self.thresholds.error_rate_max:
            alert_triggered = True
            severity = AlertSeverity.CRITICAL if value > 0.1 else AlertSeverity.HIGH
            threshold_value = self.thresholds.error_rate_max

        elif metric_type == MetricType.MEMORY_USAGE and value > self.thresholds.memory_usage_max_mb:
            alert_triggered = True
            severity = AlertSeverity.HIGH if value > 2000 else AlertSeverity.MEDIUM
            threshold_value = self.thresholds.memory_usage_max_mb

        elif metric_type == MetricType.CPU_USAGE and value > self.thresholds.cpu_usage_max_pct:
            alert_triggered = True
            severity = AlertSeverity.HIGH if value > 90 else AlertSeverity.MEDIUM
            threshold_value = self.thresholds.cpu_usage_max_pct

        if alert_triggered:
            self._trigger_alert(deployment_id, metric_type, severity, value, threshold_value)

    def _trigger_alert(
        self,
        deployment_id: str,
        metric_type: MetricType,
        severity: AlertSeverity,
        current_value: float,
        threshold_value: float,
    ):
        alert_id = f"{deployment_id}_{metric_type.value}_{int(time.time())}"

        message = f"{metric_type.value} threshold exceeded for {deployment_id}: {current_value} > {threshold_value}"

        print(f"⚠️  ALERT [{severity.value.upper()}]: {message}")

        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            metric_type=metric_type,
            message=message,
            timestamp=datetime.now().isoformat(),
            threshold_value=threshold_value,
            current_value=current_value,
            deployment_id=deployment_id,
        )

        self.alerts.append(alert)

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback failed: {e}")

        self._save_data()

    def start_monitoring(self, interval_seconds: int = 30):
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitoring_loop(self, interval_seconds: int):
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval_seconds)

    def _collect_system_metrics(self):
        try:
            import psutil

            for deployment_id in self._get_active_deployments():
                cpu_percent = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().used / (1024 * 1024)

                self.record_metric(deployment_id, MetricType.CPU_USAGE, cpu_percent)
                self.record_metric(deployment_id, MetricType.MEMORY_USAGE, memory_usage)
        except ImportError:
            pass

    def _get_active_deployments(self) -> List[str]:
        unique_deployments = set()
        for key in self.metrics.keys():
            deployment_id = key.split("_")[0]
            unique_deployments.add(deployment_id)
        return list(unique_deployments)

    def get_metrics(
        self, deployment_id: str, metric_type: Optional[MetricType] = None, hours_back: int = 24
    ) -> Dict[str, List[MetricPoint]]:
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        results = {}

        for key, points in self.metrics.items():
            key_deployment_id = key.split("_")[0]
            key_metric_type = "_".join(key.split("_")[1:])

            if key_deployment_id == deployment_id:
                if metric_type is None or key_metric_type == metric_type.value:
                    filtered_points = [p for p in points if datetime.fromisoformat(p.timestamp) > cutoff_time]
                    results[key_metric_type] = filtered_points

        return results

    def get_alerts(
        self, deployment_id: Optional[str] = None, severity: Optional[AlertSeverity] = None, hours_back: int = 24
    ) -> List[Alert]:
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        filtered_alerts = []

        for alert in self.alerts:
            if datetime.fromisoformat(alert.timestamp) < cutoff_time:
                continue

            if deployment_id and alert.deployment_id != deployment_id:
                continue

            if severity and alert.severity != severity:
                continue

            filtered_alerts.append(alert)

        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        self.alert_callbacks.append(callback)

    def get_dashboard_data(self, deployment_id: str) -> Dict[str, Any]:
        recent_metrics = self.get_metrics(deployment_id, hours_back=1)
        recent_alerts = self.get_alerts(deployment_id, hours_back=24)

        dashboard = {
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "health_status": self._calculate_health_status(deployment_id),
            "current_metrics": {},
            "trends": {},
            "alerts_summary": {
                "total": len(recent_alerts),
                "critical": len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                "high": len([a for a in recent_alerts if a.severity == AlertSeverity.HIGH]),
                "medium": len([a for a in recent_alerts if a.severity == AlertSeverity.MEDIUM]),
            },
        }

        for metric_name, points in recent_metrics.items():
            if points:
                latest_point = points[-1]
                dashboard["current_metrics"][metric_name] = latest_point.value

                if len(points) > 1:
                    values = [p.value for p in points[-10:]]
                    dashboard["trends"][metric_name] = {
                        "direction": "up" if values[-1] > values[0] else "down",
                        "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                    }

        return dashboard

    def _calculate_health_status(self, deployment_id: str) -> str:
        recent_alerts = self.get_alerts(deployment_id, hours_back=1)

        if any(a.severity == AlertSeverity.CRITICAL for a in recent_alerts):
            return "critical"
        elif any(a.severity == AlertSeverity.HIGH for a in recent_alerts):
            return "degraded"
        elif any(a.severity == AlertSeverity.MEDIUM for a in recent_alerts):
            return "warning"
        else:
            return "healthy"


class RetrainingTrigger:
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.retraining_callbacks = []
        self.trigger_conditions = {MetricType.ACCURACY: lambda x: x < 0.7, MetricType.ERROR_RATE: lambda x: x > 0.1}

    def add_retraining_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        self.retraining_callbacks.append(callback)

    def check_retraining_triggers(self, deployment_id: str) -> bool:
        metrics = self.monitor.get_metrics(deployment_id, hours_back=1)

        for metric_name, points in metrics.items():
            if not points:
                continue

            try:
                metric_type = MetricType(metric_name)
                if metric_type in self.trigger_conditions:
                    recent_values = [p.value for p in points[-5:]]
                    avg_recent = np.mean(recent_values)

                    if self.trigger_conditions[metric_type](avg_recent):
                        self._trigger_retraining(deployment_id, metric_type, avg_recent)
                        return True
            except ValueError:
                continue

        return False

    def _trigger_retraining(self, deployment_id: str, metric_type: MetricType, current_value: float):
        trigger_info = {
            "deployment_id": deployment_id,
            "trigger_metric": metric_type.value,
            "current_value": current_value,
            "timestamp": datetime.now().isoformat(),
            "reason": f"{metric_type.value} degraded to {current_value}",
        }

        for callback in self.retraining_callbacks:
            try:
                callback(deployment_id, trigger_info)
            except Exception as e:
                print(f"Retraining callback failed: {e}")


def setup_monitoring(deployment_id: str, thresholds: Optional[PerformanceThresholds] = None) -> PerformanceMonitor:
    monitor = PerformanceMonitor()

    if thresholds:
        monitor.thresholds = thresholds

    def log_alert(alert: Alert):
        print(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")

    monitor.add_alert_callback(log_alert)
    monitor.start_monitoring()

    return monitor
