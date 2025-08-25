import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import threading
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    class MockPsutil:
        @staticmethod
        def cpu_percent():
            return 50.0
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                def __init__(self):
                    self.percent = 60.0
                    self.available = 8 * 1024**3
                    self.used = 4 * 1024**3
            return MockMemory()
        
        @staticmethod
        def disk_usage(path="/"):
            class MockDisk:
                def __init__(self):
                    self.percent = 70.0
                    self.free = 100 * 1024**3
                    self.used = 50 * 1024**3
            return MockDisk()
        
        @staticmethod
        def Process():
            class MockProcess:
                def memory_info(self):
                    class MockMemInfo:
                        def __init__(self):
                            self.rss = 1024 * 1024 * 500
                    return MockMemInfo()
                
                def cpu_percent(self):
                    return 25.0
            return MockProcess()
    
    psutil = MockPsutil()

class BudgetType(Enum):
    LATENCY = "latency_ms"
    TRAINING_TIME = "training_time_seconds"
    MEMORY = "memory_mb"
    CPU_UTILIZATION = "cpu_percent"
    ACCURACY_FLOOR = "min_accuracy"
    COST = "max_cost_dollars"

class ViolationSeverity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKING = "blocking"

@dataclass
class BudgetLimit:
    type: BudgetType
    limit: Union[float, int]
    severity: ViolationSeverity = ViolationSeverity.WARNING
    tolerance_percent: float = 10.0
    measurement_window: Optional[int] = None

@dataclass
class BudgetViolation:
    budget_type: BudgetType
    limit: Union[float, int]
    actual: Union[float, int]
    severity: ViolationSeverity
    violation_percent: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    mitigation_suggestions: List[str] = field(default_factory=list)

@dataclass
class BudgetReport:
    passed_budgets: List[BudgetType]
    violations: List[BudgetViolation]
    warnings: List[BudgetViolation]
    overall_compliance: float
    resource_utilization: Dict[str, float]
    recommendations: List[str]

class PerformanceBudgetManager:
    
    def __init__(self):
        self.active_budgets: Dict[BudgetType, BudgetLimit] = {}
        self.monitoring_sessions: Dict[str, Dict] = {}
        self.measurement_history: List[Dict] = []
        self.resource_monitor = ResourceMonitor()
        
    def set_budget(self, budget_limit: BudgetLimit) -> None:
        self.active_budgets[budget_limit.type] = budget_limit
        
    def set_budgets_from_constraints(self, constraints: Dict[str, Any]) -> None:
        budget_mapping = {
            'max_latency_ms': (BudgetType.LATENCY, ViolationSeverity.CRITICAL),
            'max_training_hours': (BudgetType.TRAINING_TIME, ViolationSeverity.WARNING),
            'max_memory_gb': (BudgetType.MEMORY, ViolationSeverity.WARNING),
            'min_accuracy': (BudgetType.ACCURACY_FLOOR, ViolationSeverity.BLOCKING)
        }
        
        for constraint_key, (budget_type, severity) in budget_mapping.items():
            if constraint_key in constraints and constraints[constraint_key] is not None:
                value = constraints[constraint_key]
                
                if constraint_key == 'max_training_hours':
                    value *= 3600  # Convert to seconds
                elif constraint_key == 'max_memory_gb':
                    value *= 1024  # Convert to MB
                    
                self.set_budget(BudgetLimit(
                    type=budget_type,
                    limit=value,
                    severity=severity,
                    tolerance_percent=5.0 if severity == ViolationSeverity.CRITICAL else 15.0
                ))
    
    @contextmanager
    def monitor_session(self, session_id: str, operation_type: str = "general"):
        session_data = {
            'session_id': session_id,
            'operation_type': operation_type,
            'start_time': time.time(),
            'start_memory': self.resource_monitor.get_memory_usage(),
            'start_cpu': self.resource_monitor.get_cpu_usage(),
            'measurements': []
        }
        
        self.monitoring_sessions[session_id] = session_data
        
        try:
            yield session_data
        finally:
            session_data['end_time'] = time.time()
            session_data['duration'] = session_data['end_time'] - session_data['start_time']
            session_data['end_memory'] = self.resource_monitor.get_memory_usage()
            session_data['peak_memory'] = max(
                session_data['start_memory'],
                session_data['end_memory'],
                max([m.get('memory', 0) for m in session_data['measurements']] + [0])
            )
            
            self.measurement_history.append(session_data)
            del self.monitoring_sessions[session_id]
    
    def validate_budget_compliance(self, 
                                 metrics: Dict[str, Any],
                                 model_metadata: Dict[str, Any],
                                 session_id: Optional[str] = None) -> BudgetReport:
        
        violations = []
        warnings = []
        passed_budgets = []
        
        # Extract measurements from various sources
        measurements = self._extract_measurements(metrics, model_metadata, session_id)
        
        for budget_type, budget_limit in self.active_budgets.items():
            if budget_type in measurements:
                actual_value = measurements[budget_type]
                violation = self._check_budget_violation(budget_limit, actual_value, measurements)
                
                if violation:
                    if violation.severity == ViolationSeverity.WARNING:
                        warnings.append(violation)
                    else:
                        violations.append(violation)
                else:
                    passed_budgets.append(budget_type)
        
        overall_compliance = len(passed_budgets) / len(self.active_budgets) if self.active_budgets else 1.0
        
        resource_utilization = {
            'cpu_usage': measurements.get(BudgetType.CPU_UTILIZATION, 0),
            'memory_usage_mb': measurements.get(BudgetType.MEMORY, 0),
            'duration_seconds': measurements.get(BudgetType.TRAINING_TIME, 0)
        }
        
        recommendations = self._generate_budget_recommendations(violations, warnings, measurements)
        
        return BudgetReport(
            passed_budgets=passed_budgets,
            violations=violations,
            warnings=warnings,
            overall_compliance=overall_compliance,
            resource_utilization=resource_utilization,
            recommendations=recommendations
        )
    
    def _extract_measurements(self, 
                            metrics: Dict[str, Any], 
                            model_metadata: Dict[str, Any],
                            session_id: Optional[str]) -> Dict[BudgetType, float]:
        
        measurements = {}
        
        # Latency measurements
        prediction_time = (
            metrics.get('inference_latency_ms') or
            metrics.get('prediction_time') or
            model_metadata.get('performance_validation', {}).get('prediction_time') or
            model_metadata.get('prediction_time', 0)
        )
        if prediction_time:
            measurements[BudgetType.LATENCY] = prediction_time
        
        # Training time measurements
        training_time = (
            metrics.get('training_time') or
            model_metadata.get('performance_validation', {}).get('training_time') or
            model_metadata.get('training_time', 0)
        )
        if training_time:
            measurements[BudgetType.TRAINING_TIME] = training_time
        
        # Memory measurements
        memory_usage = (
            metrics.get('memory_usage_mb') or
            model_metadata.get('performance_validation', {}).get('memory_usage_mb') or
            model_metadata.get('memory_usage', 0)
        )
        if memory_usage:
            measurements[BudgetType.MEMORY] = memory_usage
        
        # Accuracy measurements
        accuracy = (
            metrics.get('accuracy') or
            metrics.get('f1_score') or
            metrics.get('best_score') or
            model_metadata.get('best_score') or
            metrics.get('validation_metrics', {}).get('primary_score', 0)
        )
        if accuracy:
            measurements[BudgetType.ACCURACY_FLOOR] = accuracy
        
        # CPU utilization from session monitoring
        if session_id and session_id in self.monitoring_sessions:
            session = self.monitoring_sessions[session_id]
            if session['measurements']:
                avg_cpu = sum(m.get('cpu', 0) for m in session['measurements']) / len(session['measurements'])
                measurements[BudgetType.CPU_UTILIZATION] = avg_cpu
        
        return measurements
    
    def _check_budget_violation(self, 
                              budget_limit: BudgetLimit,
                              actual_value: float,
                              all_measurements: Dict) -> Optional[BudgetViolation]:
        
        tolerance_margin = budget_limit.limit * (budget_limit.tolerance_percent / 100)
        effective_limit = budget_limit.limit + tolerance_margin
        
        # For accuracy floor, violation is when actual is BELOW limit
        if budget_limit.type == BudgetType.ACCURACY_FLOOR:
            violated = actual_value < (budget_limit.limit - tolerance_margin)
            violation_percent = ((budget_limit.limit - actual_value) / budget_limit.limit) * 100 if violated else 0
        else:
            # For other budgets, violation is when actual EXCEEDS limit
            violated = actual_value > effective_limit
            violation_percent = ((actual_value - budget_limit.limit) / budget_limit.limit) * 100 if violated else 0
        
        if not violated:
            return None
        
        # Generate context and mitigation suggestions
        context = self._generate_violation_context(budget_limit.type, actual_value, all_measurements)
        mitigation_suggestions = self._generate_mitigation_suggestions(budget_limit.type, violation_percent, context)
        
        return BudgetViolation(
            budget_type=budget_limit.type,
            limit=budget_limit.limit,
            actual=actual_value,
            severity=budget_limit.severity,
            violation_percent=abs(violation_percent),
            context=context,
            mitigation_suggestions=mitigation_suggestions
        )
    
    def _generate_violation_context(self, 
                                  budget_type: BudgetType,
                                  actual_value: float,
                                  measurements: Dict) -> Dict[str, Any]:
        
        context = {'budget_type': budget_type.value, 'actual_value': actual_value}
        
        if budget_type == BudgetType.LATENCY:
            context.update({
                'model_complexity': self._assess_model_complexity_impact(measurements),
                'data_size_factor': measurements.get(BudgetType.MEMORY, 0) / 1024  # GB
            })
        elif budget_type == BudgetType.TRAINING_TIME:
            context.update({
                'algorithm_complexity': 'high' if actual_value > 3600 else 'medium',
                'cpu_utilization': measurements.get(BudgetType.CPU_UTILIZATION, 0)
            })
        elif budget_type == BudgetType.MEMORY:
            context.update({
                'data_size_gb': actual_value / 1024,
                'memory_efficiency': 'low' if actual_value > 8000 else 'acceptable'
            })
        
        return context
    
    def _assess_model_complexity_impact(self, measurements: Dict) -> str:
        memory_gb = measurements.get(BudgetType.MEMORY, 0) / 1024
        training_time = measurements.get(BudgetType.TRAINING_TIME, 0)
        
        if memory_gb > 8 and training_time > 1800:
            return 'very_high'
        elif memory_gb > 4 or training_time > 600:
            return 'high'
        elif memory_gb > 2 or training_time > 300:
            return 'medium'
        else:
            return 'low'
    
    def _generate_mitigation_suggestions(self, 
                                       budget_type: BudgetType,
                                       violation_percent: float,
                                       context: Dict) -> List[str]:
        
        suggestions = []
        
        if budget_type == BudgetType.LATENCY:
            if violation_percent > 50:
                suggestions.extend([
                    "Consider switching to simpler, faster algorithms (linear models)",
                    "Implement model quantization or pruning",
                    "Use caching for repeated predictions"
                ])
            else:
                suggestions.extend([
                    "Optimize inference pipeline",
                    "Consider batch prediction strategies"
                ])
                
        elif budget_type == BudgetType.TRAINING_TIME:
            if violation_percent > 100:
                suggestions.extend([
                    "Reduce dataset size or use sampling",
                    "Switch to faster algorithms",
                    "Implement early stopping"
                ])
            else:
                suggestions.extend([
                    "Optimize hyperparameter search space",
                    "Use warm-start techniques"
                ])
                
        elif budget_type == BudgetType.MEMORY:
            suggestions.extend([
                "Implement data chunking strategies",
                "Use memory-efficient algorithms",
                "Consider dimensionality reduction"
            ])
            
        elif budget_type == BudgetType.ACCURACY_FLOOR:
            suggestions.extend([
                "Increase model complexity",
                "Enhance feature engineering",
                "Collect more training data"
            ])
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _generate_budget_recommendations(self, 
                                       violations: List[BudgetViolation],
                                       warnings: List[BudgetViolation],
                                       measurements: Dict) -> List[str]:
        
        recommendations = []
        
        if violations:
            critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
            if critical_violations:
                recommendations.append(f"CRITICAL: {len(critical_violations)} budget violations require immediate attention")
        
        if warnings:
            recommendations.append(f"WARNING: {len(warnings)} budgets approaching limits")
        
        # Resource utilization recommendations
        memory_gb = measurements.get(BudgetType.MEMORY, 0) / 1024
        if memory_gb > 16:
            recommendations.append("High memory usage detected. Consider optimization strategies.")
        
        training_time = measurements.get(BudgetType.TRAINING_TIME, 0)
        if training_time > 7200:  # 2 hours
            recommendations.append("Extended training time. Consider faster algorithms or early stopping.")
        
        if not violations and not warnings:
            recommendations.append("All performance budgets satisfied. System operating within constraints.")
        
        return recommendations

class ResourceMonitor:
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)  # MB
    
    def get_cpu_usage(self) -> float:
        return self.process.cpu_percent(interval=0.1)
    
    def start_continuous_monitoring(self, session_data: Dict, interval: float = 1.0):
        def monitor():
            while session_data.get('monitoring_active', True):
                measurement = {
                    'timestamp': time.time(),
                    'memory': self.get_memory_usage(),
                    'cpu': self.get_cpu_usage()
                }
                session_data['measurements'].append(measurement)
                time.sleep(interval)
        
        session_data['monitoring_active'] = True
        monitoring_thread = threading.Thread(target=monitor, daemon=True)
        monitoring_thread.start()
        return monitoring_thread