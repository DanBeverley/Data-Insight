"""
Performance Monitoring and Optimization System
Provides real-time metrics, caching, and performance analytics
"""

import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import threading
from functools import wraps
import psutil
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
try:
    from core.logger import logger
except ImportError:
    from data_scientist_chatbot.app.core.logger import logger


@dataclass
class PerformanceMetricData:
    """Performance metric data structure"""

    session_id: str
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any]


class IntelligentCache:
    """
    Intelligent caching system with TTL and usage-based eviction
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = threading.RLock()

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if valid"""
        with self.lock:
            if key not in self.cache:
                return None
            item, expiry = self.cache[key]
            # Check if expired
            if expiry and datetime.now() > expiry:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.access_counts.pop(key, None)
                return None
            # Update access statistics
            self.access_times[key] = datetime.now()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return item

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with optional TTL"""
        with self.lock:
            # Calculate expiry
            ttl = ttl or self.default_ttl
            expiry = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
            # Evict if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            self.cache[key] = (value, expiry)
            self.access_times[key] = datetime.now()
            self.access_counts[key] = 1

    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self.access_times:
            return
        # Find least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # Remove from all structures
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.access_counts.pop(lru_key, None)

    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries, optionally by pattern"""
        with self.lock:
            if not pattern:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_times.clear()
                self.access_counts.clear()
                return count
            # Pattern-based invalidation
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.access_counts.pop(key, None)
            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            avg_accesses = total_accesses / len(self.access_counts) if self.access_counts else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": None,  # Would need hit/miss tracking
                "total_accesses": total_accesses,
                "avg_accesses_per_key": avg_accesses,
                "keys": list(self.cache.keys())[:10],
            }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """

    def __init__(self):
        try:
            from src.database.connection import get_database_manager
            from src.database.models import PerformanceMetric, SystemMetrics

            self.db_manager = get_database_manager()
            self.PerformanceMetricModel = PerformanceMetric
            self.SystemMetricsModel = SystemMetrics
        except ImportError:
            logger.warning("Could not import DatabaseManager, performance monitoring may be disabled")
            self.db_manager = None

        self.cache = IntelligentCache(max_size=500, default_ttl=1800)
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()

        self._start_metrics_flush_thread()

    def record_metric(self, session_id: str, metric_name: str, value: float, context: Dict[str, Any] = None) -> None:
        """Record a performance metric"""
        metric = PerformanceMetricData(
            session_id=session_id, metric_name=metric_name, value=value, timestamp=datetime.now(), context=context or {}
        )

        with self.buffer_lock:
            self.metrics_buffer.append(metric)
            if len(self.metrics_buffer) >= 100:
                self._flush_metrics()

    def _flush_metrics(self) -> None:
        """Flush metrics buffer to database"""
        if not self.metrics_buffer or not self.db_manager:
            return

        with self.buffer_lock:
            metrics_to_flush = list(self.metrics_buffer)
            self.metrics_buffer.clear()

        try:
            with self.db_manager.get_session() as session:
                for metric in metrics_to_flush:
                    db_metric = self.PerformanceMetricModel(
                        session_id=metric.session_id,
                        metric_name=metric.metric_name,
                        metric_value=metric.value,
                        timestamp=metric.timestamp,
                        context=metric.context,
                    )
                    session.add(db_metric)
        except Exception as e:
            logger.error(f"Error flushing metrics to database: {e}")

    def _start_metrics_flush_thread(self) -> None:
        """Start background thread for periodic metrics flushing"""

        def flush_periodically():
            while True:
                time.sleep(30)  # Flush every 30 seconds
                try:
                    self._flush_metrics()
                except Exception as e:
                    logger.debug(f"Error flushing metrics: {e}")

        thread = threading.Thread(target=flush_periodically, daemon=True)
        thread.start()

    def time_function(self, session_id: str, function_name: str = None):
        """Decorator to time function execution"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    func_name = function_name or func.__name__

                    self.record_metric(
                        session_id=session_id,
                        metric_name=f"{func_name}_execution_time",
                        value=execution_time,
                        context={
                            "function": func_name,
                            "success": success,
                            "error": error,
                            "args_count": len(args),
                            "kwargs_count": len(kwargs),
                        },
                    )

                return result

            return wrapper

        return decorator

    def cache_result(self, ttl: int = 3600, key_prefix: str = ""):
        """Decorator to cache function results"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = f"{key_prefix}:{func.__name__}:" + self.cache._generate_key(*args, **kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl=ttl)

                return result

            return wrapper

        return decorator

    def record_system_metrics(self) -> None:
        """Record current system performance metrics"""
        if not self.db_manager:
            return

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            with self.db_manager.get_session() as session:
                # Record CPU
                session.add(
                    self.SystemMetricsModel(
                        metric_id=f"cpu_{datetime.now().timestamp()}",
                        metric_type="cpu_percent",
                        metric_value=cpu_percent,
                        metric_metadata={"source": "psutil"},
                    )
                )

                # Record Memory
                session.add(
                    self.SystemMetricsModel(
                        metric_id=f"mem_{datetime.now().timestamp()}",
                        metric_type="memory_percent",
                        metric_value=memory.percent,
                        metric_metadata={"total": memory.total, "available": memory.available},
                    )
                )

                # Record Disk
                session.add(
                    self.SystemMetricsModel(
                        metric_id=f"disk_{datetime.now().timestamp()}",
                        metric_type="disk_usage_percent",
                        metric_value=disk.percent,
                        metric_metadata={"path": "/"},
                    )
                )

        except Exception as e:
            logger.debug(f"Error recording system metrics: {e}")

    def get_performance_summary(self, session_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for analysis"""
        if not self.db_manager:
            return {}

        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            from sqlalchemy import select, func

            with self.db_manager.get_session() as session:
                # Query performance metrics
                query = select(
                    self.PerformanceMetricModel.metric_name,
                    func.avg(self.PerformanceMetricModel.metric_value),
                    func.count(self.PerformanceMetricModel.id),
                    func.min(self.PerformanceMetricModel.metric_value),
                    func.max(self.PerformanceMetricModel.metric_value),
                ).where(self.PerformanceMetricModel.timestamp > cutoff_time)

                if session_id:
                    query = query.where(self.PerformanceMetricModel.session_id == session_id)

                query = query.group_by(self.PerformanceMetricModel.metric_name)

                results = session.execute(query).all()

                metrics_summary = {}
                for row in results:
                    metrics_summary[row[0]] = {
                        "avg": round(row[1], 4),
                        "count": row[2],
                        "min": round(row[3], 4),
                        "max": round(row[4], 4),
                    }

                # Get system metrics averages
                sys_query = (
                    select(self.SystemMetricsModel.metric_type, func.avg(self.SystemMetricsModel.metric_value))
                    .where(self.SystemMetricsModel.timestamp > cutoff_time)
                    .group_by(self.SystemMetricsModel.metric_type)
                )

                sys_results = session.execute(sys_query).all()
                sys_avgs = {row[0]: round(row[1], 2) for row in sys_results}

                return {
                    "metrics_summary": metrics_summary,
                    "cache_stats": self.cache.get_stats(),
                    "system_averages": {
                        "cpu_percent": sys_avgs.get("cpu_percent", 0),
                        "memory_percent": sys_avgs.get("memory_percent", 0),
                        "disk_usage_percent": sys_avgs.get("disk_usage_percent", 0),
                    },
                    "time_range_hours": hours,
                    "session_id": session_id,
                }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def get_slow_operations(self, threshold_seconds: float = 1.0, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations for optimization"""
        if not self.db_manager:
            return []

        try:
            from sqlalchemy import select

            with self.db_manager.get_session() as session:
                query = (
                    select(self.PerformanceMetricModel)
                    .where(
                        self.PerformanceMetricModel.metric_name.like("%_execution_time"),
                        self.PerformanceMetricModel.metric_value > threshold_seconds,
                    )
                    .order_by(self.PerformanceMetricModel.metric_value.desc())
                    .limit(limit)
                )

                results = session.execute(query).scalars().all()

                slow_ops = []
                for row in results:
                    slow_ops.append(
                        {
                            "session_id": row.session_id,
                            "operation": row.metric_name,
                            "duration": round(row.metric_value, 4),
                            "timestamp": row.timestamp,
                            "context": row.context,
                        }
                    )
                return slow_ops
        except Exception as e:
            logger.error(f"Error getting slow operations: {e}")
            return []

    def optimize_cache(self) -> Dict[str, Any]:
        """Analyze and optimize cache performance"""
        stats = self.cache.get_stats()
        recommendations = []
        # Check cache utilization
        utilization = stats["size"] / stats["max_size"]
        if utilization > 0.9:
            recommendations.append("Consider increasing cache size - high utilization detected")
        elif utilization < 0.3:
            recommendations.append("Cache may be undersized - low utilization detected")
        # Check for frequent access patterns
        if stats["avg_accesses_per_key"] > 10:
            recommendations.append("High cache hit patterns - consider increasing TTL")
        return {"cache_stats": stats, "utilization": round(utilization, 2), "recommendations": recommendations}

    def cleanup_old_metrics(self, days_old: int = 7) -> int:
        """Clean up old performance metrics"""
        if not self.db_manager:
            return 0

        cutoff_date = datetime.now() - timedelta(days=days_old)

        try:
            from sqlalchemy import delete

            with self.db_manager.get_session() as session:
                # Delete performance metrics
                stmt1 = delete(self.PerformanceMetricModel).where(self.PerformanceMetricModel.timestamp < cutoff_date)
                result1 = session.execute(stmt1)

                # Delete system metrics
                stmt2 = delete(self.SystemMetricsModel).where(self.SystemMetricsModel.timestamp < cutoff_date)
                result2 = session.execute(stmt2)

                return result1.rowcount + result2.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
            return 0
