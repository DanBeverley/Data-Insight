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

@dataclass
class PerformanceMetric:
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
        key_data = {'args': args, 'kwargs': sorted(kwargs.items())}
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
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': None,  # Would need hit/miss tracking
                'total_accesses': total_accesses,
                'avg_accesses_per_key': avg_accesses,
                'keys': list(self.cache.keys())[:10]  
            }

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(self, db_path: str = "data_scientist_chatbot/memory/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = IntelligentCache(max_size=500, default_ttl=1800) 
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()
        self._init_database()
        
        # Start background metrics flushing
        self._start_metrics_flush_thread()
    
    def _init_database(self):
        """Initialize performance metrics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                context TEXT, -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage_percent REAL,
                active_sessions INTEGER,
                cache_hit_rate REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_session ON performance_metrics(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_perf_metric ON performance_metrics(metric_name)")
        
        conn.commit()
        conn.close()
    
    def record_metric(self, session_id: str, metric_name: str, 
                     value: float, context: Dict[str, Any] = None) -> None:
        """Record a performance metric"""
        metric = PerformanceMetric(
            session_id=session_id,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric)
            
            # Flush if buffer is full
            if len(self.metrics_buffer) >= 100:
                self._flush_metrics()
    
    def _flush_metrics(self) -> None:
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric in self.metrics_buffer:
            cursor.execute("""
                INSERT INTO performance_metrics 
                (session_id, metric_name, metric_value, timestamp, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.session_id,
                metric.metric_name,
                metric.value,
                metric.timestamp,
                json.dumps(metric.context)
            ))
        
        conn.commit()
        conn.close()
        
        self.metrics_buffer.clear()
    
    def _start_metrics_flush_thread(self) -> None:
        """Start background thread for periodic metrics flushing"""
        def flush_periodically():
            while True:
                time.sleep(30)  # Flush every 30 seconds
                with self.buffer_lock:
                    if self.metrics_buffer:
                        try:
                            self._flush_metrics()
                        except Exception as e:
                            print(f"Error flushing metrics: {e}")
        
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
                            'function': func_name,
                            'success': success,
                            'error': error,
                            'args_count': len(args),
                            'kwargs_count': len(kwargs)
                        }
                    )
                
                return result
            return wrapper
        return decorator
    
    def cache_result(self, ttl: int = 3600, key_prefix: str = ""):
        """Decorator to cache function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{func.__name__}:" + self.cache._generate_key(*args, **kwargs)
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl=ttl)
                
                return result
            return wrapper
        return decorator
    
    def record_system_metrics(self) -> None:
        """Record current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics 
                (cpu_percent, memory_percent, disk_usage_percent, active_sessions, cache_hit_rate)
                VALUES (?, ?, ?, ?, ?)
            """, (
                cpu_percent,
                memory.percent,
                disk.percent,
                0,  
                0.0  
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error recording system metrics: {e}")
    
    def get_performance_summary(self, session_id: Optional[str] = None, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Base query
        base_query = """
            SELECT metric_name, AVG(metric_value) as avg_value, 
                   COUNT(*) as count, MIN(metric_value) as min_value,
                   MAX(metric_value) as max_value
            FROM performance_metrics
            WHERE timestamp > ?
        """
        params = [cutoff_time]
        
        if session_id:
            base_query += " AND session_id = ?"
            params.append(session_id)
        
        base_query += " GROUP BY metric_name ORDER BY avg_value DESC"
        
        cursor.execute(base_query, params)
        
        metrics_summary = {}
        for row in cursor.fetchall():
            metrics_summary[row[0]] = {
                'avg': round(row[1], 4),
                'count': row[2],
                'min': round(row[3], 4),
                'max': round(row[4], 4)
            }
        
        # Get cache statistics
        cache_stats = self.cache.get_stats()
        
        # Get recent system metrics
        cursor.execute("""
            SELECT AVG(cpu_percent), AVG(memory_percent), AVG(disk_usage_percent)
            FROM system_metrics
            WHERE timestamp > ?
        """, (cutoff_time,))
        
        system_avg = cursor.fetchone()
        
        conn.close()
        
        return {
            'metrics_summary': metrics_summary,
            'cache_stats': cache_stats,
            'system_averages': {
                'cpu_percent': round(system_avg[0] or 0, 2),
                'memory_percent': round(system_avg[1] or 0, 2),
                'disk_usage_percent': round(system_avg[2] or 0, 2)
            } if system_avg else {},
            'time_range_hours': hours,
            'session_id': session_id
        }
    
    def get_slow_operations(self, threshold_seconds: float = 1.0, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations for optimization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, metric_name, metric_value, timestamp, context
            FROM performance_metrics
            WHERE metric_name LIKE '%_execution_time' AND metric_value > ?
            ORDER BY metric_value DESC
            LIMIT ?
        """, (threshold_seconds, limit))
        
        slow_ops = []
        for row in cursor.fetchall():
            context = json.loads(row[4]) if row[4] else {}
            slow_ops.append({
                'session_id': row[0],
                'operation': row[1],
                'duration': round(row[2], 4),
                'timestamp': row[3],
                'context': context
            })
        
        conn.close()
        return slow_ops
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Analyze and optimize cache performance"""
        stats = self.cache.get_stats()
        
        recommendations = []
        
        # Check cache utilization
        utilization = stats['size'] / stats['max_size']
        if utilization > 0.9:
            recommendations.append("Consider increasing cache size - high utilization detected")
        elif utilization < 0.3:
            recommendations.append("Cache may be undersized - low utilization detected")
        
        # Check for frequent access patterns
        if stats['avg_accesses_per_key'] > 10:
            recommendations.append("High cache hit patterns - consider increasing TTL")
        
        return {
            'cache_stats': stats,
            'utilization': round(utilization, 2),
            'recommendations': recommendations
        }
    
    def cleanup_old_metrics(self, days_old: int = 7) -> int:
        """Clean up old performance metrics"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM performance_metrics WHERE timestamp < ?
        """, (cutoff_date,))
        
        deleted_count = cursor.rowcount
        
        cursor.execute("""
            DELETE FROM system_metrics WHERE timestamp < ?
        """, (cutoff_date,))
        
        deleted_count += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted_count