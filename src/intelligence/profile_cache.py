"""Profile caching system to avoid re-profiling identical datasets"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProfileCache:
    """Cache data profiling results by dataset hash"""

    def __init__(self, cache_db: str = "data_profile_cache.db"):
        self.cache_db = Path(cache_db)
        self._initialize_db()

    def _initialize_db(self):
        """Create cache database and tables"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_cache (
                dataset_hash TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                row_count INTEGER,
                column_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_accessed
            ON profile_cache(last_accessed DESC)
        """
        )

        conn.commit()
        conn.close()

    def compute_dataset_hash(self, df) -> str:
        """
        Compute fast hash for dataset based on shape, columns, and sample data

        Strategy: Hash metadata + first/last 100 rows for speed
        """
        import pandas as pd

        hash_input = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        sample_size = min(100, len(df))
        if len(df) > 200:
            sample_df = pd.concat([df.head(sample_size), df.tail(sample_size)])
        else:
            sample_df = df

        hash_input["sample_values"] = sample_df.to_dict("records")

        hash_str = json.dumps(hash_input, sort_keys=True, default=str)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get_cached_profile(self, dataset_hash: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached profile if exists and not expired

        Args:
            dataset_hash: MD5 hash of dataset
            max_age_hours: Maximum age of cache in hours (default 24)

        Returns:
            Cached profile dict or None if not found/expired
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        cursor.execute(
            """
            SELECT profile_json, created_at, access_count
            FROM profile_cache
            WHERE dataset_hash = ? AND created_at > ?
        """,
            (dataset_hash, cutoff_time.isoformat()),
        )

        row = cursor.fetchone()

        if row:
            profile_json, created_at, access_count = row

            cursor.execute(
                """
                UPDATE profile_cache
                SET last_accessed = ?, access_count = ?
                WHERE dataset_hash = ?
            """,
                (datetime.now().isoformat(), access_count + 1, dataset_hash),
            )

            conn.commit()
            conn.close()

            profile = json.loads(profile_json)
            logger.info(
                f"Profile cache HIT for {dataset_hash[:8]} (age: {(datetime.now() - datetime.fromisoformat(created_at)).seconds // 60}min)"
            )
            return profile
        else:
            conn.close()
            logger.info(f"Profile cache MISS for {dataset_hash[:8]}")
            return None

    def save_profile(self, dataset_hash: str, profile: Dict[str, Any], row_count: int, column_count: int):
        """
        Save profiling results to cache

        Args:
            dataset_hash: MD5 hash of dataset
            profile: Profiling results dictionary
            row_count: Number of rows in dataset
            column_count: Number of columns in dataset
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        def sanitize_for_json(obj):
            """Recursively sanitize object for JSON serialization, handling numpy dtypes as keys"""
            import numpy as np
            import pandas as pd
            from enum import Enum

            if isinstance(obj, dict):
                return {str(k): sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, "__dtype__"):
                return str(obj)
            elif hasattr(obj, "__dict__"):
                return sanitize_for_json(obj.__dict__)
            else:
                return obj

        sanitized_profile = sanitize_for_json(profile)
        profile_json = json.dumps(sanitized_profile)

        cursor.execute(
            """
            INSERT OR REPLACE INTO profile_cache
            (dataset_hash, profile_json, row_count, column_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                dataset_hash,
                profile_json,
                row_count,
                column_count,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Saved profile to cache: {dataset_hash[:8]} ({row_count} rows × {column_count} cols)")

    def cleanup_old_entries(self, days: int = 7, max_entries: int = 1000):
        """
        Remove old/unused cache entries

        Args:
            days: Remove entries older than this many days
            max_entries: Keep only this many most recent entries
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cutoff_time = datetime.now() - timedelta(days=days)

        cursor.execute(
            """
            DELETE FROM profile_cache
            WHERE last_accessed < ?
        """,
            (cutoff_time.isoformat(),),
        )

        deleted_old = cursor.rowcount

        cursor.execute(
            """
            DELETE FROM profile_cache
            WHERE dataset_hash NOT IN (
                SELECT dataset_hash FROM profile_cache
                ORDER BY last_accessed DESC
                LIMIT ?
            )
        """,
            (max_entries,),
        )

        deleted_excess = cursor.rowcount

        conn.commit()
        conn.close()

        if deleted_old + deleted_excess > 0:
            logger.info(f"Cache cleanup: removed {deleted_old} old + {deleted_excess} excess entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) as total_entries,
                SUM(access_count) as total_accesses,
                AVG(access_count) as avg_accesses,
                MAX(created_at) as newest_entry,
                MIN(created_at) as oldest_entry
            FROM profile_cache
        """
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "total_entries": row[0],
                "total_accesses": row[1] or 0,
                "avg_accesses_per_entry": round(row[2] or 0, 2),
                "newest_entry": row[3],
                "oldest_entry": row[4],
            }
        return {}


_global_profile_cache: Optional[ProfileCache] = None


def get_profile_cache() -> ProfileCache:
    """Get or create global profile cache instance"""
    global _global_profile_cache
    if _global_profile_cache is None:
        _global_profile_cache = ProfileCache()
    return _global_profile_cache
