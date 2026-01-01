import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import sqlite3
import hashlib
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    id: str
    connection_id: str
    timestamp: str
    row_count: int
    checksum: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeRecord:
    id: str
    connection_id: str
    timestamp: str
    rows_added: int
    rows_removed: int
    metrics_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)


class SnapshotService:
    def __init__(self, db_path: str = "data/snapshots/snapshots.db", retention_days: int = 30):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._init_database()

    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    connection_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    row_count INTEGER,
                    checksum TEXT,
                    metrics TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS change_history (
                    id TEXT PRIMARY KEY,
                    connection_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    rows_added INTEGER DEFAULT 0,
                    rows_removed INTEGER DEFAULT 0,
                    metrics_changes TEXT,
                    anomalies TEXT
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_conn ON snapshots(connection_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_conn ON change_history(connection_id)")
            conn.commit()

    def create_snapshot(
        self, connection_id: str, row_count: int, metrics: Dict[str, Any], data_sample: Optional[str] = None
    ) -> Snapshot:
        import uuid

        checksum = hashlib.md5(f"{row_count}{json.dumps(metrics, sort_keys=True)}".encode()).hexdigest()[:16]

        snapshot = Snapshot(
            id=str(uuid.uuid4()),
            connection_id=connection_id,
            timestamp=datetime.utcnow().isoformat(),
            row_count=row_count,
            checksum=checksum,
            metrics=metrics,
        )

        with self._get_db() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (id, connection_id, timestamp, row_count, checksum, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    snapshot.id,
                    snapshot.connection_id,
                    snapshot.timestamp,
                    snapshot.row_count,
                    snapshot.checksum,
                    json.dumps(snapshot.metrics),
                ),
            )
            conn.commit()

        return snapshot

    def get_latest_snapshot(self, connection_id: str) -> Optional[Snapshot]:
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM snapshots WHERE connection_id = ? ORDER BY timestamp DESC LIMIT 1", (connection_id,)
            )
            row = cursor.fetchone()
            if row:
                return Snapshot(
                    id=row["id"],
                    connection_id=row["connection_id"],
                    timestamp=row["timestamp"],
                    row_count=row["row_count"],
                    checksum=row["checksum"],
                    metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                )
        return None

    def compare_and_record(
        self, connection_id: str, new_row_count: int, new_metrics: Dict[str, Any]
    ) -> Optional[ChangeRecord]:
        previous = self.get_latest_snapshot(connection_id)

        if not previous:
            self.create_snapshot(connection_id, new_row_count, new_metrics)
            return None

        rows_added = max(0, new_row_count - previous.row_count)
        rows_removed = max(0, previous.row_count - new_row_count)

        metrics_changes = {}
        for key, new_val in new_metrics.items():
            old_val = previous.metrics.get(key)
            if old_val is not None and old_val != new_val:
                try:
                    change_pct = ((float(new_val) - float(old_val)) / float(old_val)) * 100 if old_val != 0 else 0
                    metrics_changes[key] = {"before": old_val, "after": new_val, "change_pct": round(change_pct, 2)}
                except (ValueError, TypeError):
                    metrics_changes[key] = {"before": old_val, "after": new_val}

        anomalies = self._detect_anomalies(connection_id, new_metrics)

        if rows_added or rows_removed or metrics_changes or anomalies:
            import uuid

            change = ChangeRecord(
                id=str(uuid.uuid4()),
                connection_id=connection_id,
                timestamp=datetime.utcnow().isoformat(),
                rows_added=rows_added,
                rows_removed=rows_removed,
                metrics_changes=metrics_changes,
                anomalies=anomalies,
            )

            with self._get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO change_history (id, connection_id, timestamp, rows_added, rows_removed, metrics_changes, anomalies)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        change.id,
                        change.connection_id,
                        change.timestamp,
                        change.rows_added,
                        change.rows_removed,
                        json.dumps(change.metrics_changes),
                        json.dumps(change.anomalies),
                    ),
                )
                conn.commit()

            self.create_snapshot(connection_id, new_row_count, new_metrics)
            return change

        return None

    def _detect_anomalies(self, connection_id: str, current_metrics: Dict[str, Any]) -> List[Dict]:
        anomalies = []
        history = self.get_metrics_history(connection_id, days=7)

        if len(history) < 3:
            return anomalies

        for key, current_val in current_metrics.items():
            try:
                current_num = float(current_val)
                historical_vals = [float(h["metrics"].get(key, 0)) for h in history if key in h.get("metrics", {})]

                if len(historical_vals) < 3:
                    continue

                mean = sum(historical_vals) / len(historical_vals)
                variance = sum((x - mean) ** 2 for x in historical_vals) / len(historical_vals)
                std_dev = variance**0.5

                if std_dev > 0:
                    z_score = (current_num - mean) / std_dev
                    if abs(z_score) > 2:
                        anomalies.append(
                            {
                                "metric": key,
                                "value": current_num,
                                "z_score": round(z_score, 2),
                                "mean": round(mean, 2),
                                "std_dev": round(std_dev, 2),
                                "severity": "high" if abs(z_score) > 3 else "medium",
                            }
                        )
            except (ValueError, TypeError):
                continue

        return anomalies

    def get_metrics_history(self, connection_id: str, days: int = 30) -> List[Dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM snapshots WHERE connection_id = ? AND timestamp > ? ORDER BY timestamp DESC",
                (connection_id, cutoff),
            )
            return [
                {
                    "timestamp": row["timestamp"],
                    "row_count": row["row_count"],
                    "metrics": json.loads(row["metrics"]) if row["metrics"] else {},
                }
                for row in cursor.fetchall()
            ]

    def get_change_history(self, connection_id: str, days: int = 30) -> List[ChangeRecord]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM change_history WHERE connection_id = ? AND timestamp > ? ORDER BY timestamp DESC",
                (connection_id, cutoff),
            )
            return [
                ChangeRecord(
                    id=row["id"],
                    connection_id=row["connection_id"],
                    timestamp=row["timestamp"],
                    rows_added=row["rows_added"],
                    rows_removed=row["rows_removed"],
                    metrics_changes=json.loads(row["metrics_changes"]) if row["metrics_changes"] else {},
                    anomalies=json.loads(row["anomalies"]) if row["anomalies"] else [],
                )
                for row in cursor.fetchall()
            ]

    def get_agent_context(self, connection_id: str) -> str:
        changes = self.get_change_history(connection_id, days=7)
        if not changes:
            return ""

        context_parts = ["## Recent Database Changes (Last 7 Days)"]

        for change in changes[:5]:
            parts = []
            if change.rows_added:
                parts.append(f"+{change.rows_added} rows")
            if change.rows_removed:
                parts.append(f"-{change.rows_removed} rows")
            for metric, data in change.metrics_changes.items():
                pct = data.get("change_pct", 0)
                sign = "+" if pct > 0 else ""
                parts.append(f"{metric}: {sign}{pct}%")

            if parts:
                context_parts.append(f"- {change.timestamp[:16]}: {', '.join(parts)}")

            for anomaly in change.anomalies:
                context_parts.append(f"  ⚠️ Anomaly: {anomaly['metric']} (z={anomaly['z_score']})")

        return "\n".join(context_parts)

    def cleanup_old_data(self):
        cutoff = (datetime.utcnow() - timedelta(days=self.retention_days)).isoformat()
        with self._get_db() as conn:
            conn.execute("DELETE FROM snapshots WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM change_history WHERE timestamp < ?", (cutoff,))
            conn.commit()


_snapshot_service: Optional[SnapshotService] = None


def get_snapshot_service() -> SnapshotService:
    global _snapshot_service
    if _snapshot_service is None:
        _snapshot_service = SnapshotService()
    return _snapshot_service
