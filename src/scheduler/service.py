import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import sqlite3
from contextlib import contextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .models import Alert, AlertStatus, AlertCondition, AlertCheckResult

logger = logging.getLogger(__name__)


class AlertScheduler:
    def __init__(self, db_path: str = "data/databases/alerts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler = AsyncIOScheduler()
        self.alerts: Dict[str, Alert] = {}
        self._init_database()
        self._load_alerts()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    metric_query TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    cron_expression TEXT DEFAULT '0 9 * * *',
                    timezone TEXT DEFAULT 'UTC',
                    notification_type TEXT DEFAULT 'email',
                    notification_target TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    last_run TEXT,
                    last_triggered TEXT,
                    last_value REAL,
                    last_error TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    triggered INTEGER NOT NULL,
                    current_value REAL,
                    threshold REAL,
                    message TEXT,
                    checked_at TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(id)
                )
            """
            )
            conn.commit()

    def _load_alerts(self):
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE status = 'active'")
            for row in cursor.fetchall():
                alert = Alert(
                    id=row["id"],
                    name=row["name"],
                    session_id=row["session_id"],
                    metric_query=row["metric_query"],
                    metric_name=row["metric_name"],
                    condition=AlertCondition(row["condition"]),
                    threshold=row["threshold"],
                    cron_expression=row["cron_expression"],
                    timezone=row["timezone"],
                    notification_type=row["notification_type"],
                    notification_target=row["notification_target"],
                    status=AlertStatus(row["status"]),
                    last_run=datetime.fromisoformat(row["last_run"]) if row["last_run"] else None,
                    last_triggered=datetime.fromisoformat(row["last_triggered"]) if row["last_triggered"] else None,
                    last_value=row["last_value"],
                    last_error=row["last_error"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
                )
                self.alerts[alert.id] = alert
        logger.info(f"Loaded {len(self.alerts)} active alerts")

    def start(self):
        if not self.scheduler.running:
            for alert in self.alerts.values():
                self._schedule_alert(alert)
            self.scheduler.start()
            logger.info("Alert scheduler started")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Alert scheduler stopped")

    def _schedule_alert(self, alert: Alert):
        try:
            trigger = CronTrigger.from_crontab(alert.cron_expression)
            self.scheduler.add_job(
                self._check_alert,
                trigger=trigger,
                args=[alert.id],
                id=f"alert_{alert.id}",
                replace_existing=True,
                timezone=alert.timezone,
            )
            logger.info(f"Scheduled alert '{alert.name}' with cron '{alert.cron_expression}'")
        except Exception as e:
            logger.error(f"Failed to schedule alert {alert.id}: {e}")

    async def _check_alert(self, alert_id: str):
        alert = self.alerts.get(alert_id)
        if not alert or alert.status != AlertStatus.ACTIVE:
            return

        try:
            current_value = await self._execute_metric_query(alert)
            triggered = self._evaluate_condition(current_value, alert.condition, alert.threshold)

            result = AlertCheckResult(
                alert_id=alert_id,
                triggered=triggered,
                current_value=current_value,
                threshold=alert.threshold,
                condition=alert.condition,
                message=f"{alert.metric_name}: {current_value} {alert.condition} {alert.threshold}",
            )

            alert.last_run = datetime.utcnow()
            alert.last_value = current_value
            alert.last_error = None

            if triggered:
                alert.last_triggered = datetime.utcnow()
                await self._send_notification(alert, result)

            self._save_alert(alert)
            self._save_history(result)

            logger.info(f"Alert check: {alert.name} - Value: {current_value}, Triggered: {triggered}")

        except Exception as e:
            alert.last_error = str(e)
            alert.last_run = datetime.utcnow()
            self._save_alert(alert)
            logger.error(f"Alert check failed for {alert_id}: {e}")

    async def _execute_metric_query(self, alert: Alert) -> float:
        from src.api_utils.session_management import session_data_manager

        session_data = session_data_manager.get_session(alert.session_id)
        if not session_data:
            raise ValueError(f"Session {alert.session_id} not found")

        dataset_path = session_data.get("dataset_path")
        if not dataset_path:
            raise ValueError("No dataset in session")

        import pandas as pd

        df = pd.read_csv(dataset_path)

        try:
            result = eval(alert.metric_query, {"df": df, "pd": pd})
            return float(result)
        except Exception as e:
            raise ValueError(f"Query execution failed: {e}")

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        operators = {
            "lt": lambda v, t: v < t,
            "gt": lambda v, t: v > t,
            "eq": lambda v, t: v == t,
            "ne": lambda v, t: v != t,
            "lte": lambda v, t: v <= t,
            "gte": lambda v, t: v >= t,
        }
        return operators.get(condition, lambda v, t: False)(value, threshold)

    async def _send_notification(self, alert: Alert, result: AlertCheckResult):
        from src.notifications.service import get_notification_service, get_email_notifier, NotificationType

        notification_service = get_notification_service()
        notification_service.create(
            title=f"🔔 Alert: {alert.name}",
            message=result.message,
            type=NotificationType.ALERT,
            session_id=alert.session_id,
            alert_id=alert.id,
            metadata={"value": result.current_value, "threshold": result.threshold},
        )

        if alert.notification_type == "email":
            email_notifier = get_email_notifier()
            if email_notifier.is_configured:
                email_notifier.send(
                    to_email=alert.notification_target,
                    subject=f"Alert Triggered: {alert.name}",
                    body=f"{alert.metric_name}\n\nCurrent Value: {result.current_value}\nThreshold: {alert.threshold}\n\n{result.message}",
                )
        elif alert.notification_type == "webhook":
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(
                    alert.notification_target,
                    json={
                        "alert_id": alert.id,
                        "alert_name": alert.name,
                        "triggered": result.triggered,
                        "value": result.current_value,
                        "threshold": result.threshold,
                        "message": result.message,
                    },
                )

    def _save_alert(self, alert: Alert):
        alert.updated_at = datetime.utcnow()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE alerts SET
                    status = ?, last_run = ?, last_triggered = ?, 
                    last_value = ?, last_error = ?, updated_at = ?
                WHERE id = ?
            """,
                (
                    alert.status,
                    alert.last_run.isoformat() if alert.last_run else None,
                    alert.last_triggered.isoformat() if alert.last_triggered else None,
                    alert.last_value,
                    alert.last_error,
                    alert.updated_at.isoformat(),
                    alert.id,
                ),
            )
            conn.commit()

    def _save_history(self, result: AlertCheckResult):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alert_history (alert_id, triggered, current_value, threshold, message, checked_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    result.alert_id.value if hasattr(result.alert_id, "value") else result.alert_id,
                    1 if result.triggered else 0,
                    result.current_value,
                    result.threshold,
                    result.message,
                    result.checked_at.isoformat(),
                ),
            )
            conn.commit()

    def create_alert(self, alert: Alert) -> Alert:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (id, name, session_id, metric_query, metric_name, condition, 
                    threshold, cron_expression, timezone, notification_type, notification_target,
                    status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.id,
                    alert.name,
                    alert.session_id,
                    alert.metric_query,
                    alert.metric_name,
                    alert.condition,
                    alert.threshold,
                    alert.cron_expression,
                    alert.timezone,
                    alert.notification_type,
                    alert.notification_target,
                    alert.status,
                    alert.created_at.isoformat(),
                    alert.updated_at.isoformat(),
                ),
            )
            conn.commit()

        self.alerts[alert.id] = alert
        if self.scheduler.running:
            self._schedule_alert(alert)

        return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        return self.alerts.get(alert_id)

    def get_alerts_by_session(self, session_id: str) -> List[Alert]:
        return [a for a in self.alerts.values() if a.session_id == session_id]

    def get_all_alerts(self) -> List[Alert]:
        return list(self.alerts.values())

    def update_alert(self, alert_id: str, updates: Dict) -> Optional[Alert]:
        alert = self.alerts.get(alert_id)
        if not alert:
            return None

        for key, value in updates.items():
            if hasattr(alert, key) and value is not None:
                setattr(alert, key, value)

        self._save_alert(alert)

        if "cron_expression" in updates:
            self.scheduler.remove_job(f"alert_{alert_id}")
            self._schedule_alert(alert)

        return alert

    def delete_alert(self, alert_id: str) -> bool:
        if alert_id not in self.alerts:
            return False

        try:
            self.scheduler.remove_job(f"alert_{alert_id}")
        except:
            pass

        del self.alerts[alert_id]

        with self._get_connection() as conn:
            conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            conn.commit()

        return True

    def get_alert_history(self, alert_id: str, limit: int = 50) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM alert_history 
                WHERE alert_id = ? 
                ORDER BY checked_at DESC 
                LIMIT ?
            """,
                (alert_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    async def check_alert_now(self, alert_id: str) -> Optional[AlertCheckResult]:
        await self._check_alert(alert_id)
        alert = self.alerts.get(alert_id)
        if alert:
            return AlertCheckResult(
                alert_id=alert_id,
                triggered=alert.last_triggered == alert.last_run if alert.last_run else False,
                current_value=alert.last_value or 0,
                threshold=alert.threshold,
                condition=alert.condition,
                message=f"{alert.metric_name}: {alert.last_value}",
            )
        return None


_scheduler_instance: Optional[AlertScheduler] = None


def get_alert_scheduler() -> AlertScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AlertScheduler()
    return _scheduler_instance
