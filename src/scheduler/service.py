"""Quorvix Scheduler Service - APScheduler with SQLAlchemy persistence."""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from contextlib import contextmanager
import sqlite3
import json

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None

logger = logging.getLogger(__name__)

SCHEDULER_DB_PATH = Path("data/scheduler/scheduler.db")
HISTORY_DB_PATH = Path("data/scheduler/job_history.db")


class JobHistoryStore:
    def __init__(self, db_path: Path = HISTORY_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    job_name TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT DEFAULT 'running',
                    error TEXT,
                    result TEXT
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_id ON job_history(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_started_at ON job_history(started_at)")
            conn.commit()

    def record_start(self, job_id: str, job_name: str) -> int:
        with self._get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO job_history (job_id, job_name, started_at, status) VALUES (?, ?, ?, ?)",
                (job_id, job_name, datetime.utcnow().isoformat(), "running"),
            )
            conn.commit()
            return cursor.lastrowid

    def record_finish(self, history_id: int, status: str, error: str = None, result: str = None):
        with self._get_db() as conn:
            conn.execute(
                "UPDATE job_history SET finished_at = ?, status = ?, error = ?, result = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), status, error, result, history_id),
            )
            conn.commit()

    def get_history(self, limit: int = 100, job_id: str = None) -> List[Dict]:
        with self._get_db() as conn:
            if job_id:
                cursor = conn.execute(
                    "SELECT * FROM job_history WHERE job_id = ? ORDER BY started_at DESC LIMIT ?", (job_id, limit)
                )
            else:
                cursor = conn.execute("SELECT * FROM job_history ORDER BY started_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old(self, days: int = 30):
        with self._get_db() as conn:
            conn.execute("DELETE FROM job_history WHERE started_at < datetime('now', ?)", (f"-{days} days",))
            conn.commit()


class SchedulerService:
    def __init__(self):
        if not APSCHEDULER_AVAILABLE:
            raise ImportError("APScheduler required: pip install apscheduler")

        SCHEDULER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        jobstores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{SCHEDULER_DB_PATH}")}
        executors = {"default": ThreadPoolExecutor(max_workers=5)}
        job_defaults = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 300}

        self.scheduler = BackgroundScheduler(
            jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone="UTC"
        )
        self.history = JobHistoryStore()
        self._job_registry: Dict[str, Callable] = {}
        self._started = False

    def _wrap_job(self, func: Callable, job_id: str, job_name: str) -> Callable:
        def wrapped():
            history_id = self.history.record_start(job_id, job_name)
            try:
                result = func()
                self.history.record_finish(history_id, "success", result=str(result) if result else None)
                logger.info(f"[SCHEDULER] Job '{job_name}' completed successfully")
            except Exception as e:
                self.history.record_finish(history_id, "error", error=str(e))
                logger.error(f"[SCHEDULER] Job '{job_name}' failed: {e}")

        return wrapped

    def register_job(self, job_id: str, func: Callable):
        self._job_registry[job_id] = func

    def add_interval_job(
        self,
        job_id: str,
        func: Callable,
        minutes: int = None,
        hours: int = None,
        seconds: int = None,
        job_name: str = None,
    ) -> bool:
        try:
            name = job_name or job_id
            self.register_job(job_id, func)

            kwargs = {}
            if minutes:
                kwargs["minutes"] = minutes
            if hours:
                kwargs["hours"] = hours
            if seconds:
                kwargs["seconds"] = seconds

            self.scheduler.add_job(func, trigger=IntervalTrigger(**kwargs), id=job_id, name=name, replace_existing=True)
            logger.info(f"[SCHEDULER] Added interval job: {name}")
            return True
        except Exception as e:
            logger.error(f"[SCHEDULER] Failed to add job {job_id}: {e}")
            return False

    def add_cron_job(
        self,
        job_id: str,
        func: Callable,
        hour: int = None,
        minute: int = 0,
        day_of_week: str = None,
        job_name: str = None,
    ) -> bool:
        try:
            name = job_name or job_id
            self.register_job(job_id, func)

            kwargs = {"minute": minute}
            if hour is not None:
                kwargs["hour"] = hour
            if day_of_week:
                kwargs["day_of_week"] = day_of_week

            self.scheduler.add_job(func, trigger=CronTrigger(**kwargs), id=job_id, name=name, replace_existing=True)
            logger.info(f"[SCHEDULER] Added cron job: {name}")
            return True
        except Exception as e:
            logger.error(f"[SCHEDULER] Failed to add cron job {job_id}: {e}")
            return False

    def remove_job(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
            self._job_registry.pop(job_id, None)
            logger.info(f"[SCHEDULER] Removed job: {job_id}")
            return True
        except Exception:
            return False

    def pause_job(self, job_id: str) -> bool:
        try:
            self.scheduler.pause_job(job_id)
            return True
        except Exception:
            return False

    def resume_job(self, job_id: str) -> bool:
        try:
            self.scheduler.resume_job(job_id)
            return True
        except Exception:
            return False

    def run_job_now(self, job_id: str) -> bool:
        func = self._job_registry.get(job_id)
        if not func:
            return False
        try:
            wrapped = self._wrap_job(func, job_id, job_id)
            wrapped()
            return True
        except Exception as e:
            logger.error(f"[SCHEDULER] Manual run failed for {job_id}: {e}")
            return False

    def get_jobs(self) -> List[Dict]:
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger),
                    "paused": job.next_run_time is None,
                }
            )
        return jobs

    def get_job(self, job_id: str) -> Optional[Dict]:
        job = self.scheduler.get_job(job_id)
        if not job:
            return None
        return {
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
            "paused": job.next_run_time is None,
        }

    def get_history(self, limit: int = 100, job_id: str = None) -> List[Dict]:
        return self.history.get_history(limit, job_id)

    def start(self):
        if not self._started:
            self.scheduler.start()
            self._started = True
            logger.info("[SCHEDULER] Started")

    def shutdown(self):
        if self._started:
            self.scheduler.shutdown(wait=False)
            self._started = False
            logger.info("[SCHEDULER] Shutdown")


_scheduler_instance: Optional[SchedulerService] = None


def get_scheduler() -> SchedulerService:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = SchedulerService()
    return _scheduler_instance


def initialize_scheduler() -> SchedulerService:
    scheduler = get_scheduler()
    from src.scheduler.jobs import register_builtin_jobs

    register_builtin_jobs(scheduler)
    scheduler.start()
    return scheduler


def shutdown_scheduler():
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.shutdown()
        _scheduler_instance = None


ALERT_DB_PATH = Path("data/scheduler/alerts.db")


class AlertScheduler:
    def __init__(self, db_path: Path = ALERT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_db() as conn:
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
                    triggered INTEGER,
                    current_value REAL,
                    threshold REAL,
                    message TEXT,
                    checked_at TEXT
                )
            """
            )
            conn.commit()

    def create_alert(self, alert) -> Any:
        from src.scheduler.models import Alert

        with self._get_db() as conn:
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
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return alert

    def get_alert(self, alert_id: str) -> Optional[Any]:
        from src.scheduler.models import Alert

        with self._get_db() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Alert(**dict(row))

    def get_all_alerts(self) -> List[Any]:
        from src.scheduler.models import Alert

        with self._get_db() as conn:
            cursor = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC")
            return [Alert(**dict(row)) for row in cursor.fetchall()]

    def get_alerts_by_session(self, session_id: str) -> List[Any]:
        from src.scheduler.models import Alert

        with self._get_db() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE session_id = ? ORDER BY created_at DESC", (session_id,))
            return [Alert(**dict(row)) for row in cursor.fetchall()]

    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> Optional[Any]:
        from src.scheduler.models import Alert

        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [alert_id]
        with self._get_db() as conn:
            conn.execute(f"UPDATE alerts SET {set_clause} WHERE id = ?", values)
            conn.commit()
        return self.get_alert(alert_id)

    def delete_alert(self, alert_id: str) -> bool:
        with self._get_db() as conn:
            cursor = conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            conn.commit()
            return cursor.rowcount > 0

    async def check_alert_now(self, alert_id: str) -> Dict[str, Any]:
        alert = self.get_alert(alert_id)
        if not alert:
            return {"error": "Alert not found"}
        return await self._evaluate_alert(alert)

    async def _evaluate_alert(self, alert) -> Dict[str, Any]:
        from src.scheduler.models import AlertCheckResult

        try:
            current_value = await self._execute_metric_query(alert.session_id, alert.metric_query)
            triggered = self._check_condition(current_value, alert.condition, alert.threshold)

            message = f"{alert.metric_name} is {current_value} ({alert.condition} {alert.threshold})"
            if triggered:
                message = f"ALERT: {message}"
                await self._send_notification(alert, current_value)

            with self._get_db() as conn:
                conn.execute(
                    """
                    UPDATE alerts SET last_run = ?, last_value = ?, last_triggered = ?
                    WHERE id = ?
                """,
                    (
                        datetime.utcnow().isoformat(),
                        current_value,
                        datetime.utcnow().isoformat() if triggered else alert.last_triggered,
                        alert.id,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO alert_history (alert_id, triggered, current_value, threshold, message, checked_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (alert.id, int(triggered), current_value, alert.threshold, message, datetime.utcnow().isoformat()),
                )
                conn.commit()

            return {
                "alert_id": alert.id,
                "triggered": triggered,
                "current_value": current_value,
                "threshold": alert.threshold,
                "condition": alert.condition,
                "message": message,
            }
        except Exception as e:
            with self._get_db() as conn:
                conn.execute("UPDATE alerts SET last_error = ? WHERE id = ?", (str(e), alert.id))
                conn.commit()
            return {"alert_id": alert.id, "error": str(e)}

    async def _execute_metric_query(self, session_id: str, query: str) -> float:
        import builtins

        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            df = builtins._session_store[session_id].get("dataframe")
            if df is not None:
                import pandas as pd

                result = eval(query, {"df": df, "pd": pd})
                return float(result)
        raise ValueError(f"No dataset found for session {session_id}")

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        ops = {
            "lt": lambda v, t: v < t,
            "gt": lambda v, t: v > t,
            "eq": lambda v, t: v == t,
            "ne": lambda v, t: v != t,
            "lte": lambda v, t: v <= t,
            "gte": lambda v, t: v >= t,
        }
        return ops.get(condition, lambda v, t: False)(value, threshold)

    async def _send_notification(self, alert, current_value: float):
        from src.notifications.service import get_notification_service, get_email_notifier, NotificationType

        service = get_notification_service()
        service.create(
            title=f"Alert: {alert.name}",
            message=f"{alert.metric_name} is {current_value} ({alert.condition} {alert.threshold})",
            type=NotificationType.ALERT,
            session_id=alert.session_id,
            alert_id=alert.id,
        )
        if alert.notification_type == "email" and alert.notification_target:
            notifier = get_email_notifier()
            if notifier.is_configured:
                notifier.send(
                    to_email=alert.notification_target,
                    subject=f"Quorvix Alert: {alert.name}",
                    body=f"{alert.metric_name} is {current_value} ({alert.condition} {alert.threshold})",
                )

    def get_alert_history(self, alert_id: str, limit: int = 50) -> List[Dict]:
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT * FROM alert_history WHERE alert_id = ? ORDER BY checked_at DESC LIMIT ?", (alert_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def evaluate_all_alerts(self) -> int:
        import asyncio

        alerts = self.get_all_alerts()
        triggered_count = 0
        for alert in alerts:
            if alert.status == "active":
                try:
                    result = asyncio.run(self._evaluate_alert(alert))
                    if result.get("triggered"):
                        triggered_count += 1
                except Exception as e:
                    logger.error(f"Failed to evaluate alert {alert.id}: {e}")
        return triggered_count


_alert_scheduler_instance: Optional[AlertScheduler] = None


def get_alert_scheduler() -> AlertScheduler:
    global _alert_scheduler_instance
    if _alert_scheduler_instance is None:
        _alert_scheduler_instance = AlertScheduler()
    return _alert_scheduler_instance
