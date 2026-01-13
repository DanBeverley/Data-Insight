"""Quorvix Scheduler - Built-in Jobs."""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.scheduler.service import SchedulerService

logger = logging.getLogger(__name__)


def check_alerts():
    try:
        from src.scheduler.service import get_alert_scheduler

        scheduler = get_alert_scheduler()
        triggered = scheduler.evaluate_all_alerts()
        logger.info(f"[JOB] Alert check completed: {triggered} alerts triggered")
        return triggered
    except Exception as e:
        logger.error(f"[JOB] Alert check failed: {e}")
        return 0


def send_daily_digest():
    try:
        from src.notifications.service import get_notification_service, get_email_notifier, NotificationType

        service = get_notification_service()
        notifier = get_email_notifier()

        if not notifier.is_configured:
            logger.debug("[JOB] Email not configured, skipping digest")
            return 0

        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()

        with service._get_db() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM notifications WHERE created_at > ?", (yesterday,))
            count = cursor.fetchone()["count"]

        if count > 0:
            service.create(
                title="Daily Digest",
                message=f"You had {count} notifications in the last 24 hours.",
                type=NotificationType.INFO,
            )
            logger.info(f"[JOB] Daily digest created: {count} notifications")

        return count
    except Exception as e:
        logger.error(f"[JOB] Daily digest failed: {e}")
        raise


def cleanup_old_sessions():
    try:
        from pathlib import Path
        import shutil

        sessions_dir = Path("data/sessions")
        if not sessions_dir.exists():
            return 0

        cutoff = datetime.utcnow() - timedelta(days=7)
        cleaned = 0

        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                try:
                    mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                    if mtime < cutoff:
                        shutil.rmtree(session_dir)
                        cleaned += 1
                except Exception:
                    pass

        logger.info(f"[JOB] Session cleanup completed: {cleaned} old sessions removed")
        return cleaned
    except Exception as e:
        logger.error(f"[JOB] Session cleanup failed: {e}")
        raise


def check_research_interrupts():
    try:
        from data_scientist_chatbot.app.state.research_state_manager import ResearchStateManager

        manager = ResearchStateManager()
        manager.mark_interrupted_on_startup()

        interrupted = manager.get_interrupted_sessions()
        if interrupted:
            logger.info(f"[JOB] Found {len(interrupted)} interrupted research sessions")
        return len(interrupted)
    except ImportError:
        return 0
    except Exception as e:
        logger.error(f"[JOB] Research interrupt check failed: {e}")
        raise


def cleanup_job_history():
    try:
        from src.scheduler.service import get_scheduler

        scheduler = get_scheduler()
        scheduler.history.cleanup_old(days=30)
        logger.info("[JOB] Job history cleanup completed")
        return True
    except Exception as e:
        logger.error(f"[JOB] Job history cleanup failed: {e}")
        raise


def register_builtin_jobs(scheduler: "SchedulerService"):
    scheduler.add_interval_job(job_id="check_alerts", func=check_alerts, minutes=5, job_name="Alert Condition Check")

    scheduler.add_cron_job(
        job_id="daily_digest", func=send_daily_digest, hour=9, minute=0, job_name="Daily Notification Digest"
    )

    scheduler.add_cron_job(
        job_id="cleanup_sessions", func=cleanup_old_sessions, hour=3, minute=0, job_name="Old Session Cleanup"
    )

    scheduler.add_interval_job(
        job_id="check_research_interrupts",
        func=check_research_interrupts,
        minutes=15,
        job_name="Research Interrupt Monitor",
    )

    scheduler.add_cron_job(
        job_id="cleanup_job_history", func=cleanup_job_history, hour=4, minute=0, job_name="Job History Cleanup"
    )

    logger.info("[SCHEDULER] Built-in jobs registered")
