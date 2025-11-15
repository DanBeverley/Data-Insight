"""Utilities for task cancellation management"""

import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def is_task_cancelled(session_id: str) -> bool:
    """
    Check if a task has been cancelled by the user.

    Args:
        session_id: Session ID to check

    Returns:
        True if task is cancelled, False otherwise
    """
    try:
        from ..database.connection import get_database_manager
        from ..database.models import CancelledTask

        db_manager = get_database_manager()
        if not db_manager:
            return False

        db = db_manager.db

        recent_cancellation = (
            db.query(CancelledTask)
            .filter(
                CancelledTask.session_id == session_id,
                CancelledTask.cancelled_at > datetime.utcnow() - timedelta(seconds=30),
            )
            .first()
        )

        return recent_cancellation is not None

    except Exception as e:
        logger.warning(f"Error checking cancellation status: {e}")
        return False


def clear_cancellation(session_id: str) -> bool:
    """
    Clear cancellation flag for a session after task completes.

    Args:
        session_id: Session ID to clear

    Returns:
        True if cleared successfully
    """
    try:
        from ..database.connection import get_database_manager
        from ..database.models import CancelledTask

        db_manager = get_database_manager()
        if not db_manager:
            return False

        db = db_manager.db

        db.query(CancelledTask).filter(CancelledTask.session_id == session_id).delete()

        db.commit()
        logger.info(f"Cleared cancellation flag for session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Error clearing cancellation: {e}")
        return False
