import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ALERT = "alert"


@dataclass
class Notification:
    id: str
    title: str
    message: str
    type: NotificationType = NotificationType.INFO
    session_id: Optional[str] = None
    alert_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationService:
    def __init__(self, db_path: str = "data/notifications/notifications.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending_notifications: List[Notification] = []
        self._subscribers: Dict[str, List] = {}
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
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    type TEXT DEFAULT 'info',
                    session_id TEXT,
                    alert_id TEXT,
                    created_at TEXT,
                    read INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    endpoint TEXT NOT NULL,
                    p256dh TEXT,
                    auth TEXT,
                    created_at TEXT
                )
            """
            )
            conn.commit()

    def create(
        self,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        session_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Notification:
        import uuid

        notification = Notification(
            id=str(uuid.uuid4()),
            title=title,
            message=message,
            type=type,
            session_id=session_id,
            alert_id=alert_id,
            metadata=metadata or {},
        )

        with self._get_db() as conn:
            conn.execute(
                """
                INSERT INTO notifications (id, title, message, type, session_id, alert_id, created_at, read, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    notification.id,
                    notification.title,
                    notification.message,
                    notification.type.value,
                    notification.session_id,
                    notification.alert_id,
                    notification.created_at,
                    0,
                    json.dumps(notification.metadata),
                ),
            )
            conn.commit()

        self._pending_notifications.append(notification)
        return notification

    def get_pending(self) -> List[Notification]:
        pending = self._pending_notifications.copy()
        self._pending_notifications.clear()
        return pending

    def get_unread(self, session_id: Optional[str] = None, limit: int = 50) -> List[Notification]:
        with self._get_db() as conn:
            if session_id:
                cursor = conn.execute(
                    "SELECT * FROM notifications WHERE read = 0 AND (session_id = ? OR session_id IS NULL) ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM notifications WHERE read = 0 ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            return [self._row_to_notification(row) for row in cursor.fetchall()]

    def mark_read(self, notification_id: str) -> bool:
        with self._get_db() as conn:
            cursor = conn.execute("UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,))
            conn.commit()
            return cursor.rowcount > 0

    def mark_all_read(self, session_id: Optional[str] = None):
        with self._get_db() as conn:
            if session_id:
                conn.execute("UPDATE notifications SET read = 1 WHERE session_id = ?", (session_id,))
            else:
                conn.execute("UPDATE notifications SET read = 1")
            conn.commit()

    def _row_to_notification(self, row) -> Notification:
        return Notification(
            id=row["id"],
            title=row["title"],
            message=row["message"],
            type=NotificationType(row["type"]),
            session_id=row["session_id"],
            alert_id=row["alert_id"],
            created_at=row["created_at"],
            read=bool(row["read"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class EmailNotifier:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("SMTP_FROM", self.smtp_user)

    @property
    def is_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password)

    def send(self, to_email: str, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        if not self.is_configured:
            logger.warning("Email not configured - SMTP credentials missing")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email

            msg.attach(MIMEText(body, "plain"))
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


_notification_service: Optional[NotificationService] = None
_email_notifier: Optional[EmailNotifier] = None


def get_notification_service() -> NotificationService:
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


def get_email_notifier() -> EmailNotifier:
    global _email_notifier
    if _email_notifier is None:
        _email_notifier = EmailNotifier()
    return _email_notifier


class PushNotifier:
    def __init__(self):
        self.vapid_private_key = os.getenv("VAPID_PRIVATE_KEY", "")
        self.vapid_public_key = os.getenv("VAPID_PUBLIC_KEY", "")
        self.vapid_email = os.getenv("VAPID_EMAIL", "admin@example.com")

    @property
    def is_configured(self) -> bool:
        return bool(self.vapid_private_key and self.vapid_public_key)

    def save_subscription(self, endpoint: str, p256dh: str, auth: str, user_id: Optional[str] = None):
        import uuid

        service = get_notification_service()
        with service._get_db() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
            conn.execute(
                """
                INSERT INTO push_subscriptions (id, user_id, endpoint, p256dh, auth, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (str(uuid.uuid4()), user_id, endpoint, p256dh, auth, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def remove_subscription(self, endpoint: str):
        service = get_notification_service()
        with service._get_db() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
            conn.commit()

    def get_all_subscriptions(self) -> List[Dict[str, str]]:
        service = get_notification_service()
        with service._get_db() as conn:
            cursor = conn.execute("SELECT endpoint, p256dh, auth FROM push_subscriptions")
            return [dict(row) for row in cursor.fetchall()]

    def send_push(
        self, title: str, message: str, url: str = "/", alert_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> int:
        if not self.is_configured:
            logger.warning("Push notifications not configured - VAPID keys missing")
            return 0

        try:
            from pywebpush import webpush, WebPushException
        except ImportError:
            logger.warning("pywebpush not installed - push notifications disabled")
            return 0

        subscriptions = self.get_all_subscriptions()
        sent = 0

        payload = json.dumps(
            {
                "title": title,
                "message": message,
                "url": url,
                "alertId": alert_id,
                "sessionId": session_id,
                "tag": f"alert-{alert_id}" if alert_id else "notification",
            }
        )

        for sub in subscriptions:
            try:
                webpush(
                    subscription_info={
                        "endpoint": sub["endpoint"],
                        "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                    },
                    data=payload,
                    vapid_private_key=self.vapid_private_key,
                    vapid_claims={"sub": f"mailto:{self.vapid_email}"},
                )
                sent += 1
            except Exception as e:
                logger.error(f"Push notification failed: {e}")
                if "410" in str(e) or "404" in str(e):
                    self.remove_subscription(sub["endpoint"])

        return sent


_push_notifier: Optional[PushNotifier] = None


def get_push_notifier() -> PushNotifier:
    global _push_notifier
    if _push_notifier is None:
        _push_notifier = PushNotifier()
    return _push_notifier
