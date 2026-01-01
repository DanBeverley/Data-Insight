import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.notifications.service import get_notification_service, NotificationType, Notification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class CreateNotificationRequest(BaseModel):
    title: str
    message: str
    type: str = "info"
    session_id: Optional[str] = None


class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    type: str
    session_id: Optional[str]
    created_at: str
    read: bool


@router.get("/pending")
async def get_pending_notifications() -> List[dict]:
    service = get_notification_service()
    notifications = service.get_pending()
    return [
        {"id": n.id, "title": n.title, "message": n.message, "type": n.type.value, "created_at": n.created_at}
        for n in notifications
    ]


@router.get("/unread")
async def get_unread_notifications(session_id: Optional[str] = None, limit: int = 50) -> List[dict]:
    service = get_notification_service()
    notifications = service.get_unread(session_id, limit)
    return [
        {
            "id": n.id,
            "title": n.title,
            "message": n.message,
            "type": n.type.value,
            "session_id": n.session_id,
            "created_at": n.created_at,
            "read": n.read,
        }
        for n in notifications
    ]


@router.post("/")
async def create_notification(request: CreateNotificationRequest):
    service = get_notification_service()
    try:
        notification_type = NotificationType(request.type)
    except ValueError:
        notification_type = NotificationType.INFO

    notification = service.create(
        title=request.title, message=request.message, type=notification_type, session_id=request.session_id
    )

    return {
        "id": notification.id,
        "title": notification.title,
        "message": notification.message,
        "type": notification.type.value,
        "created_at": notification.created_at,
    }


@router.post("/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    service = get_notification_service()
    success = service.mark_read(notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"success": True}


@router.post("/read-all")
async def mark_all_read(session_id: Optional[str] = None):
    service = get_notification_service()
    service.mark_all_read(session_id)
    return {"success": True}


class PushSubscribeRequest(BaseModel):
    endpoint: str
    keys: dict


@router.get("/vapid-public-key")
async def get_vapid_public_key():
    from src.notifications.service import get_push_notifier

    push = get_push_notifier()
    if not push.vapid_public_key:
        raise HTTPException(status_code=503, detail="Push notifications not configured")
    return {"publicKey": push.vapid_public_key}


@router.post("/subscribe")
async def subscribe_push(request: PushSubscribeRequest):
    from src.notifications.service import get_push_notifier

    push = get_push_notifier()
    push.save_subscription(
        endpoint=request.endpoint, p256dh=request.keys.get("p256dh", ""), auth=request.keys.get("auth", "")
    )
    return {"success": True}


@router.post("/unsubscribe")
async def unsubscribe_push(request: dict):
    from src.notifications.service import get_push_notifier

    push = get_push_notifier()
    push.remove_subscription(request.get("endpoint", ""))
    return {"success": True}


@router.post("/test-push")
async def test_push_notification():
    from src.notifications.service import get_push_notifier

    push = get_push_notifier()
    if not push.is_configured:
        raise HTTPException(status_code=503, detail="Push notifications not configured")

    sent = push.send_push(title="Test Notification", message="Push notifications are working!", url="/")
    return {"success": True, "sent": sent}
