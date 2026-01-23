import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.scheduler.models import Alert, AlertCreateRequest, AlertUpdateRequest, AlertCheckResult
from src.scheduler.service import get_alert_scheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/", response_model=List[Alert])
async def get_all_alerts():
    scheduler = get_alert_scheduler()
    return scheduler.get_all_alerts()


@router.get("/session/{session_id}", response_model=List[Alert])
async def get_session_alerts(session_id: str):
    scheduler = get_alert_scheduler()
    return scheduler.get_alerts_by_session(session_id)


@router.get("/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str):
    scheduler = get_alert_scheduler()
    alert = scheduler.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("/", response_model=Alert)
async def create_alert(request: AlertCreateRequest):
    scheduler = get_alert_scheduler()

    alert = Alert(
        name=request.name,
        session_id=request.session_id,
        metric_query=request.metric_query,
        metric_name=request.metric_name,
        condition=request.condition,
        threshold=request.threshold,
        cron_expression=request.cron_expression,
        notification_type=request.notification_type,
        notification_target=request.notification_target,
    )

    created = scheduler.create_alert(alert)
    logger.info(f"Created alert: {created.name} for session {created.session_id}")
    return created


@router.put("/{alert_id}", response_model=Alert)
async def update_alert(alert_id: str, request: AlertUpdateRequest):
    scheduler = get_alert_scheduler()

    updates = request.model_dump(exclude_unset=True)
    updated = scheduler.update_alert(alert_id, updates)

    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")

    return updated


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str):
    scheduler = get_alert_scheduler()
    success = scheduler.delete_alert(alert_id)

    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert deleted successfully"}


@router.post("/{alert_id}/check")
async def check_alert_now(alert_id: str):
    scheduler = get_alert_scheduler()

    alert = scheduler.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    result = await scheduler.check_alert_now(alert_id)
    return result


@router.get("/{alert_id}/history")
async def get_alert_history(alert_id: str, limit: int = 50):
    scheduler = get_alert_scheduler()

    alert = scheduler.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    history = scheduler.get_alert_history(alert_id, limit)
    return {"alert_id": alert_id, "history": history}


@router.post("/{alert_id}/pause")
async def pause_alert(alert_id: str):
    scheduler = get_alert_scheduler()
    updated = scheduler.update_alert(alert_id, {"status": "paused"})

    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert paused", "alert": updated}


@router.post("/{alert_id}/resume")
async def resume_alert(alert_id: str):
    scheduler = get_alert_scheduler()
    updated = scheduler.update_alert(alert_id, {"status": "active"})

    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert resumed", "alert": updated}
