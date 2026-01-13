"""Quorvix Scheduler API Router."""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])
logger = logging.getLogger(__name__)


class CreateJobRequest(BaseModel):
    job_id: str
    job_type: str  # "interval" or "cron"
    job_name: Optional[str] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = 0
    day_of_week: Optional[str] = None
    action: str  # "check_alerts", "daily_digest", "cleanup_sessions", etc.


@router.get("/jobs")
async def list_jobs():
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    return {"jobs": scheduler.get_jobs()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs")
async def create_job(request: CreateJobRequest):
    from src.scheduler.service import get_scheduler
    from src.scheduler import jobs

    scheduler = get_scheduler()

    func = getattr(jobs, request.action, None)
    if not func:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    if request.job_type == "interval":
        if not (request.minutes or request.hours):
            raise HTTPException(status_code=400, detail="Interval job requires minutes or hours")
        success = scheduler.add_interval_job(
            job_id=request.job_id, func=func, minutes=request.minutes, hours=request.hours, job_name=request.job_name
        )
    elif request.job_type == "cron":
        success = scheduler.add_cron_job(
            job_id=request.job_id,
            func=func,
            hour=request.hour,
            minute=request.minute,
            day_of_week=request.day_of_week,
            job_name=request.job_name,
        )
    else:
        raise HTTPException(status_code=400, detail="job_type must be 'interval' or 'cron'")

    if not success:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {"status": "created", "job_id": request.job_id}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    success = scheduler.remove_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    success = scheduler.pause_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "paused", "job_id": job_id}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    success = scheduler.resume_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "resumed", "job_id": job_id}


@router.post("/jobs/{job_id}/run")
async def run_job_now(job_id: str):
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    success = scheduler.run_job_now(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or no registered function")
    return {"status": "executed", "job_id": job_id}


@router.get("/history")
async def get_job_history(limit: int = 100, job_id: Optional[str] = None):
    from src.scheduler.service import get_scheduler

    scheduler = get_scheduler()
    return {"history": scheduler.get_history(limit=limit, job_id=job_id)}


@router.get("/status")
async def get_scheduler_status():
    from src.scheduler.service import get_scheduler

    try:
        scheduler = get_scheduler()
        jobs = scheduler.get_jobs()
        return {
            "status": "running" if scheduler._started else "stopped",
            "job_count": len(jobs),
            "jobs": [j["name"] for j in jobs],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
