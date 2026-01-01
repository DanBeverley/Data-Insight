from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class AlertCondition(str, Enum):
    LESS_THAN = "lt"
    GREATER_THAN = "gt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN_OR_EQUAL = "gte"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TRIGGERED = "triggered"
    ERROR = "error"


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    session_id: str

    metric_query: str
    metric_name: str
    condition: AlertCondition
    threshold: float

    cron_expression: str = "0 9 * * *"
    timezone: str = "UTC"

    notification_type: str = "email"
    notification_target: str

    status: AlertStatus = AlertStatus.ACTIVE
    last_run: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    last_value: Optional[float] = None
    last_error: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class AlertCreateRequest(BaseModel):
    name: str
    session_id: str
    metric_query: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    cron_expression: str = "0 9 * * *"
    notification_type: str = "email"
    notification_target: str


class AlertUpdateRequest(BaseModel):
    name: Optional[str] = None
    condition: Optional[AlertCondition] = None
    threshold: Optional[float] = None
    cron_expression: Optional[str] = None
    notification_target: Optional[str] = None
    status: Optional[AlertStatus] = None


class AlertCheckResult(BaseModel):
    alert_id: str
    triggered: bool
    current_value: float
    threshold: float
    condition: str
    message: str
    checked_at: datetime = Field(default_factory=datetime.utcnow)
