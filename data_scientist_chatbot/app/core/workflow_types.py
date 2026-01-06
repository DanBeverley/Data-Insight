from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

try:
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    from src.api_utils.artifact_tracker import ArtifactCategory
except ImportError:
    from enum import Enum

    class ArtifactCategory(Enum):
        VISUALIZATION = "visualization"
        DATASET = "dataset"
        MODEL = "model"
        REPORT = "report"
        OTHER = "other"


class WorkflowStage(Enum):
    ROUTING = "routing"
    HANDS_EXECUTION = "hands"
    BRAIN_INTERPRETATION = "brain"
    REPORTING_ANALYSIS = "reporting_analysis"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"


class Artifact(BaseModel):
    filename: str
    category: str
    local_path: Optional[str] = None
    artifact_id: Optional[str] = None
    cloud_url: Optional[str] = None
    presigned_url: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("presigned_url", "cloud_url")
    @classmethod
    def validate_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class ExecutionResult(BaseModel):
    success: bool = Field(..., description="Whether execution succeeded")
    code: str = Field(default="", description="Generated code")
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    artifacts: List[Artifact] = Field(default_factory=list)
    execution_time: float = Field(default=0.0)
    error_details: Optional[str] = None

    class Config:
        use_enum_values = True


class DataSummary(BaseModel):
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    memory_mb: float
    numeric_cols: List[str] = Field(default_factory=list)
    categorical_cols: List[str] = Field(default_factory=list)


class InterpretationContext(BaseModel):
    user_request: str
    execution_result: Optional[ExecutionResult] = None
    artifacts: List[Artifact] = Field(default_factory=list)
    data_summary: Optional[DataSummary] = None
    previous_context: Optional[str] = None
