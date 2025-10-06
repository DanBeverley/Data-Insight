from pydantic import BaseModel
from typing import Optional, List


class DataIngestionRequest(BaseModel):
    url: str
    data_type: str = "csv"
    enable_profiling: bool = True
    session_id: Optional[str] = None


class ProfilingRequest(BaseModel):
    deep_analysis: bool = True
    include_relationships: bool = True
    include_domain_detection: bool = True


class FeatureRecommendationRequest(BaseModel):
    target_column: Optional[str] = None
    max_recommendations: int = 10
    priority_filter: Optional[str] = None


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    response: str
    plots: List[str] = []


class AgentChatRequest(BaseModel):
    message: str
    session_id: str
    async_execution: bool = False
