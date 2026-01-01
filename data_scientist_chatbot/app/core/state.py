from typing import TypedDict, List, Optional, Dict, Any, Literal, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


class Task(TypedDict):
    """Represents a single unit of work in the plan."""

    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    assigned_to: Literal["hands", "brain", "verifier"]
    result: Optional[str]
    artifacts: List[str]
    error: Optional[str]


class GlobalState(TypedDict):
    """The shared blackboard state for the autonomous agent system."""

    # Core LangGraph Messages
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Session Context
    session_id: str
    user_request: str

    # Planning & Orchestration
    plan: List[Task]
    current_task_index: int

    # Data & Artifacts
    dataset_metadata: Dict[str, Any]
    artifacts: List[Dict[str, Any]]  # Global registry of all generated artifacts
    agent_insights: List[Dict[str, Any]]  # Structured insights from Hands agent

    # Error Handling
    errors: List[str]
    retry_count: int

    scratchpad: str
    workflow_stage: Optional[str]
    current_agent: str
    last_agent_sequence: List[str]
    execution_result: Optional[dict]
    next_agent: Optional[str]
    current_task_description: Optional[str]
    report_url: Optional[str]
    report_path: Optional[str]
