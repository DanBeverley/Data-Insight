from typing import TypedDict, List, Optional, Dict, Any, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


class GlobalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    artifacts: List[Dict[str, Any]]
    agent_insights: List[Dict[str, Any]]
    execution_result: Optional[dict]
    retry_count: int
    workflow_stage: Optional[str]
    current_agent: str
    last_agent_sequence: List[str]
    current_task_description: Optional[str]
    report_url: Optional[str]
    report_path: Optional[str]
    search_config: Optional[Dict[str, str]]
