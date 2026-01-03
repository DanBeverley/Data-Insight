"""Core constants and enums for the agent workflow"""

from enum import Enum
from typing import Set


class NodeName(str, Enum):
    BRAIN = "brain"
    HANDS = "hands"
    VERIFIER = "verifier"
    PARSER = "parser"
    ACTION = "action"
    ANALYST = "analyst"
    ARCHITECT = "architect"
    PRESENTER = "presenter"
    ROUTER = "router"


WORKFLOW_NODES: Set[str] = {n.value for n in NodeName}

EXECUTION_NODES: Set[str] = {
    NodeName.BRAIN.value,
    NodeName.HANDS.value,
    NodeName.PARSER.value,
    NodeName.ACTION.value,
    NodeName.ANALYST.value,
    NodeName.ARCHITECT.value,
    NodeName.PRESENTER.value,
}

SAFE_CHECKPOINT_NODES: Set[str] = {
    NodeName.ARCHITECT.value,
    NodeName.PRESENTER.value,
}

TOOL_NODES: Set[str] = {
    NodeName.PARSER.value,
    NodeName.ACTION.value,
}


class WorkflowStage(str, Enum):
    DELEGATING = "delegating"
    REPORTING = "reporting"
    EXECUTING = "executing"
    COMPLETE = "complete"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
