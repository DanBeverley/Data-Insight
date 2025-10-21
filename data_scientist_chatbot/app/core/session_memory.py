"""Progressive session memory that learns from successes and failures"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class Turn:
    turn_id: int
    timestamp: str
    user_request: str
    code_executed: str
    success: bool
    output: str
    error: str
    artifacts: List[str] = field(default_factory=list)
    self_corrected: bool = False
    attempts: int = 1


@dataclass
class SessionMemory:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    successful_patterns: List[Dict[str, Any]] = field(default_factory=list)
    failed_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def add_turn(self, turn_data: Dict[str, Any]):
        turn = Turn(
            turn_id=len(self.turns) + 1,
            timestamp=datetime.now().isoformat(),
            user_request=turn_data.get("user_request", ""),
            code_executed=turn_data.get("code", ""),
            success=turn_data.get("success", False),
            output=turn_data.get("output", ""),
            error=turn_data.get("error", ""),
            artifacts=turn_data.get("artifacts", []),
            self_corrected=turn_data.get("self_corrected", False),
            attempts=turn_data.get("attempts", 1),
        )

        self.turns.append(turn)

        if turn.success:
            self.successful_patterns.append(
                {
                    "request_type": self._categorize_request(turn.user_request),
                    "code": turn.code_executed,
                    "timestamp": turn.timestamp,
                }
            )
        else:
            self.failed_patterns.append(
                {
                    "request_type": self._categorize_request(turn.user_request),
                    "code": turn.code_executed,
                    "error": turn.error[:200],
                    "timestamp": turn.timestamp,
                }
            )

    def get_learning_context(self) -> str:
        if not self.turns:
            return ""

        recent_turns = self.turns[-5:]

        context_parts = []

        if len(self.turns) > 0:
            total_success = sum(1 for t in self.turns if t.success)
            self_corrected = sum(1 for t in self.turns if t.self_corrected)

            context_parts.append(
                f"SESSION LEARNING ({len(self.turns)} turns, {total_success} successful, {self_corrected} self-corrected)"
            )

        recent_successes = [t for t in recent_turns if t.success]
        if recent_successes:
            context_parts.append("\nRECENT SUCCESSFUL APPROACHES:")
            for turn in recent_successes[-3:]:
                context_parts.append(f"  ✓ {turn.user_request[:60]}")
                if turn.self_corrected:
                    context_parts.append(f"    (auto-fixed in {turn.attempts} attempts)")

        recent_failures = [t for t in recent_turns if not t.success]
        if recent_failures:
            context_parts.append("\nRECENT FAILURES TO AVOID:")
            for turn in recent_failures[-2:]:
                context_parts.append(f"  ✗ {turn.user_request[:60]}")
                context_parts.append(f"    Error: {turn.error[:100]}")

        total_artifacts = sum(len(t.artifacts) for t in self.turns)
        if total_artifacts > 0:
            context_parts.append(f"\nCREATED {total_artifacts} artifacts this session")

        return "\n".join(context_parts)

    def _categorize_request(self, request: str) -> str:
        request_lower = request.lower()

        if any(kw in request_lower for kw in ["train", "model", "fit", "predict"]):
            return "modeling"
        elif any(kw in request_lower for kw in ["plot", "visualiz", "chart", "graph"]):
            return "visualization"
        elif any(kw in request_lower for kw in ["correlat", "statistic", "distribut"]):
            return "analysis"
        else:
            return "general"


class SessionMemoryStore:
    def __init__(self):
        import builtins

        if not hasattr(builtins, "_session_memories"):
            builtins._session_memories = {}
        self.store = builtins._session_memories

    def get_memory(self, session_id: str) -> SessionMemory:
        if session_id not in self.store:
            self.store[session_id] = SessionMemory(session_id=session_id)
        return self.store[session_id]

    def save_turn(self, session_id: str, turn_data: Dict[str, Any]):
        memory = self.get_memory(session_id)
        memory.add_turn(turn_data)


memory_store = SessionMemoryStore()


def get_session_memory(session_id: str) -> SessionMemory:
    return memory_store.get_memory(session_id)


def record_execution(session_id: str, turn_data: Dict[str, Any]):
    memory_store.save_turn(session_id, turn_data)
