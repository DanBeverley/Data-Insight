"""Research Brain - Deep research agent with time-aware exploration."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

from data_scientist_chatbot.app.core.logger import logger


@dataclass
class ResearchFinding:
    """A single research finding with source."""

    content: str
    source_url: str
    source_title: str
    query: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchFindings:
    """Accumulated research findings from deep exploration."""

    original_query: str
    findings: List[ResearchFinding] = field(default_factory=list)
    explored_subtopics: List[str] = field(default_factory=list)
    pending_subtopics: List[str] = field(default_factory=list)
    visited_urls: Set[str] = field(default_factory=set)
    start_time: float = field(default_factory=time.time)
    time_budget_seconds: int = 600
    iterations: int = 0

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def time_remaining_seconds(self) -> float:
        return max(0, self.time_budget_seconds - self.elapsed_seconds())

    def is_time_exhausted(self) -> bool:
        return self.elapsed_seconds() >= self.time_budget_seconds

    def add_finding(self, finding: ResearchFinding) -> None:
        self.findings.append(finding)
        if finding.source_url:
            self.visited_urls.add(finding.source_url)

    def to_summary(self) -> str:
        """Generate a summary for handoff to Main Brain."""
        if not self.findings:
            return f"No findings for query: {self.original_query}"

        summary_parts = [
            f"## Research Summary: {self.original_query}\n",
            f"**Duration:** {int(self.elapsed_seconds())} seconds",
            f"**Iterations:** {self.iterations}",
            f"**Subtopics explored:** {len(self.explored_subtopics)}",
            f"**Sources consulted:** {len(self.visited_urls)}",
            f"**Findings:** {len(self.findings)}\n",
            "---\n",
        ]

        for i, finding in enumerate(self.findings, 1):
            summary_parts.append(
                f"### Finding {i}: {finding.query}\n"
                f"{finding.content}\n"
                f"*Source: [{finding.source_title}]({finding.source_url})*\n"
            )

        return "\n".join(summary_parts)


class ResearchBrain:
    """Time-aware research agent that explores topics deeply."""

    CHECKPOINT_INTERVAL = 3

    def __init__(self, session_id: str, time_budget_minutes: int = 10, search_config: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.time_budget_seconds = time_budget_minutes * 60
        self.search_config = search_config or {"provider": "duckduckgo"}
        self._cancelled = False
        self._last_checkpoint_iteration = 0

    def cancel(self) -> None:
        """Signal the research to stop gracefully."""
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        """Check if research has been cancelled via database flag."""
        from src.api_utils.cancellation import is_task_cancelled

        return self._cancelled or is_task_cancelled(self.session_id)

    async def _generate_subtopics(self, query: str, existing_findings: List[ResearchFinding]) -> List[str]:
        """Use LLM to decompose query into sub-questions for exploration."""
        from data_scientist_chatbot.app.core.agent_factory import create_brain_agent

        context = ""
        if existing_findings:
            context = "\n".join([f"- {f.query}: {f.content[:200]}..." for f in existing_findings[-5:]])

        prompt = f"""You are a research strategist. Given this research query and any existing findings, 
generate 3-5 focused sub-questions that would help explore this topic more deeply.

Original Query: {query}

Existing Findings:
{context if context else "None yet"}

Output ONLY a JSON array of sub-questions, nothing else:
["question 1", "question 2", "question 3"]"""

        llm = create_brain_agent(mode="chat")
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        try:
            import json

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse subtopics: {e}")
            return [f"What are the key aspects of {query}?"]

    async def _search_topic(self, query: str) -> List[ResearchFinding]:
        """Execute web search and extract findings."""
        from data_scientist_chatbot.app.tools.web_search import web_search

        try:
            results_text = await web_search(query, self.search_config, num_results=5)

            # Parse results into findings
            findings = []
            if "## Web Search Results" in results_text:
                lines = results_text.split("\n")
                current_finding = None

                for line in lines:
                    if line.startswith(("1.", "2.", "3.", "4.", "5.")):
                        if current_finding:
                            findings.append(current_finding)
                        title_match = line.split("**")
                        title = title_match[1] if len(title_match) > 1 else "Unknown"
                        current_finding = ResearchFinding(content="", source_url="", source_title=title, query=query)
                    elif current_finding and line.strip().startswith("`"):
                        domain = line.strip().strip("`")
                        current_finding.source_url = f"https://{domain}"
                    elif current_finding and line.strip() and not line.startswith("**"):
                        current_finding.content += line.strip() + " "

                if current_finding:
                    findings.append(current_finding)

            return findings[:3]

        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []

    async def _emit_progress(self, findings: ResearchFindings, current_topic: str, phase: str = "exploring") -> None:
        """Emit SSE progress event via session store."""
        import builtins

        if not hasattr(builtins, "_session_store"):
            builtins._session_store = {}

        if self.session_id not in builtins._session_store:
            builtins._session_store[self.session_id] = {}

        builtins._session_store[self.session_id]["research_progress"] = {
            "phase": phase,
            "iteration": findings.iterations,
            "time_remaining": int(findings.time_remaining_seconds()),
            "current_topic": current_topic[:100],
            "findings_count": len(findings.findings),
            "subtopics_explored": len(findings.explored_subtopics),
        }

    def _should_checkpoint(self, iteration: int) -> bool:
        """Check if we should save a checkpoint at this iteration."""
        return (iteration - self._last_checkpoint_iteration) >= self.CHECKPOINT_INTERVAL

    def _checkpoint_to_knowledge(self, findings: ResearchFindings, is_final: bool = False) -> None:
        """Save current findings to knowledge store as checkpoint."""
        if not findings.findings:
            return

        try:
            from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

            store = KnowledgeStore(self.session_id)

            content_parts = [f"# Research: {findings.original_query}\n"]
            content_parts.append(f"Status: {'Complete' if is_final else 'In Progress'}")
            content_parts.append(f"Iterations: {findings.iterations}")
            content_parts.append(f"Findings: {len(findings.findings)}\n")

            for i, finding in enumerate(findings.findings, 1):
                content_parts.append(f"## {i}. {finding.query}")
                content_parts.append(finding.content)
                content_parts.append(f"Source: {finding.source_url}\n")

            source_name = f"Research: {findings.original_query[:50]}"
            if not is_final:
                source_name += f" (checkpoint @{findings.iterations})"

            store.add_document(content="\n".join(content_parts), source="research", source_name=source_name)

            self._last_checkpoint_iteration = findings.iterations
            logger.info(f"[RESEARCH] Checkpointed {len(findings.findings)} findings to knowledge store")
        except Exception as e:
            logger.warning(f"[RESEARCH] Checkpoint failed: {e}")

    @traceable(name="research_brain_execution", tags=["research", "deep"])
    async def research(self, query: str) -> ResearchFindings:
        """
        Execute deep research with time-aware exploration.

        Continues exploring subtopics until time budget is exhausted,
        then returns accumulated findings for Main Brain to synthesize.
        """
        findings = ResearchFindings(original_query=query, time_budget_seconds=self.time_budget_seconds)

        logger.info(f"[RESEARCH] Starting deep research: '{query}' ({self.time_budget_seconds}s budget)")

        # Initial subtopic generation
        await self._emit_progress(findings, query, "decomposing")
        subtopics = await self._generate_subtopics(query, [])
        findings.pending_subtopics = subtopics

        logger.info(f"[RESEARCH] Generated {len(subtopics)} initial subtopics")

        # Main exploration loop
        while not findings.is_time_exhausted() and not self._is_cancelled():
            if not findings.pending_subtopics:
                # Generate more subtopics based on current findings
                if findings.findings:
                    new_subtopics = await self._generate_subtopics(query, findings.findings)
                    findings.pending_subtopics = [s for s in new_subtopics if s not in findings.explored_subtopics]

                if not findings.pending_subtopics:
                    logger.info("[RESEARCH] No more subtopics to explore")
                    break

            current_topic = findings.pending_subtopics.pop(0)
            findings.explored_subtopics.append(current_topic)
            findings.iterations += 1

            await self._emit_progress(findings, current_topic, "searching")
            logger.info(
                f"[RESEARCH] Iteration {findings.iterations}: '{current_topic}' "
                f"({int(findings.time_remaining_seconds())}s remaining)"
            )

            # Search and extract findings
            topic_findings = await self._search_topic(current_topic)
            for f in topic_findings:
                if f.source_url not in findings.visited_urls:
                    findings.add_finding(f)

            if self._should_checkpoint(findings.iterations):
                self._checkpoint_to_knowledge(findings, is_final=False)

            await asyncio.sleep(0.1)

        await self._emit_progress(findings, "", "complete")

        self._checkpoint_to_knowledge(findings, is_final=True)

        reason = (
            "cancelled" if self._is_cancelled() else ("time_exhausted" if findings.is_time_exhausted() else "complete")
        )
        logger.info(
            f"[RESEARCH] Finished ({reason}): {len(findings.findings)} findings, "
            f"{findings.iterations} iterations, {int(findings.elapsed_seconds())}s elapsed"
        )

        return findings
