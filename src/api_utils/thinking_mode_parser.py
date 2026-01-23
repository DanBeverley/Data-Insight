"""
DeepSeek V3.1 thinking mode parser for status streaming

Extracts chain-of-thought reasoning from <think> tags and converts
them into user-friendly status updates for real-time streaming.
"""

import re
import logging
from typing import Generator, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThinkingUpdate:
    """Represents a thinking/reasoning update"""

    type: str  # 'thinking_start', 'thinking_content', 'thinking_end', 'response'
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ThinkingModeParser:
    """Parses DeepSeek V3.1 thinking mode outputs for status streaming"""

    THINK_START_PATTERN = re.compile(r"<think>")
    THINK_END_PATTERN = re.compile(r"</think>")

    REASONING_INDICATORS = [
        "analyzing",
        "considering",
        "evaluating",
        "processing",
        "examining",
        "determining",
        "calculating",
        "reasoning",
        "thinking about",
        "first",
        "second",
        "next",
        "then",
        "therefore",
        "because",
        "since",
        "this means",
        "let me",
        "i need to",
        "i should",
    ]

    def __init__(self):
        self.in_think_block = False
        self.thinking_buffer = []
        self.total_thinking_chars = 0

    def parse_streaming_chunk(self, chunk: str) -> Generator[ThinkingUpdate, None, None]:
        """
        Parse a streaming chunk and extract thinking updates

        Args:
            chunk: Text chunk from streaming response

        Yields:
            ThinkingUpdate objects representing reasoning steps
        """
        if self.THINK_START_PATTERN.search(chunk):
            self.in_think_block = True
            self.thinking_buffer = []
            yield ThinkingUpdate(
                type="thinking_start", content="Starting analysis...", metadata={"event": "thinking_mode_activated"}
            )

            chunk = self.THINK_START_PATTERN.sub("", chunk)

        if self.in_think_block:
            if self.THINK_END_PATTERN.search(chunk):
                self.in_think_block = False

                remaining_content = self.THINK_END_PATTERN.split(chunk)[0]
                if remaining_content.strip():
                    self.thinking_buffer.append(remaining_content)

                full_thinking = "".join(self.thinking_buffer)

                if full_thinking.strip():
                    summary = self._summarize_thinking(full_thinking)
                    yield ThinkingUpdate(
                        type="thinking_end",
                        content=summary,
                        metadata={
                            "event": "thinking_mode_completed",
                            "thinking_length": len(full_thinking),
                            "steps_identified": self._count_reasoning_steps(full_thinking),
                        },
                    )

                post_think_content = self.THINK_END_PATTERN.split(chunk)
                if len(post_think_content) > 1 and post_think_content[1].strip():
                    yield ThinkingUpdate(
                        type="response", content=post_think_content[1], metadata={"event": "response_start"}
                    )

            else:
                self.thinking_buffer.append(chunk)

                reasoning_step = self._extract_reasoning_step(chunk)
                if reasoning_step:
                    yield ThinkingUpdate(
                        type="thinking_content", content=reasoning_step, metadata={"event": "reasoning_step"}
                    )

        else:
            if chunk.strip():
                yield ThinkingUpdate(type="response", content=chunk, metadata={"event": "response_content"})

    def _extract_reasoning_step(self, text: str) -> Optional[str]:
        """Extract a human-readable reasoning step from thinking text"""
        text_lower = text.lower()

        for indicator in self.REASONING_INDICATORS:
            if indicator in text_lower:
                sentences = re.split(r"[.!?]\s+", text)

                for sentence in sentences:
                    if indicator in sentence.lower() and len(sentence.strip()) > 20:
                        cleaned = sentence.strip()

                        if cleaned and not cleaned.startswith("<"):
                            return self._format_reasoning_step(cleaned)

        return None

    def _format_reasoning_step(self, step: str) -> str:
        """Format reasoning step for user display"""
        step = step.strip()

        if not step[0].isupper():
            step = step.capitalize()

        if not step.endswith((".", "!", "?")):
            step += "..."

        if len(step) > 150:
            step = step[:147] + "..."

        return step

    def _summarize_thinking(self, thinking_text: str) -> str:
        """Summarize the complete thinking process"""
        sentences = re.split(r"[.!?]\s+", thinking_text)

        key_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in self.REASONING_INDICATORS[:5]:
                if indicator in sentence_lower and len(sentence) > 30:
                    key_sentences.append(sentence.strip())
                    break

            if len(key_sentences) >= 3:
                break

        if key_sentences:
            summary = " → ".join(key_sentences[:3])
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return f"Analysis complete: {summary}"
        else:
            step_count = self._count_reasoning_steps(thinking_text)
            return f"Completed reasoning process ({step_count} steps analyzed)"

    def _count_reasoning_steps(self, text: str) -> int:
        """Count approximate number of reasoning steps"""
        indicators_found = sum(1 for indicator in self.REASONING_INDICATORS if indicator in text.lower())
        return max(indicators_found, 1)

    def reset(self):
        """Reset parser state for new response"""
        self.in_think_block = False
        self.thinking_buffer = []
        self.total_thinking_chars = 0


def create_thinking_status_message(update: ThinkingUpdate) -> Dict[str, Any]:
    """Convert ThinkingUpdate to SSE-compatible status message"""
    if update.type == "thinking_start":
        return {"type": "status", "message": "🧠 Analyzing request...", "category": "thinking"}

    elif update.type == "thinking_content":
        return {"type": "status", "message": f"💭 {update.content}", "category": "reasoning"}

    elif update.type == "thinking_end":
        return {"type": "status", "message": f"✓ {update.content}", "category": "thinking_complete"}

    elif update.type == "response":
        return {"type": "content", "message": update.content, "category": "response"}

    return {"type": "status", "message": update.content}


async def stream_with_thinking_mode(
    streaming_generator, thinking_mode_enabled: bool = False
) -> Generator[Dict[str, Any], None, None]:
    """
    Wrapper that adds thinking mode parsing to any streaming generator

    Args:
        streaming_generator: Async generator yielding response chunks
        thinking_mode_enabled: Whether to parse <think> tags

    Yields:
        Dict messages compatible with SSE streaming
    """
    if not thinking_mode_enabled:
        async for chunk in streaming_generator:
            yield chunk
        return

    parser = ThinkingModeParser()

    try:
        async for chunk in streaming_generator:
            if isinstance(chunk, dict):
                yield chunk
                continue

            if isinstance(chunk, str):
                for update in parser.parse_streaming_chunk(chunk):
                    status_msg = create_thinking_status_message(update)
                    yield status_msg

    finally:
        parser.reset()
