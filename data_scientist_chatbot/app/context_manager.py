"""
Enhanced Context Management System for Data Scientist Agent
Provides conversation memory, context optimization, and cross-session learning
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass
import re


@dataclass
class ConversationContext:
    """Represents conversation context with metadata"""

    session_id: str
    summary: str
    key_topics: List[str]
    data_insights: List[str]
    code_patterns: List[str]
    user_preferences: Dict[str, Any]
    timestamp: datetime
    importance_score: float


class ContextManager:
    """
    Advanced context management for persistent conversation memory
    and cross-session learning capabilities
    """

    def __init__(self, db_path: str = "data_scientist_chatbot/memory/context.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for context storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Conversation contexts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT,
                key_topics TEXT, -- JSON array
                data_insights TEXT, -- JSON array
                code_patterns TEXT, -- JSON array
                user_preferences TEXT, -- JSON object
                timestamp DATETIME,
                importance_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Cross-session learning patterns
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT, -- JSON object
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                last_used DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Performance metrics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_conversation_context(self, context: ConversationContext) -> None:
        """Save conversation context to persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversation_contexts 
            (session_id, summary, key_topics, data_insights, code_patterns, 
             user_preferences, timestamp, importance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                context.session_id,
                context.summary,
                json.dumps(context.key_topics),
                json.dumps(context.data_insights),
                json.dumps(context.code_patterns),
                json.dumps(context.user_preferences),
                context.timestamp,
                context.importance_score,
            ),
        )

        conn.commit()
        conn.close()

    def get_session_context(self, session_id: str, limit: int = 5) -> List[ConversationContext]:
        """Retrieve recent conversation contexts for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT session_id, summary, key_topics, data_insights, code_patterns,
                   user_preferences, timestamp, importance_score
            FROM conversation_contexts
            WHERE session_id = ?
            ORDER BY importance_score DESC, timestamp DESC
            LIMIT ?
        """,
            (session_id, limit),
        )

        contexts = []
        for row in cursor.fetchall():
            contexts.append(
                ConversationContext(
                    session_id=row[0],
                    summary=row[1],
                    key_topics=json.loads(row[2]) if row[2] else [],
                    data_insights=json.loads(row[3]) if row[3] else [],
                    code_patterns=json.loads(row[4]) if row[4] else [],
                    user_preferences=json.loads(row[5]) if row[5] else {},
                    timestamp=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
                    importance_score=row[7],
                )
            )

        conn.close()
        return contexts

    def extract_conversation_insights(self, messages: List) -> ConversationContext:
        """Extract insights from conversation messages using pattern analysis"""
        # Extract key information from messages
        user_messages = [msg for msg in messages if hasattr(msg, "type") and msg.type == "human"]
        ai_messages = [msg for msg in messages if hasattr(msg, "type") and msg.type == "ai"]

        # Analyze conversation content
        key_topics = self._extract_topics(user_messages + ai_messages)
        data_insights = self._extract_data_insights(ai_messages)
        code_patterns = self._extract_code_patterns(ai_messages)
        user_preferences = self._analyze_user_preferences(user_messages)

        # Generate summary
        summary = self._generate_summary(user_messages, ai_messages)

        # Calculate importance score
        importance_score = self._calculate_importance_score(len(user_messages), len(data_insights), len(code_patterns))

        return ConversationContext(
            session_id="",  # Will be set by caller
            summary=summary,
            key_topics=key_topics,
            data_insights=data_insights,
            code_patterns=code_patterns,
            user_preferences=user_preferences,
            timestamp=datetime.now(),
            importance_score=importance_score,
        )

    def _extract_topics(self, messages: List) -> List[str]:
        """Extract key topics from messages using keyword analysis"""
        topics = set()
        data_science_keywords = {
            "visualization",
            "plot",
            "chart",
            "graph",
            "correlation",
            "regression",
            "clustering",
            "classification",
            "prediction",
            "model",
            "feature",
            "dataset",
            "data",
            "analysis",
            "statistics",
            "machine learning",
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "sklearn",
        }

        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content.lower()
                for keyword in data_science_keywords:
                    if keyword in content:
                        topics.add(keyword)

        return list(topics)[:10]  # Limit to top 10 topics

    def _extract_data_insights(self, ai_messages: List) -> List[str]:
        """Extract data insights from AI responses"""
        insights = []
        insight_patterns = [
            r"correlation.*?(\d+\.?\d*)",
            r"distribution.*?(normal|skewed|bimodal)",
            r"outliers.*?(\d+)",
            r"missing values.*?(\d+\.?\d*%?)",
            r"accuracy.*?(\d+\.?\d*%?)",
        ]

        for msg in ai_messages:
            if hasattr(msg, "content"):
                content = msg.content
                for pattern in insight_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    insights.extend([f"Found {match}" for match in matches])

        return insights[:5]  # Limit to top 5 insights

    def _extract_code_patterns(self, ai_messages: List) -> List[str]:
        """Extract successful code patterns from AI messages"""
        patterns = []
        code_blocks = []

        for msg in ai_messages:
            if hasattr(msg, "content"):
                # Extract code blocks
                content = msg.content
                code_matches = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
                code_blocks.extend(code_matches)

        # Analyze code patterns
        for code in code_blocks:
            if "plt.figure" in code or "sns." in code:
                patterns.append("visualization")
            if "df.corr()" in code:
                patterns.append("correlation_analysis")
            if "train_test_split" in code:
                patterns.append("ml_modeling")
            if "df.describe()" in code or "df.info()" in code:
                patterns.append("exploratory_data_analysis")

        return list(set(patterns))  # Remove duplicates

    def _analyze_user_preferences(self, user_messages: List) -> Dict[str, Any]:
        """Analyze user preferences from their messages"""
        preferences = {
            "visualization_preference": "matplotlib",  # Default
            "analysis_depth": "standard",
            "explanation_level": "detailed",
        }

        for msg in user_messages:
            if hasattr(msg, "content"):
                content = msg.content.lower()
                if "plotly" in content or "interactive" in content:
                    preferences["visualization_preference"] = "plotly"
                elif "seaborn" in content:
                    preferences["visualization_preference"] = "seaborn"

                if "quick" in content or "brief" in content:
                    preferences["explanation_level"] = "brief"
                elif "detailed" in content or "explain" in content:
                    preferences["explanation_level"] = "detailed"

        return preferences

    def _generate_summary(self, user_messages: List, ai_messages: List) -> str:
        """Generate conversation summary"""
        if not user_messages:
            return "No user interaction recorded"

        # Extract main user request from first message
        first_request = ""
        if user_messages and hasattr(user_messages[0], "content"):
            first_request = user_messages[0].content[:100]

        # Count interactions
        interaction_count = len(user_messages)

        return f"Session with {interaction_count} interactions. Main request: {first_request}..."

    def _calculate_importance_score(self, user_messages_count: int, insights_count: int, patterns_count: int) -> float:
        """Calculate importance score for conversation prioritization"""
        base_score = min(user_messages_count * 0.1, 1.0)  # Max 1.0 for interactions
        insight_bonus = min(insights_count * 0.2, 1.0)  # Max 1.0 for insights
        pattern_bonus = min(patterns_count * 0.15, 0.5)  # Max 0.5 for patterns

        return min(base_score + insight_bonus + pattern_bonus, 3.0)

    def get_cross_session_patterns(self, pattern_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve successful patterns from across sessions for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if pattern_type:
            cursor.execute(
                """
                SELECT pattern_type, pattern_data, success_count, confidence_score
                FROM learning_patterns
                WHERE pattern_type = ? AND confidence_score > 0.5
                ORDER BY confidence_score DESC, success_count DESC
                LIMIT ?
            """,
                (pattern_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT pattern_type, pattern_data, success_count, confidence_score
                FROM learning_patterns
                WHERE confidence_score > 0.5
                ORDER BY confidence_score DESC, success_count DESC
                LIMIT ?
            """,
                (limit,),
            )

        patterns = []
        for row in cursor.fetchall():
            patterns.append({"type": row[0], "data": json.loads(row[1]), "success_count": row[2], "confidence": row[3]})

        conn.close()
        return patterns

    def record_pattern_usage(self, pattern_type: str, pattern_data: Dict[str, Any], success: bool) -> None:
        """Record usage of a pattern for learning"""
        pattern_hash = hashlib.md5(json.dumps(pattern_data, sort_keys=True).encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if pattern exists
        cursor.execute(
            """
            SELECT id, success_count, failure_count FROM learning_patterns
            WHERE pattern_type = ? AND pattern_data = ?
        """,
            (pattern_type, json.dumps(pattern_data)),
        )

        existing = cursor.fetchone()

        if existing:
            # Update existing pattern
            new_success = existing[1] + (1 if success else 0)
            new_failure = existing[2] + (0 if success else 1)
            confidence = new_success / (new_success + new_failure) if (new_success + new_failure) > 0 else 0

            cursor.execute(
                """
                UPDATE learning_patterns 
                SET success_count = ?, failure_count = ?, confidence_score = ?, last_used = ?
                WHERE id = ?
            """,
                (new_success, new_failure, confidence, datetime.now(), existing[0]),
            )
        else:
            # Create new pattern
            success_count = 1 if success else 0
            failure_count = 0 if success else 1
            confidence = 1.0 if success else 0.0

            cursor.execute(
                """
                INSERT INTO learning_patterns 
                (pattern_type, pattern_data, success_count, failure_count, confidence_score, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (pattern_type, json.dumps(pattern_data), success_count, failure_count, confidence, datetime.now()),
            )

        conn.commit()
        conn.close()

    def cleanup_old_contexts(self, days_old: int = 30) -> int:
        """Clean up old conversation contexts to manage storage"""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM conversation_contexts 
            WHERE timestamp < ? AND importance_score < 1.0
        """,
            (cutoff_date,),
        )

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def get_context_summary_for_session(self, session_id: str) -> str:
        """Generate a context summary for the agent to use"""
        contexts = self.get_session_context(session_id, limit=3)

        if not contexts:
            return "No previous conversation context available."

        summary_parts = []
        for ctx in contexts:
            summary_parts.append(f"Previous session summary: {ctx.summary}")
            if ctx.key_topics:
                summary_parts.append(f"Key topics discussed: {', '.join(ctx.key_topics[:5])}")
            if ctx.user_preferences:
                prefs = [f"{k}: {v}" for k, v in ctx.user_preferences.items()]
                summary_parts.append(f"User preferences: {', '.join(prefs)}")

        # Add successful patterns
        patterns = self.get_cross_session_patterns(limit=3)
        if patterns:
            pattern_summaries = [f"{p['type']} (confidence: {p['confidence']:.1f})" for p in patterns]
            summary_parts.append(f"Proven successful patterns: {', '.join(pattern_summaries)}")

        return " | ".join(summary_parts)
