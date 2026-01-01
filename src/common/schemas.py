"""Shared schemas for agent-driven insights and data structures"""

from typing import List, Dict, Literal, Optional
from dataclasses import dataclass

InsightType = Literal["summary", "quality", "health", "pattern", "anomaly", "recommendation", "info", "warning"]


@dataclass
class DataInsight:
    label: str
    value: str
    type: InsightType
    source: str = "Agent-Analysis"
    details: Optional[str] = None

    def to_dict(self) -> Dict:
        result = {"label": self.label, "value": self.value, "type": self.type, "source": self.source}
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class ReportSummary:
    title: str
    key_findings: List[str]
    visualizations_summary: str
    recommendations: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        result = {
            "title": self.title,
            "key_findings": self.key_findings,
            "visualizations_summary": self.visualizations_summary,
        }
        if self.recommendations:
            result["recommendations"] = self.recommendations
        return result
