"""Image Analyzer Service - Extract structured data from images using vision model"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    Analyzes images using vision model to extract structured data.
    Supports charts, tables, and documents.
    """

    EXTRACTION_PROMPT = """Analyze this image and extract all quantitative data in structured format.

OUTPUT FORMAT (respond with ONLY valid JSON, no markdown):
{
  "image_type": "chart|table|document|mixed",
  "title": "detected title or null",
  "columns": ["column1", "column2", ...],
  "rows": [
    {"column1": value1, "column2": value2},
    ...
  ],
  "metadata": {
    "chart_type": "bar|line|pie|scatter|heatmap|null",
    "x_axis": "label or null",
    "y_axis": "label or null",
    "source": "detected source or null"
  },
  "confidence": 0.0 to 1.0,
  "notes": "any extraction warnings or null"
}

RULES:
- Extract ALL visible data points
- Preserve exact column names from labels
- Numbers should be numeric, not strings
- If data is unclear, estimate and lower confidence
- For charts without data labels, estimate values from visual position"""

    TABLE_PROMPT = """This image contains a table. Extract ALL data from the table.

OUTPUT FORMAT (respond with ONLY valid JSON):
{
  "image_type": "table",
  "title": "table title or null",
  "columns": ["col1", "col2", ...],
  "rows": [
    {"col1": "value1", "col2": "value2"},
    ...
  ],
  "confidence": 0.0 to 1.0,
  "notes": "any issues or null"
}

Extract every row and column. Preserve headers exactly as shown."""

    CHART_PROMPT = """This image is a chart/graph. Extract all data points.

OUTPUT FORMAT (respond with ONLY valid JSON):
{
  "image_type": "chart",
  "chart_type": "bar|line|pie|scatter|area|heatmap",
  "title": "chart title or null",
  "columns": ["category", "value"],
  "rows": [
    {"category": "Label1", "value": 123.4},
    ...
  ],
  "metadata": {
    "x_axis": "x-axis label",
    "y_axis": "y-axis label"
  },
  "confidence": 0.0 to 1.0,
  "notes": "any estimation notes"
}

For pie charts: columns are ["segment", "percentage"].
For line/scatter: columns are ["x", "y"] or include series name.
Estimate values if not labeled."""

    def __init__(self):
        self._cached_extractions: Dict[str, Dict] = {}

    async def extract_data(self, image_path: str, analysis_type: str = "auto", session_id: str = "") -> Dict[str, Any]:
        """
        Extract structured data from image using vision model.

        Args:
            image_path: Path to the image file
            analysis_type: 'auto', 'chart', 'table', or 'document'
            session_id: Session ID for caching

        Returns:
            Extracted data with columns, rows, and metadata
        """
        cache_key = f"{session_id}:{Path(image_path).name}"

        if cache_key in self._cached_extractions:
            logger.info(f"[IMAGE] Using cached extraction for {cache_key}")
            return self._cached_extractions[cache_key]

        prompt = self._get_prompt_for_type(analysis_type)

        try:
            from data_scientist_chatbot.app.core.agent_factory import create_vision_agent

            vision_agent = create_vision_agent()

            with open(image_path, "rb") as f:
                image_data = f.read()

            import base64

            image_b64 = base64.b64encode(image_data).decode("utf-8")

            file_ext = Path(image_path).suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(file_ext, "image/png")

            from langchain_core.messages import HumanMessage

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ]
            )

            response = await vision_agent.ainvoke([message])

            result = self._parse_vision_response(response.content)

            result["row_count"] = len(result.get("rows", []))
            result["preview_rows"] = result.get("rows", [])[:10]

            self._cached_extractions[cache_key] = result

            logger.info(f"[IMAGE] Extracted {result['row_count']} rows from {image_path}")

            return result

        except Exception as e:
            logger.error(f"[IMAGE] Vision extraction failed: {e}")
            return {
                "image_type": "unknown",
                "title": None,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "preview_rows": [],
                "confidence": 0.0,
                "notes": f"Extraction failed: {str(e)}",
            }

    def _get_prompt_for_type(self, analysis_type: str) -> str:
        prompts = {
            "auto": self.EXTRACTION_PROMPT,
            "table": self.TABLE_PROMPT,
            "chart": self.CHART_PROMPT,
            "document": self.EXTRACTION_PROMPT,
        }
        return prompts.get(analysis_type, self.EXTRACTION_PROMPT)

    def _parse_vision_response(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return {
                "image_type": "unknown",
                "title": None,
                "columns": [],
                "rows": [],
                "confidence": 0.1,
                "notes": f"Failed to parse JSON response: {content[:200]}...",
            }

    async def create_dataframe(self, image_id: str, session_id: str, adjustments: Optional[Dict] = None) -> str:
        """
        Create a DataFrame from extracted data and save as CSV.

        Args:
            image_id: ID of the analyzed image
            session_id: Session ID
            adjustments: Optional column renames or data corrections

        Returns:
            Path to saved CSV file
        """
        cache_key = f"{session_id}:{image_id}"

        found_key = None
        for key in self._cached_extractions:
            if image_id in key:
                found_key = key
                break

        if not found_key:
            raise ValueError(f"No extraction found for image {image_id}")

        extraction = self._cached_extractions[found_key]

        columns = extraction.get("columns", [])
        rows = extraction.get("rows", [])

        if not columns or not rows:
            raise ValueError("No data extracted from image")

        if adjustments:
            if "column_renames" in adjustments:
                renames = adjustments["column_renames"]
                columns = [renames.get(c, c) for c in columns]
                rows = [{renames.get(k, k): v for k, v in row.items()} for row in rows]

        df = pd.DataFrame(rows)

        if list(df.columns) != columns:
            df.columns = columns[: len(df.columns)]

        output_dir = Path(f"data/uploads/{session_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"extracted_{image_id}.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"[IMAGE] Created DataFrame at {output_path}: {len(df)} rows, {len(df.columns)} columns")

        return str(output_path)

    def validate_and_clean_data(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        rows = extraction.get("rows", [])
        columns = extraction.get("columns", [])

        if not rows:
            return extraction

        cleaned_rows = []
        for row in rows:
            if isinstance(row, dict):
                cleaned_row = {}
                for col in columns:
                    val = row.get(col)
                    cleaned_row[col] = self._coerce_value(val)
                cleaned_rows.append(cleaned_row)

        extraction["rows"] = cleaned_rows
        extraction["row_count"] = len(cleaned_rows)
        return extraction

    def _coerce_value(self, val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            val = val.strip()
            if val.lower() in ("null", "none", "n/a", "-", ""):
                return None
            val_clean = val.replace(",", "").replace("%", "").replace("$", "")
            try:
                if "." in val_clean:
                    return float(val_clean)
                return int(val_clean)
            except ValueError:
                return val
        return val

    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        type_map = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                type_map[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_map[col] = "datetime"
            else:
                type_map[col] = "categorical"
        return type_map
