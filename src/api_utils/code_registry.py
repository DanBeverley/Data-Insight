from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


class CodeRegistry:
    STORAGE_FILE = Path(__file__).parent.parent.parent / "data" / "metadata" / "code_registry.json"

    def __init__(self):
        self.session_code: Dict[str, List[Dict]] = {}
        self._load_from_file()

    def _load_from_file(self):
        if self.STORAGE_FILE.exists():
            try:
                with open(self.STORAGE_FILE, "r") as f:
                    self.session_code = json.load(f)
            except Exception:
                self.session_code = {}
        else:
            self.session_code = {}

    def _save_to_file(self):
        try:
            self.STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.STORAGE_FILE, "w") as f:
                json.dump(self.session_code, f, indent=2)
        except Exception:
            pass

    def add_code(
        self,
        session_id: str,
        code: str,
        description: str = "",
        artifact_filename: Optional[str] = None,
        execution_result: Optional[str] = None,
    ):
        if session_id not in self.session_code:
            self.session_code[session_id] = []

        entry = {
            "id": f"{session_id}_{len(self.session_code[session_id])}",
            "code": code,
            "description": description,
            "artifact_filename": artifact_filename,
            "execution_result": execution_result,
            "created_at": datetime.now().isoformat(),
        }

        self.session_code[session_id].append(entry)
        self._save_to_file()
        return entry

    def get_session_code(self, session_id: str) -> List[Dict]:
        self._load_from_file()
        return self.session_code.get(session_id, [])

    def get_code_for_artifact(self, session_id: str, artifact_filename: str) -> Optional[Dict]:
        self._load_from_file()
        codes = self.session_code.get(session_id, [])
        for entry in codes:
            if entry.get("artifact_filename") == artifact_filename:
                return entry
        return None

    def export_as_script(self, session_id: str, include_imports: bool = True) -> str:
        codes = self.get_session_code(session_id)
        if not codes:
            return ""

        lines = []
        if include_imports:
            lines.append("import pandas as pd")
            lines.append("import numpy as np")
            lines.append("import plotly.express as px")
            lines.append("import plotly.graph_objects as go")
            lines.append("")
            lines.append("")

        for i, entry in enumerate(codes):
            if entry.get("description"):
                lines.append(f"# {entry['description']}")
            lines.append(entry["code"])
            lines.append("")

        return "\n".join(lines)

    def export_as_notebook(self, session_id: str) -> str:
        codes = self.get_session_code(session_id)

        nb = new_notebook()

        nb.cells.append(
            new_markdown_cell(
                f"# Data Analysis Session\n\nGenerated from session: `{session_id[:8]}`\n\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        )

        nb.cells.append(
            new_code_cell(
                "import pandas as pd\nimport numpy as np\nimport plotly.express as px\nimport plotly.graph_objects as go"
            )
        )

        for entry in codes:
            if entry.get("description"):
                nb.cells.append(new_markdown_cell(f"## {entry['description']}"))
            nb.cells.append(new_code_cell(entry["code"]))

        return nbformat.writes(nb)

    def clear_session(self, session_id: str):
        if session_id in self.session_code:
            del self.session_code[session_id]
            self._save_to_file()


_code_registry = None


def get_code_registry() -> CodeRegistry:
    global _code_registry
    if _code_registry is None:
        _code_registry = CodeRegistry()
    return _code_registry
