import pytest
import json
from typing import Dict, Any


def parse_tool_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    from data_scientist_chatbot.app.tools.parsers import parse_tool_call_arguments
    return parse_tool_call_arguments(tool_call)


@pytest.mark.unit
class TestToolArgumentParsing:

    def test_parse_valid_json_arguments(self, mock_tool_call):
        result = json.loads(mock_tool_call["arguments"])
        assert result["code"] == "print(1+1)"
        assert result["session_id"] == "test_123"

    def test_parse_python_execution_args(self):
        tool_call = {
            "name": "python_code_execution",
            "arguments": '{"code": "import pandas as pd\\ndf = pd.read_csv(\\"data.csv\\")", "session_id": "abc"}'
        }
        result = json.loads(tool_call["arguments"])
        assert "import pandas" in result["code"]
        assert result["session_id"] == "abc"

    def test_parse_web_search_args(self):
        tool_call = {
            "name": "web_search",
            "arguments": '{"query": "housing market trends 2024"}'
        }
        result = json.loads(tool_call["arguments"])
        assert result["query"] == "housing market trends 2024"

    def test_parse_knowledge_graph_query_args(self):
        tool_call = {
            "name": "knowledge_graph_query",
            "arguments": '{"query": "MATCH (c:Column) WHERE c.name = \\"price\\" RETURN c"}'
        }
        result = json.loads(tool_call["arguments"])
        assert "MATCH" in result["query"]

    def test_parse_malformed_json_raises_error(self):
        tool_call = {
            "name": "test_tool",
            "arguments": '{"code": "test", invalid_json}'
        }
        with pytest.raises(json.JSONDecodeError):
            json.loads(tool_call["arguments"])

    def test_parse_empty_arguments(self):
        tool_call = {
            "name": "test_tool",
            "arguments": '{}'
        }
        result = json.loads(tool_call["arguments"])
        assert result == {}

    def test_parse_nested_json_arguments(self):
        tool_call = {
            "name": "complex_tool",
            "arguments": '{"config": {"timeout": 30, "retries": 3}, "data": [1, 2, 3]}'
        }
        result = json.loads(tool_call["arguments"])
        assert result["config"]["timeout"] == 30
        assert result["data"] == [1, 2, 3]

    def test_parse_arguments_with_special_characters(self):
        tool_call = {
            "name": "test_tool",
            "arguments": json.dumps({"query": "SELECT * FROM table WHERE name = \"O'Brien\""})
        }
        result = json.loads(tool_call["arguments"])
        assert "O'Brien" in result["query"]

    def test_parse_multiline_code_arguments(self):
        code = "def hello():\\n    print('world')\\n    return 42"
        tool_call = {
            "name": "python_code_execution",
            "arguments": json.dumps({"code": code, "session_id": "test"})
        }
        result = json.loads(tool_call["arguments"])
        assert "def hello()" in result["code"]
        assert "return 42" in result["code"]

    def test_parse_unicode_arguments(self):
        tool_call = {
            "name": "test_tool",
            "arguments": '{"text": "Hello 世界 🌍"}'
        }
        result = json.loads(tool_call["arguments"])
        assert result["text"] == "Hello 世界 🌍"
