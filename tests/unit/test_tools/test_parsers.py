import pytest
from unittest.mock import Mock
from data_scientist_chatbot.app.tools.parsers import parse_message_to_tool_call


@pytest.mark.unit
class TestMessageToolCallParsing:

    def test_parse_clean_json_format(self):
        message = Mock()
        message.tool_calls = None
        message.content = '{"name": "python_code_interpreter", "arguments": {"code": "print(1+1)"}}'

        result = parse_message_to_tool_call(message)

        assert result is True
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0]["name"] == "python_code_interpreter"
        assert message.tool_calls[0]["args"]["code"] == "print(1+1)"
        assert message.content == ""

    def test_parse_json_with_markdown_backticks(self):
        message = Mock()
        message.tool_calls = None
        message.content = """```json
        {"name": "web_search", "arguments": {"query": "housing trends"}}
        ```"""

        result = parse_message_to_tool_call(message)

        assert result is True
        assert message.tool_calls[0]["name"] == "web_search"
        assert message.tool_calls[0]["args"]["query"] == "housing trends"

    def test_parse_json_embedded_in_text(self):
        message = Mock()
        message.tool_calls = None
        message.content = 'Let me search for that. {"name": "web_search", "arguments": {"query": "test"}}'

        result = parse_message_to_tool_call(message)

        assert result is True
        assert message.tool_calls[0]["name"] == "web_search"

    def test_returns_true_if_tool_calls_already_exist(self):
        message = Mock()
        message.tool_calls = [{"name": "existing_tool"}]
        message.content = "some content"

        result = parse_message_to_tool_call(message)

        assert result is True

    def test_returns_false_for_invalid_json(self):
        message = Mock()
        message.tool_calls = None
        message.content = '{"name": invalid json}'

        result = parse_message_to_tool_call(message)

        assert result is False

    def test_returns_false_for_json_without_name_field(self):
        message = Mock()
        message.tool_calls = None
        message.content = '{"arguments": {"code": "test"}}'

        result = parse_message_to_tool_call(message)

        assert result is False

    def test_returns_false_for_json_without_arguments_field(self):
        message = Mock()
        message.tool_calls = None
        message.content = '{"name": "test_tool"}'

        result = parse_message_to_tool_call(message)

        assert result is False

    def test_returns_false_for_plain_text(self):
        message = Mock()
        message.tool_calls = None
        message.content = "This is just plain text without any JSON"

        result = parse_message_to_tool_call(message)

        assert result is False

    def test_custom_tool_id_prefix(self):
        message = Mock()
        message.tool_calls = None
        message.content = '{"name": "python_code_interpreter", "arguments": {"code": "test"}}'

        result = parse_message_to_tool_call(message, tool_id_prefix="custom")

        assert result is True
        assert message.tool_calls[0]["id"] == "custom_python_code_interpreter"

    def test_preserves_complex_nested_arguments(self):
        message = Mock()
        message.tool_calls = None
        message.content = '{"name": "test", "arguments": {"nested": {"key": "value"}, "list": [1, 2, 3]}}'

        result = parse_message_to_tool_call(message)

        assert result is True
        assert message.tool_calls[0]["args"]["nested"]["key"] == "value"
        assert message.tool_calls[0]["args"]["list"] == [1, 2, 3]
