import json
import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage
from .helpers import create_agent_input, create_workflow_status_context
from .agent_response import extract_agent_response
from data_scientist_chatbot.app.core.constants import NodeName, WORKFLOW_NODES, SAFE_CHECKPOINT_NODES
from data_scientist_chatbot.app.core.state_extractor import extract_report_url_from_messages
from data_scientist_chatbot.app.core.agent_factory import get_model_name


def strip_model_tokens(content: str) -> str:
    """Strip special model formatting tokens from output"""
    if not content or not isinstance(content, str):
        return content

    # Strip model special tokens: <|start|>, <|channel|>, <|message|>, etc.
    content = re.sub(r"<\|[^|]+\|>", "", content)

    # Strip any remaining angle bracket tokens
    content = re.sub(r"<\|", "", content)
    content = re.sub(r"\|>", "", content)

    return content.strip()


def format_agent_output(content: str) -> str:
    """Universal formatter for various data types in agent output"""
    if not content or not isinstance(content, str):
        return content

    # Try pandas DataFrame
    df_result = _try_format_dataframe(content)
    if df_result:
        return df_result[0]

    # Try other formats (skip markdown table - let frontend handle it)
    json_result = _try_format_json(content)
    if json_result:
        return json_result[0]

    list_result = _try_format_list(content)
    if list_result:
        return list_result[0]

    code_result = _try_format_code_block(content)
    if code_result:
        return code_result[0]

    stats_result = _try_format_statistics(content)
    if stats_result:
        return stats_result[0]

    return content


def _try_format_markdown_table(text: str):
    """Detect and format markdown tables (| col1 | col2 |)"""
    lines = text.split("\n")

    # Find first line with pipe separator
    table_start = -1
    for i, line in enumerate(lines):
        if "|" in line and line.count("|") >= 2:
            table_start = i
            break

    if table_start == -1:
        return None

    # Extract table lines (consecutive lines with pipes)
    table_lines = []
    for i in range(table_start, len(lines)):
        if "|" in lines[i]:
            table_lines.append(lines[i])
        elif table_lines:
            break

    if len(table_lines) < 2:
        return None

    try:
        # Parse header row
        header_line = table_lines[0]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        # Skip separator line (usually line 1: |---|---|)
        data_start = 2 if len(table_lines) > 1 and "-" in table_lines[1] else 1

        html = '<div class="table-responsive-container">\n<div class="table-scroll-wrapper">\n<table class="data-table markdown-table">\n<thead>\n<tr>'
        for header in headers:
            html += f"<th>{header}</th>"
        html += "</tr>\n</thead>\n<tbody>\n"

        # Parse data rows
        for line in table_lines[data_start:]:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                html += "<tr>"
                for cell in cells:
                    html += f"<td>{cell}</td>"
                html += "</tr>\n"

        html += "</tbody>\n</table>\n</div>\n</div>"

        # Include any text before the table
        prefix = "\n".join(lines[:table_start]) if table_start > 0 else ""
        # Include any text after the table
        table_end = table_start + len(table_lines)
        suffix = "\n".join(lines[table_end:]) if table_end < len(lines) else ""

        result = f"{prefix}\n{html}\n{suffix}".strip()
        print(f"DEBUG MD TABLE: Formatted {len(table_lines)} rows with {len(headers)} columns")

        return result, len(text)
    except Exception as e:
        print(f"DEBUG MD TABLE: Failed to parse: {e}")
        return None


def _try_format_dataframe(text: str):
    """Detect and format pandas DataFrame text"""
    lines = text.split("\n")
    if len(lines) < 2:
        print(f"DEBUG DF: Not enough lines ({len(lines)})")
        return None

    # Handle pandas line continuation (backslash at end)
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if line ends with backslash (continuation)
        while line.rstrip().endswith("\\") and i + 1 < len(lines):
            line = line.rstrip()[:-1] + " " + lines[i + 1].lstrip()
            i += 1
        cleaned_lines.append(line)
        i += 1

    lines = cleaned_lines
    first_line = lines[0]

    print(f"DEBUG DF: First line after continuation merge: {first_line[:100]}")

    # DataFrame detection: multiple columns separated by 2+ spaces
    if not re.search(r"\S+\s{2,}\S+", first_line):
        print(f"DEBUG DF: No 2+ space separation detected")
        return None

    # Check if looks like tabular data
    has_index = False
    for line in lines[1 : min(4, len(lines))]:
        if line.strip() and line.split()[0].replace("-", "").isdigit():
            has_index = True
            break

    print(f"DEBUG DF: has_index={has_index}")

    try:
        # Split by 2+ spaces to get columns
        columns = re.split(r"\s{2,}", first_line.strip())
        columns = [col.strip() for col in columns if col.strip()]

        print(f"DEBUG DF: Found {len(columns)} columns: {columns[:5]}")

        html = '<div class="table-responsive-container">\n<div class="table-scroll-wrapper">\n<table class="data-table">\n<thead>\n<tr>'

        if has_index:
            html += '<th class="index-col">Index</th>'

        for col in columns:
            html += f"<th>{col}</th>"
        html += "</tr>\n</thead>\n<tbody>\n"

        row_count = 0
        for line in lines[1:]:
            if not line.strip():
                continue
            # Split by whitespace but handle leading index
            parts = re.split(r"\s+", line.strip())
            if not parts:
                continue

            html += "<tr>"
            for i, val in enumerate(parts):
                css_class = "index-col" if has_index and i == 0 else "data-col"
                html += f'<td class="{css_class}">{val}</td>'
            html += "</tr>\n"
            row_count += 1

        html += "</tbody>\n</table>\n</div>\n</div>"

        print(f"DEBUG DF: Successfully formatted {row_count} rows")
        consumed = len(text)  # Consume entire DataFrame
        return html, consumed
    except Exception as e:
        print(f"DEBUG DF: Exception during parsing: {e}")
        return None


def _try_format_json(text: str):
    """Detect and format JSON objects/arrays"""
    text_stripped = text.strip()

    if not (text_stripped.startswith("{") or text_stripped.startswith("[")):
        return None

    try:
        # Try to find complete JSON
        depth = 0
        in_string = False
        escape = False
        end_pos = 0

        for i, char in enumerate(text_stripped):
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

        if end_pos > 0:
            json_str = text_stripped[:end_pos]
            parsed = json.loads(json_str)

            formatted_json = json.dumps(parsed, indent=2)
            html = f'<div class="json-output">\n<pre class="json-content">{formatted_json}</pre>\n</div>'

            return html, len(json_str)
    except:
        return None

    return None


def _try_format_list(text: str):
    """Detect and format list outputs"""
    lines = text.split("\n")

    # Check for list pattern: starts with -, *, or numbered items
    list_pattern = re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[0])
    if not list_pattern:
        return None

    html_items = []
    line_count = 0
    is_ordered = lines[0].strip()[0].isdigit()

    for line in lines:
        if not line.strip():
            break
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            item_text = re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)
            html_items.append(f"<li>{item_text}</li>")
            line_count += 1
        else:
            break

    if html_items:
        tag = "ol" if is_ordered else "ul"
        html = f'<div class="list-output">\n<{tag}>\n' + "\n".join(html_items) + f"\n</{tag}>\n</div>"
        consumed = sum(len(lines[i]) + 1 for i in range(line_count))
        return html, consumed

    return None


def _try_format_code_block(text: str):
    """Detect and format code blocks (```python ... ```)"""
    match = re.match(r"^```(\w+)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        language = match.group(1) or "python"
        code = match.group(2)
        html = f'<div class="code-block">\n<pre class="language-{language}"><code>{code}</code></pre>\n</div>'
        return html, match.end()

    return None


def _try_format_statistics(text: str):
    """Detect and format statistical summaries"""
    # Check for patterns like "Mean: 123.45" or "Count: 100"
    lines = text.split("\n")
    stats_pattern = re.compile(r"^([A-Z][a-zA-Z\s]+):\s*([0-9.,\-]+|[a-zA-Z]+)$")

    stats_found = []
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        match = stats_pattern.match(line.strip())
        if match:
            stats_found.append((match.group(1).strip(), match.group(2).strip()))
        elif stats_found:
            break
        elif i > 0:
            break

    if len(stats_found) >= 2:
        html = '<div class="statistics-output">\n<dl class="stats-list">\n'
        for key, value in stats_found:
            html += f"<dt>{key}</dt>\n<dd>{value}</dd>\n"
        html += "</dl>\n</div>"

        consumed = sum(len(lines[i]) + 1 for i in range(len(stats_found)))
        return html, consumed

    return None


async def _generate_report_summary(insights: list, artifact_count: int, report_path: str, user_message: str) -> str:
    try:
        from data_scientist_chatbot.app.core.agent_factory import create_brain_agent

        brain_agent = create_brain_agent()

        formatted_insights = []
        for i in insights[:10]:
            if isinstance(i, dict):
                label = i.get("label", "")
                value = i.get("value", "")
                if label and value:
                    formatted_insights.append(f"- **{label}**: {value}")

        insights_block = (
            "\n".join(formatted_insights)
            if formatted_insights
            else "No specific insights were extracted from this analysis."
        )

        prompt = f"""You are summarizing a data analysis that was just completed. Based on the insights below, write a natural conversational response.

**User's Request**: {user_message[:300]}

**Analysis Results**:
- Generated {artifact_count} visualizations
- Key Insights Found:
{insights_block}

**Instructions**:
1. Write 2-4 sentences summarizing the key findings in a friendly, conversational tone
2. Be SPECIFIC - mention actual numbers, trends, or patterns from the insights above
3. At the end, add this exact link on its own line: **[📄 View Full Report](report:{report_path})**

Write your response now:"""

        logging.info(f"[STREAM] Calling Brain for summary with {len(formatted_insights)} insights")
        result = await brain_agent.ainvoke([HumanMessage(content=prompt)])
        content = result.content if hasattr(result, "content") else str(result)

        logging.info(
            f"[STREAM] Brain summary generated ({len(content) if content else 0} chars): {content[:200] if content else 'None'}..."
        )

        if content and len(content) > 20:
            if f"report:{report_path}" not in content:
                content += f"\n\n**[📄 View Full Report](report:{report_path})**"
            return content
        else:
            logging.warning(f"[STREAM] Brain summary too short or empty")
    except Exception as e:
        logging.warning(f"[STREAM] Brain summary generation failed: {e}")
        import traceback

        logging.warning(f"[STREAM] Traceback: {traceback.format_exc()}")

    insight_labels = [i.get("label", "") for i in insights[:5] if isinstance(i, dict) and i.get("label")]
    insight_text = f" Key findings include: {', '.join(insight_labels)}." if insight_labels else ""
    return f"## 📊 Analysis Complete\n\nGenerated {artifact_count} visualizations from your analysis.{insight_text}\n\n**[📄 View Full Report](report:{report_path})**"


async def stream_agent_chat(
    message: str,
    session_id: str,
    agent,
    session_store: Dict[str, Any],
    session_agents=None,
    regenerate: bool = False,
    message_id: str = None,
) -> AsyncGenerator[str, None]:
    from .message_storage import save_message, save_message_version
    import uuid

    if regenerate and message_id:
        pass
    else:
        save_message(session_id, "human", message)
        message_id = str(uuid.uuid4())

    thinking_mode = session_store.get(session_id, {}).get("thinking_mode", False)

    try:
        logging.info(f"[STREAMING] Using .astream() method (async streaming) for session {session_id}")
        try:
            async for event_data in _stream_fallback(message, session_id, agent):
                yield event_data
        except KeyError as ke:
            if "__start__" in str(ke):
                logging.warning(f"Checkpoint error for session {session_id}: {ke}")

                from .session_management import clean_checkpointer_state

                cleaned = clean_checkpointer_state(session_id, "checkpoint_error")

                if cleaned:
                    logging.info(f"Checkpoint cleaned, recreating agent with fresh state for session {session_id}")

                    from data_scientist_chatbot.app.core.graph_builder import create_enhanced_agent_executor

                    agent = create_enhanced_agent_executor(session_id)

                    if session_agents is not None and session_id in session_agents:
                        session_agents[session_id] = agent
                        logging.info(f"Agent cache updated for session {session_id}")
                else:
                    logging.warning(f"Checkpoint cleanup failed, proceeding anyway for session {session_id}")

                async for event_data in _stream_fallback(message, session_id, agent):
                    yield event_data
            else:
                raise
        except Exception as stream_error:
            logging.error(f"[STREAMING] .astream() failed for session {session_id}: {stream_error}")

            if hasattr(agent, "astream_events"):
                logging.warning(f"[STREAMING] Falling back to astream_events() for session {session_id}")
                try:
                    async for event_data in _stream_with_events(message, session_id, agent, thinking_mode):
                        yield event_data
                except Exception as e:
                    logging.error(f"[STREAMING] astream_events() also failed: {e}")
                    raise
            else:
                raise

    except Exception as e:
        logging.error(f"Streaming error for session {session_id}: {e}")

        # Save error message to prevent message corruption on reload
        from .message_storage import save_message

        error_content = f"An error occurred while processing your request: {str(e)}"
        try:
            save_message(session_id, "ai", error_content)
        except Exception as save_error:
            logging.error(f"Failed to save error message: {save_error}")

        yield f"data: {json.dumps({'type': 'error', 'message': f'Streaming error: {str(e)}'})}\n\n"


async def _stream_with_events(
    message: str, session_id: str, agent, thinking_mode: bool = False
) -> AsyncGenerator[str, None]:
    await asyncio.sleep(0.01)

    config = {"configurable": {"thread_id": session_id}}
    input_data = create_agent_input(message, session_id, thinking_mode=thinking_mode)

    logging.info(f"[STREAM_EVENTS] Starting for session {session_id}")
    logging.info(f"[STREAM_EVENTS] Agent type: {type(agent)}")
    logging.info(f"[STREAM_EVENTS] Config: {config}")

    plots = []
    final_response = None
    current_node = None

    workflow_context = {
        "current_agent": None,
        "current_action": None,
        "tool_calls": [],
        "agent_decisions": [],
        "user_goal": message,
        "execution_progress": {},
        "session_id": session_id,
    }

    thinking_parser = None
    if thinking_mode:
        from .thinking_mode_parser import ThinkingModeParser, create_thinking_status_message

        thinking_parser = ThinkingModeParser()
        logging.info(f"[STREAM_EVENTS] Thinking mode ENABLED for session {session_id}")

    try:
        logging.info(f"[STREAM_EVENTS] Calling agent.astream_events...")
        async for event in agent.astream_events(input_data, config=config, version="v2"):
            from .cancellation import is_task_cancelled

            if is_task_cancelled(session_id):
                yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Task cancelled by user'})}\n\n"
                break

            event_name = event.get("event")
            event_data = event.get("data", {})
            event_node_name = event.get("name", "unknown")

            logging.debug(f"[EVENT] {event_name} | node={event_node_name} | tags={event.get('tags', [])}")

            if event_name == "on_chain_start":
                current_node = event.get("name", "")
                workflow_context["current_agent"] = current_node
                workflow_context["current_action"] = "starting"

                if current_node == "hands":
                    logging.info(f"[STREAM_EVENTS] Hands node started")
                    # Try to get plan from state to mark as in-progress
                    # Note: event_data might not have the full state, but let's check input
                    # LangGraph v0.2+ passes input state in 'data' -> 'input'
                    input_state = event_data.get("input", {})
                    if input_state and isinstance(input_state, dict):
                        plan = input_state.get("plan", [])
                        current_idx = input_state.get("current_task_index", 0)

                        if plan and current_idx < len(plan):
                            # Create a local copy to modify status for UI
                            import copy

                            ui_plan = copy.deepcopy(plan)
                            ui_plan[current_idx]["status"] = "in_progress"
                            logging.info(f"[STREAM_EVENTS] Emitting in-progress plan for task {current_idx}")
                            yield f"data: {json.dumps({'type': 'plan', 'plan': ui_plan})}\n\n"

                # [REPORT UI TRIGGER] Detect Architect start
                if current_node == "architect":
                    logging.info(f"[STREAM] Architect node started, triggering UI loading state")
                    yield f"data: {json.dumps({'type': 'report_generation_started'})}\n\n"

            elif event_name == "on_chat_model_stream":
                if event_data.get("chunk"):
                    chunk_content = str(event_data["chunk"].content)
                    workflow_context["execution_progress"][current_node] = chunk_content[:100]

                    if thinking_parser and chunk_content:
                        for thinking_update in thinking_parser.parse_streaming_chunk(chunk_content):
                            status_msg = create_thinking_status_message(thinking_update)
                            yield f"data: {json.dumps(status_msg)}\n\n"
                            await asyncio.sleep(0.05)

            elif event_name == "on_tool_start":
                tool_name = event.get("name", "")
                workflow_context["tool_calls"].append({"tool": tool_name, "status": "starting"})
                workflow_context["current_action"] = f"executing {tool_name}"

            elif event_name == "on_tool_end":
                tool_name = event.get("name", "")
                tool_output = event_data.get("output", "")

                for tool_info in workflow_context["tool_calls"]:
                    if tool_info["tool"] == tool_name and tool_info["status"] == "starting":
                        tool_info["status"] = "completed"
                        break

                workflow_context["current_action"] = f"completed {tool_name}"

                if tool_name == "python_code_interpreter" and "PLOT_SAVED:" in str(tool_output):
                    import re

                    plot_files = re.findall(r"PLOT_SAVED:([^\s]+\.png)", str(tool_output))
                    for plot_file in plot_files:
                        plots.append(f"/static/plots/{plot_file}")
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Visualization created successfully!'})}\n\n"
                    await asyncio.sleep(0.2)

            elif event_name == "on_chain_end":
                node_name = event.get("name", "unknown")
                logging.info(f"[EVENT] on_chain_end for node: {node_name}")

                if event_data.get("output"):
                    output_keys = (
                        list(event_data["output"].keys()) if isinstance(event_data["output"], dict) else "not_dict"
                    )
                    logging.debug(f"[STREAM_EVENTS] Node {node_name} output keys: {output_keys}")

                if node_name == "__start__":
                    logging.info(f"[STREAM_EVENTS] Graph execution completed (__start__ ended)")
                    final_response = event_data.get("output")
                    break

                if event_data.get("output") and isinstance(event_data["output"], dict):
                    plan = event_data["output"].get("plan")
                    if plan:
                        logging.info(f"[STREAM_EVENTS] Found plan in node {node_name} output with {len(plan)} tasks")
                        workflow_context["plan"] = plan
                        yield f"data: {json.dumps({'type': 'plan', 'plan': plan})}\n\n"

                if node_name == "architect":
                    logging.info(f"[STREAM_EVENTS] Architect node completed")
                    output = event_data.get("output") or {}

                    # Check for report_url in multiple locations:
                    # 1. Directly in output (state root) - how architect returns it
                    # 2. Nested in execution_result - legacy compatibility
                    report_path = None

                    # First check direct state keys
                    if output.get("report_url"):
                        report_path = output.get("report_url")
                        logging.info(f"[STREAM_EVENTS] Found report_url at state root: {report_path}")
                    elif output.get("report_path"):
                        raw_path = output.get("report_path")
                        import os

                        filename = os.path.basename(raw_path)
                        report_path = f"/reports/{filename}"
                        logging.info(f"[STREAM_EVENTS] Found report_path at state root, converted: {report_path}")
                    # Then check nested in execution_result
                    elif output.get("execution_result"):
                        exec_result = output.get("execution_result")
                        report_path = exec_result.get("report_url")
                        if not report_path and exec_result.get("report_path"):
                            raw_path = exec_result.get("report_path")
                            import os

                            filename = os.path.basename(raw_path)
                            report_path = f"/reports/{filename}"

                    if report_path:
                        logging.info(f"[STREAM_EVENTS] Sending report_generated event: {report_path}")
                        yield f"data: {json.dumps({'type': 'report_generated', 'report_path': report_path})}\n\n"
                    else:
                        logging.warning(
                            f"[STREAM_EVENTS] Architect completed but no report_url found in output: {list(output.keys())}"
                        )
                else:
                    logging.debug(f"[EVENT] Ignoring on_chain_end for node: {node_name}")

            else:
                logging.debug(f"[EVENT] Unhandled event type: {event_name} | node={event_node_name}")

    except KeyError as ke:
        logging.error(f"[STREAM_EVENTS] KeyError occurred: {ke}")
        logging.error(f"[STREAM_EVENTS] Error type: {type(ke)}")
        import traceback

        logging.error(f"[STREAM_EVENTS] Traceback:\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logging.error(f"[STREAM_EVENTS] Unexpected error: {e}")
        import traceback

        logging.error(f"[STREAM_EVENTS] Traceback:\n{traceback.format_exc()}")
        raise

    logging.info("[STREAM_EVENTS] Event loop completed, processing final response...")

    from .cancellation import is_task_cancelled, clear_cancellation

    if is_task_cancelled(session_id):
        clear_cancellation(session_id)
        yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Task cancelled by user'})}\n\n"
        return

    if final_response and final_response.get("messages"):
        final_message = final_response["messages"][-1]
        content = final_message.content if hasattr(final_message, "content") else str(final_message)
        logging.info(f"[STREAM_EVENTS] Final response content length: {len(content)}")
    else:
        content = "Task completed successfully."
        logging.warning("[STREAM_EVENTS] No final response messages, using default content")

    content = format_agent_output(content)

    from src.api import session_store as api_session_store

    token_streaming = api_session_store.get(session_id, {}).get("token_streaming", True)

    if token_streaming:
        yield f"data: {json.dumps({'type': 'start_tokens'})}\n\n"

        async for token_event in _stream_brain_tokens(content, session_id):
            yield token_event

        yield f"data: {json.dumps({'type': 'end_tokens'})}\n\n"

    from .message_storage import save_message, save_message_version

    metadata = {"plots": plots} if plots else None

    if message_id:
        version = save_message_version(session_id, message_id, content, metadata=metadata)
    else:
        save_message(session_id, "ai", content, metadata=metadata)
        version = 1

    clear_cancellation(session_id)

    yield f"data: {json.dumps({'type': 'final_response', 'response': content, 'plots': plots, 'message_id': message_id, 'version': version})}\n\n"


async def _stream_brain_tokens(content: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream content token-by-token for ChatGPT-like experience.
    Speed: 100 tokens/sec (10ms delay) - faster than ChatGPT's 50 tokens/sec
    """
    import os

    token_delay = float(os.getenv("TOKEN_STREAM_DELAY", "0.01"))

    words = content.split()
    for word in words:
        from .cancellation import is_task_cancelled

        if is_task_cancelled(session_id):
            break

        yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
        await asyncio.sleep(token_delay)


async def _stream_fallback(message: str, session_id: str, agent) -> AsyncGenerator[str, None]:
    from langchain_core.messages import HumanMessage
    import builtins

    config = {"configurable": {"thread_id": session_id}}
    final_response = None
    all_messages = []
    final_brain_response = None
    action_outputs = []

    logging.info(f"[STREAM] Starting async streaming for session {session_id}")
    logging.info(f"[STREAM] Agent type: {type(agent)}")
    logging.info(f"[STREAM] Config: {config}")

    from src.api import session_store as api_session_store

    if not hasattr(builtins, "_session_store"):
        builtins._session_store = {}
    if session_id not in builtins._session_store:
        builtins._session_store[session_id] = {}
    if session_id in api_session_store and "search_config" in api_session_store[session_id]:
        builtins._session_store[session_id]["search_config"] = api_session_store[session_id]["search_config"]

    original_user_message = HumanMessage(content=message)
    current_state = {"messages": [original_user_message], "session_id": session_id}

    try:
        logging.info(f"[STREAM] Calling agent.astream (async streaming)...")

        import time

        start_time = time.time()
        max_stream_time = 1800  # 30 minutes maximum (increased for complex reports)
        grace_period = 300  # 5 minutes grace after soft timeout (increased for complex reports)
        soft_timeout_hit = False
        soft_timeout_time = None
        current_pipeline_phase = "init"  # init → hands → verifier → brain
        safe_checkpoint_nodes = {"brain", "__end__", "presenter", "architect"}

        async for event in agent.astream(create_agent_input(message, session_id), config=config):
            elapsed = time.time() - start_time

            # Track current pipeline phase for smart timeout
            for node_name in event.keys():
                if node_name in WORKFLOW_NODES:
                    current_pipeline_phase = node_name

            # Smart timeout logic
            if elapsed > max_stream_time:
                if not soft_timeout_hit:
                    soft_timeout_hit = True
                    soft_timeout_time = time.time()
                    logging.warning(
                        f"[STREAM] Soft timeout hit at {elapsed:.0f}s. Phase: {current_pipeline_phase}. Grace period started."
                    )

                grace_elapsed = time.time() - soft_timeout_time
                at_safe_checkpoint = any(node in event for node in SAFE_CHECKPOINT_NODES)

                if at_safe_checkpoint:
                    logging.info(
                        f"[STREAM] Safe checkpoint reached ({list(event.keys())}). Completing stream gracefully."
                    )
                    break
                elif grace_elapsed > grace_period:
                    logging.error(
                        f"[STREAM] Grace period ({grace_period}s) exceeded. Forcing termination at phase: {current_pipeline_phase}"
                    )
                    break

            # Log ALL events to debug
            logging.debug(f"[STREAM] Received event: {list(event.keys())}")

            # Check if this is a special END event (LangGraph marks completed graphs)
            if "__end__" in event:
                logging.info(f"[STREAM] Graph reached END, completing stream")
                final_response = {"messages": all_messages}
                break

            for node_name, node_data in event.items():
                if node_name in WORKFLOW_NODES:
                    logging.info(f"[STREAM] Node completed: {node_name}")

                    if node_name in {NodeName.BRAIN.value, NodeName.HANDS.value}:
                        model_name = get_model_name(node_name)
                        yield f"data: {json.dumps({'type': 'thinking_start', 'agent': node_name, 'model_name': model_name})}\n\n"

                    messages = node_data.get("messages", []) if node_data else []
                    if messages and isinstance(messages, list):
                        all_messages.extend(messages)
                        current_state["messages"] = all_messages

                        if node_name == NodeName.HANDS.value:
                            logging.info(f"[STREAM] Hands node completed, messages count: {len(messages)}")
                            if node_data.get("artifacts"):
                                logging.info(f"[STREAM] Hands generated {len(node_data.get('artifacts'))} artifacts")
                            yield f"data: {json.dumps({'type': 'thinking_complete', 'agent': NodeName.HANDS.value})}\n\n"
                            yield f"data: {json.dumps({'type': 'task_update', 'status': 'complete'})}\n\n"
                            logging.info(f"[STREAM] Waiting for next node (should route to brain)...")

                        elif node_name == NodeName.BRAIN.value:
                            logging.info(f"[STREAM] Brain node executing")
                            if messages:
                                last_brain_msg = messages[-1]
                                brain_content = (
                                    str(last_brain_msg.content)
                                    if hasattr(last_brain_msg, "content")
                                    else str(last_brain_msg)
                                )
                                brain_content = strip_model_tokens(brain_content)

                                has_tool_calls = hasattr(last_brain_msg, "tool_calls") and last_brain_msg.tool_calls
                                if has_tool_calls:
                                    for tc in last_brain_msg.tool_calls:
                                        tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                                        tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                                        if tc_name == "delegate_coding_task":
                                            task_desc = tc_args.get("task_description", "Executing task...")
                                            yield f"data: {json.dumps({'type': 'task', 'description': task_desc, 'status': 'pending'})}\n\n"
                                            yield f"data: {json.dumps({'type': 'thinking_complete', 'agent': 'brain'})}\n\n"
                                        elif tc_name == "web_search":
                                            query = tc_args.get("query", "")
                                            yield f"data: {json.dumps({'type': 'search_status', 'action': 'searching', 'query': query})}\n\n"
                                else:
                                    final_brain_response = brain_content
                                    yield f"data: {json.dumps({'type': 'thinking_complete', 'agent': 'brain'})}\n\n"
                                    logging.info(
                                        f"[STREAM] Brain generated final response, length: {len(brain_content)}"
                                    )
                            else:
                                logging.warning(f"[STREAM] Brain node completed but no messages found")

                        elif node_name == NodeName.ACTION.value:
                            last_action_msg = messages[-1]
                            if hasattr(last_action_msg, "type") and last_action_msg.type == "tool":
                                action_content = str(last_action_msg.content)

                                # Check for search_status in session store
                                import builtins

                                if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                                    search_status = builtins._session_store[session_id].get("search_status")
                                    if search_status:
                                        yield f"data: {json.dumps({'type': 'search_status', **search_status})}\n\n"
                                        builtins._session_store[session_id]["search_status"] = None

                                # Parse SEARCH_DATA_JSON from web search results for accurate count/sources
                                if "SEARCH_DATA_JSON" in action_content:
                                    try:
                                        json_match = re.search(
                                            r"<!-- SEARCH_DATA_JSON\n(.*?)\nSEARCH_DATA_END -->",
                                            action_content,
                                            re.DOTALL,
                                        )
                                        if json_match:
                                            search_data = json.loads(json_match.group(1))
                                            result_count = search_data.get("result_count", 0)
                                            sources = search_data.get("sources", [])
                                            yield f"data: {json.dumps({'type': 'search_status', 'action': 'complete', 'resultCount': result_count, 'sources': sources[:5]})}\n\n"
                                    except Exception as e:
                                        logging.error(f"[STREAM] Failed to parse search data: {e}")

                                if action_content and action_content.strip():
                                    if '"event": "report_generated"' in action_content:
                                        try:
                                            json_match = re.search(
                                                r'\{.*"event":\s*"report_generated".*\}', action_content, re.DOTALL
                                            )
                                            if json_match:
                                                report_data = json.loads(json_match.group(0))
                                                report_id = report_data.get("report_id")
                                                if report_id:
                                                    logging.info(
                                                        f"[STREAM] Detected report {report_id}, triggering UI event"
                                                    )
                                                    yield f"data: {json.dumps({'type': 'report_generated', 'report_id': report_id})}\n\n"
                                        except Exception as e:
                                            logging.error(f"[STREAM] Failed to parse report event: {e}")

                                    # Skip web search results - they're shown via status bar, not in response
                                    if (
                                        "## Web Search Results" not in action_content
                                        and "SEARCH_DATA_JSON" not in action_content
                                    ):
                                        action_outputs.append(action_content)

                        elif node_name == NodeName.ANALYST.value:
                            logging.info(f"[STREAM] Analyst node completed")

                        elif node_name == NodeName.ARCHITECT.value:
                            logging.info(f"[STREAM] Architect node completed")

                            report_path = node_data.get("report_url") if node_data else None

                            if not report_path and messages:
                                for msg in reversed(messages):
                                    if hasattr(msg, "additional_kwargs"):
                                        report_path = msg.additional_kwargs.get("report_url")
                                        if report_path:
                                            logging.info(f"[STREAM] Extracted report_url from message kwargs")
                                            break

                            if not report_path and node_data:
                                if node_data.get("report_path"):
                                    import os

                                    raw_path = node_data.get("report_path")
                                    filename = os.path.basename(raw_path)
                                    report_path = f"/reports/{filename}"

                            if report_path:
                                logging.info(f"[STREAM] Captured report URL: {report_path}")
                                yield f"data: {json.dumps({'type': 'report_generated', 'report_path': report_path})}\n\n"

                                insights = node_data.get("agent_insights", []) if node_data else []
                                from .artifact_tracker import get_artifact_tracker

                                tracker = get_artifact_tracker()
                                tracker_result = tracker.get_session_artifacts(session_id)
                                tracked_artifacts = (
                                    tracker_result.get("artifacts", []) if isinstance(tracker_result, dict) else []
                                )
                                artifact_count = len(tracked_artifacts) if tracked_artifacts else 0

                                brain_context = await _generate_report_summary(
                                    insights, artifact_count, report_path, message
                                )
                                if brain_context:
                                    final_brain_response = brain_context
                            else:
                                logging.warning(f"[STREAM] Architect completed but no report_url found")

                            logging.info(f"[STREAM] Presenter node completed")

                    await asyncio.sleep(0.1)

        # Set final_response after loop completes
        if "final_response" not in locals():
            final_response = {"messages": all_messages}

        logging.info(f"[STREAM] Stream loop exited, processing final response...")

        # Set final_response after loop completes if not already set
        if not final_response:
            logging.info(f"[STREAM] Loop finished without __end__ event, using accumulated messages")
            final_response = {"messages": all_messages}

    except KeyError as ke:
        logging.error(f"[STREAM] KeyError occurred: {ke}")
        import traceback

        logging.error(f"[STREAM] Traceback:\n{traceback.format_exc()}")

        if "__start__" in str(ke):
            logging.error(f"[STREAM] Checkpoint still broken after cleanup for {session_id}: {ke}")
            raise
        else:
            raise
    except Exception as stream_error:
        logging.error(f"[STREAM] astream() failed for {session_id}: {stream_error}")
        import traceback

        logging.error(f"[STREAM] Traceback:\n{traceback.format_exc()}")

        try:
            logging.info(f"[STREAM] Attempting agent.ainvoke() as last resort...")
            final_response = await agent.ainvoke(create_agent_input(message, session_id), config=config)
        except Exception as invoke_error:
            logging.error(f"[STREAM] ainvoke() also failed for {session_id}: {invoke_error}")
            import traceback

            logging.error(f"[STREAM] ainvoke() traceback:\n{traceback.format_exc()}")
            raise

    logging.info(f"[STREAM] Processing final response...")

    try:
        response = final_response if final_response else {}
        messages = response.get("messages", []) if response else []
        logging.info(f"[STREAM] Extracting from {len(messages)} messages...")

        _, plots = extract_agent_response(messages, recent_count=3)
        logging.info(f"[STREAM] Extracted {len(plots)} plots")

        if plots:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Visualization generated!'})}\n\n"
            await asyncio.sleep(0.3)

        logging.info(f"[STREAM] Building response content...")
        logging.info(
            f"[STREAM] Available: action_outputs={len(action_outputs)}, final_brain_response={bool(final_brain_response)}, messages={len(messages)}"
        )

        content_parts = []

        # Use action outputs (hands execution results)
        if action_outputs:
            formatted_outputs = [format_agent_output(output) for output in action_outputs]
            content_parts.extend(formatted_outputs)
            logging.info(f"[STREAM] Added {len(formatted_outputs)} action outputs")

        # Use brain response if available
        if final_brain_response:
            content_parts.append(format_agent_output(final_brain_response))
            logging.info(f"[STREAM] Added brain response ({len(final_brain_response)} chars)")

        if not content_parts:
            logging.warning(f"[STREAM] No content_parts built, using fallback extraction")
            fallback_content = None
            for msg in reversed(all_messages):
                if hasattr(msg, "type") and msg.type in ["ai", "tool"]:
                    fallback_content = msg.content
                    break

            if not fallback_content or not str(fallback_content).strip():
                try:
                    from src.api_utils.artifact_tracker import get_artifact_tracker

                    tracker = get_artifact_tracker()
                    session_artifacts = tracker.get_session_artifacts_with_urls(session_id)

                    if session_artifacts:
                        all_artifacts = session_artifacts.get("artifacts", [])
                        viz_artifacts = [a for a in all_artifacts if a.get("category") == "visualization"]
                        if viz_artifacts:
                            fallback_content = f"## Analysis Complete\n\nI've generated **{len(viz_artifacts)} visualizations** for your analysis:\n\n"
                            for artifact in viz_artifacts[:10]:
                                filename = artifact.get("filename", "")
                                url = artifact.get("url", f"/static/plots/{filename}")
                                if filename.endswith(".html"):
                                    fallback_content += f"- [📊 {filename}]({url})\n"
                                elif filename.endswith((".png", ".jpg")):
                                    fallback_content += f"![{filename}]({url})\n"
                            if len(viz_artifacts) > 10:
                                fallback_content += f"\n...and {len(viz_artifacts) - 10} more visualizations.\n"
                            fallback_content += "\n*Please review the generated charts above. If you need specific insights, ask me to analyze them.*"
                        else:
                            fallback_content = "Task completed successfully."
                    else:
                        fallback_content = "Task completed successfully."
                except Exception as e:
                    logging.warning(f"[STREAM] Could not get artifacts for fallback: {e}")
                    fallback_content = "Task completed successfully."

            content_parts.append(format_agent_output(fallback_content))

        content = "\n\n".join(content_parts) if content_parts else "Task completed successfully."
        logging.info(f"[STREAM] Content length: {len(content)} chars")

        from .message_storage import save_message

        metadata = {"plots": plots} if plots else None
        save_message(session_id, "ai", content, metadata=metadata)
        logging.info(f"[STREAM] Message saved to storage")

        # [REPORT UI TRIGGER] Extract report_id from action outputs for final response
        report_id = None
        for output in action_outputs:
            if '"event": "report_generated"' in output:
                try:
                    json_match = re.search(r'\{.*"event":\s*"report_generated".*\}', output, re.DOTALL)
                    if json_match:
                        report_data = json.loads(json_match.group(0))
                        report_id = report_data.get("report_id")
                except:
                    pass

        final_json = {"type": "final_response", "response": content, "plots": plots}
        if report_id:
            final_json["report_id"] = report_id

        from src.api import session_store as api_session_store

        token_streaming = api_session_store.get(session_id, {}).get("token_streaming", True)

        if token_streaming:
            yield f"data: {json.dumps({'type': 'start_tokens'})}\n\n"
            async for token_event in _stream_brain_tokens(content, session_id):
                yield token_event
            yield f"data: {json.dumps({'type': 'end_tokens'})}\n\n"

        logging.info(f"[STREAM] Yielding final response to frontend...")
        yield f"data: {json.dumps(final_json)}\n\n"
        logging.info(f"[STREAM] Final response yielded successfully!")

    except Exception as final_error:
        logging.error(f"[STREAM] Error in final response processing: {final_error}")
        import traceback

        logging.error(f"[STREAM] Traceback:\n{traceback.format_exc()}")

        error_response = {"type": "final_response", "response": "Analysis completed.", "plots": []}
        yield f"data: {json.dumps(error_response)}\n\n"
