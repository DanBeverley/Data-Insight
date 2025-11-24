import json
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, Any, AsyncGenerator
from langchain_core.messages import HumanMessage
from .helpers import create_agent_input, create_workflow_status_context
from .agent_response import extract_agent_response


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


async def stream_agent_chat(
    message: str, session_id: str, agent, session_store: Dict[str, Any], status_agent_runnable=None, session_agents=None
) -> AsyncGenerator[str, None]:
    from .message_storage import save_message

    save_message(session_id, "human", message)

    thinking_mode = session_store.get(session_id, {}).get("thinking_mode", False)

    try:
        logging.info(f"[STREAMING] Using .astream() method (async streaming) for session {session_id}")
        try:
            async for event_data in _stream_fallback(message, session_id, agent, status_agent_runnable):
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

                async for event_data in _stream_fallback(message, session_id, agent, status_agent_runnable):
                    yield event_data
            else:
                raise
        except Exception as stream_error:
            logging.error(f"[STREAMING] .astream() failed for session {session_id}: {stream_error}")

            if hasattr(agent, "astream_events"):
                logging.warning(f"[STREAMING] Falling back to astream_events() for session {session_id}")
                try:
                    async for event_data in _stream_with_events(
                        message, session_id, agent, status_agent_runnable, thinking_mode
                    ):
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
    message: str, session_id: str, agent, status_agent_runnable, thinking_mode: bool = False
) -> AsyncGenerator[str, None]:
    yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...'})}\n\n"
    await asyncio.sleep(0.01)

    config = {"configurable": {"thread_id": session_id}}
    input_data = create_agent_input(message, session_id)

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

                if status_agent_runnable:
                    status_msg = await _generate_status(status_agent_runnable, workflow_context, event, current_node)
                    if status_msg:
                        yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"

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

                if status_agent_runnable:
                    status_msg = await _generate_status(status_agent_runnable, workflow_context, event, current_node)
                    if status_msg:
                        yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"
                        await asyncio.sleep(0.1)

            elif event_name == "on_tool_end":
                tool_name = event.get("name", "")
                tool_output = event_data.get("output", "")

                for tool_info in workflow_context["tool_calls"]:
                    if tool_info["tool"] == tool_name and tool_info["status"] == "starting":
                        tool_info["status"] = "completed"
                        break

                workflow_context["current_action"] = f"completed {tool_name}"

                if status_agent_runnable:
                    status_msg = await _generate_status(status_agent_runnable, workflow_context, event, current_node)
                    if status_msg:
                        yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"
                        await asyncio.sleep(0.1)

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

                if node_name == "__start__":
                    logging.info(f"[STREAM_EVENTS] Graph execution completed (__start__ ended)")
                    final_response = event_data.get("output")
                    break
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

    from .message_storage import save_message

    metadata = {"plots": plots} if plots else None
    save_message(session_id, "ai", content, metadata=metadata)

    clear_cancellation(session_id)

    yield f"data: {json.dumps({'type': 'final_response', 'response': content, 'plots': plots})}\n\n"


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


async def _stream_fallback(message: str, session_id: str, agent, status_agent_runnable) -> AsyncGenerator[str, None]:
    from langchain_core.messages import HumanMessage

    yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing agent...'})}\n\n"
    await asyncio.sleep(0.01)

    config = {"configurable": {"thread_id": session_id}}
    final_response = None
    all_messages = []
    final_brain_response = None
    action_outputs = []

    logging.info(f"[STREAM] Starting async streaming for session {session_id}")
    logging.info(f"[STREAM] Agent type: {type(agent)}")
    logging.info(f"[STREAM] Config: {config}")

    original_user_message = HumanMessage(content=message)
    current_state = {"messages": [original_user_message], "session_id": session_id}

    try:
        logging.info(f"[STREAM] Calling agent.astream (async streaming)...")

        import time

        start_time = time.time()
        max_stream_time = 300  # 5 minutes maximum

        async for event in agent.astream(create_agent_input(message, session_id), config=config):
            # Check timeout
            if time.time() - start_time > max_stream_time:
                logging.error(f"[STREAM] Stream exceeded max time ({max_stream_time}s), forcing termination")
                break

            # Log ALL events to debug
            logging.debug(f"[STREAM] Received event: {list(event.keys())}")

            # Check if this is a special END event (LangGraph marks completed graphs)
            if "__end__" in event:
                logging.info(f"[STREAM] Graph reached END, completing stream")
                final_response = {"messages": all_messages}
                break

            for node_name, node_data in event.items():
                if node_name in ["router", "brain", "hands", "parser", "action"]:
                    logging.info(f"[STREAM] Node completed: {node_name}")
                    messages = node_data.get("messages", []) if node_data else []
                    if messages and isinstance(messages, list):
                        all_messages.extend(messages)
                        current_state["messages"] = all_messages

                        if node_name == "hands":
                            logging.info(f"[STREAM] Hands node completed, messages count: {len(messages)}")
                            if node_data.get("artifacts"):
                                logging.info(f"[STREAM] Hands generated {len(node_data.get('artifacts'))} artifacts")
                            logging.info(f"[STREAM] Waiting for next node (should route to brain)...")

                        if node_name == "brain":
                            logging.info(f"[STREAM] Brain node executing")
                            last_brain_msg = messages[-1]
                            brain_content = (
                                str(last_brain_msg.content)
                                if hasattr(last_brain_msg, "content")
                                else str(last_brain_msg)
                            )
                            if not (hasattr(last_brain_msg, "tool_calls") and last_brain_msg.tool_calls):
                                final_brain_response = brain_content
                                logging.info(f"[STREAM] Brain generated final response, length: {len(brain_content)}")

                        elif node_name == "action":
                            last_action_msg = messages[-1]
                            if hasattr(last_action_msg, "type") and last_action_msg.type == "tool":
                                action_content = str(last_action_msg.content)
                                if action_content and action_content.strip():
                                    action_outputs.append(action_content)

                    if status_agent_runnable:
                        status_msg = await _generate_fallback_status(
                            status_agent_runnable, message, node_name, session_id
                        )
                        if status_msg:
                            yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"

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

        # Fallback: Extract from messages if content_parts is still empty
        if not content_parts:
            logging.warning(f"[STREAM] No content_parts built, using fallback extraction")
            final_message = messages[-1] if messages else None
            fallback_content = (
                final_message.content
                if final_message and hasattr(final_message, "content")
                else "Task completed successfully."
            )
            content_parts.append(format_agent_output(fallback_content))
            logging.info(f"[STREAM] Used fallback content ({len(str(fallback_content))} chars)")

        content = "\n\n".join(content_parts) if content_parts else "Task completed successfully."
        logging.info(f"[STREAM] Content length: {len(content)} chars")

        from .message_storage import save_message

        metadata = {"plots": plots} if plots else None
        save_message(session_id, "ai", content, metadata=metadata)
        logging.info(f"[STREAM] Message saved to storage")

        final_json = {"type": "final_response", "response": content, "plots": plots}
        logging.info(f"[STREAM] Yielding final response to frontend...")
        yield f"data: {json.dumps(final_json)}\n\n"
        logging.info(f"[STREAM] Final response yielded successfully!")

    except Exception as final_error:
        logging.error(f"[STREAM] Error in final response processing: {final_error}")
        import traceback

        logging.error(f"[STREAM] Traceback:\n{traceback.format_exc()}")

        error_response = {"type": "final_response", "response": "Analysis completed.", "plots": []}
        yield f"data: {json.dumps(error_response)}\n\n"


async def _generate_status(status_agent_runnable, workflow_context, event, current_node):
    try:
        status_context = create_workflow_status_context(workflow_context, event)
        from data_scientist_chatbot.app.agent import get_status_agent_prompt

        status_prompt_template = get_status_agent_prompt()
        status_formatted = status_prompt_template.format(
            current_agent=status_context.get("current_agent", "unknown"),
            user_goal=status_context.get("user_goal", "processing"),
        )
        status_response = await asyncio.wait_for(status_agent_runnable.ainvoke(status_formatted), timeout=10.0)
        return status_response.content.strip()
    except asyncio.TimeoutError:
        print(f"ERROR: Status agent timeout for node={current_node}")
        return None
    except Exception as e:
        print(f"ERROR: Status generation failed for node={current_node}: {type(e).__name__}: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return None


async def _generate_fallback_status(status_agent_runnable, message, node_name, session_id):
    try:
        workflow_context = {
            "current_agent": node_name,
            "current_action": "processing",
            "user_goal": message,
            "session_id": session_id,
            "execution_progress": {},
            "tool_calls": [],
        }
        fake_event = {"event": "on_chain_start", "name": node_name}
        status_context = create_workflow_status_context(workflow_context, fake_event)

        from data_scientist_chatbot.app.agent import get_status_agent_prompt

        status_prompt_template = get_status_agent_prompt()
        status_formatted = status_prompt_template.format(
            current_agent=status_context.get("current_agent", "unknown"),
            user_goal=status_context.get("user_goal", "processing"),
        )
        status_response = await asyncio.wait_for(status_agent_runnable.ainvoke(status_formatted), timeout=10.0)
        return status_response.content.strip() if hasattr(status_response, "content") else str(status_response).strip()
    except asyncio.TimeoutError:
        print(f"ERROR: Fallback status agent timeout for node={node_name}")
        return None
    except Exception as e:
        print(f"ERROR: Fallback status agent failed for node={node_name}: {type(e).__name__}: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return None
