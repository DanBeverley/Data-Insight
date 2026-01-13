"""Tool execution nodes and utility functions."""

from typing import Dict, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from data_scientist_chatbot.app.core.state import GlobalState
from data_scientist_chatbot.app.core.workflow_types import WorkflowStage
from data_scientist_chatbot.app.core.logger import logger
from data_scientist_chatbot.app.tools import execute_tool
from data_scientist_chatbot.app.utils.helpers import has_tool_calls
from data_scientist_chatbot.app.utils.text_processing import parse_message_to_tool_call, sanitize_output
from data_scientist_chatbot.app.core.exceptions import ToolExecutionError


def parse_tool_calls(state: GlobalState) -> Dict[str, Any]:
    """Parse tool calls from the last message."""
    last_message = state["messages"][-1]
    parse_message_to_tool_call(last_message, "json")
    return {"messages": [last_message]}


def execute_tools_node(state: GlobalState) -> Dict[str, Any]:
    """Execute tool calls from the last message."""
    messages = state.get("messages", [])
    logger.info(f"[ACTION] execute_tools_node received {len(messages)} messages")
    msg_types = [f"{type(m).__name__}" for m in messages]
    logger.info(f"[ACTION] Message types: {msg_types}")

    if not messages:
        logger.warning("[ACTION] No messages in state")
        return {"messages": []}

    last_message = messages[-1]

    if not has_tool_calls(last_message):
        logger.debug("execute_tools: No tool calls found")
        return {"messages": messages}

    tool_calls = last_message.tool_calls or []
    logger.info(
        f"[ACTION] Found {len(tool_calls)} tool calls: {[tc.get('name', getattr(tc, 'name', 'unknown')) if isinstance(tc, dict) else tc.name for tc in tool_calls]}"
    )

    tool_responses = []
    for tool_call in tool_calls:
        session_id = state.get("session_id")

        if isinstance(tool_call, dict):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", tool_call.get("arguments", {}))
            tool_id = tool_call.get("id", f"call_{tool_name}")
        else:
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_id = tool_call.id

        logger.debug(f"execute_tools: Processing {tool_name} with args: {tool_args}")

        if not session_id:
            content = "Error: Session ID is missing."
        else:
            try:
                if tool_name == "generate_comprehensive_report":
                    logger.info(f"[REPORT] Triggering Newsroom Pipeline. Args: {tool_args}")
                    return {
                        "messages": [
                            ToolMessage(
                                content="[REPORT] Data analysis and report generation pipeline started.",
                                tool_call_id=tool_id,
                            )
                        ],
                        "current_agent": "action",
                        "workflow_stage": WorkflowStage.REPORTING.value,
                    }

                elif tool_name == "delegate_coding_task":
                    task_desc = tool_args.get("task_description", "")
                    logger.info(f"[ACTION] Delegating task: {task_desc}")
                    execute_tool(tool_name, tool_args, session_id)
                    return {
                        "messages": [ToolMessage(content="", tool_call_id=tool_id)],
                        "current_task_description": task_desc,
                        "workflow_stage": "delegating",
                        "current_agent": "action",
                    }

                elif tool_name == "web_search":
                    import asyncio
                    import builtins
                    from data_scientist_chatbot.app.tools.web_search import web_search as do_web_search

                    query = tool_args.get("query", "")
                    logger.info(f"[WEB_SEARCH] Executing search for: {query}")

                    search_config = {}
                    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                        search_config = builtins._session_store[session_id].get("search_config", {})
                        builtins._session_store[session_id]["search_status"] = {
                            "action": "searching",
                            "query": query,
                            "provider": search_config.get("provider", "duckduckgo"),
                        }
                    logger.info(f"[WEB_SEARCH] Using config: {search_config}")

                    try:
                        loop = asyncio.get_running_loop()
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, do_web_search(query, search_config))
                            content = future.result(timeout=30)
                    except RuntimeError:
                        content = asyncio.run(do_web_search(query, search_config))

                    import re
                    import json as json_lib

                    result_count = 0
                    json_match = re.search(r"<!-- SEARCH_DATA_JSON\n(.*?)\nSEARCH_DATA_END -->", content, re.DOTALL)
                    if json_match:
                        try:
                            search_data = json_lib.loads(json_match.group(1))
                            result_count = search_data.get("result_count", 0)
                        except:
                            result_count = content.count("[Source]")

                    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                        builtins._session_store[session_id]["search_status"] = {
                            "action": "complete",
                            "query": query,
                            "result_count": result_count,
                        }

                    logger.info(f"[WEB_SEARCH] Result length: {len(content)} chars, count: {result_count}")

                elif tool_name == "delegate_research_task":
                    import asyncio
                    import builtins
                    from data_scientist_chatbot.app.agents.research_brain import ResearchBrain

                    query = tool_args.get("query", "")
                    time_budget = tool_args.get("time_budget_minutes", 10)
                    logger.info(f"[RESEARCH] Starting deep research: '{query}' ({time_budget}m budget)")

                    search_config = {}
                    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                        search_config = builtins._session_store[session_id].get("search_config", {})

                    research_brain = ResearchBrain(
                        session_id=session_id, time_budget_minutes=time_budget, search_config=search_config
                    )

                    try:
                        loop = asyncio.get_running_loop()
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, research_brain.research(query))
                            findings = future.result(timeout=time_budget * 60 + 30)
                    except RuntimeError:
                        findings = asyncio.run(research_brain.research(query))

                    content = findings.to_summary()
                    logger.info(
                        f"[RESEARCH] Complete: {len(findings.findings)} findings, " f"{findings.iterations} iterations"
                    )

                elif tool_name == "save_to_knowledge":
                    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

                    store = KnowledgeStore(session_id)
                    doc_id = store.add_document(
                        content=tool_args.get("content", ""),
                        source="agent",
                        source_name=tool_args.get("source_name", "Agent Knowledge"),
                    )
                    content = f"Saved to knowledge store with ID: {doc_id}" if doc_id else "Failed to save"
                    logger.info(f"[KNOWLEDGE] Saved document: {doc_id}")

                elif tool_name == "query_knowledge":
                    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

                    store = KnowledgeStore(session_id)
                    results = store.query(tool_args.get("query", ""), k=tool_args.get("k", 5))
                    if results:
                        content = "KNOWLEDGE STORE RESULTS:\n"
                        for i, r in enumerate(results, 1):
                            content += (
                                f"\n{i}. [{r['metadata'].get('source_name', 'Unknown')}]\n{r['content'][:500]}...\n"
                            )
                    else:
                        content = "No relevant knowledge found."
                    logger.info(f"[KNOWLEDGE] Query returned {len(results)} results")

                elif tool_name == "save_file_to_knowledge":
                    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore
                    import builtins

                    store = KnowledgeStore(session_id)
                    file_path = tool_args.get("file_path", "")
                    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                        uploads_dir = builtins._session_store[session_id].get("uploads_dir", "")
                        if uploads_dir:
                            file_path = f"{uploads_dir}/{file_path}"
                    doc_id = store.add_file(file_path, source_type="user_upload")
                    content = f"File ingested to knowledge with ID: {doc_id}" if doc_id else "Failed to ingest file"
                    logger.info(f"[KNOWLEDGE] Ingested file: {file_path} -> {doc_id}")

                elif tool_name == "list_datasets":
                    from data_scientist_chatbot.app.utils.dataset_registry import DatasetRegistry

                    registry = DatasetRegistry(session_id)
                    datasets = registry.list_all()
                    if datasets:
                        content = "AVAILABLE DATASETS:\n"
                        for ds in datasets:
                            status = "✓ profiled" if ds.profiled else "○ not profiled"
                            content += f"  • {ds.filename} ({ds.rows}×{ds.columns}) [{status}]\n"
                        content += "\nNote: Text documents (.txt, .pdf, .docx) are in Knowledge Store. Use query_knowledge() to search them."
                    else:
                        content = (
                            "No tabular datasets in this session. Use query_knowledge() to search uploaded documents."
                        )
                    logger.info(f"[DATASET] Listed {len(datasets)} datasets")

                elif tool_name == "load_dataset":
                    from data_scientist_chatbot.app.utils.dataset_registry import DatasetRegistry
                    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore
                    from data_scientist_chatbot.app.tools import execute_python_in_sandbox

                    filename = tool_args.get("filename", "")
                    registry = DatasetRegistry(session_id)
                    info = registry.get(filename)

                    if not info:
                        content = f"Dataset '{filename}' not found. Use list_datasets() to see available."
                    else:
                        df = registry.load_dataframe(filename)
                        if df is None:
                            content = f"Failed to load dataset '{filename}'."
                        else:
                            # Load into sandbox
                            csv_data = df.to_csv(index=False)
                            load_code = f"""
import pandas as pd
from io import StringIO
csv_data = '''{csv_data}'''
df = pd.read_csv(StringIO(csv_data))
df.to_csv('dataset.csv', index=False)
print(f"Loaded {filename}: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
"""
                            execute_python_in_sandbox(load_code, session_id)

                            # Lazy profiling: run if not profiled yet
                            if not info.profiled:
                                try:
                                    from src.intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent

                                    profile = generate_dataset_profile_for_agent(df, {"filename": filename})

                                    # Save to RAG
                                    store = KnowledgeStore(session_id)
                                    profile_content = f"# Dataset Profile: {filename}\n"
                                    profile_content += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                                    profile_content += f"Columns: {', '.join(df.columns.tolist())}\n"
                                    if hasattr(profile, "dataset_insights"):
                                        profile_content += f"Domain: {profile.dataset_insights.detected_domains}\n"
                                    if hasattr(profile, "quality_assessment"):
                                        score = profile.quality_assessment.get("overall_score", "N/A")
                                        profile_content += f"Quality Score: {score}\n"

                                    store.add_document(profile_content, "dataset_profile", f"Profile: {filename}")
                                    registry.mark_profiled(filename)
                                    logger.info(f"[DATASET] Profiled and saved to RAG: {filename}")
                                except Exception as e:
                                    logger.warning(f"[DATASET] Profiling failed: {e}")

                            content = f"Dataset '{filename}' loaded ({df.shape[0]}×{df.shape[1]}). Available as 'df' and saved as 'dataset.csv'."
                    logger.info(f"[DATASET] Load: {filename}")

                elif tool_name == "get_dataset_info":
                    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

                    filename = tool_args.get("filename", "")
                    store = KnowledgeStore(session_id)
                    results = store.query(f"Dataset Profile: {filename}", k=1)
                    if results:
                        content = f"DATASET INFO FOR {filename}:\n{results[0]['content']}"
                    else:
                        content = f"No profiling information found for '{filename}'. Load it first with load_dataset()."
                    logger.info(f"[DATASET] Get info: {filename}")

                elif tool_name == "python_code_interpreter":
                    logger.debug(f"Received clean code from Pydantic schema ({len(tool_args.get('code', ''))} chars)")
                    content = execute_tool(tool_name, tool_args, session_id)
                else:
                    content = execute_tool(tool_name, tool_args, session_id)
            except Exception as e:
                logger.exception(f"Tool execution failed for {tool_name}")
                raise ToolExecutionError(f"Execution failed in tool node: {tool_name}") from e

        logger.debug(f"execute_tools: Tool {tool_name} result ({len(str(content))} chars)")

        sanitized_content = sanitize_output(content)
        tool_responses.append(ToolMessage(content=sanitized_content, tool_call_id=tool_id))

    python_executions = state.get("python_executions") or 0
    for tc in tool_calls:
        name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        if name == "python_code_interpreter":
            python_executions += 1
            break

    result_state = {
        "messages": messages + tool_responses,
        "current_agent": "action",
        "python_executions": python_executions,
        "retry_count": 0,
        "last_agent_sequence": (state.get("last_agent_sequence") or []) + ["action"],
        "artifacts": state.get("artifacts") or [],
        "agent_insights": state.get("agent_insights") or [],
        "workflow_stage": "tool_execution",
    }

    for tool_call in tool_calls:
        name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        if name == "generate_comprehensive_report":
            logger.info("[WORKFLOW] Triggering Newsroom Pipeline (Action -> Analyst)")
            result_state["workflow_stage"] = WorkflowStage.REPORTING.value
        elif name == "delegate_coding_task":
            logger.info("[WORKFLOW] Delegating to Hands Agent")
            result_state["workflow_stage"] = "delegating"

    logger.info(
        f"[ACTION] Returning state with {len(result_state['messages'])} messages (added {len(tool_responses)} ToolMessages)"
    )
    return result_state


@tool
def submit_dashboard_insights(insights_json: str, session_id: str) -> str:
    """
    Submit structured insights for the dashboard.

    Args:
        insights_json: JSON string of insights array. Each item: {label, value, type (info/warning/success)}
        session_id: The current session ID.
    """
    import json
    import builtins

    try:
        insights = json.loads(insights_json)
        if not isinstance(insights, list):
            return "Error: insights_json must be a JSON array."

        if not hasattr(builtins, "_session_store"):
            builtins._session_store = {}
        if session_id not in builtins._session_store:
            builtins._session_store[session_id] = {}

        builtins._session_store[session_id]["insights"] = insights
        logger.info(f"[TOOL] Submitted {len(insights)} insights for session {session_id}")
        return f"Successfully submitted {len(insights)} insights to dashboard."
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"
    except Exception as e:
        return f"Error submitting insights: {e}"


@tool
def create_alert(
    name: str,
    metric_query: str,
    metric_name: str,
    condition: str,
    threshold: float,
    notification_email: str,
    session_id: str,
) -> str:
    """
    Create an alert to notify user when a metric condition is met.

    Args:
        name: Alert name (e.g., 'Low Sales Alert')
        metric_query: Python expression to evaluate on dataframe (e.g., 'df["sales"].sum()')
        metric_name: Human-readable metric name (e.g., 'Total Sales')
        condition: Comparison operator - lt, gt, eq, ne, lte, gte
        threshold: Threshold value to compare against
        notification_email: Email address to send notification
        session_id: Current session ID
    """
    try:
        from src.scheduler.service import get_alert_scheduler
        from src.scheduler.models import Alert

        scheduler = get_alert_scheduler()
        alert = Alert(
            name=name,
            session_id=session_id,
            metric_query=metric_query,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            notification_type="email",
            notification_target=notification_email,
        )
        created = scheduler.create_alert(alert)
        logger.info(f"[TOOL] Created alert: {name} for session {session_id}")
        return f"Alert '{name}' created. You will be notified at {notification_email} when {metric_name} {condition} {threshold}."
    except Exception as e:
        logger.error(f"[TOOL] Failed to create alert: {e}")
        return f"Failed to create alert: {e}"


@tool
def list_my_alerts(session_id: str) -> str:
    """
    List all alerts for the current session.

    Args:
        session_id: Current session ID
    """
    try:
        from src.scheduler.service import get_alert_scheduler

        scheduler = get_alert_scheduler()
        alerts = scheduler.get_alerts_by_session(session_id)

        if not alerts:
            return "No alerts configured for this session."

        lines = [f"Found {len(alerts)} alert(s):"]
        for a in alerts:
            status_icon = "✅" if a.status == "active" else "⏸️"
            lines.append(f"{status_icon} {a.name}: {a.metric_name} {a.condition} {a.threshold}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"[TOOL] Failed to list alerts: {e}")
        return f"Failed to list alerts: {e}"


@tool
def delete_alert(alert_id: str, session_id: str) -> str:
    """
    Delete an alert by ID.

    Args:
        alert_id: The alert ID to delete
        session_id: Current session ID (for validation)
    """
    try:
        from src.scheduler.service import get_alert_scheduler

        scheduler = get_alert_scheduler()
        alert = scheduler.get_alert(alert_id)

        if not alert:
            return f"Alert {alert_id} not found."
        if alert.session_id != session_id:
            return "Cannot delete alert from another session."

        scheduler.delete_alert(alert_id)
        return f"Alert '{alert.name}' deleted."
    except Exception as e:
        return f"Failed to delete alert: {e}"
