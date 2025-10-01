"""Tool execution logic and helper functions"""

import time
import sys
import os
import logging

logger = logging.getLogger(__name__)

try:
    from ..tools import execute_python_in_sandbox
    from ..context_manager import ContextManager
    from ..performance_monitor import PerformanceMonitor
except ImportError:
    try:
        from tools import execute_python_in_sandbox
        from context_manager import ContextManager
        from performance_monitor import PerformanceMonitor
    except ImportError:
        raise ImportError("Could not import required modules")

context_manager = ContextManager()
performance_monitor = PerformanceMonitor()


import json
import re
import builtins

def execute_tool(tool_name: str, tool_args: dict, session_id: str) -> str:
    """Shared tool executor for both main graph and subgraph"""
    if tool_name == 'python_code_interpreter':
        code = tool_args["code"]
        start_time = time.time()

        from src.mlops.monitoring import PerformanceMonitor, MetricType
        mlops_monitor = PerformanceMonitor()

        result = execute_python_in_sandbox(code, session_id)

        execution_time = time.time() - start_time
        mlops_monitor.record_metric(
            deployment_id=session_id,
            metric_type=MetricType.LATENCY,
            value=execution_time * 1000,
            metadata={"code_length": len(code), "tool": "python_code_interpreter"}
        )

        try:
            from src.learning.adaptive_system import AdaptiveLearningSystem
            adaptive_system = AdaptiveLearningSystem()
            adaptive_system.capture_execution_data(
                session_id=session_id,
                code=code,
                execution_time=execution_time,
                success=result.get('success', False),
                output=result.get('stdout', ''),
                error=result.get('stderr', ''),
                context={'tool': 'python_code_interpreter'}
            )
        except Exception as adaptive_error:
            logger.warning(f"Adaptive learning capture failed: {adaptive_error}")

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="code_execution_time",
            value=execution_time,
            context={"code_length": len(code), "success": True}
        )

        stdout_content = result.get('stdout', '')
        stderr_content = result.get('stderr', '')
        plots = result.get('plots', [])

        analysis_results = {}
        for line in stdout_content.split('\n'):
            if line.startswith('ANALYSIS_RESULTS:'):
                try:
                    analysis_results = json.loads(line.replace('ANALYSIS_RESULTS:', ''))
                except:
                    pass

        execution_result = {
            "status": "ERROR" if stderr_content else "SUCCESS",
            "analysis_data": analysis_results,
            "plots": plots,
            "raw_output": stdout_content
        }

        if hasattr(builtins, '_session_store') and session_id in builtins._session_store:
            builtins._session_store[session_id]['last_execution'] = execution_result

        output_parts = []
        if stdout_content:
            output_parts.append(stdout_content)
        if stderr_content:
            output_parts.append(f"Error: {stderr_content}")
        if plots:
            output_parts.append(f"\n📊 Generated {len(plots)} visualization(s): {plots}")

        content = "\n".join(output_parts) if output_parts else "Code executed successfully, but no output was generated."
        return content

    elif tool_name == 'retrieve_historical_patterns':
        task_description = tool_args["task_description"]
        return retrieve_historical_patterns_logic(task_description, session_id)
    elif tool_name == 'knowledge_graph_query':
        query = tool_args["query"]
        return knowledge_graph_query_logic(query, session_id)
    elif tool_name == 'access_learning_data':
        query = tool_args["query"]
        return access_learning_data_logic(query, session_id)
    else:
        raise ValueError(f"Unknown tool '{tool_name}'")


def generate_business_insights(code: str, output: str, session_id: str) -> str:
    try:
        import builtins
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

        from src.insights.business_translator import BusinessTranslator, StakeholderType
        from src.insights.narrative_generator import NarrativeGenerator, ReportTone

        if hasattr(builtins, '_session_store') and session_id in builtins._session_store:
            data_profile = builtins._session_store[session_id].get('data_profile')
            if data_profile:
                translator = BusinessTranslator()
                metrics = translator.extract_business_metrics_from_analysis(code, output, data_profile)
                stakeholder_view = translator.generate_stakeholder_view(metrics, StakeholderType.BUSINESS_ANALYST)
                narrative = NarrativeGenerator().generate_insight_narrative(stakeholder_view, ReportTone.BUSINESS)
                return narrative[:200] + "..." if len(narrative) > 200 else narrative
    except:
        pass
    return ""


def generate_explainability_insights(code: str, session_id: str) -> str:
    try:
        import builtins
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

        from src.explainability.bias_detector import BiasDetector

        if hasattr(builtins, '_session_store') and session_id in builtins._session_store:
            df = builtins._session_store[session_id].get('dataframe')
            if df is not None:
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    detector = BiasDetector(sensitive_attributes=categorical_cols[:1])
                    bias_results = detector.detect_data_bias(df)
                    if bias_results:
                        return f"Data analysis: {bias_results[0].description}"
    except:
        pass
    return ""


def knowledge_graph_query_logic(query: str, session_id: str) -> str:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
        from src.knowledge_graph.service import KnowledgeGraphService, SessionDataStorage

        storage = SessionDataStorage()
        graph_data = storage.get_all_data()

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="graph_query",
            value=1.0,
            context={"query": query}
        )

        if graph_data:
            return f"Knowledge graph data found: {graph_data}"
        else:
            return "No relevant historical patterns or relationships found in the knowledge graph."

    except Exception as e:
        return f"Graph query failed: {str(e)}"


def access_learning_data_logic(query: str, session_id: str) -> str:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
        from src.learning.adaptive_system import AdaptiveLearningSystem

        adaptive_system = AdaptiveLearningSystem()
        execution_history = adaptive_system.get_execution_history(success_only=True)

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="learning_data_access",
            value=1.0,
            context={"query": query}
        )

        if execution_history:
            return f"Learning data found: {len(execution_history)} successful executions. Recent examples: {execution_history[-5:]}"
        else:
            return "No learning data available yet. Execute some code first to build learning history."

    except Exception as e:
        return f"Learning data access failed: {str(e)}"


def retrieve_historical_patterns_logic(task_description: str, session_id: str) -> str:
    try:
        patterns = context_manager.get_cross_session_patterns(
            pattern_type=task_description,
            limit=3
        )

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="pattern_retrieval",
            value=len(patterns),
            context={"task_description": task_description}
        )

        if not patterns:
            return f"No historical patterns found for '{task_description}'. Now execute the task using python_code_interpreter tool with standard best practices."

        pattern_summaries = []
        for pattern in patterns:
            summary = f"**{pattern['type']}** (confidence: {pattern['confidence']:.2f}, used {pattern['success_count']} times)"
            if pattern.get('data'):
                data_preview = str(pattern['data'])[:200]
                summary += f"\nPattern details: {data_preview}{'...' if len(str(pattern['data'])) > 200 else ''}"
            pattern_summaries.append(summary)

        result = f"Found {len(patterns)} proven patterns for '{task_description}':\n\n" + "\n\n".join(pattern_summaries)
        result += "\n\nNow use python_code_interpreter tool to implement the task, adapting these successful patterns."

        return result

    except Exception as e:
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="pattern_retrieval_error",
            value=1.0,
            context={"error": str(e), "task_description": task_description}
        )
        return f"Error retrieving patterns: {str(e)}"