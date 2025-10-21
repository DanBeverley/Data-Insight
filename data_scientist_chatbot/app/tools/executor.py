"""Tool execution logic and helper functions"""
import time
import sys
import os
import logging

logger = logging.getLogger(__name__)

try:
    import importlib.util
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    tools_file_path = os.path.join(parent_dir, 'tools.py')
    spec = importlib.util.spec_from_file_location("parent_tools_module", tools_file_path)
    parent_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_tools)

    execute_python_in_sandbox = parent_tools.execute_python_in_sandbox
    azure_gpu_train = parent_tools.azure_gpu_train
    aws_gpu_train = parent_tools.aws_gpu_train

    sys.path.insert(0, parent_dir)
    from context_manager import ContextManager
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    raise ImportError(f"Could not import required modules in executor.py: {e}")

context_manager = ContextManager()
performance_monitor = PerformanceMonitor()

import json
import re
import builtins

def format_training_result(result: dict) -> str:
    """Format training executor result into string for agent consumption"""
    env = result.get('execution_environment', 'unknown')
    reasoning = result.get('decision_reasoning', '')

    output_parts = []

    # Add environment info
    env_label = {
        'cpu': 'CPU (E2B Sandbox)',
        'gpu_azure': 'GPU (Azure ML)',
        'gpu_aws': 'GPU (AWS SageMaker)'
    }.get(env, env)

    output_parts.append(f"✓ Execution Environment: {env_label}")

    if reasoning:
        output_parts.append(f"ℹ Decision: {reasoning}")

    # Add main output
    if result.get('success'):
        if result.get('stdout'):
            output_parts.append(f"\n{result['stdout']}")
        if result.get('plots'):
            plots_str = ", ".join(result['plots'])
            output_parts.append(f"\n📊 Generated visualizations: {plots_str}")
    else:
        error = result.get('stderr', 'Unknown error')
        output_parts.append(f"\n❌ Error: {error}")

    return "\n".join(output_parts)

def execute_tool(tool_name: str, tool_args: dict, session_id: str) -> str:
    """Shared tool executor for both main graph and subgraph"""
    if tool_name == 'python_code_interpreter':
        code = tool_args.get("code", "")
        if not code:
            return f"Error: No code provided in tool arguments. Received args: {tool_args}"

        # Check if this is a training task - route to unified training executor
        try:
            from data_scientist_chatbot.app.core.training_executor import training_executor, should_use_training_executor

            if should_use_training_executor(code):
                logger.info(f"Training task detected, using unified training executor")
                result = training_executor.execute_training(
                    code=code,
                    session_id=session_id,
                    user_request=None,  # TODO: Pass user request from context
                    model_type=""
                )

                # Format result for consistency with sandbox execution
                return format_training_result(result)
        except ImportError as e:
            logger.warning(f"Could not import training_executor: {e}, falling back to standard execution")

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
    elif tool_name == 'web_search':
        query = tool_args["query"]
        return web_search_logic(query, session_id)
    elif tool_name == 'zip_artifacts':
        artifact_ids = tool_args["artifact_ids"]
        description = tool_args.get("description")
        return zip_artifacts_logic(artifact_ids, description, session_id)
    elif tool_name == 'load_trained_model':
        model_type = tool_args.get("model_type")
        model_id = tool_args.get("model_id")
        return load_trained_model_logic(model_type, model_id, session_id)
    elif tool_name == "aws_gpu_train":
        return aws_gpu_train(tool_args["code"], tool_args.get("session_id"))
    elif tool_name == "azure_gpu_train":
        return azure_gpu_train(tool_args["code"], tool_args.get("session_id"))
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
            context={"query": query})
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
            context={"query": query})

        if execution_history:
            return f"Learning data found: {len(execution_history)} successful executions. Recent examples: {execution_history[-5:]}"
        else:
            return "No learning data available yet. Execute some code first to build learning history."
    except Exception as e:
        return f"Learning data access failed: {str(e)}"

def web_search_logic(query: str, session_id: str) -> str:
    try:
        session_store = getattr(builtins, '_session_store', None)
        if session_store and session_id in session_store:
            web_search_enabled = session_store[session_id].get("web_search_enabled", False)
            if not web_search_enabled:
                return "Web search is currently disabled. Please enable it in settings to use this feature."

        from .web_search import search_web
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="web_search_used",
            value=1.0,
            context={"query": query})
        return search_web(query)
    except Exception as e:
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="web_search_error",
            value=1.0,
            context={"error": str(e), "query": query}
        )
        return f"Web search failed: {str(e)}"

def retrieve_historical_patterns_logic(task_description: str, session_id: str) -> str:
    try:
        patterns = context_manager.get_cross_session_patterns(
            pattern_type=task_description,
            limit=3)
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="pattern_retrieval",
            value=len(patterns),
            context={"task_description": task_description})
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


def zip_artifacts_logic(artifact_ids: list, description: str, session_id: str) -> str:
    try:
        import requests
        from pathlib import Path

        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
        from src.api_utils.artifact_tracker import artifact_tracker

        artifacts_data = artifact_tracker.get_session_artifacts(session_id)
        all_artifacts = artifacts_data.get('artifacts', [])
        selected_artifacts = [a for a in all_artifacts if a['artifact_id'] in artifact_ids]
        if not selected_artifacts:
            return "Error: None of the specified artifacts were found in the current session."

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="zip_artifacts_request",
            value=len(selected_artifacts),
            context={"artifact_count": len(selected_artifacts)})
        artifact_names = [a['filename'] for a in selected_artifacts]
        artifact_categories = {}
        for a in selected_artifacts:
            cat = a.get('category', 'other')
            artifact_categories[cat] = artifact_categories.get(cat, 0) + 1

        category_summary = ", ".join([f"{count} {cat}" for cat, count in artifact_categories.items()])

        zip_url = f"/api/data/{session_id}/artifacts/zip"

        result = f"Successfully prepared zip archive with {len(selected_artifacts)} artifact(s): {', '.join(artifact_names[:3])}"
        if len(artifact_names) > 3:
            result += f" and {len(artifact_names) - 3} more"

        result += f"\n\nContents: {category_summary}"
        result += f"\n\nThe zip file has been created and is ready for download. You can access it via the artifact storage panel or by visiting: {zip_url}"

        if description:
            result += f"\n\nDescription: {description}"

        return result
    except Exception as e:
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="zip_artifacts_error",
            value=1.0,
            context={"error": str(e)}
        )
        return f"Error creating zip file: {str(e)}"


def load_trained_model_logic(model_type: str, model_id: str, session_id: str) -> str:
    """
    Load a previously trained model from object storage and upload to sandbox
    """
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
        from src.storage.blob_service import BlobStorageService
        from src.storage.model_registry import ModelRegistryService
        from src.storage.model_loader import ModelLoader
        from src.database.service import get_database_service
        from src.config import settings

        storage_config = settings.get('object_storage', {})
        if not storage_config.get('enabled', False):
            return "Model persistence is not enabled. Please configure object_storage in config.yaml."

        # Initialize services
        blob_service = BlobStorageService(
            container_name=storage_config.get('container_name', 'datainsight-models')
        )
        db_service = get_database_service()
        registry = ModelRegistryService(db_service)
        loader = ModelLoader(blob_service, registry)

        # Get sandbox for this session
        import builtins
        session_sandboxes = getattr(builtins, '_persistent_sandboxes', {})

        if session_id not in session_sandboxes:
            return "No active sandbox found for this session. Please execute some code first to initialize the sandbox."

        sandbox = session_sandboxes[session_id]

        # Upload model to sandbox
        sandbox_path = loader.upload_model_to_sandbox(
            sandbox=sandbox,
            session_id=session_id,
            model_type=model_type if model_type else None,
            model_id=model_id if model_id else None
        )

        if not sandbox_path:
            # Try to find what models are available
            available_models = registry.list_session_models(session_id)

            if not available_models:
                return f"No trained models found for session {session_id}. Please train a model first."

            model_list = "\n".join([
                f"- {m['model_type']} (ID: {m['model_id']}, trained at {m['created_at']})"
                for m in available_models[:5]
            ])

            return f"Could not load the requested model. Available models:\n{model_list}\n\nUse the model_type or model_id to specify which model to load."

        filename = Path(sandbox_path).name

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="model_loaded",
            value=1.0,
            context={"model_type": model_type, "model_id": model_id}
        )

        return f"Model loaded successfully into sandbox at: {sandbox_path}\n\nYou can now use it in your code with:\n```python\nimport joblib\nmodel = joblib.load('{filename}')\n```"

    except Exception as e:
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="model_load_error",
            value=1.0,
            context={"error": str(e)}
        )
        return f"Error loading model: {str(e)}"