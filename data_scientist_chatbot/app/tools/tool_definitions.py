"""Tool definitions for agent system"""
from pydantic import BaseModel, Field
from langchain.tools import tool
from typing import List, Optional

class CodeInput(BaseModel):
    code: str = Field(description="Raw Python code to execute in the sandbox. Must be valid Python syntax.")

class PatternInput(BaseModel):
    task_description: str = Field(description="Description of the data science task to retrieve patterns for")

class GraphQueryInput(BaseModel):
    query: str = Field(description="Natural language query about data relationships, feature lineage, or past analysis patterns")

class LearningDataInput(BaseModel):
    query: str = Field(description="Query for execution history or patterns - e.g., 'show successful visualization code' or 'get recent analysis patterns'")

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for external domain knowledge or current information")

class ZipArtifactsInput(BaseModel):
    artifact_ids: List[str] = Field(description="List of artifact IDs to include in the zip file")
    description: Optional[str] = Field(default=None, description="Optional description for the zip archive")

class CodingTask(BaseModel):
    task_description: str = Field(description="Clear description of the coding task for the specialized coding agent")

class LoadModelInput(BaseModel):
    model_type: Optional[str] = Field(default=None, description="Type of model to load (e.g., 'linear_regression', 'random_forest'). If not specified, loads most recent model.")
    model_id: Optional[str] = Field(default=None, description="Specific model ID to load. Takes precedence over model_type.")

@tool(args_schema=CodeInput)
def python_code_interpreter(code: str) -> str:
    """
    Executes Python code in a stateful sandbox to perform data analysis,
    manipulation, and visualization. The sandbox maintains state, so you can
    define a variable or load data in one call and use it in the next.
    Always use this tool to inspect, transform, and visualize data.
    When creating plots, they will be saved automatically. Make sure to
    inform the user that you have generated a plot.
    """
    return "This is a placeholder. Execution happens in the graph node."

@tool(args_schema=PatternInput)
def retrieve_historical_patterns(task_description: str) -> str:
    """
    Retrieve proven successful code patterns from past sessions for similar tasks.
    Call this before performing common data science tasks like EDA, visualization,
    modeling, or data cleaning to leverage learned patterns and best practices.

    Args:
        task_description: Description of the task you're about to perform
                         (e.g., 'visualization', 'correlation_analysis', 'ml_modeling')
    """
    return "This is a placeholder. Pattern retrieval happens in the graph node."

@tool(args_schema=CodingTask)
def delegate_coding_task(task_description: str) -> str:
    """
    Delegate computational work to specialized coding agent. Use ONLY when the user has:
    - Uploaded a dataset and needs analysis, visualization, or modeling
    - Specific technical requests requiring data processing or machine learning
    - Questions that require code execution to answer properly

    DO NOT use for:
    - Conversational exchanges, greetings, or general questions
    - Requests when no dataset is available
    - Explanations that don't require computation
    """
    return "Delegation confirmed. Coding agent will execute this task."

@tool(args_schema=GraphQueryInput)
def knowledge_graph_query(query: str) -> str:
    """
    Query relationships between datasets, features, and analysis patterns from previous work.
    Use ONLY when you need to reference or compare with historical data analysis patterns.

    DO NOT use for:
    - General conversation or greetings
    - First-time questions about capabilities
    - Simple responses that don't require historical context

    Args:
        query: Natural language query about data relationships from past analyses
    """
    return "This is a placeholder. Graph query happens in the graph node."

@tool(args_schema=LearningDataInput)
def access_learning_data(query: str) -> str:
    """
    Access successful code patterns and execution strategies from previous sessions.
    Use ONLY when you need to optimize approach based on historical performance data.

    DO NOT use for:
    - Casual conversation or capability questions
    - Initial interactions without specific technical needs
    - General responses that don't require historical learning context

    Args:
        query: Description of specific learning patterns needed for current technical task
    """
    return "This is a placeholder. Learning data access happens in the graph node."

@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search the web for current market data, trends, or domain-specific context.

    Use for:
    - Current market trends or benchmarks ("Does this reflect current market trends?")
    - Industry-specific standards or regulations
    - Recent events or news affecting the data domain
    - Domain expertise beyond general data science knowledge

    Don't use for:
    - General data science or ML concepts (use existing knowledge)
    - Questions about uploaded datasets (analyze them directly)
    - Historical patterns (use access_learning_data instead)
    - Normal exploratory data analysis requests

    Args:
        query: Specific search query for external information
    """
    return "This is a placeholder. Web search happens in the graph node."

@tool(args_schema=ZipArtifactsInput)
def zip_artifacts(artifact_ids: List[str], description: Optional[str] = None) -> str:
    """
    Package multiple artifacts into a single downloadable zip file.
    Artifact IDs are provided in your context under "AVAILABLE ARTIFACTS".

    Examples:
    - "Zip those 3 correlation plots" → Check context for visualization artifact IDs → Call with matching IDs
    - "Package all visualizations" → Extract all visualization IDs from context → Call with those IDs
    - "Download the EDA outputs" → Identify EDA-related artifact IDs from context → Call with IDs

    Args:
        artifact_ids: List of artifact IDs from context (format: session_id_number)
        description: Optional description for the zip archive
    """
    return "This is a placeholder. Zip creation happens in the graph node."

@tool(args_schema=LoadModelInput)
def load_trained_model(model_type: Optional[str] = None, model_id: Optional[str] = None) -> str:
    """
    Load a previously trained model from object storage into the sandbox for reuse.
    Use this when you need to:
    - Make predictions with a model trained earlier in the session
    - Create visualizations using a trained model (e.g., decision boundaries, regression planes)
    - Compare new data against a trained model
    - Continue training from a saved checkpoint

    The model will be downloaded from cloud storage and uploaded to the sandbox,
    making it available for joblib.load(), pickle.load(), or framework-specific loading.

    Args:
        model_type: Type identifier of the model (e.g., 'linear_regression_model'). Loads most recent if not specified.
        model_id: Specific model ID for exact model selection. Overrides model_type if provided.

    Returns:
        Path to the loaded model file in the sandbox
    """
    return "This is a placeholder. Model loading happens in the graph node."
