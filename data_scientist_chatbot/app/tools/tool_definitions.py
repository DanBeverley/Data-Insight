"""Tool definitions for agent system"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import List, Optional
from .report_generation_tool import generate_comprehensive_report, ReportGenerationInput


class CodeInput(BaseModel):
    code: str = Field(description="Raw Python code to execute in the sandbox. Must be valid Python syntax.")


class PatternInput(BaseModel):
    task_description: str = Field(description="Description of the data science task to retrieve patterns for")


class GraphQueryInput(BaseModel):
    query: str = Field(
        description="Natural language query about data relationships, feature lineage, or past analysis patterns"
    )


class LearningDataInput(BaseModel):
    query: str = Field(
        description="Query for execution history or patterns - e.g., 'show successful visualization code' or 'get recent analysis patterns'"
    )


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for external domain knowledge or current information")


class ZipArtifactsInput(BaseModel):
    artifact_ids: List[str] = Field(description="List of artifact IDs to include in the zip file")
    description: str = Field(default="", description="Description for the zip archive")


class CodingTask(BaseModel):
    task_description: str = Field(description="Clear description of the coding task for the specialized coding agent")


class LoadModelInput(BaseModel):
    model_type: Optional[str] = Field(
        default=None,
        description="Type of model to load (e.g., 'linear_regression', 'random_forest'). If not specified, loads most recent model.",
    )
    model_id: Optional[str] = Field(
        default=None, description="Specific model ID to load. Takes precedence over model_type."
    )


@tool(args_schema=CodeInput)
def python_code_interpreter(code: str) -> str:
    """
    Executes Python code in a stateful sandbox to perform data analysis,
    manipulation, and visualization. The sandbox maintains state, so you can
    define a variable or load data in one call and use it in the next.

    CRITICAL RULES:
    1. NEVER use `plt.show()` or `fig.show()`. They do not work.
    2. YOU MUST SAVE PLOTS EXPLICITLY:
       - Matplotlib: `plt.savefig('filename.png')`
       - Plotly: `fig.write_html('filename.html')`
    3. DO NOT wrap your code in a `def main():` function. Execute logic at the top level
       so variables remain available in the global scope for the next step.
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
    Delegate computational work to specialized coding agent. Use when the user needs:
    - Data analysis, visualization, or statistical exploration
    - Data preprocessing (cleaning, encoding, imputation, scaling)
    - Model training and evaluation
    - Complete workflows (preprocess + train + evaluate)

    The coding agent can generate multiple artifacts (processed datasets, models, visualizations).
    For modeling tasks, consider data quality - delegate preprocessing if needed.

    DO NOT use for:
    - Conversational exchanges or general questions
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


class ResearchTaskInput(BaseModel):
    query: str = Field(description="The research question or topic to investigate deeply")
    time_budget_minutes: int = Field(
        default=10, description="Time budget in minutes for the research (5-60 minutes recommended)"
    )


@tool(args_schema=ResearchTaskInput)
def delegate_research_task(query: str, time_budget_minutes: int = 10) -> str:
    """
    Delegate a deep research task to the specialized Research Brain.

    Use when the user explicitly requests thorough research, or when the query
    requires exploring multiple subtopics, gathering diverse sources, and
    synthesizing comprehensive findings.

    The Research Brain will:
    - Decompose the query into sub-questions
    - Search and explore each subtopic iteratively
    - Accumulate findings with source citations
    - Return structured results for synthesis

    Args:
        query: The research question to investigate
        time_budget_minutes: How long to spend researching (default 10 minutes)
    """
    return "This is a placeholder. Research delegation happens in the graph node."


class SaveToKnowledgeInput(BaseModel):
    content: str = Field(description="Text content to save to the knowledge store")
    source_name: str = Field(description="Name/label for this knowledge item (e.g., 'Climate Research 2026')")


class QueryKnowledgeInput(BaseModel):
    query: str = Field(description="Search query to find relevant knowledge")
    k: int = Field(default=5, description="Number of results to retrieve")


class IngestFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to ingest (from uploads directory)")


@tool(args_schema=SaveToKnowledgeInput)
def save_to_knowledge(content: str, source_name: str) -> str:
    """
    Save content to the session's knowledge store for future retrieval.

    Use when you have gathered valuable information (research findings,
    extracted insights, user-provided context) that should be remembered
    for this conversation session.

    Args:
        content: The text content to store
        source_name: A descriptive name for this knowledge item
    """
    return "This is a placeholder. Knowledge saving happens in the graph node."


@tool(args_schema=QueryKnowledgeInput)
def query_knowledge(query: str, k: int = 5) -> str:
    """
    Search the knowledge store for relevant information.

    Use BEFORE starting research to check if you already have relevant
    information stored. Also use when the user asks about previously
    discussed topics.

    Args:
        query: What to search for
        k: How many results to return (default 5)
    """
    return "This is a placeholder. Knowledge query happens in the graph node."


@tool(args_schema=IngestFileInput)
def ingest_file_to_knowledge(file_path: str) -> str:
    """
    Add an uploaded file to the knowledge store.

    Use when the user explicitly asks to "add this file to knowledge" or
    "remember this document". The file content will be extracted and indexed.

    Args:
        file_path: Path to the file (relative to uploads directory)
    """
    return "This is a placeholder. File ingestion happens in the graph node."


@tool
def zip_artifacts(artifact_ids: str, description: str = "") -> str:
    """
    Package multiple artifacts into a single downloadable zip file.

    Args:
        artifact_ids: Comma-separated artifact IDs (e.g., 'id1,id2,id3')
        description: Description for the zip archive
    """
    return "This is a placeholder. Zip creation happens in the graph node."


@tool
def load_trained_model(model_type: str = "", model_id: str = "") -> str:
    """
    Load a previously trained model from object storage into the sandbox for reuse.

    Args:
        model_type: Type identifier (e.g., 'linear_regression_model'). Loads most recent if empty.
        model_id: Specific model ID. Overrides model_type if provided.

    Returns:
        Path to the loaded model file in the sandbox
    """
    return "This is a placeholder. Model loading happens in the graph node."


class DatasetExplorerInput(BaseModel):
    folder: Optional[str] = Field(default=None, description="Filter by folder name (e.g., 'stocks', 'etfs')")
    extension: Optional[str] = Field(default=None, description="Filter by file extension (e.g., '.csv', '.json')")


class FilePathInput(BaseModel):
    file_path: str = Field(description="Relative path to the file within the uploaded dataset")


class FileCombineInput(BaseModel):
    file_pattern: str = Field(description="Glob pattern to match files (e.g., 'stocks/*.csv', '*.json')")


@tool
def inspect_dataset() -> str:
    """
    See what files and structure exist in the uploaded dataset.
    Use this to understand what data is available before analyzing.

    Returns JSON with:
    - Total files and size
    - File types present
    - Folder structure
    - Sample file listing

    Perfect for multi-file uploads, ZIP archives, or complex folder structures.
    """
    return "This is a placeholder. Dataset inspection happens in the graph node."


@tool
def list_files(folder: str = "", extension: str = "") -> str:
    """
    List files in the uploaded dataset. Optionally filter by folder or extension.

    Args:
        folder: Only show files in this folder (e.g., 'stocks', 'root')
        extension: Only show files with this extension (e.g., '.csv', '.json')

    Returns JSON with file paths, names, and sizes.
    """
    return "This is a placeholder. File listing happens in the graph node."


@tool(args_schema=FilePathInput)
def load_file(file_path: str) -> str:
    """
    Load a specific file from the uploaded dataset intelligently.

    Tries to load as CSV first, falls back to text if needed.
    Automatically loads into session dataframe if successful.

    Args:
        file_path: Relative path to file (e.g., 'stocks/AAPL.csv', 'metadata.csv')

    Returns preview of loaded data and confirmation it's ready for analysis.
    """
    return "This is a placeholder. File loading happens in the graph node."


@tool(args_schema=FileCombineInput)
def combine_files(file_pattern: str) -> str:
    """
    Combine multiple files matching a pattern into single dataframe.

    Useful for analyzing groups of related files together.
    Adds 'source_file' column to track origin.

    Args:
        file_pattern: Glob pattern (e.g., 'stocks/*.csv' to combine all stock data)

    Returns combined dataframe info and loads it into session.
    """
    return "This is a placeholder. File combination happens in the graph node."
