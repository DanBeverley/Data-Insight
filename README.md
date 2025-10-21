# Data-Insight

Multi-agent AI system for conversational data analysis with self-learning capabilities and intelligent execution orchestration.

## Demo
<p align = "center">
<video src="https://github.com/user-attachments/assets/209b6418-e65f-4893-b568-5bba87fedf6a" controls width="720"></video>
</p>

## Core Capabilities

### 1. Mixture of Experts Architecture
LangGraph-based multi-agent system with specialized roles:

**Router Agent** (`phi3:3.8b`) - Fast query classification with complexity analysis
- Analyzes task complexity (1-10 scale) using LLM reasoning
- Routes to optimal execution strategy: `direct`, `standard`, or `collaborative`
- Sub-second response for simple queries

**Brain Agent** (`gpt-oss:120b`) - Business reasoning and interpretation
- Generates insights from execution results
- Delegates coding tasks to Hands agent
- Synthesizes multi-step workflows

**Hands Agent** (`qwen3-coder:480b`) - Code generation and execution
- Generates Python code in E2B sandbox
- Self-corrects failures (max 3 attempts with LLM diagnosis)
- Automatic plot/model detection and artifact storage

### 2. Self-Learning System
Progressive session memory that improves over time:
- **Turn tracking**: Records every user request → code → outcome
- **Pattern learning**: Categorizes successful and failed approaches
- **Context injection**: Provides relevant history to agents for better code generation
- **Persistent checkpoints**: SQLite-based conversation state (LangGraph)

### 3. Self-Correction Execution
LLM-driven error recovery without manual intervention:
```python
Attempt 1: Code fails → Extract error
         ↓
LLM diagnoses issue and generates fix
         ↓
Attempt 2: Fixed code executes
         ↓
Success or repeat (max 3 attempts)
```

### 4. Intelligent Training Decisions
Hybrid CPU/GPU routing based on dataset characteristics:
- **Fast path**: Obvious cases (<1M or >100M data points)
- **Profiling**: Borderline cases sampled and profiled 
- **Multi-cloud**: Azure GPU, AWS GPU, or E2B CPU execution

### 5. Privacy-Aware Data Profiling
Multi-factor PII detection with intelligent sensitivity scoring:
- **Entropy-based analysis**: Shannon entropy for uniqueness detection
- **Statistical profiling**: Cardinality, uniqueness, distribution analysis
- **Risk assessment**: Privacy score, re-identification risk, compliance recommendations
- **User consent flow**: Detected PII triggers protection dialog

### 6. Knowledge Graph Integration
Session-aware relationship mapping:
- **Execution lineage**: Dataset → Code → Models → Artifacts
- **Feature importance**: Tracks which features drive predictions
- **Cross-session patterns**: Discovers recurring analysis workflows
- **Neo4j/PostgreSQL**: Dual backend support

### 7. Artifact Management
Automatic tracking and categorization:
- **Auto-detection**: Plots (`PLOT_SAVED:`), Models (`MODEL_SAVED:`)
- **Categorization**: Smart detection (extension + description-based)
- **Blob storage**: Azure Blob for models, local filesystem for plots
- **Metadata registry**: SQLite tracking with SHA256 checksums
- **Hover previews**: 250x250px thumbnails for images, metadata for models

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Router  │ (Complexity Analysis)
                    │ Agent   │ phi3:3.8b
                    └────┬────┘
                         │
           ┌─────────────┼─────────────┐
           │                           │
      ┌────▼────┐               ┌─────▼─────┐
      │  Brain  │◄──────────────┤   Hands   │
      │  Agent  │   Delegates   │   Agent   │
      │ 120b    │   ─────────►  │  480b     │
      └────┬────┘               └─────┬─────┘
           │                           │
           │                      ┌────▼────────┐
           │                      │ E2B Sandbox │
           │                      │ Code Exec   │
           │                      └────┬────────┘
           │                           │
           │                      ┌────▼────────┐
           │                      │ Self-Correct│
           │                      │ (3 retries) │
           │                      └────┬────────┘
           │                           │
           └───────────────────────────┤
                                       │
                              ┌────────▼─────────┐
                              │ Session Memory   │
                              │ Knowledge Graph  │
                              │ Artifact Tracker │
                              └──────────────────┘
```

## Tech Stack

**Backend**
- FastAPI (async web framework)
- LangGraph (agent orchestration with checkpointing)
- LangChain (LLM abstraction)
- Ollama (local + cloud LLM inference)
- E2B Code Interpreter (sandboxed execution)
- SQLAlchemy (ORM)

**Databases**
- PostgreSQL (primary storage)
- Neo4j (knowledge graph)
- SQLite (sessions, checkpoints, artifacts)
- Azure Blob Storage (model registry)

**Frontend**
- Vanilla JavaScript (no framework)
- Server-Sent Events (streaming responses)
- Three.js (background visualization)

**LLMs**
- Router: `phi3:3.8b-mini-128k-instruct-q4_K_M` (local)
- Brain: `gpt-oss:120b-cloud` (cloud)
- Hands: `qwen3-coder:480b-cloud` (cloud)

## Project Structure

```
Data-Insight/
├── data_scientist_chatbot/          # Multi-agent orchestration
│   └── app/
│       ├── agent.py                 # Agent state and execution
│       ├── core/
│       │   ├── graph_builder.py     # LangGraph workflow
│       │   ├── router.py            # Routing logic
│       │   ├── complexity_analyzer.py  # Task complexity LLM
│       │   ├── self_correction.py   # Auto-retry with LLM fixes
│       │   ├── session_memory.py    # Progressive learning
│       │   ├── training_decision.py # CPU/GPU routing
│       │   ├── model_manager.py     # Model switching
│       │   └── agent_factory.py     # Agent initialization
│       ├── tools/
│       │   ├── executor.py          # Sandbox execution
│       │   └── tool_definitions.py  # Tool schemas
│       └── utils/
│           └── format_parser.py     # Response formatting
│
├── src/                             # Core application
│   ├── api.py                       # FastAPI app (20+ endpoints)
│   ├── routers/
│   │   ├── data_router.py           # Data upload, profiling, artifacts
│   │   └── session_router.py        # Session CRUD, persistence
│   ├── api_utils/
│   │   ├── artifact_tracker.py      # Artifact categorization
│   │   ├── streaming_service.py     # SSE streaming
│   │   └── upload_handler.py        # File ingestion + privacy
│   ├── intelligence/
│   │   ├── hybrid_data_profiler.py  # Comprehensive profiling
│   │   ├── data_profiler.py         # Semantic type detection
│   │   ├── semantic_profiler.py     # Pattern recognition
│   │   └── relationship_discovery.py # Correlation analysis
│   ├── data_quality/
│   │   ├── anomaly_detector.py      # Multi-layer outlier detection
│   │   ├── quality_assessor.py      # 6-dimension scoring
│   │   └── drift_monitor.py         # Distribution changes
│   ├── security/
│   │   ├── privacy_engine.py        # PII detection + k-anonymity
│   │   └── compliance_manager.py    # GDPR/HIPAA checks
│   ├── knowledge_graph/
│   │   └── service.py               # Neo4j/PostgreSQL abstraction
│   ├── storage/
│   │   ├── blob_service.py          # Azure Blob integration
│   │   └── model_registry.py        # Model versioning
│   └── database/
│       ├── models.py                # SQLAlchemy ORM
│       └── migrations.py            # Schema management
│
├── static/                          # Frontend SPA
│   ├── index.html                   # Main UI
│   ├── styles.css                   # Dark theme
│   ├── js/
│   │   ├── app.js                   # Application controller
│   │   ├── session-manager.js       # Session persistence
│   │   ├── artifact-storage.js      # Artifact dropdown
│   │   ├── chat-interface.js        # Chat UI
│   │   └── blackhole.js             # Three.js visualization
│   ├── plots/                       # Generated visualizations
│   └── models/                      # Downloaded model files
│
├── tests/
│   ├── unit/                        # Component tests
│   ├── integration/                 # Multi-component tests
│   │   ├── test_agent_flows/        # Brain-Hands collaboration
│   │   ├── test_sandbox_execution/  # GPU training tests
│   │   └── test_knowledge_graph/    # Graph operations
│   ├── e2e/                         # End-to-end scenarios
│   └── chaos/                       # Failure injection
│
├── docker/                          # Containerization
├── .github/workflows/               # CI/CD pipelines
├── config.yaml                      # Agent configuration
├── requirements.txt                 # Python dependencies
└── run_app.py                       # Application entrypoint
```

## Installation

```bash
# Clone repository
git clone <repository-url> && cd Data-Insight

# Install dependencies
pip install -r requirements.txt

# Pull LLM models (requires Ollama)
ollama pull phi3:3.8b-mini-128k-instruct-q4_K_M
ollama pull gpt-oss:120b-cloud
ollama pull qwen3-coder:480b-cloud

# Configure environment
cp .env.example .env
# Edit .env with:
#   - DATABASE_URL (PostgreSQL)
#   - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
#   - E2B_API_KEY
#   - AZURE_STORAGE_CONNECTION_STRING

# Run database migrations
alembic upgrade head

# Start application
python run_app.py
```

Access at `http://localhost:8000`

## Usage

**Basic Workflow**
1. Upload dataset (CSV, Excel, Parquet, JSON)
2. Privacy consent dialog (if PII detected)
3. Ask questions in natural language
4. Agent executes code in sandbox
5. View results, plots, and trained models

**Example Queries**
- "Analyze correlation between price and area"
- "Build a random forest to predict housing prices"
- "Create a scatter plot with regression line"
- "Compare accuracy of 3 different classifiers"

**Key Endpoints**
- `POST /api/sessions/new` - Create session
- `POST /api/upload` - Upload dataset with profiling
- `GET /api/agent/chat-stream` - SSE streaming chat
- `GET /api/sessions/{id}/messages` - Load history
- `GET /api/data/{id}/artifacts` - List generated files

## Development

**Testing**
```bash
# Unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=html

# Integration tests (requires E2B API key)
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v
```

**Code Quality**
```bash
black --line-length 120 .
pylint src/ --fail-under=7.0
mypy src/ --ignore-missing-imports
```

**Docker**
```bash
docker compose -f docker/docker-compose.ci.yml up --build
```

### Session Persistence
- **Sessions**: Stored in `sessions_metadata.db` with titles
- **Messages**: LangGraph checkpoints in `context.db` (SQLite)
- **Artifacts**: Tracked in `artifact_storage.json` with metadata
- **Models**: Versioned in Azure Blob + local registry

### Artifact Hover Previews


## Configuration

`config.yaml` - Agent model assignments:
```yaml
router_model: "phi3:3.8b-mini-128k-instruct-q4_K_M"
brain_model: "gpt-oss:120b-cloud"
hands_model: "qwen3-coder:480b-cloud"
```

`data_scientist_chatbot/app/core/model_manager.py` - Temperature settings:
- Router: 0.0 (deterministic)
- Brain: 0.6 (creative)
- Hands: 0.0 (for code generation)

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Run code quality checks
6. Submit pull request

Branch protection on `main` requires:
- Passing CI/CD checks
- Code review approval
- Test coverage >80%

## License

See LICENSE file for details.
