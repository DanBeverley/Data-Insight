# Data Insight

Multi-agent AI platform for conversational data analysis with secure code execution and knowledge graph integration.

## Overview

Data Insight combines LangGraph-based multi-agent orchestration with comprehensive data intelligence for interactive data analysis. The system features specialized AI agents, secure Python sandbox execution, and persistent knowledge graph storage.

## Status

**Operational**
- Multi-agent conversation system (Router, Brain, Hands agents)
- Secure Python execution in E2B sandbox with stateful memory
- Comprehensive data profiling with semantic type detection
- Knowledge graph integration (Neo4j/PostgreSQL)
- Real-time streaming responses via SSE
- Web search integration
- Privacy protection and PII detection
- Modern web interface with dark mode

**In Development**
- CI/CD pipeline with GitHub Actions
- Comprehensive test coverage (unit, integration, e2e)
- LLM-generated adversarial test scenarios
- MLOps monitoring and deployment automation

## Architecture

**Multi-Agent System**
- Router Agent: Fast query classification and delegation
- Brain Agent: Business reasoning and insight generation
- Hands Agent: Code generation and execution
- Status Agent: Real-time progress updates

**Tech Stack**
- Backend: FastAPI, LangChain, LangGraph, SQLAlchemy
- Databases: PostgreSQL, Neo4j, SQLite
- LLM: Ollama with Qwen models
- Sandbox: E2B Code Interpreter
- Frontend: Vanilla JS, Three.js, SSE

**Core Components**
- `src/api.py`: FastAPI application (20+ endpoints)
- `data_scientist_chatbot/`: Agent system with graph builder
- `src/intelligence/`: Data profiling and semantic analysis
- `src/knowledge_graph/`: Graph database integration
- `src/database/`: ORM models and migrations
- `src/security/`: Privacy protection and compliance
- `static/`: SPA frontend with real-time chat

## Features

**Data Analysis**
- Semantic type detection (email, phone, currency, datetime, etc.)
- Business domain classification (finance, healthcare, e-commerce)
- Data quality assessment with anomaly detection
- Relationship discovery (keys, correlations)
- Missing value pattern analysis

**Code Execution**
- Secure E2B sandbox environment
- Stateful execution (variables persist)
- Automatic plot generation and download
- Support for pandas, numpy, scikit-learn, matplotlib

**Knowledge Management**
- Graph-based analysis history
- Relationship mapping between datasets and models
- Feature importance ranking
- Execution lineage tracking

**Agent Tools**
- `python_code_interpreter`: Execute code in sandbox
- `web_search`: External information retrieval
- `delegate_coding_task`: Transfer to coding agent
- `knowledge_graph_query`: Query analysis graph
- `access_learning_data`: Historical patterns

## Installation

```bash
git clone <repository-url> && cd Data-Insight
pip install -r requirements.txt
ollama pull qwen2.5:1.5b qwen2.5:14b qwen3-coder:480b-cloud
cp .env.example .env  # Edit with credentials
alembic upgrade head
python run_app.py  # Access at http://localhost:8000
```

**Environment**: `DATABASE_URL`, `NEO4J_URI`, `E2B_API_KEY`
**Config** (`config.yaml`): Router (qwen2.5:1.5b), Brain (qwen2.5:14b), Hands (qwen3-coder:480b-cloud)

## Usage

**Workflow**: Create session → Upload data (CSV/Excel/Parquet/JSON) → Ask questions → Agent executes code → View results

**Example**: "Analyze correlation between price and area", "Create scatter plot", "Detect anomalies"

**Key Endpoints**: `/api/sessions/create`, `/api/upload`, `/api/chat/{session_id}`, `/api/data/{session_id}/profile`, `/api/download/{session_id}/{filename}`

## Development

**Testing**: `pytest tests/unit/ -v -m unit --cov=src` or `pytest tests/ -v --cov=src --cov-report=html`

**Quality**: `black --line-length 120 .` | `pylint src/ --fail-under=7.0` | `mypy src/`

**Docker**: `docker compose -f docker/docker-compose.ci.yml up --build`

## CI/CD Pipeline

GitHub Actions workflow with:
1. Code quality checks (Black, Pylint, MyPy)
2. Unit tests with coverage
3. Integration tests
4. E2E tests
5. Codecov reporting

Branch protection on `main` requires passing checks and code review.

## Project Structure

```
Data-Insight/
├── data_scientist_chatbot/    # Multi-agent system
├── src/                       # Core application
│   ├── api.py                 # FastAPI application
│   ├── intelligence/          # Data profiling
│   ├── database/              # ORM and migrations
│   ├── knowledge_graph/       # Graph integration
│   └── security/              # Privacy protection
├── static/                    # Frontend SPA
├── tests/                     # Test suite
├── docker/                    # Docker configs
└── .github/workflows/         # CI/CD
```

## Roadmap

**Phase 1** (Current): Testing infrastructure and CI/CD
**Phase 2** (Q2 2025): MLOps enhancement and monitoring
**Phase 3** (Q3 2025): Advanced feature engineering
**Phase 4** (Q4 2025): Enterprise features and multi-tenancy

## Contributing

1. Fork repository
2. Create feature branch
3. Ensure tests pass and code quality checks
4. Submit pull request
