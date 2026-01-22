# Quorvix

> An AI data science platform with multi-agent architecture, self-learning capabilities, deep research, and intelligent cloud GPU training.

<p align="center">
<video src="https://github.com/user-attachments/assets/209b6418-e65f-4893-b568-5bba87fedf6a" controls width="720"></video>
</p>

---

## ✨ Features

### 🤖 Multi-Agent Architecture (LangGraph)

| Agent | Role | Model |
|-------|------|-------|
| **Brain** | Strategic reasoning, task delegation, insight synthesis 
| **Hands** | Code generation, sandbox execution, artifact creation 
| **Verifier** | Quality control, approval-based retry logic, deliverable validation 
| **Research Brain** | Deep web research with multi-iteration exploration 
| **Reporter** | Automated report generation with visualizations and statistics

The system uses LangGraph for orchestration with persistent checkpoints, enabling conversation recovery and session continuity.

---

### 📊 Intelligent Data Profiling

- **Hybrid Profiler** - Statistical + semantic analysis with AI-generated insights
- **Domain Detection** - Automatic dataset categorization (finance, healthcare, e-commerce, etc.)
- **Feature Intelligence** - Correlation discovery, importance scoring, transformation suggestions
- **Relationship Discovery** - Cross-column and cross-dataset relationship mapping
- **Profile Caching** - SHA256-based caching for instant re-profiling

---

### 🛡️ Data Quality Engine

| Module | Capabilities |
|--------|--------------|
| **Quality Assessor** | 6-dimension scoring (completeness, accuracy, consistency, timeliness, validity, uniqueness) |
| **Anomaly Detector** | Multi-layer outlier detection (IQR, Z-score, Isolation Forest) |
| **Drift Monitor** | Distribution shift detection across time windows |
| **Missing Value Intelligence** | Pattern analysis and imputation recommendations |
| **Validator** | Schema validation and constraint checking |

---

### 🔒 Privacy & Security

- **PII Detection** - Multi-factor sensitivity scoring with entropy analysis
- **Compliance Engine** - GDPR/CCPA/HIPAA compliance recommendations
- **User Consent Flow** - Interactive dialog when sensitive data detected
- **K-Anonymity Assessment** - Re-identification risk scoring
- **Google OAuth** - Secure authentication with session management

---

### 🔍 Deep Research Mode

- Multi-iteration web research (configurable 5-60 minute time budgets)
- Automatic subtopic generation and source synthesis
- **Pausable/Resumable** - State persistence for interrupted research
- Real-time progress streaming with source tracking
- Integration with Tavily, Google CSE, Brave Search APIs

---

### 🧠 RAG Knowledge System

- Session-specific vector stores with semantic embeddings
- **15+ file format support**: CSV, Excel, JSON, Parquet, PDF, DOCX, HTML, EML, etc.
- Save analysis results, research findings, and insights for future queries
- Cross-session knowledge persistence
- Automatic chunking and embedding optimization

---

### ☁️ Cloud GPU Training

| Provider | GPU | Status |
|----------|-----|--------|
| **Google Cloud Vertex AI** | T4, V100, A100 | Primary |
| **Azure Machine Learning** | GPU clusters | Fallback |
| **AWS SageMaker** | GPU instances | Fallback |
| **E2B Sandbox** | CPU | Default |

**Intelligent Routing**:
- Complexity analysis based on data size and model type
- Automatic deep learning detection (PyTorch, TensorFlow, Keras)
- Profiling-based decision for borderline cases (1M-100M data points)

---

### 🔔 Notifications & Alerts

- **Natural language alerts**: *"Notify me if sales drops below $50k"*
- Scheduled condition checks via APScheduler with SQLite persistence
- Email notifications (SMTP) with configurable schedules
- In-app notification center with read/unread states

---

### 📈 Knowledge Graph

- **Execution Lineage**: Dataset → Code → Models → Artifacts
- **Feature Tracking**: Which features drive predictions
- **Cross-Session Patterns**: Recurring analysis workflow discovery
- Neo4j and PostgreSQL backend support

---

### 📝 Automated Reporting

- AI-generated executive summaries
- Interactive visualizations with Plotly
- PDF/HTML export with branded templates
- Artifact embedding (plots, tables, models)

---

## ⚡ Optimization & Performance

| Technique | Description |
|-----------|-------------|
| **SHA256 Profile Caching** | Instant re-profiling for unchanged datasets |
| **Dynamic Max Turns** | Agent iterations scale with task complexity (5-8 turns) |
| **Hybrid Profiling** | Statistical + AI-powered semantic analysis in parallel |
| **Lazy Loading** | Datasets loaded on-demand, not preloaded |
| **Artifact Streaming** | Real-time visualization delivery during analysis |
| **Connection Pooling** | Database connections reused across requests |

### Multi-Dataset Intelligence

- **Automatic Relationship Detection**: FK/PK discovery across tables
- **Join Cardinality Analysis**: 1:1, 1:N, N:M relationship mapping
- **Cross-Table Correlations**: Significant correlations across joined datasets
- **Semantic Layer**: Domain detection (finance, e-commerce, healthcare) with business term mapping



## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | FastAPI, LangGraph, LangChain, APScheduler, SQLAlchemy, ChromaDB |
| **Frontend** | React 18, TypeScript, Tailwind CSS, Shadcn/UI |
| **LLMs** | Google Gemini, Ollama |
| **Databases** | PostgreSQL, SQLite, Neo4j |
| **Storage** | Cloudflare R2, Azure Blob, Local filesystem |
| **Execution** | E2B Sandbox, GCP Vertex AI, Azure ML, AWS SageMaker |
| **Search** | DuckDuckGo, Google Custom Search, Brave Search |
| **Monitoring** | LangSmith |

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/DanBeverley/Data-Insight.git
cd Data-Insight
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run
python run_app.py
```

Access at `http://localhost:8000`

---

## 📁 Project Structure

```
Quorvix/
├── data_scientist_chatbot/          # Multi-agent orchestration
│   └── app/
│       ├── agents/                  # Brain, Hands, Verifier, Research, Reporter
│       ├── core/                    # Graph builder, training executor, model manager
│       ├── tools/                   # Tool definitions and execution
│       └── utils/                   # Helpers, knowledge store, text processing
│
├── src/                             # Core backend
│   ├── api.py                       # FastAPI app (50+ endpoints)
│   ├── routers/                     # REST API endpoints
│   ├── intelligence/                # Profiling, clustering, feature intelligence
│   ├── data_quality/                # Quality assessment, anomaly detection, drift
│   ├── knowledge_graph/             # Neo4j/PostgreSQL graph service
│   ├── scheduler/                   # APScheduler + alert system
│   ├── notifications/               # Email + in-app notifications
│   ├── auth/                        # Google OAuth + session management
│   ├── connectors/                  # Database connectors (PostgreSQL, MySQL, etc.)
│   ├── storage/                     # R2, Azure Blob, local storage
│   ├── reporting/                   # Report generation engine
│   └── mlops/                       # Performance monitoring
│
├── frontend/                        # React + TypeScript
│   └── src/
│       ├── components/              # Chat, Reports, Artifacts UI
│       └── lib/                     # API clients, utilities
│
└── data/                            # Local storage
    ├── sessions/                    # Session data
    ├── uploads/                     # Uploaded datasets
    └── scheduler/                   # Alert and job databases
```

---

## 🔑 Environment Variables

```env
# LLM Providers
OPENAI_API_KEY=
GOOGLE_API_KEY=

# Execution
E2B_API_KEY=

# Search
TAVILY_API_KEY=

# Database
DATABASE_URL=postgresql://...

# Storage
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=

# Auth
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

# Notifications
SMTP_HOST=smtp.gmail.com
SMTP_USER=
SMTP_PASSWORD=

# Cloud GPU (optional)
GCP_PROJECT_ID=
AZURE_STORAGE_CONN_STR=
SAGEMAKER_ROLE=
```

---

## 📖 Usage Examples

```
# Data Analysis
"Show me the correlation matrix and identify the top 5 predictive features"

# Model Training
"Train a gradient boosting model to predict churn with hyperparameter tuning"

# Deep Research
"Research the latest trends in federated learning for healthcare (20 min)"

# Alerts
"Send me an email at john@example.com when daily revenue drops below $10k"

# Report Generation
"Generate an executive summary report of this quarter's sales performance"
```

---

## 🎥 Demo Videos

### Web Search & Analysis
[![Web Search Demo](https://img.youtube.com/vi/KWoL7fQmvKw/maxresdefault.jpg)](https://www.youtube.com/watch?v=KWoL7fQmvKw)

### Database Connector
[![Database Connector Demo](https://img.youtube.com/vi/fRPgBizcclY/maxresdefault.jpg)](https://www.youtube.com/watch?v=fRPgBizcclY)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/DanBeverley">Dan Beverley</a></b>
</p>
