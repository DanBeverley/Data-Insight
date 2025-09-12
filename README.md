# Data Insight

AI-powered data analysis platform with automated ML pipelines and intelligent dataset profiling.

## Current Status

**Active Features:**
- Dataset-aware AI agent with conversational data analysis
- Session-persistent visualization generation (matplotlib/seaborn)
- Comprehensive data profiling (anomalies, quality, semantic analysis)
- Multi-chart support (heatmaps, scatter plots, bar charts, bubble charts)
- E2B sandbox code execution for secure Python analysis

**Architecture:**
- FastAPI + Flask dual backend with LangGraph agent system
- Hybrid data profiler integrating 6+ intelligence modules
- Persistent session management with stateful code execution
- Smart graph termination preventing recursive tool calls

## Core Functionality

### Data Intelligence & Analysis
- **Semantic Data Profiling**: Detects 16 semantic types (email, phone, currency, datetime, etc.) with confidence scoring
- **Domain Detection**: Automatic classification into business domains (finance, healthcare, e-commerce, etc.)
- **Relationship Discovery**: Identifies primary/foreign keys, correlations, and data dependencies
- **Data Quality Assessment**: Comprehensive validation with anomaly detection and drift monitoring

### Feature Engineering & Selection
- **Automated Feature Generation**: Uses FeatureTools for time-based features, aggregations, and transformations
- **Intelligent Feature Selection**: Multi-method selection (statistical, model-based, correlation-based)
- **Missing Value Intelligence**: Advanced imputation strategies based on semantic types
- **Categorical Grouping**: ML-powered categorical variable optimization

### Model Selection & Training
- **Intelligent Algorithm Selection**: Automated model recommendation based on dataset characteristics
- **Hyperparameter Optimization**: Automated tuning with multiple optimization strategies
- **Performance Validation**: Cross-validation with comprehensive metrics
- **Support for**: Classification, regression, clustering, time-series, and basic NLP

### MLOps Integration
- **Pipeline Orchestration**: 15-stage production pipeline with error recovery and checkpointing
- **Model Deployment**: Automated deployment with version control and rollback capabilities
- **Monitoring System**: Performance tracking, drift detection, and alert management
- **Auto-scaling**: Load-based scaling for deployed models

### Security & Privacy
- **PII Detection**: Automatic identification and masking of sensitive data
- **Privacy Protection**: Multiple anonymization techniques (k-anonymity, differential privacy)
- **Compliance Checking**: GDPR/CCPA validation with violation reporting
- **Access Control**: Role-based permissions and audit logging

### Explainability & Bias Detection
- **Model Explanations**: SHAP/LIME-based local and global explanations
- **Bias Assessment**: Fairness metrics across demographic groups
- **Trust Scoring**: Model reliability and calibration metrics
- **Business Insights**: Automated insight generation from model results

## Technical Architecture

### Application Structure
- **FastAPI Backend** (`src/api.py`): REST API with 20+ endpoints for data processing and ML operations
- **Streamlit Interface** (`src/app.py`): Interactive web UI for data science workflows  
- **RobustPipelineOrchestrator** (`src/core/pipeline_orchestrator.py`): Production-grade pipeline with error recovery and caching
- **Modern Web Dashboard** (`static/dashboard.html`): Professional frontend with real-time progress tracking

### Key Modules
- **Intelligence Engine** (`src/intelligence/`): Data profiling, domain detection, semantic analysis
- **Data Quality** (`src/data_quality/`): Validation, anomaly detection, drift monitoring, quality assessment
- **Feature Engineering** (`src/feature_selector/`): Automated generation and intelligent selection
- **Model Selection** (`src/model_selection/`): Algorithm recommendation, hyperparameter optimization, performance validation
- **MLOps** (`src/mlops/`): Deployment automation, monitoring, scaling, version control
- **Security** (`src/security/`): PII detection, privacy protection, compliance management
- **Explainability** (`src/explainability/`): Model interpretation, bias detection, trust metrics

### Pipeline Execution Flow
1. **Data Ingestion & Validation** - Multi-format support with error handling
2. **Intelligence Profiling** - Semantic type detection and domain classification
3. **Quality Assessment** - Comprehensive data quality scoring and issue identification
4. **Data Cleaning** - Automated preprocessing based on intelligence insights
5. **Feature Engineering** - Context-aware feature generation and selection
6. **Model Selection & Training** - Intelligent algorithm selection with hyperparameter tuning
7. **Performance Validation** - Cross-validation and metric computation
8. **Explainability Analysis** - Model interpretation and insight generation
9. **Bias & Fairness Assessment** - Multi-dimensional fairness evaluation
10. **Security & Privacy Scanning** - PII detection and protection measures
11. **Compliance Verification** - Regulatory requirement validation
12. **Deployment Preparation** - Model packaging and deployment automation

## API Endpoints

### Data Operations
- `POST /api/upload` - File upload with automatic profiling and validation
- `POST /api/ingest-url` - Data ingestion from web URLs
- `GET /api/data/{session_id}/preview` - Dataset preview and basic statistics
- `GET /api/data/{session_id}/columns` - Column information and metadata

### Pipeline Execution
- `POST /api/data/{session_id}/process` - Execute full ML pipeline with configuration
- `GET /api/data/{session_id}/status` - Real-time pipeline execution status
- `POST /api/data/{session_id}/recover` - Pipeline error recovery and resume
- `GET /api/data/{session_id}/logs` - Detailed execution logs and debug information

### Intelligence & Analysis
- `GET /api/data/{session_id}/profile` - Comprehensive data profiling results
- `GET /api/data/{session_id}/eda` - Exploratory data analysis report
- `GET /api/data/{session_id}/feature-recommendations` - AI-powered feature suggestions
- `GET /api/data/{session_id}/relationship-graph` - Data relationship visualization

### Results & Artifacts
- `GET /api/data/{session_id}/results` - Complete pipeline results and metrics
- `GET /api/data/{session_id}/download/{artifact_type}` - Download processed data, models, reports
- `GET /api/data/{session_id}/model-info` - Model performance and configuration details

### Learning & Feedback
- `POST /api/learning/feedback` - User feedback for adaptive learning system
- `GET /api/learning/recommendations` - Personalized workflow recommendations

## Installation

1. **Clone repository**
```bash
git clone <repository-url>
cd Data-Insight
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run application**
```bash
python run_app.py
```

4. **Access interface** - `http://localhost:8000`

## Usage

### Basic Workflow
1. **Data Upload**: Upload CSV/Excel files or provide URLs for data ingestion
2. **Automatic Profiling**: System performs semantic analysis and quality assessment
3. **Pipeline Configuration**: Select ML task type (classification, regression, clustering, etc.)
4. **Feature Engineering**: Enable automated feature generation and selection
5. **Model Training**: Automated algorithm selection and hyperparameter optimization
6. **Results Review**: Access model performance, explanations, and quality reports
7. **Artifact Download**: Download processed data, trained models, and analysis reports

### Advanced Features
- **Interactive Dashboard**: Real-time pipeline monitoring with progress tracking
- **Relationship Analysis**: Visualize data relationships and correlation networks
- **Quality Assessment**: Comprehensive data quality scoring with issue recommendations
- **Adaptive Learning**: System learns from previous executions to improve recommendations

## Configuration Options

### Task Types
- **Classification**: Binary and multi-class prediction tasks
- **Regression**: Continuous value prediction
- **Clustering**: Unsupervised pattern discovery
- **Time Series**: Forecasting with temporal data
- **NLP**: Basic text analysis and processing

### Feature Engineering
- **Automated Generation**: Time-based features, aggregations, polynomial features
- **Intelligent Selection**: Statistical, model-based, and correlation-based methods
- **Domain-Specific**: Context-aware features based on detected business domain

### Security & Privacy Options
- **PII Detection**: Automatic identification of sensitive data fields
- **Data Masking**: Multiple anonymization techniques available
- **Compliance Checking**: GDPR/CCPA validation with detailed reports
- **Access Control**: Session-based permissions and audit logging

## Supported Data

### Input Formats
- **File Types**: CSV, Excel (.xlsx, .xls), Parquet, JSON
- **Data Sources**: Direct file upload, web URLs, database connections
- **Size Limits**: Configurable based on system resources

### Data Types
- **Tabular Data**: Structured datasets with mixed column types
- **Time Series**: Sequential data with temporal patterns
- **Text Data**: Unstructured text for basic NLP processing
- **Mixed Types**: Datasets combining numerical, categorical, and text data

## Output Artifacts

### Data Products
- **Processed Dataset**: Cleaned data with generated features
- **Feature Metadata**: Detailed feature engineering documentation
- **Quality Report**: Data quality assessment with improvement recommendations

### Models & Analysis
- **Trained Models**: Optimized ML models with hyperparameter configurations
- **Performance Metrics**: Comprehensive evaluation including cross-validation results
- **Explainability Reports**: Model interpretation with feature importance and SHAP analysis

### Business Intelligence
- **Domain Insights**: Business context analysis and recommendations
- **Relationship Maps**: Data dependency and correlation visualizations
- **Executive Summary**: High-level findings and actionable insights

## Current Status

**Production-Ready ML Automation Platform**
- ✅ Complete pipeline orchestration with error recovery
- ✅ Intelligent data profiling and quality assessment
- ✅ Automated feature engineering and model selection  
- ✅ MLOps integration with deployment capabilities
- ✅ Security and privacy protection features
- ✅ Explainability and bias detection
- ✅ Modern web interface with API access
- ✅ Adaptive learning system