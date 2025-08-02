# DataInsight AI

Automated data preprocessing and feature engineering platform with intelligent data understanding capabilities.

## Current Implementation

### Core Features
- **Intelligent Data Profiling**: Detects 27 semantic data types beyond basic dtypes
- **Domain Detection**: Identifies 10 business domains (ecommerce, finance, healthcare, etc.)  
- **Relationship Discovery**: Discovers 7 types of column relationships with statistical validation
- **Automated Feature Engineering**: Context-aware feature generation and selection
- **Robust Pipeline**: Production-grade orchestration with error recovery
- **Adaptive Learning**: Learns from execution patterns to improve recommendations

### Architecture

**Backend (FastAPI)**
- REST API with comprehensive intelligence endpoints
- Session-based data management
- Real-time pipeline monitoring
- Adaptive learning integration

**Frontend (HTML/CSS/JS)**
- Modern minimalistic interface (black/grey/green)
- Interactive intelligence tabs (semantic types, relationships, recommendations)
- Real-time pipeline status monitoring
- Relationship graph visualization with D3.js

**Intelligence System**
- `IntelligentDataProfiler`: Semantic type detection and column profiling
- `DomainDetector`: Business domain identification with confidence scoring
- `RelationshipDiscovery`: Statistical relationship detection between columns
- `AdvancedFeatureIntelligence`: Context-aware feature engineering recommendations
- `AdaptiveLearningSystem`: Pattern learning from execution feedback

**Pipeline Architecture**
- `RobustPipelineOrchestrator`: Production-grade pipeline with error handling
- Stage-based execution (ingestion → profiling → cleaning → feature engineering → modeling)
- Automatic fallback to legacy orchestrator
- Manual recovery triggers

### API Endpoints

**Core Workflow**
- `POST /api/upload` - Upload with intelligent profiling
- `POST /api/ingest-url` - URL ingestion with intelligence
- `POST /api/data/{session_id}/process` - Robust pipeline execution
- `GET /api/data/{session_id}/download/{artifact}` - Download results

**Intelligence Features**
- `POST /api/data/{session_id}/profile` - On-demand deep profiling
- `GET /api/data/{session_id}/feature-recommendations` - AI feature suggestions
- `POST /api/data/{session_id}/apply-features` - Apply recommendations
- `GET /api/data/{session_id}/relationship-graph` - Interactive graph data

**Monitoring & Learning**
- `GET /api/data/{session_id}/pipeline-status` - Real-time monitoring
- `POST /api/data/{session_id}/pipeline-recovery` - Manual recovery
- `POST /api/learning/feedback` - Learning system feedback
- `GET /api/learning/insights` - Learning system status

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python run_app.py
```

3. Access the interface at `http://localhost:8000`

## Usage

1. **Upload Data**: Upload CSV/Excel files or provide URLs
2. **Review Intelligence**: Examine semantic types, domain detection, and relationships
3. **Configure Task**: Select ML task type and advanced options
4. **Process**: Execute robust pipeline with real-time monitoring
5. **Download Results**: Access processed data and intelligence reports

## Technical Details

**Supported Data Types**
- CSV, Excel files
- URL data sources
- Time series data
- Text/NLP data

**ML Tasks**
- Classification
- Regression  
- Clustering
- Time-series forecasting
- NLP text analysis

**Intelligence Capabilities**
- 27 semantic types (email, currency, keys, temporal, geographic, etc.)
- 10 business domains with pattern matching
- 7 relationship types (correlations, associations, dependencies, hierarchies)
- Domain-specific feature engineering strategies
- Statistical validation and confidence scoring

**Export Formats**
- Processed data (CSV)
- Enhanced data with engineered features (CSV)
- ML pipeline (Joblib)
- Intelligence reports (JSON)
- Pipeline metadata (JSON)

## Development Status

**Completed**
- ✅ Phase 1: Core data processing and basic automation
- ✅ Phase 2A: Enhanced pipelines (time-series, NLP, automated FE)
- ✅ Phase 2B: Intelligence & robustness (semantic understanding, domain detection)
- ✅ Phase 3A-1: API integration with intelligence features
- ✅ Phase 3A-2: Dedicated intelligence endpoints and monitoring
- ✅ Phase 3A-3: Frontend integration with intelligence UI

**Current Focus**
- Enhanced JavaScript functionality for intelligence features
- Comprehensive testing and validation
- Performance optimization

The system operates as an intelligent data preprocessing platform that automatically understands data context and applies appropriate transformations without manual intervention.