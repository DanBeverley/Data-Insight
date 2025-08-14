# Data Insight

Intelligent data processing platform with automated ML pipeline, explainability, and enterprise security.

## Features

### Intelligence & Automation
- **Intelligent Data Profiling**: 27 semantic data types, domain detection, relationship discovery
- **Automated Feature Engineering**: Context-aware feature generation and selection  
- **Model Selection**: Intelligent algorithm selection based on data characteristics
- **Pipeline Orchestration**: Production-grade execution with error recovery and caching

### Explainability & Trust
- **Model Explanations**: SHAP/LIME explanations with intelligent fallbacks
- **Bias Detection**: Multi-dimensional fairness assessment across demographic groups
- **Trust Metrics**: Reliability, consistency, robustness, and calibration scoring

### Security & Compliance
- **Data Protection**: PII detection, masking, and anonymization (k-anonymity, differential privacy)
- **Access Control**: Role-based permissions with session management and audit logging
- **Compliance**: GDPR/CCPA automation with violation detection and reporting
- **Privacy Engine**: Risk assessment and comprehensive privacy protection

## Architecture

### Core Components
- **Pipeline Orchestrator**: Stage-based execution with error handling and recovery
- **Intelligence Engine**: Data profiling, domain detection, relationship discovery
- **Feature Engineering**: Automated feature generation and intelligent selection
- **Model Selection**: Algorithm recommendation based on data characteristics
- **Explainability Engine**: SHAP/LIME explanations with business insights
- **Security Manager**: PII detection, data masking, access control
- **Compliance Manager**: GDPR/CCPA automation and violation monitoring
- **Privacy Engine**: Risk assessment and privacy-preserving transformations

### Pipeline Stages
1. **Data Ingestion** - Load and validate data sources
2. **Profiling** - Semantic analysis and domain detection  
3. **Quality Assessment** - Data quality scoring and anomaly detection
4. **Cleaning** - Intelligent data cleaning and preprocessing
5. **Feature Engineering** - Automated feature generation and selection
6. **Model Selection** - Algorithm selection and hyperparameter optimization
7. **Validation** - Performance assessment and quality metrics
8. **Explainability** - Model interpretation and business insights
9. **Bias Assessment** - Fairness evaluation across sensitive attributes
10. **Security Scan** - PII detection and data protection
11. **Privacy Protection** - Risk assessment and anonymization
12. **Compliance Check** - Regulatory compliance validation

## API

### Core Endpoints
- `POST /api/upload` - Data upload with intelligence profiling
- `POST /api/data/{session_id}/process` - Execute complete pipeline
- `GET /api/data/{session_id}/results` - Retrieve results and artifacts
- `GET /api/data/{session_id}/status` - Pipeline execution status

### Intelligence & Analysis
- `POST /api/data/{session_id}/profile` - Intelligent data profiling
- `GET /api/data/{session_id}/explanations` - Model explanations and insights
- `GET /api/data/{session_id}/bias-report` - Fairness assessment report
- `GET /api/data/{session_id}/trust-metrics` - Trust and reliability scores

### Security & Compliance
- `GET /api/data/{session_id}/security-scan` - Security assessment results
- `GET /api/data/{session_id}/privacy-report` - Privacy risk analysis
- `GET /api/data/{session_id}/compliance-status` - Regulatory compliance report

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

1. **Upload data** - CSV/Excel files or provide data URLs
2. **Configure pipeline** - Select task type and security/privacy settings
3. **Execute processing** - Automated pipeline with intelligence and security
4. **Review results** - Model performance, explanations, bias, and compliance reports
5. **Download artifacts** - Processed data, models, and comprehensive reports

## Configuration

### Security Levels
- `basic` - Standard protection with PII detection
- `standard` - Enhanced security with access control  
- `high` - Strong encryption and audit logging
- `maximum` - Maximum protection with strict policies

### Privacy Levels
- `low` - Basic anonymization
- `medium` - K-anonymity and data masking
- `high` - Differential privacy and l-diversity
- `maximum` - Comprehensive privacy protection

### Compliance
- GDPR - EU General Data Protection Regulation
- CCPA - California Consumer Privacy Act  
- HIPAA - Health Insurance Portability and Accountability Act

## Supported Data
- **Formats**: CSV, Excel, Parquet, JSON
- **Sources**: File upload, URLs, databases
- **Types**: Tabular, time-series, text data
- **Tasks**: Classification, regression, clustering, forecasting, NLP

## Output Artifacts
- Processed datasets with feature engineering
- Trained models with hyperparameter optimization
- Explanation reports with SHAP/LIME analysis
- Bias assessment with fairness metrics
- Security scan results with PII detection
- Privacy protection reports with risk analysis
- Compliance reports with violation detection

## MVP Status

✅ **Complete** - Intelligent pipeline with explainability and enterprise security ready for deployment