from typing import Dict, List, Any
from datetime import datetime


E2B_MOCK_RESPONSES: Dict[str, Dict[str, Any]] = {
    "successful_execution": {
        "success": True,
        "stdout": "Result: 42\n",
        "stderr": "",
        "execution_time": 0.123,
        "memory_used": 15.2,
        "plots": []
    },

    "execution_with_plot": {
        "success": True,
        "stdout": "Plot saved to correlation_heatmap.html\n",
        "stderr": "",
        "execution_time": 1.456,
        "memory_used": 32.8,
        "plots": ["correlation_heatmap.html", "scatter_plot.png"]
    },

    "syntax_error": {
        "success": False,
        "stdout": "",
        "stderr": "SyntaxError: invalid syntax (line 2)",
        "execution_time": 0.012,
        "memory_used": 8.1,
        "plots": []
    },

    "runtime_error": {
        "success": False,
        "stdout": "",
        "stderr": "NameError: name 'undefined_var' is not defined",
        "execution_time": 0.045,
        "memory_used": 10.3,
        "plots": []
    },

    "timeout_error": {
        "success": False,
        "stdout": "",
        "stderr": "TimeoutError: Execution exceeded 30 seconds",
        "execution_time": 30.001,
        "memory_used": 50.0,
        "plots": []
    },

    "memory_limit_error": {
        "success": False,
        "stdout": "",
        "stderr": "MemoryError: Process exceeded 512MB memory limit",
        "execution_time": 2.345,
        "memory_used": 512.0,
        "plots": []
    },

    "dataframe_analysis": {
        "success": True,
        "stdout": """<class 'pandas.core.frame.DataFrame'>
RangeIndex: 545 entries, 0 to 544
Data columns (total 13 columns):
price       545 non-null int64
area        545 non-null int64
bedrooms    545 non-null int64
""",
        "stderr": "",
        "execution_time": 0.234,
        "memory_used": 18.5,
        "plots": []
    },

    "correlation_analysis": {
        "success": True,
        "stdout": "Correlation matrix:\n          price      area\nprice   1.000000  0.540000\narea    0.540000  1.000000\n",
        "stderr": "",
        "execution_time": 0.567,
        "memory_used": 22.1,
        "plots": ["correlation_heatmap.html"]
    },

    "model_training": {
        "success": True,
        "stdout": """Model training complete.
Train R² Score: 0.87
Test R² Score: 0.84
RMSE: 45231.23
""",
        "stderr": "",
        "execution_time": 5.234,
        "memory_used": 125.6,
        "plots": ["residual_plot.png", "feature_importance.png"]
    },

    "security_violation": {
        "success": False,
        "stdout": "",
        "stderr": "SecurityError: Network access not allowed in sandbox",
        "execution_time": 0.001,
        "memory_used": 5.0,
        "plots": []
    }
}


DUCKDUCKGO_MOCK_RESPONSES: List[Dict[str, str]] = [
    {
        "title": "Housing Market Trends 2024 - Real Estate Analysis",
        "body": "According to recent data, housing prices have increased by 4.2% year-over-year. Urban areas show stronger growth compared to suburban regions. Interest rates remain a key factor affecting market dynamics.",
        "href": "https://example.com/housing-trends-2024"
    },
    {
        "title": "Data Science Best Practices - Machine Learning Guide",
        "body": "Essential data science practices include proper train-test splitting, cross-validation, feature scaling, and hyperparameter tuning. Always validate model assumptions and check for data leakage.",
        "href": "https://example.com/ds-best-practices"
    },
    {
        "title": "Python Pandas Documentation - DataFrame Operations",
        "body": "Pandas DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. It provides powerful data manipulation capabilities including groupby, merge, and pivot operations.",
        "href": "https://pandas.pydata.org/docs"
    },
    {
        "title": "Statistical Hypothesis Testing Explained",
        "body": "Hypothesis testing involves formulating null and alternative hypotheses, choosing significance level (α), computing test statistic, and making decisions based on p-values. Common tests include t-test, chi-square, and ANOVA.",
        "href": "https://example.com/hypothesis-testing"
    },
    {
        "title": "Customer Churn Prediction Strategies",
        "body": "Effective churn prediction combines behavioral data, engagement metrics, and customer demographics. Random Forest and Gradient Boosting often perform well. Feature engineering is crucial for model success.",
        "href": "https://example.com/churn-prediction"
    },
    {
        "title": "Time Series Forecasting Methods",
        "body": "Popular time series methods include ARIMA, Exponential Smoothing, Prophet, and LSTM neural networks. Choose based on data characteristics like seasonality, trend, and stationarity.",
        "href": "https://example.com/time-series-forecasting"
    },
    {
        "title": "A/B Testing Statistical Guidelines",
        "body": "Proper A/B testing requires adequate sample size, randomization, single metric focus, and appropriate statistical tests. Account for multiple testing problem when running many experiments.",
        "href": "https://example.com/ab-testing-guide"
    },
    {
        "title": "Feature Engineering Techniques for ML",
        "body": "Advanced feature engineering includes creating polynomial features, interaction terms, domain-specific transformations, and automated feature selection. Can significantly boost model performance.",
        "href": "https://example.com/feature-engineering"
    },
    {
        "title": "Handling Imbalanced Datasets in Classification",
        "body": "Techniques for imbalanced data: SMOTE oversampling, undersampling majority class, class weights adjustment, ensemble methods, and using appropriate metrics like F1-score and AUC-ROC.",
        "href": "https://example.com/imbalanced-data"
    },
    {
        "title": "Deep Learning for Tabular Data",
        "body": "While tree-based models excel on tabular data, deep learning approaches like TabNet and Neural Oblivious Decision Ensembles show promise. Proper preprocessing and architecture selection are critical.",
        "href": "https://example.com/dl-tabular"
    }
]


NEO4J_MOCK_RESPONSES: Dict[str, Any] = {
    "dataset_relationships": {
        "nodes": [
            {"id": 1, "label": "Dataset", "properties": {"name": "housing_data", "rows": 545}},
            {"id": 2, "label": "Column", "properties": {"name": "price", "type": "numeric"}},
            {"id": 3, "label": "Column", "properties": {"name": "area", "type": "numeric"}},
            {"id": 4, "label": "Analysis", "properties": {"type": "correlation", "value": 0.54}}
        ],
        "relationships": [
            {"source": 1, "target": 2, "type": "HAS_COLUMN"},
            {"source": 1, "target": 3, "type": "HAS_COLUMN"},
            {"source": 2, "target": 4, "type": "CORRELATED_WITH"},
            {"source": 3, "target": 4, "type": "CORRELATED_WITH"}
        ]
    },

    "feature_lineage": {
        "nodes": [
            {"id": 1, "label": "RawFeature", "properties": {"name": "date"}},
            {"id": 2, "label": "DerivedFeature", "properties": {"name": "day_of_week"}},
            {"id": 3, "label": "DerivedFeature", "properties": {"name": "month"}},
            {"id": 4, "label": "Model", "properties": {"name": "sales_predictor", "accuracy": 0.89}}
        ],
        "relationships": [
            {"source": 1, "target": 2, "type": "DERIVED_FROM"},
            {"source": 1, "target": 3, "type": "DERIVED_FROM"},
            {"source": 2, "target": 4, "type": "USED_IN_MODEL"},
            {"source": 3, "target": 4, "type": "USED_IN_MODEL"}
        ]
    },

    "analysis_history": {
        "analyses": [
            {
                "id": "analysis_1",
                "type": "correlation",
                "timestamp": "2024-01-15T10:30:00Z",
                "dataset": "housing_data",
                "results": {"correlation_price_area": 0.54}
            },
            {
                "id": "analysis_2",
                "type": "regression",
                "timestamp": "2024-01-15T11:45:00Z",
                "dataset": "housing_data",
                "results": {"r2_score": 0.78, "rmse": 45231.23}
            },
            {
                "id": "analysis_3",
                "type": "clustering",
                "timestamp": "2024-01-15T14:20:00Z",
                "dataset": "customer_data",
                "results": {"n_clusters": 4, "silhouette_score": 0.67}
            }
        ]
    }
}


POSTGRESQL_MOCK_RESPONSES: Dict[str, Any] = {
    "session_data": {
        "session_id": "test_session_123",
        "created_at": datetime(2024, 1, 15, 10, 0, 0),
        "last_activity": datetime(2024, 1, 15, 12, 30, 0),
        "dataset_name": "housing_data.csv",
        "row_count": 545,
        "column_count": 13
    },

    "execution_history": [
        {
            "id": 1,
            "session_id": "test_session_123",
            "code": "df.describe()",
            "success": True,
            "execution_time": 0.123,
            "timestamp": datetime(2024, 1, 15, 10, 15, 0)
        },
        {
            "id": 2,
            "session_id": "test_session_123",
            "code": "df.corr()",
            "success": True,
            "execution_time": 0.567,
            "timestamp": datetime(2024, 1, 15, 10, 30, 0)
        }
    ],

    "learning_patterns": [
        {
            "pattern_id": "pattern_viz_001",
            "task_type": "visualization",
            "code_template": "import seaborn as sns\nsns.heatmap(df.corr(), annot=True)",
            "success_count": 45,
            "avg_execution_time": 1.234,
            "confidence_score": 0.92
        },
        {
            "pattern_id": "pattern_ml_001",
            "task_type": "regression",
            "code_template": "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()",
            "success_count": 32,
            "avg_execution_time": 3.456,
            "confidence_score": 0.88
        }
    ]
}


def get_e2b_mock_response(scenario: str) -> Dict[str, Any]:
    return E2B_MOCK_RESPONSES.get(scenario, E2B_MOCK_RESPONSES["successful_execution"])


def get_duckduckgo_mock_results(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    return DUCKDUCKGO_MOCK_RESPONSES[:num_results]


def get_neo4j_mock_response(query_type: str) -> Dict[str, Any]:
    return NEO4J_MOCK_RESPONSES.get(query_type, {})


def get_postgres_mock_response(query_type: str) -> Any:
    return POSTGRESQL_MOCK_RESPONSES.get(query_type, {})
