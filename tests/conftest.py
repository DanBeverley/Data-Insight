import pytest
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def sample_session_id() -> str:
    session_id = "test_session_123"
    yield session_id

    import builtins

    if hasattr(builtins, "_persistent_sandboxes") and session_id in builtins._persistent_sandboxes:
        try:
            sandbox = builtins._persistent_sandboxes[session_id]
            sandbox.close()
            del builtins._persistent_sandboxes[session_id]
        except Exception as e:
            pass


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def datasets_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "datasets"


@pytest.fixture
def housing_dataset(datasets_dir: Path) -> pd.DataFrame:
    csv_path = datasets_dir / "housing_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "price": [300000, 450000, 250000, 500000, 350000],
            "area": [1500, 2000, 1200, 2200, 1800],
            "bedrooms": [3, 4, 2, 4, 3],
            "bathrooms": [2, 3, 1, 3, 2],
            "stories": [2, 2, 1, 2, 2],
            "mainroad": ["yes", "yes", "no", "yes", "yes"],
            "guestroom": ["no", "yes", "no", "yes", "no"],
            "basement": ["no", "yes", "no", "yes", "no"],
            "hotwaterheating": ["no", "no", "no", "no", "no"],
            "airconditioning": ["yes", "yes", "no", "yes", "yes"],
            "parking": [2, 3, 0, 3, 2],
            "prefarea": ["yes", "no", "no", "yes", "yes"],
            "furnishingstatus": ["furnished", "furnished", "unfurnished", "semi-furnished", "furnished"],
        }
    )


@pytest.fixture
def flight_dataset(datasets_dir: Path) -> pd.DataFrame:
    csv_path = datasets_dir / "flight_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "flight_id": range(1, 6),
            "airline": ["AA", "UA", "DL", "SW", "AA"],
            "departure_delay": [15, -5, 30, 0, 45],
            "arrival_delay": [10, -10, 25, -5, 40],
            "distance": [500, 1200, 800, 300, 1500],
        }
    )


@pytest.fixture
def coffee_sales_dataset(datasets_dir: Path) -> pd.DataFrame:
    csv_path = datasets_dir / "coffee_sales.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100),
            "product": ["Espresso", "Latte", "Cappuccino"] * 33 + ["Espresso"],
            "quantity": [5, 10, 8, 12, 6] * 20,
            "revenue": [15.0, 30.0, 24.0, 36.0, 18.0] * 20,
        }
    )


@pytest.fixture
def customer_churn_dataset(datasets_dir: Path) -> pd.DataFrame:
    csv_path = datasets_dir / "customer_churn.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "customer_id": range(1, 11),
            "tenure_months": [12, 24, 6, 36, 3, 18, 48, 9, 15, 60],
            "monthly_charges": [50, 70, 45, 80, 40, 65, 90, 55, 60, 95],
            "total_charges": [600, 1680, 270, 2880, 120, 1170, 4320, 495, 900, 5700],
            "churn": [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        }
    )


@pytest.fixture
def mock_tool_call() -> Dict[str, Any]:
    return {"name": "python_code_execution", "arguments": '{"code": "print(1+1)", "session_id": "test_123"}'}


@pytest.fixture
def mock_ollama_response() -> str:
    return "Based on the data analysis, the correlation between price and area is 0.54, indicating a moderate positive relationship."


@pytest.fixture
def mock_web_search_results() -> list:
    return [
        {
            "title": "Housing Market Trends 2024",
            "body": "Average home prices increased by 5% this year",
            "href": "https://example.com/housing-trends",
        },
        {
            "title": "Real Estate Analysis",
            "body": "Location remains the key factor in property valuation",
            "href": "https://example.com/real-estate",
        },
    ]


@pytest.fixture
def sample_graph_state() -> Dict[str, Any]:
    return {
        "messages": [
            {"type": "human", "content": "What's my name? It's Alice."},
            {"type": "ai", "content": "Your name is Alice."},
        ],
        "session_id": "test_123",
        "current_agent": "brain",
        "iteration_count": 1,
    }


@pytest.fixture(scope="session", autouse=True)
def configure_ollama_deterministic():
    os.environ["OLLAMA_SEED"] = "42"
    os.environ["OLLAMA_TEMPERATURE"] = "0.0"
    os.environ["OLLAMA_NUM_PREDICT"] = "512"
    yield
    os.environ.pop("OLLAMA_SEED", None)
    os.environ.pop("OLLAMA_TEMPERATURE", None)
    os.environ.pop("OLLAMA_NUM_PREDICT", None)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-api-key"],
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "cassette_library_dir": "tests/fixtures/vcr_cassettes",
    }


@pytest.fixture
def vcr_cassette_dir(fixtures_dir: Path) -> Path:
    cassette_dir = fixtures_dir / "vcr_cassettes"
    cassette_dir.mkdir(exist_ok=True)
    return cassette_dir


@pytest.fixture(scope="function", autouse=True)
def cleanup_sandboxes():
    import builtins

    initial_sandboxes = set()
    if hasattr(builtins, "_persistent_sandboxes"):
        initial_sandboxes = set(builtins._persistent_sandboxes.keys())

    yield

    if hasattr(builtins, "_persistent_sandboxes"):
        new_sandboxes = set(builtins._persistent_sandboxes.keys()) - initial_sandboxes
        for session_id in new_sandboxes:
            try:
                sandbox = builtins._persistent_sandboxes[session_id]
                sandbox.close()
                del builtins._persistent_sandboxes[session_id]
            except Exception:
                pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_all_sandboxes_on_exit():
    yield

    import builtins

    if hasattr(builtins, "_persistent_sandboxes"):
        sandboxes_to_close = list(builtins._persistent_sandboxes.items())
        for session_id, sandbox in sandboxes_to_close:
            try:
                sandbox.close()
            except Exception:
                pass
        builtins._persistent_sandboxes.clear()
