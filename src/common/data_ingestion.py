"""
Centralized Data Ingestion Module for DataInsight AI

This module provides a unified function, `ingest_data`, to load data from
various sources including file uploads, local paths, and remote URLs. It is
designed to be robust and extensible.
"""

import io
import json
import logging
import requests
from pathlib import Path
from typing import Any, Optional, Union, Dict, List
from urllib.parse import urlparse

import pandas as pd

try:
    import gdown
except ImportError:
    logging.warning("`gdown` not found. To ingest from Google Drive, run: pip install gdown")
    gdown = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _ingest_from_file_object(file_obj: Any, filename: str = None) -> Optional[pd.DataFrame]:
    """Ingests data from a file-like object with enhanced error handling."""
    try:
        file_name = filename or getattr(file_obj, "name", "unknown")
        file_name = file_name.lower()

        # Enhanced CSV reading with encoding detection
        if file_name.endswith(".csv"):
            try:
                return pd.read_csv(file_obj, encoding="utf-8")
            except UnicodeDecodeError:
                file_obj.seek(0)
                return pd.read_csv(file_obj, encoding="latin-1")

        # Excel files
        elif file_name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file_obj, engine="openpyxl" if file_name.endswith(".xlsx") else "xlrd")

        # JSON files
        elif file_name.endswith(".json"):
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            data = json.loads(content)
            return pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])

        # TSV files
        elif file_name.endswith(".tsv"):
            return pd.read_csv(file_obj, sep="\t")

        else:
            logging.error(f"Unsupported file type: {file_name}")
            return None

    except Exception as e:
        logging.error(f"Error reading file '{file_name}': {e}")
        return None


def _ingest_from_path(path: Path) -> Optional[pd.DataFrame]:
    """Ingests data from a local file path."""
    if not path.is_file():
        logging.error(f"File not found at path: {path}")
        return None
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix in [".xls", ".xlsx"]:
            return pd.read_excel(path)
        else:
            logging.error(f"Unsupported file type at path: {path.suffix}")
            return None
    except Exception as e:
        logging.error(f"Error reading file from path '{path}': {e}")
        return None


def _ingest_from_google_drive(url: str) -> Optional[pd.DataFrame]:
    """Handle Google Drive URLs."""
    if not gdown:
        logging.error("Cannot ingest from Google Drive; `gdown` is not installed.")
        return None

    try:
        output_path = gdown.download(url, quiet=True, fuzzy=True)
        df = _ingest_from_path(Path(output_path))
        Path(output_path).unlink()  # Clean up downloaded file
        return df
    except Exception as e:
        logging.error(f"Failed to download from Google Drive: {e}")
        return None


def _fetch_and_parse_url(url: str, data_type: str) -> Optional[pd.DataFrame]:
    """Fetch data from URL and parse based on type."""
    headers = {"User-Agent": "DataInsight-AI/2.0 (Data Analysis Tool)"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    content = io.BytesIO(response.content)

    if data_type == "csv" or url.endswith(".csv"):
        try:
            return pd.read_csv(content, encoding="utf-8")
        except UnicodeDecodeError:
            content.seek(0)
            return pd.read_csv(content, encoding="latin-1")

    elif data_type == "json" or url.endswith(".json"):
        data = response.json()
        return pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])

    elif data_type == "excel" or url.endswith((".xls", ".xlsx")):
        return pd.read_excel(content)

    elif data_type == "tsv" or url.endswith(".tsv"):
        return pd.read_csv(content, sep="\t")

    else:
        # Default to CSV
        logging.warning(f"Unknown data type '{data_type}', attempting CSV parsing")
        content.seek(0)
        return pd.read_csv(content)


def _ingest_from_url(url: str, data_type: str = "csv") -> Optional[pd.DataFrame]:
    """Enhanced URL ingestion with multiple data source support."""
    logging.info(f"Attempting to ingest from URL: {url} (type: {data_type})")
    parsed_url = urlparse(url)

    try:
        # Special handling for different services
        if "drive.google.com" in parsed_url.netloc:
            return _ingest_from_google_drive(url)
        elif "github.com" in parsed_url.netloc and "/blob/" in url:
            # Convert GitHub blob URL to raw URL
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            return _fetch_and_parse_url(raw_url, data_type)
        elif "kaggle.com" in parsed_url.netloc:
            logging.warning("Kaggle datasets require API authentication. Use direct file URLs.")
            return None
        else:
            return _fetch_and_parse_url(url, data_type)

    except Exception as e:
        logging.error(f"Failed to ingest from URL '{url}': {e}")
        return None


def ingest_data(source: Union[str, Path, Any], filename: str = None) -> Optional[pd.DataFrame]:
    """Enhanced data ingestion with better error handling and format support.

    Args:
        source: Data source (file object, path, or URL)
        filename: Optional filename for file objects

    Returns:
        DataFrame if successful, None otherwise
    """
    if hasattr(source, "read"):
        logging.info("Source identified as a file-like object.")
        return _ingest_from_file_object(source, filename)

    elif isinstance(source, Path):
        logging.info("Source identified as a pathlib.Path object.")
        return _ingest_from_path(source)

    elif isinstance(source, str):
        if urlparse(source).scheme in ["http", "https"]:
            logging.info("Source identified as a URL.")
            return _ingest_from_url(source)
        else:
            logging.info("Source identified as a local path string.")
            return _ingest_from_path(Path(source))

    else:
        logging.error(f"Unsupported source type: {type(source)}")
        return None


def ingest_from_url(url: str, data_type: str = "csv") -> Optional[pd.DataFrame]:
    """Public API for URL ingestion with explicit data type."""
    return _ingest_from_url(url, data_type)


def validate_data_source(source: Union[str, Path, Any]) -> Dict[str, Any]:
    """Validate data source and return metadata."""
    result = {"valid": False, "type": "unknown", "size_estimate": None, "format": None, "error": None}

    try:
        if hasattr(source, "read"):
            result["type"] = "file_object"
            result["format"] = getattr(source, "name", "").split(".")[-1] if hasattr(source, "name") else "unknown"
            result["valid"] = True

        elif isinstance(source, (str, Path)):
            if isinstance(source, str) and urlparse(source).scheme in ["http", "https"]:
                result["type"] = "url"
                result["format"] = source.split(".")[-1] if "." in source.split("/")[-1] else "unknown"

                # Quick HEAD request to check if URL is accessible
                response = requests.head(source, timeout=10)
                if response.status_code == 200:
                    result["valid"] = True
                    result["size_estimate"] = response.headers.get("Content-Length")
                else:
                    result["error"] = f"HTTP {response.status_code}"

            else:
                path = Path(source)
                result["type"] = "file_path"
                result["format"] = path.suffix.lstrip(".")
                result["valid"] = path.exists() and path.is_file()
                if result["valid"]:
                    result["size_estimate"] = path.stat().st_size
                else:
                    result["error"] = "File not found"

    except Exception as e:
        result["error"] = str(e)

    return result


class DataIngestion:
    """
    Object-oriented wrapper for data ingestion functions.

    Provides a consistent interface for loading data from various sources
    with enhanced error handling and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataIngestion with optional configuration.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary for ingestion parameters
        """
        default_config = {"timeout": 30, "max_retries": 3, "encoding": "utf-8", "validate_source": True}

        self.config = {**default_config, **(config or {})}
        self.last_source_info = None

    def load(self, source: Union[str, Path, Any], filename: str = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load data from a source with comprehensive error handling.

        Parameters
        ----------
        source : str, Path, or file-like object
            Data source to load from
        filename : str, optional
            Optional filename for file objects
        **kwargs : additional arguments
            Additional arguments for pandas reading functions

        Returns
        -------
        pd.DataFrame or None
            Loaded data or None if failed
        """
        # Validate source if enabled
        if self.config["validate_source"]:
            validation = validate_data_source(source)
            self.last_source_info = validation

            if not validation["valid"]:
                logging.error(f"Source validation failed: {validation['error']}")
                return None

        # Load data using the appropriate function
        try:
            if isinstance(source, str) and urlparse(source).scheme in ["http", "https"]:
                data_type = kwargs.get("data_type", "csv")
                df = ingest_from_url(source, data_type)
            else:
                df = ingest_data(source, filename)

            if df is not None:
                logging.info(f"Successfully loaded data with shape: {df.shape}")

            return df

        except Exception as e:
            logging.error(f"Data ingestion failed: {e}")
            return None

    def load_from_file(self, file_obj: Any, filename: str = None) -> Optional[pd.DataFrame]:
        """Load data from a file-like object."""
        return _ingest_from_file_object(file_obj, filename)

    def load_from_path(self, path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Load data from a local file path."""
        return _ingest_from_path(Path(path))

    def load_from_url(self, url: str, data_type: str = "csv") -> Optional[pd.DataFrame]:
        """Load data from a URL."""
        return _ingest_from_url(url, data_type)

    def validate_source(self, source: Union[str, Path, Any]) -> Dict[str, Any]:
        """Validate a data source."""
        return validate_data_source(source)

    def get_source_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last validated source."""
        return self.last_source_info

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ["csv", "xlsx", "xls", "json", "tsv"]
