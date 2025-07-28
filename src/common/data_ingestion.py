"""
Centralized Data Ingestion Module for DataInsight AI

This module provides a unified function, `ingest_data`, to load data from
various sources including file uploads, local paths, and remote URLs. It is
designed to be robust and extensible.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import pandas as pd

try:
    import gdown
except ImportError:
    logging.warning("`gdown` not found. To ingest from Google Drive, run: pip install gdown")
    gdown = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _ingest_from_file_object(file_obj: Any) -> Optional[pd.DataFrame]:
    """Ingests data from a file-like object (e.g., Streamlit's UploadedFile)."""
    try:
        file_name = file_obj.name.lower()
        if file_name.endswith('.csv'):
            return pd.read_csv(file_obj)
        elif file_name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_obj)
        else:
            logging.error(f"Unsupported file type for file object: {file_name}")
            return None
    except Exception as e:
        logging.error(f"Error reading file object '{file_obj.name}': {e}")
        return None

def _ingest_from_path(path: Path) -> Optional[pd.DataFrame]:
    """Ingests data from a local file path."""
    if not path.is_file():
        logging.error(f"File not found at path: {path}")
        return None
    try:
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix in ['.xls', '.xlsx']:
            return pd.read_excel(path)
        else:
            logging.error(f"Unsupported file type at path: {path.suffix}")
            return None
    except Exception as e:
        logging.error(f"Error reading file from path '{path}': {e}")
        return None

def _ingest_from_url(url: str) -> Optional[pd.DataFrame]:
    """Ingests data from a remote URL, with special handling for services."""
    logging.info(f"Attempting to ingest from URL: {url}")
    parsed_url = urlparse(url)

    if 'drive.google.com' in parsed_url.netloc:
        if gdown:
            try:
                # gdown downloads the file and returns the path; we read it from there
                output_path = gdown.download(url, quiet=True, fuzzy=True)
                df = _ingest_from_path(Path(output_path))
                Path(output_path).unlink()
                return df
            except Exception as e:
                logging.error(f"Failed to download from Google Drive URL '{url}': {e}")
                return None
        else:
            logging.error("Cannot ingest from Google Drive; `gdown` is not installed.")
            return None

    # Default handler for direct links to CSV/Excel files
    try:
        if url.endswith('.csv'):
            return pd.read_csv(url)
        elif url.endswith(('.xls', '.xlsx')):
            return pd.read_excel(url)
        else:
            logging.warning("URL has no clear extension, attempting to read as CSV.")
            return pd.read_csv(url)
    except Exception as e:
        logging.error(f"Failed to read from direct URL '{url}': {e}")
        return None

def ingest_data(source: Union[str, Path, Any]) -> Optional[pd.DataFrame]:
    """
    Primary data ingestion function.

    Acts as a dispatcher to the correct ingestion helper based on the
    source type.

    Args:
        source: The data source. Can be:
                - A string (URL or local file path).
                - A pathlib.Path object (local file path).
                - A file-like object (from Streamlit's file_uploader).

    Returns:
        A pandas DataFrame if successful, otherwise None.
    """
    if hasattr(source, 'name') and hasattr(source, 'read'):
        # File-like object from Streamlit
        logging.info("Source identified as a file-like object.")
        return _ingest_from_file_object(source)
    
    elif isinstance(source, Path):
        logging.info("Source identified as a pathlib.Path object.")
        return _ingest_from_path(source)
    
    elif isinstance(source, str):
        # Could be a URL or a local path string
        if urlparse(source).scheme in ['http', 'https']:
            logging.info("Source identified as a URL.")
            return _ingest_from_url(source)
        else:
            logging.info("Source identified as a local path string.")
            return _ingest_from_path(Path(source))
    
    else:
        logging.error(f"Unsupported source type: {type(source)}")
        return None
