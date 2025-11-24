"""Standardized logging configuration for Data Insight application"""

import logging
import sys
import os


def configure_logging(name: str = "DataInsight") -> logging.Logger:
    """
    Configure standardized logging for the application.

    Args:
        name: Logger name (default: "DataInsight")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    level = logging.DEBUG if os.getenv("DEBUG") == "true" else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("pandas_profiling").setLevel(logging.ERROR)
    logging.getLogger("ydata_profiling").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    return logger


logger = configure_logging()
