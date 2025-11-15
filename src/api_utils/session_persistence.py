"""Session data persistence to survive server restarts"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SessionDataStore:
    """
    Persistent storage for session data including DataFrames
    Uses pickle files to serialize session state
    """

    def __init__(self, storage_dir: str = "session_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"Session data store initialized at {self.storage_dir}")

    def save_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Save session data to disk

        Args:
            session_id: Session identifier
            data: Session data dictionary

        Returns:
            True if successful
        """
        try:
            session_file = self.storage_dir / f"{session_id}.pkl"

            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    serializable_data[key] = {"_type": "dataframe", "_data": value.to_dict("tight")}
                else:
                    serializable_data[key] = value

            with open(session_file, "wb") as f:
                pickle.dump(serializable_data, f)

            logger.debug(f"Saved session data for {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session data for {session_id}: {e}")
            return False

    def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from disk

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None if not found
        """
        try:
            session_file = self.storage_dir / f"{session_id}.pkl"

            if not session_file.exists():
                logger.debug(f"No saved data found for session {session_id}")
                return None

            with open(session_file, "rb") as f:
                serialized_data = pickle.load(f)

            restored_data = {}
            for key, value in serialized_data.items():
                if isinstance(value, dict) and value.get("_type") == "dataframe":
                    restored_data[key] = pd.DataFrame.from_dict(value["_data"], orient="tight")
                else:
                    restored_data[key] = value

            logger.info(f"Loaded session data for {session_id}")
            return restored_data

        except Exception as e:
            logger.error(f"Failed to load session data for {session_id}: {e}")
            return None

    def delete_session_data(self, session_id: str) -> bool:
        """
        Delete session data from disk

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            session_file = self.storage_dir / f"{session_id}.pkl"

            if session_file.exists():
                session_file.unlink()
                logger.info(f"Deleted session data for {session_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete session data for {session_id}: {e}")
            return False

    def list_sessions(self) -> list:
        """
        List all stored session IDs

        Returns:
            List of session IDs
        """
        try:
            return [f.stem for f in self.storage_dir.glob("*.pkl")]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []


session_data_store = SessionDataStore()
