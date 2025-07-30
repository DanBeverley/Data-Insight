"""
Validated Configuration Loader for DataInsight AI

This module uses Pydantic to load, validate, and provide access to the
application settings from `config.yml`. It ensures all configurations are
structurally and typographically correct upon application startup.
"""

import yaml
from pathlib import Path
from typing import Any

from ..config import AppConfig

class Settings:
    """Singleton class to hold the validated application settings."""
    _instance: 'Settings' = None
    config: AppConfig

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load_and_validate_config()
        return cls._instance

    def _load_and_validate_config(self) -> None:
        """Loads YAML, parses with Pydantic, and resolves paths."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "config.yml"

        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        self.config = AppConfig(**raw_config)
        self._resolve_paths(base_dir)

    def _resolve_paths(self, base_dir: Path) -> None:
        """Converts relative paths from config into absolute paths."""
        for key, value in self.config.paths.dict().items():
            absolute_path = (base_dir / value).resolve()
            absolute_path.mkdir(parents=True, exist_ok=True)
            setattr(self.config.paths, key, str(absolute_path))

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the loaded Pydantic config model."""
        return getattr(self.config, name)

settings = Settings()