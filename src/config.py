"""
Configuration Loader for DataInsight AI

This module reads the main `config.yaml` file, validates its structure,
and provides a singleton-like object (`settings`) for easy access
to configuration parameters throughout the application.
"""
import yaml
from pathlib import Path
from typing import Any, Dict

class AppConfig:
    """Singleton class to hold and provide access to the application settings"""
    _instance = None
    _config_data: Dict[str, Any] = {}
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Loads config from YAML file and set up paths"""
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir/"config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, "r") as f:
            self._config_data = yaml.safe_load(f)
        self._resolve_paths(base_dir)
    
    def _resolve_paths(self, base_dir:Path) -> None:
        """Converts relative paths from config into absolute paths"""
        if "paths" in self._config_data:
            for key, value in self._config_data["paths"].items():
                self._config_data["paths"][key] = base_dir/value
                self._config_data["paths"][key].mkdir(parents=True, exist_ok=True)
    
    def get(self, key:str, default:Any=None)->Any:
        """Retrives a configuration section / value"""
        return self._config_data.get(key, default)
    
    def __getitem__(self, key:str) -> Any:
        """Allow dict-style access to configuration sections"""
        if key not in self._config_data:
            raise KeyError(f"Configuration section '{key}' not found")
        return self._config_data[key]


settings = AppConfig()
