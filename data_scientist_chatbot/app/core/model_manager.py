"""Model management and configuration"""

import os
import asyncio
import psutil
from typing import Dict, Optional
import httpx

try:
    from .logger import logger
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger


class ModelManager:
    """Manages model configurations and switching for different agent types"""

    _http_client: Optional[httpx.AsyncClient] = None
    _health_check_task: Optional[asyncio.Task] = None
    _health_check_running: bool = False

    def __init__(self):
        is_test_env = os.getenv("ENVIRONMENT") == "test"
        logger.debug(f"ModelManager: ENVIRONMENT={os.getenv('ENVIRONMENT')}, is_test_env={is_test_env}")
        unified_model = "deepseek-v3.2:cloud"
        self.brain_model = "phi3:3.8b-mini-128k-instruct-q4_K_M" if is_test_env else unified_model
        self.hands_model = "phi3:3.8b-mini-128k-instruct-q4_K_M" if is_test_env else unified_model
        self.vision_model = "qwen3-vl:235b-instruct-cloud"
        self.status_model = "ministral-3:8b-cloud"
        self.verifier_model = "phi3:3.8b-mini-128k-instruct-q4_K_M" if is_test_env else "gpt-oss:20b-cloud"
        self.current_model = self.brain_model
        self.switch_count = 0
        self.num_cores = psutil.cpu_count(logical=False) or 8
        self.num_ctx = 131072
        self.num_predict_hands = 32768
        self.num_predict_brain = 4096
        self.temperature_brain = 0.6
        self.temperature_hands = 0.0
        self.temperature_status = 0.3
        self.ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.health_check_interval = int(os.getenv("MODEL_HEALTH_CHECK_INTERVAL", "60"))
        self._last_health_status: Dict[str, bool] = {}

    @classmethod
    def get_http_client(cls) -> httpx.AsyncClient:
        if cls._http_client is None or cls._http_client.is_closed:
            cls._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            logger.info("[POOL] HTTP connection pool initialized (max=20, keepalive=10)")
        return cls._http_client

    @classmethod
    async def close_http_client(cls):
        if cls._http_client and not cls._http_client.is_closed:
            await cls._http_client.aclose()
            cls._http_client = None
            logger.info("[POOL] HTTP connection pool closed")

    async def health_check_ping(self, model_name: str) -> bool:
        try:
            client = self.get_http_client()
            response = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": model_name, "prompt": "ping", "stream": False, "options": {"num_predict": 1}},
                timeout=30.0,
            )
            is_healthy = response.status_code == 200
            self._last_health_status[model_name] = is_healthy
            return is_healthy
        except Exception as e:
            logger.warning(f"[HEALTH] Ping failed for {model_name}: {e}")
            self._last_health_status[model_name] = False
            return False

    async def _health_check_loop(self):
        models_to_check = [self.brain_model, self.status_model, self.verifier_model]
        logger.info(f"[HEALTH] Starting health check loop (interval={self.health_check_interval}s)")

        while self._health_check_running:
            for model in models_to_check:
                if not self._health_check_running:
                    break
                healthy = await self.health_check_ping(model)
                status = "healthy" if healthy else "unhealthy"
                logger.debug(f"[HEALTH] {model}: {status}")

            await asyncio.sleep(self.health_check_interval)

    def start_health_check_loop(self):
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("[HEALTH] Health check loop started")

    def stop_health_check_loop(self):
        self._health_check_running = False
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            logger.info("[HEALTH] Health check loop stopped")

    def get_model(self, agent_type: str) -> str:
        model_map = {
            "brain": self.brain_model,
            "hands": self.hands_model,
            "vision": self.vision_model,
            "status": self.status_model,
            "verifier": self.verifier_model,
        }
        target_model = model_map.get(agent_type, self.brain_model)

        if self.current_model != target_model:
            prev_agent = getattr(self, "current_agent_type", "unknown")
            logger.info(f"Switching model: {self.current_model} [{prev_agent}] -> {target_model} [{agent_type}]")
            self.current_model = target_model
            self.current_agent_type = agent_type
            self.switch_count += 1
        else:
            self.current_agent_type = agent_type
        return self.current_model

    def get_ollama_config(self, agent_type: str) -> Dict[str, any]:
        temp_map = {
            "brain": self.temperature_brain,
            "hands": self.temperature_hands,
            "vision": 0.1,
            "status": self.temperature_status,
            "verifier": 0.1,
        }

        predict_map = {
            "hands": self.num_predict_hands,
            "brain": self.num_predict_brain,
            "vision": 4096,
            "status": self.num_predict_brain,
            "verifier": 2048,
        }

        model_name = self.get_model(agent_type)

        if model_name.endswith("-cloud"):
            config = {
                "model": model_name,
                "base_url": self.ollama_base_url,
                "temperature": temp_map.get(agent_type, 0.1),
                "timeout": 120,
            }
            config["format"] = ""
            config["options"] = {
                "num_predict": predict_map.get(agent_type, 4096),
                "stop": ["<｜end of sentence｜>", "</think>", "<｜/thought｜>"],
            }
            if agent_type == "hands":
                logger.debug(
                    f"Hands config with num_predict={config['options']['num_predict']}, think={config['options'].get('think', False)}"
                )
            return config

        config = {
            "model": model_name,
            "base_url": self.ollama_base_url,
            "temperature": temp_map.get(agent_type, 0.1),
            "num_thread": self.num_cores,
            "num_ctx": self.num_ctx,
        }
        config["format"] = ""
        config["options"] = {"num_predict": predict_map.get(agent_type, 4096)}
        return config
