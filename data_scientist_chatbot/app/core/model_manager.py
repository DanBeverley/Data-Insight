"""Model management and configuration"""

import psutil


class ModelManager:
    """Manages model configurations and switching for different agent types"""

    def __init__(self):
        self.brain_model = 'gpt-oss:120b-cloud'
        self.hands_model = 'qwen3-coder:480b-cloud'
        self.router_model = 'phi3:3.8b-mini-128k-instruct-q4_K_M'
        self.status_model = 'gpt-oss:20b-cloud'
        self.current_model = None
        self.switch_count = 0

        self.num_cores = psutil.cpu_count(logical=False) or 8
        self.num_ctx = 2048
        self.temperature_router = 0.0
        self.temperature_brain = 0.6
        self.temperature_hands = 0.0
        self.temperature_status = 0.3

    def get_model(self, agent_type: str) -> str:
        model_map = {
            'brain': self.brain_model,
            'hands': self.hands_model,
            'router': self.router_model,
            'status': self.status_model
        }
        target_model = model_map.get(agent_type, self.brain_model)

        if self.current_model != target_model:
            print(f"🧠 Switching model: {self.current_model} → {target_model}")
            self.current_model = target_model
            self.switch_count += 1

        return self.current_model

    def get_ollama_config(self, agent_type: str) -> dict:
        temp_map = {
            'router': self.temperature_router,
            'brain': self.temperature_brain,
            'hands': self.temperature_hands,
            'status': self.temperature_status
        }

        model_name = self.get_model(agent_type)

        if model_name.endswith('-cloud'):
            return {
                'model': model_name,
                'base_url': "http://localhost:11434",
                'temperature': temp_map.get(agent_type, 0.1)
            }

        return {
            'model': model_name,
            'base_url': "http://localhost:11434",
            'temperature': temp_map.get(agent_type, 0.1),
            'num_thread': self.num_cores,
            'num_ctx': self.num_ctx
        }

    def _optimize_context_for_coder(self, state: dict) -> dict:
        optimized_state = state.copy()
        messages = state.get("messages", [])
        if messages is None:
            messages = []

        if len(messages) > 5:
            optimized_state["messages"] = messages[-5:]

        return optimized_state