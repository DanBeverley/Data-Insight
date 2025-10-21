"""Model management and configuration"""
import os
import psutil

class ModelManager:
    """Manages model configurations and switching for different agent types"""
    def __init__(self):
        # Use local model for testing environment to avoid cloud auth issues
        is_test_env = os.getenv('ENVIRONMENT') == 'test'
        print(f"DEBUG ModelManager: ENVIRONMENT={os.getenv('ENVIRONMENT')}, is_test_env={is_test_env}")
        self.brain_model = 'phi3:3.8b-mini-128k-instruct-q4_K_M' if is_test_env else 'gpt-oss:120b-cloud'
        self.hands_model = 'phi3:3.8b-mini-128k-instruct-q4_K_M' if is_test_env else 'qwen3-coder:480b-cloud'
        self.router_model = 'phi3:3.8b-mini-128k-instruct-q4_K_M'
        self.status_model = 'phi3:3.8b-mini-128k-instruct-q4_K_M' if is_test_env else 'gpt-oss:20b-cloud'
        self.current_model = None
        self.switch_count = 0
        self.num_cores = psutil.cpu_count(logical=False) or 8
        self.num_ctx = 131072
        self.num_predict_hands = 16384
        self.num_predict_brain = 4096
        self.num_predict_router = 512
        self.temperature_router = 0.0
        self.temperature_brain = 0.6
        self.temperature_hands = 0.0
        self.temperature_status = 0.3

        self.ollama_base_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

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

        predict_map = {
            'hands': self.num_predict_hands,
            'brain': self.num_predict_brain,
            'router': self.num_predict_router,
            'status': self.num_predict_brain
        }

        model_name = self.get_model(agent_type)

        if agent_type == 'router':
            return {
                'model': 'gpt-oss:20b-cloud',
                'base_url': self.ollama_base_url,
                'temperature': 0.0,
                'num_predict': 512
            }

        if model_name.endswith('-cloud'):
            config = {
                'model': model_name,
                'base_url': self.ollama_base_url,
                'temperature': temp_map.get(agent_type, 0.1)
            }
            config['format'] = ''
            config['options'] = {'num_predict': predict_map.get(agent_type, 4096)}
            if agent_type == 'hands':
                print(f"DEBUG: Hands config with num_predict={config['options']['num_predict']}")
            return config

        config = {
            'model': model_name,
            'base_url': self.ollama_base_url,
            'temperature': temp_map.get(agent_type, 0.1),
            'num_thread': self.num_cores,
            'num_ctx': self.num_ctx
        }
        config['format'] = ''
        config['options'] = {'num_predict': predict_map.get(agent_type, 4096)}
        return config