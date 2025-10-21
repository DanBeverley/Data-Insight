"""
Smart GPU training wrapper with auto format detection
Deployed to Azure ML and AWS SageMaker for intelligent model saving
"""
import sys
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Tuple


class ModelFormatDetector:
    """Detects optimal save format based on model type introspection"""

    def __init__(self):
        self.format_map = {
            'sklearn': ('.joblib', self._save_sklearn),
            'torch': ('.pt', self._save_pytorch),
            'tensorflow.python': ('.h5', self._save_tensorflow),
            'tensorflow.keras': ('.h5', self._save_keras),
            'keras': ('.h5', self._save_keras),
            'xgboost': ('.json', self._save_xgboost),
            'lightgbm': ('.txt', self._save_lightgbm),
            'catboost': ('.cbm', self._save_catboost),
        }

    def detect_and_save(
        self,
        model: Any,
        output_path: str,
        user_format: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Detect model type and save with optimal format

        Args:
            model: Trained model object
            output_path: Base path for saving
            user_format: Optional user-specified format override

        Returns:
            Tuple of (saved_path, format_used)
        """
        model_module = type(model).__module__
        model_class = type(model).__name__

        print(f"[ModelDetector] Model type: {model_module}.{model_class}")

        # Parse user format override
        if user_format:
            detected_format = self._parse_user_format(user_format)
            if detected_format:
                print(f"[ModelDetector] User specified format: {detected_format}")
                ext, save_func = detected_format
                final_path = f"{output_path}{ext}"
                save_func(model, final_path)
                return final_path, ext

        # Auto-detect based on module
        for module_key, (ext, save_func) in self.format_map.items():
            if module_key in model_module:
                print(f"[ModelDetector] Auto-detected format: {ext} (from {module_key})")
                final_path = f"{output_path}{ext}"
                save_func(model, final_path)
                return final_path, ext

        # Fallback to pickle
        print(f"[ModelDetector] Unknown model type, using pickle fallback")
        ext = '.pkl'
        final_path = f"{output_path}{ext}"
        self._save_pickle(model, final_path)
        return final_path, ext

    def _parse_user_format(self, user_format: str) -> Optional[Tuple[str, callable]]:
        """Parse natural language format specification"""
        format_lower = user_format.lower()

        if 'onnx' in format_lower:
            return ('.onnx', self._save_onnx)
        elif 'joblib' in format_lower:
            return ('.joblib', self._save_sklearn)
        elif 'pickle' in format_lower or 'pkl' in format_lower:
            return ('.pkl', self._save_pickle)
        elif 'pytorch' in format_lower or '.pt' in format_lower:
            return ('.pt', self._save_pytorch)
        elif 'h5' in format_lower or 'hdf5' in format_lower:
            return ('.h5', self._save_keras)
        elif 'savedmodel' in format_lower:
            return ('', self._save_tensorflow_savedmodel)

        return None

    def _save_sklearn(self, model: Any, path: str):
        """Save sklearn model with joblib"""
        import joblib
        joblib.dump(model, path)
        print(f"[ModelDetector] Saved with joblib: {path}")

    def _save_pytorch(self, model: Any, path: str):
        """Save PyTorch model state dict"""
        import torch
        torch.save(model.state_dict(), path)
        print(f"[ModelDetector] Saved PyTorch state_dict: {path}")

    def _save_tensorflow(self, model: Any, path: str):
        """Save TensorFlow/Keras model"""
        model.save(path)
        print(f"[ModelDetector] Saved TensorFlow model: {path}")

    def _save_keras(self, model: Any, path: str):
        """Save Keras model"""
        model.save(path)
        print(f"[ModelDetector] Saved Keras model: {path}")

    def _save_xgboost(self, model: Any, path: str):
        """Save XGBoost model"""
        model.save_model(path)
        print(f"[ModelDetector] Saved XGBoost model: {path}")

    def _save_lightgbm(self, model: Any, path: str):
        """Save LightGBM model"""
        model.save_model(path)
        print(f"[ModelDetector] Saved LightGBM model: {path}")

    def _save_catboost(self, model: Any, path: str):
        """Save CatBoost model"""
        model.save_model(path)
        print(f"[ModelDetector] Saved CatBoost model: {path}")

    def _save_pickle(self, model: Any, path: str):
        """Fallback pickle save"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[ModelDetector] Saved with pickle: {path}")

    def _save_onnx(self, model: Any, path: str):
        """Save model as ONNX format"""
        try:
            import onnx
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            # Attempt sklearn to ONNX conversion
            initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            with open(path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"[ModelDetector] Saved as ONNX: {path}")
        except Exception as e:
            print(f"[ModelDetector] ONNX conversion failed: {e}, falling back to pickle")
            self._save_pickle(model, path.replace('.onnx', '.pkl'))

    def _save_tensorflow_savedmodel(self, model: Any, path: str):
        """Save TensorFlow SavedModel format"""
        model.save(path, save_format='tf')
        print(f"[ModelDetector] Saved TensorFlow SavedModel: {path}")


def find_trained_model(exec_globals: dict) -> Optional[Any]:
    """
    Find trained model in execution globals
    Tries common variable names
    """
    common_names = ['model', 'trained_model', 'clf', 'classifier', 'regressor', 'estimator', 'pipeline']

    for var_name in common_names:
        if var_name in exec_globals:
            obj = exec_globals[var_name]
            # Check if it has fit method (likely a model)
            if hasattr(obj, 'fit') or hasattr(obj, 'predict'):
                print(f"[Wrapper] Found model in variable: {var_name}")
                return obj

    return None


def train_wrapper(
    user_code: str,
    output_dir: str = '/opt/ml/model',
    user_format: Optional[str] = None
) -> dict:
    """
    Execute user training code and save model with auto format detection

    Args:
        user_code: Python training code from user
        output_dir: Directory to save model artifacts
        user_format: Optional natural language format specification

    Returns:
        Dict with model_path, format, and metadata
    """
    print("[Wrapper] Starting smart training wrapper")
    print(f"[Wrapper] User format specification: {user_format or 'Auto-detect'}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    exec_globals = {}

    try:
        print("[Wrapper] Executing user training code...")
        exec(user_code, exec_globals)
        print("[Wrapper] Training code executed successfully")

    except Exception as e:
        print(f"[Wrapper] Training execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    model = find_trained_model(exec_globals)

    if model is None:
        raise ValueError(
            "No trained model found. Please assign your trained model to a variable named 'model'. "
            "Example: model = LinearRegression().fit(X_train, y_train)"
        )

    detector = ModelFormatDetector()
    model_path, format_used = detector.detect_and_save(
        model=model,
        output_path=f"{output_dir}/model",
        user_format=user_format
    )

    model_module = type(model).__module__
    model_class = type(model).__name__

    metadata = {
        'model_type': f'{model_module}.{model_class}',
        'format': format_used,
        'user_specified_format': user_format,
        'model_path': model_path
    }

    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Wrapper] MODEL_SAVED:{model_path}")
    print(f"[Wrapper] MODEL_FORMAT:{format_used}")
    print(f"[Wrapper] METADATA_SAVED:{metadata_path}")

    return {
        'model_path': model_path,
        'format': format_used,
        'metadata': metadata
    }


if __name__ == "__main__":
    """
    Entry point when deployed to cloud GPU services
    Reads code and format from environment or command line
    """
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--code-file', type=str, help='Path to training code file')
    parser.add_argument('--format', type=str, default=None, help='Desired save format')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model', help='Output directory')

    args = parser.parse_args()

    if args.code_file and os.path.exists(args.code_file):
        with open(args.code_file, 'r') as f:
            training_code = f.read()
    else:
        training_code = os.getenv('TRAINING_CODE')
        if not training_code:
            raise ValueError("No training code provided via --code-file or TRAINING_CODE env var")

    result = train_wrapper(
        user_code=training_code,
        output_dir=args.output_dir,
        user_format=args.format
    )

    print("[Wrapper] Training completed successfully")
    print(json.dumps(result, indent=2))
