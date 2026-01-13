import os
import logging
import importlib.util
import multiprocessing


class DataFrameFactory:
    _library = None
    _library_name = "pandas"

    @classmethod
    def get_library(cls):
        if cls._library is not None:
            return cls._library

        try:
            if importlib.util.find_spec("cudf"):
                import cudf
                import pynvml

                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        cls._library = cudf
                        cls._library_name = "cudf"
                        logging.info("Hardware Acceleration: Enabled (NVIDIA GPU via cuDF)")
                        return cls._library
                except Exception:
                    pass
        except ImportError:
            pass

        import pandas as pd

        cls._library = pd
        cls._library_name = "pandas"
        return cls._library

    @classmethod
    def get_library_for_file(cls, file_path: str):
        try:
            if os.path.getsize(file_path) > 1024 * 1024 * 1024:  # 1GB
                if importlib.util.find_spec("modin.pandas"):
                    import modin.pandas as pd

                    logging.info(f"Large file detected ({os.path.getsize(file_path) / 1e9:.2f} GB). Using Modin.")
                    return pd
        except Exception:
            pass

        return cls.get_library()

    @classmethod
    def get_library_name(cls):
        if cls._library is None:
            cls.get_library()
        return cls._library_name
