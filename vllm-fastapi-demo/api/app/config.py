from functools import lru_cache
import os

from dotenv import load_dotenv

# Load .env when the API is started directly on the host.
load_dotenv()


class Settings:
    """Small environment-backed settings object; no extra settings framework needed."""

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local-vllm-key")
    served_model_name: str = os.getenv("SERVED_MODEL_NAME", "qwen2.5-1.5b")
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8080"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
