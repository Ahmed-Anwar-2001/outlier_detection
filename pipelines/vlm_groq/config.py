"""
Configuration for Groq VLM Pipeline using Qwen 3.6 27B.
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VlmConfig:
    dataset_dir: Path = Path("dataset/dataset")
    output_path: Path = Path("results/results_vlm.json")
    cache_dir: Path = Path(".cache/vlm_v2")

    # Groq Model
    model_name: str = "qwen/qwen3.6-27b"
    fallback_model: str = "qwen/qwen3.8-27b"
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"

    # Image processing
    max_image_dim: int = 768  # balanced resolution for visual landmarks and signage
    jpeg_quality: int = 85

    # Rate limiting & retries
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_delay: float = 2.0

    # API key helper
    @classmethod
    def get_api_key(cls, explicit_key: str = None) -> str:
        if explicit_key:
            return explicit_key
        if os.environ.get("GROQ_API_KEY"):
            return os.environ["GROQ_API_KEY"]
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        return line.split("=", 1)[1].strip()
        raise ValueError("GROQ_API_KEY not found in environment, argument, or .env file.")
