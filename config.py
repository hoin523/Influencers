from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    default_llm: str = "anthropic"  # "anthropic" or "openai"

    # ComfyUI
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_timeout: int = 120

    # Database
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'influencers.db'}"

    # Paths
    personas_dir: Path = BASE_DIR / "personas"
    prompts_dir: Path = BASE_DIR / "prompts"
    workflows_dir: Path = BASE_DIR / "workflows"
    assets_dir: Path = BASE_DIR / "assets"

    # Image generation
    image_gen_retries: int = 3
    llm_retries: int = 3
    face_similarity_threshold: float = 0.6

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
