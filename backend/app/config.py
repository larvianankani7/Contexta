from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:4b"
    ollama_timeout: float = Field(default=120.0, gt=0, le=300)
    ollama_num_ctx: int = Field(default=8192, ge=512, le=131072)
    ollama_num_predict: int = Field(default=8192, ge=16, le=8192)
    ollama_seed: int = 42
    database_path: Path = Path("backend/data/context.db")
    artifact_dir: Path = Path("backend/data/artifacts")
    max_recovery_attempts: int = Field(default=2, ge=0, le=5)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    def project_path(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
