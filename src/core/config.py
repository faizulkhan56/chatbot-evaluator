import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    default_model: str
    default_evaluator_model: str
    default_embedding_model: str
    default_chunk_size: int
    default_chunk_overlap: int
    default_top_k: int
    data_dir: Path
    datasets_dir: Path
    results_dir: Path
    documents_dir: Path
    vectorstores_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = _path_from_env("DATA_DIR", PROJECT_ROOT / "data")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        default_evaluator_model=os.getenv("DEFAULT_EVALUATOR_MODEL", "gpt-4o-mini"),
        default_embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"),
        default_chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "500")),
        default_chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "6")),
        data_dir=data_dir,
        datasets_dir=_path_from_env("DATASETS_DIR", data_dir / "datasets"),
        results_dir=_path_from_env("RESULTS_DIR", data_dir / "results"),
        documents_dir=_path_from_env("DOCUMENTS_DIR", data_dir / "documents"),
        vectorstores_dir=_path_from_env("VECTORSTORES_DIR", data_dir / "vectorstores"),
    )
