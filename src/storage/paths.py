from pathlib import Path

from src.core.config import get_settings


def ensure_storage_dirs() -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstores_dir.mkdir(parents=True, exist_ok=True)


def datasets_dir() -> Path:
    ensure_storage_dirs()
    return get_settings().datasets_dir


def results_dir() -> Path:
    ensure_storage_dirs()
    return get_settings().results_dir
