from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import get_settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def cleanup_dataset(settings):
    created = []

    def track(name: str) -> str:
        created.append(name)
        return name

    yield track

    for name in created:
        path = settings.datasets_dir / f"{name}.json"
        if path.exists():
            path.unlink()


@pytest.fixture
def result_store_tmp(tmp_path):
    return tmp_path / "results"


def cleanup_result_files(settings, prefix: str) -> None:
    for path in Path(settings.results_dir).glob(f"{prefix}*"):
        if path.is_file():
            path.unlink()
