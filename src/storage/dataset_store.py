import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.schemas.datasets import DatasetCreateRequest, DatasetIO, DatasetResponse
from src.storage.paths import datasets_dir


DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class DatasetStoreError(Exception):
    """Base exception for dataset storage errors."""


class InvalidDatasetNameError(DatasetStoreError):
    """Raised when a dataset name is not safe for local file storage."""


class DatasetAlreadyExistsError(DatasetStoreError):
    """Raised when creating a dataset that already exists."""


class DatasetNotFoundError(DatasetStoreError):
    """Raised when a dataset file does not exist."""


class LocalDatasetStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or datasets_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_name(self, name: str) -> None:
        if not DATASET_NAME_PATTERN.match(name):
            raise InvalidDatasetNameError(
                "Dataset name must contain only letters, numbers, underscores, and hyphens."
            )

    def _path_for(self, name: str) -> Path:
        self._validate_name(name)
        return self.base_dir / f"{name}.json"

    def create(self, request: DatasetCreateRequest) -> DatasetResponse:
        path = self._path_for(request.name)
        if path.exists() and not request.overwrite:
            raise DatasetAlreadyExistsError(f"Dataset already exists: {request.name}")

        dataset = DatasetIO(name=request.name, examples=request.examples)
        path.write_text(
            json.dumps(self._dump_dataset(dataset), indent=2),
            encoding="utf-8",
        )
        return self._to_response(dataset, path)

    def get(self, name: str) -> DatasetResponse:
        path = self._path_for(name)
        if not path.exists():
            raise DatasetNotFoundError(f"Dataset not found: {name}")

        dataset = self._load_dataset(path)
        return self._to_response(dataset, path)

    def list(self) -> List[DatasetResponse]:
        datasets = []
        for path in sorted(self.base_dir.glob("*.json")):
            dataset = self._load_dataset(path)
            datasets.append(self._to_response(dataset, path))
        return datasets

    def _load_dataset(self, path: Path) -> DatasetIO:
        raw = path.read_text(encoding="utf-8")
        if hasattr(DatasetIO, "model_validate_json"):
            return DatasetIO.model_validate_json(raw)
        return DatasetIO.parse_raw(raw)

    def _dump_dataset(self, dataset: DatasetIO) -> Dict[str, Any]:
        if hasattr(dataset, "model_dump"):
            return dataset.model_dump(mode="json")
        return dataset.dict()

    def _to_response(self, dataset: DatasetIO, path: Path) -> DatasetResponse:
        return DatasetResponse(
            name=dataset.name,
            examples=dataset.examples,
            example_count=len(dataset.examples),
            path=str(path),
        )
