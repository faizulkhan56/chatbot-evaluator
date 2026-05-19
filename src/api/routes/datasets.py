from fastapi import APIRouter, HTTPException, status

from src.schemas.datasets import DatasetCreateRequest, DatasetListResponse, DatasetResponse
from src.storage.dataset_store import (
    DatasetAlreadyExistsError,
    DatasetNotFoundError,
    InvalidDatasetNameError,
    LocalDatasetStore,
)


router = APIRouter(prefix="/datasets", tags=["datasets"])


def get_dataset_store() -> LocalDatasetStore:
    return LocalDatasetStore()


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    datasets = get_dataset_store().list()
    return DatasetListResponse(datasets=datasets, count=len(datasets))


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(request: DatasetCreateRequest) -> DatasetResponse:
    try:
        return get_dataset_store().create(request)
    except InvalidDatasetNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatasetAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/{name}", response_model=DatasetResponse)
def get_dataset(name: str) -> DatasetResponse:
    try:
        return get_dataset_store().get(name)
    except InvalidDatasetNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
