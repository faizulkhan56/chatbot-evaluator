from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.schemas.results import ResultListResponse
from src.storage.result_store import (
    InvalidResultFileNameError,
    LocalResultStore,
    ResultFileNotFoundError,
    UnsupportedResultFileTypeError,
)


router = APIRouter(prefix="/results", tags=["results"])


@router.get("", response_model=ResultListResponse)
def list_results() -> ResultListResponse:
    results = LocalResultStore().list_results()
    return ResultListResponse(results=results)


@router.get("/{result_file_name}")
def get_result(result_file_name: str) -> Any:
    try:
        return LocalResultStore().load_result(result_file_name)
    except InvalidResultFileNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ResultFileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except UnsupportedResultFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
