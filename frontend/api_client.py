import os
from typing import Any, Dict, Iterable, List, Tuple

import requests


FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120


class APIClientError(RuntimeError):
    """Raised when a backend request fails."""


def get_health() -> Dict[str, Any]:
    return _get("/health")


def list_datasets() -> Dict[str, Any]:
    return _get("/datasets")


def create_dataset(name: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _post(
        "/datasets",
        {
            "name": name,
            "examples": examples,
            "overwrite": True,
        },
    )


def evaluate_chatbot(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/evaluate/chatbot", payload)


def upload_documents(files: Iterable[Tuple[str, bytes, str]]) -> Dict[str, Any]:
    request_files = [("files", file_tuple) for file_tuple in files]
    return _post_files("/documents/upload", request_files)


def list_documents() -> Dict[str, Any]:
    return _get("/documents")


def ingest_rag(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/rag/ingest", payload)


def query_rag(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/rag/query", payload)


def evaluate_rag(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/evaluate/rag", payload)


def compare_models(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _post("/evaluate/compare", payload)


def list_results() -> Dict[str, Any]:
    return _get("/results")


def get_result(file_name: str) -> Any:
    return _get(f"/results/{file_name}")


def _get(path: str) -> Any:
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.ConnectionError as exc:
        raise APIClientError(f"Backend is not reachable at {FASTAPI_BASE_URL}. Start FastAPI with: uvicorn app:app --reload") from exc
    except requests.RequestException as exc:
        raise APIClientError(f"Backend request failed: {exc}") from exc
    return _handle_response(response)


def _post(path: str, payload: Dict[str, Any]) -> Any:
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.ConnectionError as exc:
        raise APIClientError(f"Backend is not reachable at {FASTAPI_BASE_URL}. Start FastAPI with: uvicorn app:app --reload") from exc
    except requests.RequestException as exc:
        raise APIClientError(f"Backend request failed: {exc}") from exc
    return _handle_response(response)


def _post_files(path: str, files) -> Any:
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}{path}", files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.ConnectionError as exc:
        raise APIClientError(f"Backend is not reachable at {FASTAPI_BASE_URL}. Start FastAPI with: uvicorn app:app --reload") from exc
    except requests.RequestException as exc:
        raise APIClientError(f"Backend request failed: {exc}") from exc
    return _handle_response(response)


def _handle_response(response: requests.Response) -> Any:
    if response.ok:
        if not response.content:
            return {}
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if response.status_code == 404:
        raise APIClientError(f"Not found: {detail}")
    if response.status_code == 422:
        raise APIClientError(f"Validation error: {detail}")
    raise APIClientError(f"Backend error {response.status_code}: {detail}")
