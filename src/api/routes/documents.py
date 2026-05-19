from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.schemas.documents import DocumentListResponse, DocumentUploadResponse
from src.storage.document_store import (
    InvalidDocumentFileNameError,
    LocalDocumentStore,
    UnsupportedDocumentTypeError,
)


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    documents = LocalDocumentStore().list_documents()
    return DocumentListResponse(documents=documents)


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_documents(files: List[UploadFile] = File(...)) -> DocumentUploadResponse:
    store = LocalDocumentStore()
    uploaded_files = []

    for uploaded_file in files:
        try:
            uploaded_files.append(store.save_upload(uploaded_file.filename or "", uploaded_file.file))
        except (InvalidDocumentFileNameError, UnsupportedDocumentTypeError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentUploadResponse(uploaded_files=uploaded_files)
