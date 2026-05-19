from typing import List

from pydantic import BaseModel


class UploadedDocumentInfo(BaseModel):
    file_name: str
    path: str
    size_bytes: int


class DocumentFileInfo(BaseModel):
    file_name: str
    path: str
    file_type: str
    size_bytes: int


class DocumentUploadResponse(BaseModel):
    uploaded_files: List[UploadedDocumentInfo]


class DocumentListResponse(BaseModel):
    documents: List[DocumentFileInfo]
