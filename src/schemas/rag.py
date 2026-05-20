from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from src.core.config import get_settings


settings = get_settings()


class RagIngestRequest(BaseModel):
    source_dir: str = "data/documents"
    chunk_size: int = Field(default=settings.default_chunk_size, gt=0)
    chunk_overlap: int = Field(default=settings.default_chunk_overlap, ge=0)
    embedding_model: str = settings.default_embedding_model


class RagIngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    embedding_model: str


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    provider: Literal["openai", "huggingface"] = settings.default_provider
    model_name: str = settings.default_model
    top_k: int = Field(default=settings.default_top_k, gt=0)
    source_filter: str | None = None


class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RagQueryResponse(BaseModel):
    question: str
    source_filter: str | None = None
    answer: str
    retrieved_documents: List[RetrievedDocument]
