from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from src.core.config import get_settings
from src.llm.openai_client import MissingOpenAIAPIKeyError


class VectorStoreBuildError(RuntimeError):
    """Raised when embedding or vector store creation fails."""


def build_vectorstore(
    documents: List[Document],
    embedding_model: str = "text-embedding-3-small",
) -> InMemoryVectorStore:
    settings = get_settings()
    if not settings.openai_api_key:
        raise MissingOpenAIAPIKeyError("OPENAI_API_KEY is missing. Add it to .env before ingesting documents.")
    if not documents:
        raise VectorStoreBuildError("Cannot build vector store with no documents.")

    try:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.openai_api_key,
        )
        return InMemoryVectorStore.from_documents(documents, embeddings)
    except Exception as exc:
        raise VectorStoreBuildError(f"Failed to build vector store: {exc}") from exc
