from typing import Any


_active_vectorstore: Any = None


class VectorStoreNotInitializedError(RuntimeError):
    """Raised when querying before /rag/ingest has initialized the active store."""


class InvalidTopKError(ValueError):
    """Raised when top_k is invalid."""


class SourceFilterNoMatchError(LookupError):
    """Raised when retrieved chunks do not match the requested source filter."""


def set_active_vectorstore(vectorstore: Any) -> None:
    global _active_vectorstore
    _active_vectorstore = vectorstore


def get_active_vectorstore() -> Any:
    if _active_vectorstore is None:
        raise VectorStoreNotInitializedError("RAG vector store is not initialized. Please call /rag/ingest first.")
    return _active_vectorstore


def is_vectorstore_ready() -> bool:
    return _active_vectorstore is not None


def get_retriever(top_k: int = 6) -> Any:
    if top_k <= 0:
        raise InvalidTopKError("top_k must be greater than 0.")
    vectorstore = get_active_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
