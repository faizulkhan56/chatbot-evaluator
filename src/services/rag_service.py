from typing import Any, Dict

from src.rag.loaders import load_local_documents
from src.rag.rag_chain import rag_answer
from src.rag.retriever import set_active_vectorstore
from src.rag.splitter import split_documents
from src.rag.vectorstore import build_vectorstore


def ingest_documents(
    source_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> Dict[str, Any]:
    documents = load_local_documents(source_dir)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectorstore = build_vectorstore(chunks, embedding_model=embedding_model)
    set_active_vectorstore(vectorstore)

    return {
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "embedding_model": embedding_model,
    }


def query_rag(
    question: str,
    model_name: str,
    top_k: int,
) -> Dict[str, Any]:
    return rag_answer(
        question=question,
        model_name=model_name,
        top_k=top_k,
    )
