from typing import Any, Dict, List

from src.llm.openai_client import LLMRequestError
from src.llm.providers import generate_response, normalize_provider
from src.rag.retriever import SourceFilterNoMatchError, get_retriever


def rag_answer(
    question: str,
    model_name: str = "gpt-4o-mini",
    top_k: int = 6,
    provider: str = "openai",
    source_filter: str | None = None,
) -> Dict[str, Any]:
    provider = normalize_provider(provider)
    candidate_k = top_k * 5 if source_filter else top_k
    retriever = get_retriever(top_k=candidate_k)
    candidate_docs = retriever.invoke(question)
    retrieved_docs = _filter_documents(candidate_docs, source_filter, top_k)
    docs_string = _format_documents(retrieved_docs)

    instructions = f"""
You are a helpful assistant who is good at analyzing source information and answering questions.

Use the following source documents to answer the user's question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}
"""

    try:
        answer = generate_response(
            provider=provider,
            model_name=model_name,
            question=question,
            instructions=instructions,
            temperature=0,
        )
    except Exception as exc:
        raise LLMRequestError(f"RAG generation failed: {exc}") from exc

    return {
        "question": question,
        "source_filter": source_filter,
        "answer": answer,
        "retrieved_documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ],
    }


def _filter_documents(documents: List[Any], source_filter: str | None, top_k: int) -> List[Any]:
    if not source_filter:
        return documents[:top_k]

    normalized_filter = source_filter.lower()
    filtered = [
        doc
        for doc in documents
        if normalized_filter in str(doc.metadata.get("source", "")).lower()
    ]
    if not filtered:
        raise SourceFilterNoMatchError(
            f"No retrieved chunks matched source_filter='{source_filter}'. "
            "Please confirm the document was uploaded and ingested."
        )
    return filtered[:top_k]


def _format_documents(documents: List[Any]) -> str:
    if not documents:
        return "No source documents were retrieved."

    formatted = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Document {index} source={source}\n{doc.page_content}")
    return "\n\n".join(formatted)
