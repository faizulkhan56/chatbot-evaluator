from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.llm.openai_client import LLMRequestError, MissingOpenAIAPIKeyError
from src.rag.retriever import get_retriever


def rag_answer(
    question: str,
    model_name: str = "gpt-4o-mini",
    top_k: int = 6,
) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise MissingOpenAIAPIKeyError("OPENAI_API_KEY is missing. Add it to .env before querying RAG.")

    retriever = get_retriever(top_k=top_k)
    retrieved_docs = retriever.invoke(question)
    docs_string = _format_documents(retrieved_docs)

    prompt = f"""
You are a helpful assistant who is good at analyzing source information and answering questions.

Use the following source documents to answer the user's question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}

Question:
{question}
"""

    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        response = llm.invoke(prompt)
    except Exception as exc:
        raise LLMRequestError(f"RAG generation failed: {exc}") from exc

    return {
        "question": question,
        "answer": response.content,
        "retrieved_documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ],
    }


def _format_documents(documents: List[Any]) -> str:
    if not documents:
        return "No source documents were retrieved."

    formatted = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"Document {index} source={source}\n{doc.page_content}")
    return "\n\n".join(formatted)
