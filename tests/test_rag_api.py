def test_rag_ingest_api_with_mocked_service(client, monkeypatch):
    from src.api.routes import rag as rag_route

    def fake_ingest_documents(source_dir, chunk_size, chunk_overlap, embedding_model):
        return {
            "status": "success",
            "documents_loaded": 1,
            "chunks_created": 3,
            "embedding_model": embedding_model,
        }

    monkeypatch.setattr(rag_route, "ingest_documents", fake_ingest_documents)

    response = client.post(
        "/rag/ingest",
        json={
            "source_dir": "data/documents",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_model": "text-embedding-3-small",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunks_created"] == 3


def test_rag_query_api_with_mocked_service(client, monkeypatch):
    from src.api.routes import rag as rag_route

    def fake_query_rag(question, model_name, top_k):
        return {
            "question": question,
            "answer": "It observes tool outputs and updates reasoning.",
            "retrieved_documents": [
                {
                    "content": "ReAct agents reason, act, observe, and update reasoning.",
                    "metadata": {"source": "rag_sample.txt"},
                }
            ],
        }

    monkeypatch.setattr(rag_route, "query_rag", fake_query_rag)

    response = client.post(
        "/rag/query",
        json={
            "question": "How does ReAct use self-reflection?",
            "model_name": "gpt-4o-mini",
            "top_k": 6,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "It observes tool outputs and updates reasoning."
    assert data["retrieved_documents"][0]["metadata"]["source"] == "rag_sample.txt"


def test_rag_query_before_ingest_returns_409(client, monkeypatch):
    from src.api.routes import rag as rag_route
    from src.rag.retriever import VectorStoreNotInitializedError

    def fake_query_rag(question, model_name, top_k):
        raise VectorStoreNotInitializedError("RAG vector store is not initialized. Please call /rag/ingest first.")

    monkeypatch.setattr(rag_route, "query_rag", fake_query_rag)

    response = client.post(
        "/rag/query",
        json={"question": "How does ReAct use self-reflection?"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "RAG vector store is not initialized. Please call /rag/ingest first."


def test_rag_query_invalid_top_k(client):
    response = client.post(
        "/rag/query",
        json={
            "question": "How does ReAct use self-reflection?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
