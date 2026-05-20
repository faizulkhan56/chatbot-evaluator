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

    def fake_query_rag(question, provider, model_name, top_k, source_filter=None):
        assert provider == "openai"
        assert source_filter is None
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

    def fake_query_rag(question, provider, model_name, top_k, source_filter=None):
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


def test_rag_query_accepts_provider_field(client, monkeypatch):
    from src.api.routes import rag as rag_route

    def fake_query_rag(question, provider, model_name, top_k, source_filter=None):
        assert provider == "huggingface"
        assert model_name == "meta-llama/Llama-3.1-8B-Instruct"
        return {
            "question": question,
            "answer": "Mock Hugging Face answer.",
            "retrieved_documents": [],
        }

    monkeypatch.setattr(rag_route, "query_rag", fake_query_rag)

    response = client.post(
        "/rag/query",
        json={
            "question": "What is the document about?",
            "provider": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Mock Hugging Face answer."


def test_rag_query_accepts_source_filter(client, monkeypatch):
    from src.api.routes import rag as rag_route

    def fake_query_rag(question, provider, model_name, top_k, source_filter=None):
        assert source_filter == "Security as Code.pdf"
        return {
            "question": question,
            "source_filter": source_filter,
            "answer": "Store IaC files in version control.",
            "retrieved_documents": [
                {
                    "content": "Store IaC files in a version control system.",
                    "metadata": {"source": "data/documents/Security as Code.pdf"},
                }
            ],
        }

    monkeypatch.setattr(rag_route, "query_rag", fake_query_rag)

    response = client.post(
        "/rag/query",
        json={
            "question": "Where should IaC files be stored?",
            "source_filter": "Security as Code.pdf",
        },
    )

    assert response.status_code == 200
    assert response.json()["source_filter"] == "Security as Code.pdf"
    assert "Security as Code.pdf" in response.json()["retrieved_documents"][0]["metadata"]["source"]


def test_rag_query_source_filter_no_match_returns_409(client, monkeypatch):
    from src.api.routes import rag as rag_route
    from src.rag.retriever import SourceFilterNoMatchError

    def fake_query_rag(question, provider, model_name, top_k, source_filter=None):
        raise SourceFilterNoMatchError(
            "No retrieved chunks matched source_filter='Security as Code.pdf'. Please confirm the document was uploaded and ingested."
        )

    monkeypatch.setattr(rag_route, "query_rag", fake_query_rag)

    response = client.post(
        "/rag/query",
        json={
            "question": "Where should IaC files be stored?",
            "source_filter": "Security as Code.pdf",
        },
    )

    assert response.status_code == 409
    assert "No retrieved chunks matched source_filter" in response.json()["detail"]


def test_rag_chain_filters_retrieved_documents_by_source(monkeypatch):
    from src.rag import rag_chain

    class FakeDocument:
        def __init__(self, content, source):
            self.page_content = content
            self.metadata = {"source": source}

    class FakeRetriever:
        def invoke(self, question):
            return [
                FakeDocument("wrong content", "data/documents/DevOps Webinar Visual Artifact.pdf"),
                FakeDocument("right content", "data/documents/Security as Code.pdf"),
            ]

    def fake_get_retriever(top_k):
        assert top_k == 10
        return FakeRetriever()

    def fake_generate_response(provider, model_name, question, instructions, temperature):
        assert "right content" in instructions
        assert "wrong content" not in instructions
        return "Store IaC files in version control."

    monkeypatch.setattr(rag_chain, "get_retriever", fake_get_retriever)
    monkeypatch.setattr(rag_chain, "generate_response", fake_generate_response)

    result = rag_chain.rag_answer(
        question="Where should IaC files be stored?",
        model_name="gpt-4o-mini",
        top_k=2,
        provider="openai",
        source_filter="Security as Code.pdf",
    )

    assert result["source_filter"] == "Security as Code.pdf"
    assert len(result["retrieved_documents"]) == 1
    assert "Security as Code.pdf" in result["retrieved_documents"][0]["metadata"]["source"]
