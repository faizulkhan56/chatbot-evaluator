def test_rag_evaluation_api_with_mocked_service(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    def fake_evaluate_rag_dataset(**kwargs):
        return {
            "mode": "rag",
            "dataset_name": "rag_eval_sample",
            "model_name": "gpt-4o-mini",
            "evaluator_model": "gpt-4o-mini",
            "total_examples": 1,
            "summary": {
                "correctness_score": 1.0,
                "groundedness_score": 1.0,
                "relevance_score": 1.0,
                "retrieval_relevance_score": 1.0,
            },
            "results": [
                {
                    "question": "How does ReAct use self-reflection?",
                    "reference_answer": "ReAct observes and reasons about tool outputs.",
                    "rag_answer": "It observes results and updates reasoning.",
                    "correctness": True,
                    "groundedness": True,
                    "relevance": True,
                    "retrieval_relevance": True,
                    "retrieved_docs_count": 1,
                }
            ],
            "saved_result_path": None,
            "saved_csv_path": None,
        }

    monkeypatch.setattr(evaluation_route, "evaluate_rag_dataset", fake_evaluate_rag_dataset)

    response = client.post(
        "/evaluate/rag",
        json={
            "dataset_name": "rag_eval_sample",
            "top_k": 6,
            "save_result": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "rag"
    assert data["summary"]["groundedness_score"] == 1.0
    assert len(data["results"]) == 1


def test_rag_evaluation_invalid_top_k(client):
    response = client.post(
        "/evaluate/rag",
        json={
            "dataset_name": "rag_eval_sample",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_rag_evaluation_accepts_huggingface_provider(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    def fake_evaluate_rag_dataset(**kwargs):
        assert kwargs["provider"] == "huggingface"
        assert kwargs["evaluator_provider"] == "openai"
        return {
            "mode": "rag",
            "dataset_name": "rag_eval_sample",
            "provider": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "evaluator_provider": "openai",
            "evaluator_model": "gpt-4o-mini",
            "total_examples": 1,
            "summary": {
                "correctness_score": 1.0,
                "groundedness_score": 1.0,
                "relevance_score": 1.0,
                "retrieval_relevance_score": 1.0,
            },
            "results": [
                {
                    "question": "How does ReAct use self-reflection?",
                    "reference_answer": "ReAct observes and reasons about tool outputs.",
                    "rag_answer": "It observes results and updates reasoning.",
                    "correctness": True,
                    "groundedness": True,
                    "relevance": True,
                    "retrieval_relevance": True,
                    "retrieved_docs_count": 1,
                }
            ],
            "saved_result_path": None,
            "saved_csv_path": None,
        }

    monkeypatch.setattr(evaluation_route, "evaluate_rag_dataset", fake_evaluate_rag_dataset)

    response = client.post(
        "/evaluate/rag",
        json={
            "dataset_name": "rag_eval_sample",
            "provider": "huggingface",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "top_k": 6,
            "save_result": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "huggingface"


def test_rag_evaluation_accepts_source_filter(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    def fake_evaluate_rag_dataset(**kwargs):
        assert kwargs["source_filter"] == "Security as Code.pdf"
        return {
            "mode": "rag",
            "dataset_name": "rag_eval_sample",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "evaluator_provider": "openai",
            "evaluator_model": "gpt-4o-mini",
            "source_filter": "Security as Code.pdf",
            "total_examples": 1,
            "summary": {
                "correctness_score": 1.0,
                "groundedness_score": 1.0,
                "relevance_score": 1.0,
                "retrieval_relevance_score": 1.0,
            },
            "results": [
                {
                    "question": "Where should IaC files be stored?",
                    "reference_answer": "In version control.",
                    "rag_answer": "Store IaC files in version control.",
                    "correctness": True,
                    "groundedness": True,
                    "relevance": True,
                    "retrieval_relevance": True,
                    "retrieved_docs_count": 1,
                }
            ],
            "saved_result_path": None,
            "saved_csv_path": None,
        }

    monkeypatch.setattr(evaluation_route, "evaluate_rag_dataset", fake_evaluate_rag_dataset)

    response = client.post(
        "/evaluate/rag",
        json={
            "dataset_name": "rag_eval_sample",
            "source_filter": "Security as Code.pdf",
            "save_result": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["source_filter"] == "Security as Code.pdf"
