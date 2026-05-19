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
