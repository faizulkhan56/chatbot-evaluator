def _mock_compare_response(mode):
    return {
        "mode": mode,
        "dataset_name": "chatbot_eval_sample" if mode == "chatbot" else "rag_eval_sample",
        "evaluator_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4.1-mini"],
        "summary_by_model": [
            {
                "model_name": "gpt-4o-mini",
                "total_examples": 1,
                "correctness_score": 1.0,
                "concision_score": 1.0 if mode == "chatbot" else None,
                "groundedness_score": 1.0 if mode == "rag" else None,
                "relevance_score": 1.0 if mode == "rag" else None,
                "retrieval_relevance_score": 1.0 if mode == "rag" else None,
            }
        ],
        "results": [
            {
                "model_name": "gpt-4o-mini",
                "results": [],
            }
        ],
        "saved_result_path": None,
        "saved_csv_path": None,
    }


def test_compare_chatbot_api_with_mocked_service(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    monkeypatch.setattr(evaluation_route, "compare_models", lambda **kwargs: _mock_compare_response("chatbot"))

    response = client.post(
        "/evaluate/compare",
        json={
            "mode": "chatbot",
            "dataset_name": "chatbot_eval_sample",
            "models": ["gpt-4o-mini", "gpt-4.1-mini"],
            "save_result": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "chatbot"


def test_compare_rag_api_with_mocked_service(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    monkeypatch.setattr(evaluation_route, "compare_models", lambda **kwargs: _mock_compare_response("rag"))

    response = client.post(
        "/evaluate/compare",
        json={
            "mode": "rag",
            "dataset_name": "rag_eval_sample",
            "models": ["gpt-4o-mini", "gpt-4.1-mini"],
            "save_result": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "rag"


def test_compare_api_validation_errors(client):
    invalid_mode = client.post(
        "/evaluate/compare",
        json={
            "mode": "bad",
            "dataset_name": "chatbot_eval_sample",
            "models": ["gpt-4o-mini"],
        },
    )
    empty_models = client.post(
        "/evaluate/compare",
        json={
            "mode": "chatbot",
            "dataset_name": "chatbot_eval_sample",
            "models": [],
        },
    )

    assert invalid_mode.status_code == 422
    assert empty_models.status_code == 422
