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


def test_compare_accepts_provider_model_objects(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    def fake_compare_models(**kwargs):
        assert kwargs["models"][0].provider == "openai"
        assert kwargs["models"][1].provider == "huggingface"
        return {
            "mode": "compare_chatbot",
            "dataset_name": "chatbot_eval_sample",
            "evaluator_provider": "openai",
            "evaluator_model": "gpt-4o-mini",
            "models": [
                {"provider": "openai", "model": "gpt-4o-mini"},
                {"provider": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
            ],
            "summary_by_model": [
                {
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                    "total_examples": 1,
                    "correctness_score": 1.0,
                    "concision_score": 1.0,
                }
            ],
            "results": [
                {
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                    "results": [],
                }
            ],
            "saved_result_path": None,
            "saved_csv_path": None,
        }

    monkeypatch.setattr(evaluation_route, "compare_models", fake_compare_models)

    response = client.post(
        "/evaluate/compare",
        json={
            "mode": "chatbot",
            "dataset_name": "chatbot_eval_sample",
            "models": [
                {"provider": "openai", "model": "gpt-4o-mini"},
                {"provider": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
            ],
            "save_result": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["models"][1]["provider"] == "huggingface"
