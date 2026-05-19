def test_chatbot_evaluation_api_with_mocked_service(client, monkeypatch):
    from src.api.routes import evaluation as evaluation_route

    def fake_evaluate_chatbot_dataset(**kwargs):
        return {
            "mode": "chatbot",
            "dataset_name": "chatbot_eval_sample",
            "model_name": "gpt-4o-mini",
            "evaluator_model": "gpt-4o-mini",
            "total_examples": 1,
            "summary": {
                "correctness_score": 1.0,
                "concision_score": 1.0,
            },
            "results": [
                {
                    "question": "What is LangChain?",
                    "reference_answer": "A framework for building LLM applications",
                    "model_response": "LangChain is a framework for LLM apps.",
                    "correctness": True,
                    "concision": True,
                }
            ],
            "saved_result_path": None,
            "saved_csv_path": None,
        }

    monkeypatch.setattr(evaluation_route, "evaluate_chatbot_dataset", fake_evaluate_chatbot_dataset)

    response = client.post(
        "/evaluate/chatbot",
        json={
            "dataset_name": "chatbot_eval_sample",
            "save_result": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "chatbot"
    assert data["summary"]["correctness_score"] == 1.0
    assert len(data["results"]) == 1


def test_chatbot_evaluation_invalid_concision_threshold(client):
    response = client.post(
        "/evaluate/chatbot",
        json={
            "dataset_name": "chatbot_eval_sample",
            "concision_threshold": 0,
        },
    )

    assert response.status_code == 422
