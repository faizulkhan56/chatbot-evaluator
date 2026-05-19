from pathlib import Path

from src.storage.result_store import LocalResultStore


def test_result_api_list_and_retrieve_json_csv(client, settings):
    prefix = "test_result_api_sample"
    store = LocalResultStore()
    bundle = store.save_result_bundle(
        prefix,
        {
            "mode": "chatbot",
            "dataset_name": "test",
            "created_at": "2026-05-19T00:00:00Z",
            "results": [{"question": "What is testing?", "correctness": True}],
            "saved_result_path": None,
            "saved_csv_path": None,
        },
    )

    try:
        list_response = client.get("/results")
        assert list_response.status_code == 200
        names = {item["file_name"] for item in list_response.json()["results"]}
        assert Path(bundle["json_path"]).name in names
        assert Path(bundle["csv_path"]).name in names

        json_response = client.get(f"/results/{Path(bundle['json_path']).name}")
        assert json_response.status_code == 200
        assert json_response.json()["mode"] == "chatbot"

        csv_response = client.get(f"/results/{Path(bundle['csv_path']).name}")
        assert csv_response.status_code == 200
        assert csv_response.json()[0]["question"] == "What is testing?"
    finally:
        for file_path in Path(settings.results_dir).glob(f"{prefix}*"):
            if file_path.is_file():
                file_path.unlink()


def test_result_api_missing_and_path_traversal(client):
    assert client.get("/results/test_result_api_missing.json").status_code == 404

    response = client.get("/results/..%2F.env")
    assert response.status_code in {400, 404}
