def test_dataset_api_create_get_and_list(client, cleanup_dataset):
    dataset_name = cleanup_dataset("test_dataset_api_sample")
    payload = {
        "name": dataset_name,
        "examples": [
            {
                "inputs": {"question": "What is testing?"},
                "outputs": {"answer": "Testing verifies behavior."},
            }
        ],
        "overwrite": True,
    }

    list_response = client.get("/datasets")
    assert list_response.status_code == 200

    create_response = client.post("/datasets", json=payload)
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["name"] == dataset_name
    assert created["example_count"] == 1

    get_response = client.get(f"/datasets/{dataset_name}")
    assert get_response.status_code == 200
    fetched = get_response.json()
    assert fetched["name"] == dataset_name
    assert fetched["examples"][0]["inputs"]["question"] == "What is testing?"


def test_dataset_api_missing_dataset_returns_404(client):
    response = client.get("/datasets/test_dataset_api_missing")

    assert response.status_code == 404


def test_dataset_api_rejects_unsafe_name(client):
    response = client.post(
        "/datasets",
        json={
            "name": "../unsafe",
            "examples": [],
        },
    )

    assert response.status_code == 400
