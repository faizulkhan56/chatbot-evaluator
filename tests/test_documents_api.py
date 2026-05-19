from pathlib import Path


def _cleanup_document(settings, file_name: str) -> None:
    path = settings.documents_dir / file_name
    if path.exists():
        path.unlink()


def test_upload_txt_file_and_list_documents(client, settings):
    file_name = "test_upload_document.txt"
    _cleanup_document(settings, file_name)

    try:
        response = client.post(
            "/documents/upload",
            files=[
                (
                    "files",
                    (file_name, b"Uploaded test document for RAG ingestion.", "text/plain"),
                )
            ],
        )

        assert response.status_code == 200
        uploaded = response.json()["uploaded_files"][0]
        assert uploaded["file_name"] == file_name
        assert uploaded["path"] == f"data\\documents\\{file_name}" or uploaded["path"] == f"data/documents/{file_name}"
        assert uploaded["size_bytes"] > 0

        list_response = client.get("/documents")
        assert list_response.status_code == 200
        names = {document["file_name"] for document in list_response.json()["documents"]}
        assert file_name in names
    finally:
        _cleanup_document(settings, file_name)


def test_upload_unsupported_extension_returns_400(client):
    response = client.post(
        "/documents/upload",
        files=[
            (
                "files",
                ("unsupported.exe", b"not allowed", "application/octet-stream"),
            )
        ],
    )

    assert response.status_code == 400


def test_upload_unsafe_filename_is_rejected(client):
    response = client.post(
        "/documents/upload",
        files=[
            (
                "files",
                ("../unsafe.txt", b"unsafe", "text/plain"),
            )
        ],
    )

    assert response.status_code == 400


def test_rag_ingest_still_works_with_mocked_vectorstore(client, settings, monkeypatch):
    from src.services import rag_service

    file_name = "test_ingest_uploaded_document.txt"
    path = settings.documents_dir / file_name
    path.write_text("ReAct agents reason, act, observe, and update reasoning.", encoding="utf-8")

    class DummyVectorStore:
        def as_retriever(self, search_kwargs=None):
            return self

    try:
        monkeypatch.setattr(rag_service, "build_vectorstore", lambda documents, embedding_model: DummyVectorStore())

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
        assert response.json()["documents_loaded"] >= 1
    finally:
        if path.exists():
            path.unlink()
