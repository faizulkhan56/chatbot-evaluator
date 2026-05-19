from fastapi import FastAPI

from src.api.routes import datasets, documents, evaluation, health, rag, results
from src.storage.paths import ensure_storage_dirs


def create_app() -> FastAPI:
    ensure_storage_dirs()

    application = FastAPI(
        title="Chatbot Evaluator",
        version="0.1.0",
        description="Local chatbot and RAG evaluation API.",
    )
    application.include_router(health.router)
    application.include_router(datasets.router)
    application.include_router(documents.router)
    application.include_router(evaluation.router)
    application.include_router(rag.router)
    application.include_router(results.router)
    return application


app = create_app()
