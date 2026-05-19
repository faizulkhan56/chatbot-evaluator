from fastapi import APIRouter, HTTPException, status

from src.llm.openai_client import LLMRequestError, MissingOpenAIAPIKeyError
from src.rag.loaders import DocumentLoadError, NoSupportedDocumentsError, SourceDirectoryNotFoundError
from src.rag.retriever import InvalidTopKError, VectorStoreNotInitializedError
from src.rag.splitter import InvalidChunkSettingsError
from src.rag.vectorstore import VectorStoreBuildError
from src.schemas.rag import RagIngestRequest, RagIngestResponse, RagQueryRequest, RagQueryResponse
from src.services.rag_service import ingest_documents, query_rag


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ingest", response_model=RagIngestResponse)
def ingest_rag_documents(request: RagIngestRequest) -> RagIngestResponse:
    try:
        result = ingest_documents(
            source_dir=request.source_dir,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
        )
        return RagIngestResponse(**result)
    except SourceDirectoryNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NoSupportedDocumentsError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidChunkSettingsError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except (DocumentLoadError, VectorStoreBuildError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/query", response_model=RagQueryResponse)
def query_rag_documents(request: RagQueryRequest) -> RagQueryResponse:
    try:
        result = query_rag(
            question=request.question,
            model_name=request.model_name,
            top_k=request.top_k,
        )
        return RagQueryResponse(**result)
    except InvalidTopKError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except VectorStoreNotInitializedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
