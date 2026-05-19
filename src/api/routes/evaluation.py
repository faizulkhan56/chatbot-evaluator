from fastapi import APIRouter, HTTPException, status

from src.llm.openai_client import LLMRequestError, MissingOpenAIAPIKeyError
from src.rag.retriever import InvalidTopKError, VectorStoreNotInitializedError
from src.schemas.evaluation import (
    ChatbotEvaluationRequest,
    ChatbotEvaluationResponse,
    CompareEvaluationRequest,
    CompareEvaluationResponse,
    RagEvaluationRequest,
    RagEvaluationResponse,
)
from src.services.chatbot_eval_service import (
    EmptyDatasetError,
    InvalidEvaluationDatasetError,
    evaluate_chatbot_dataset,
)
from src.services.compare_eval_service import CompareEvaluationError, compare_models
from src.services.rag_eval_service import evaluate_rag_dataset
from src.storage.dataset_store import DatasetNotFoundError, InvalidDatasetNameError


router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("/chatbot", response_model=ChatbotEvaluationResponse)
def evaluate_chatbot(request: ChatbotEvaluationRequest) -> ChatbotEvaluationResponse:
    try:
        result = evaluate_chatbot_dataset(
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            evaluator_model=request.evaluator_model,
            instructions=request.instructions,
            concision_threshold=request.concision_threshold,
            save_result=request.save_result,
        )
        return ChatbotEvaluationResponse(**result)
    except InvalidDatasetNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmptyDatasetError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidEvaluationDatasetError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/compare", response_model=CompareEvaluationResponse)
def compare_evaluations(request: CompareEvaluationRequest) -> CompareEvaluationResponse:
    try:
        result = compare_models(
            mode=request.mode,
            dataset_name=request.dataset_name,
            models=request.models,
            evaluator_model=request.evaluator_model,
            save_result=request.save_result,
            instructions=request.instructions,
            concision_threshold=request.concision_threshold,
            top_k=request.top_k,
        )
        return CompareEvaluationResponse(**result)
    except CompareEvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except InvalidDatasetNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmptyDatasetError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidEvaluationDatasetError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except InvalidTopKError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except VectorStoreNotInitializedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/rag", response_model=RagEvaluationResponse)
def evaluate_rag(request: RagEvaluationRequest) -> RagEvaluationResponse:
    try:
        result = evaluate_rag_dataset(
            dataset_name=request.dataset_name,
            model_name=request.model_name,
            evaluator_model=request.evaluator_model,
            top_k=request.top_k,
            save_result=request.save_result,
        )
        return RagEvaluationResponse(**result)
    except InvalidDatasetNameError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EmptyDatasetError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InvalidEvaluationDatasetError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except InvalidTopKError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except VectorStoreNotInitializedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except MissingOpenAIAPIKeyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
