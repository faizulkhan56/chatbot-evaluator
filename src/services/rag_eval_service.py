from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import ValidationError

from src.llm.evaluators import (
    evaluate_answer_relevance,
    evaluate_correctness,
    evaluate_groundedness,
    evaluate_retrieval_relevance,
)
from src.rag.retriever import InvalidTopKError, VectorStoreNotInitializedError
from src.schemas.evaluation import RagEvaluationResponse
from src.services.chatbot_eval_service import EmptyDatasetError, InvalidEvaluationDatasetError
from src.services.rag_service import query_rag
from src.storage.dataset_store import LocalDatasetStore
from src.storage.result_store import LocalResultStore


def evaluate_rag_dataset(
    dataset_name: str,
    model_name: str,
    evaluator_model: str,
    top_k: int,
    save_result: bool,
) -> Dict[str, Any]:
    if top_k <= 0:
        raise InvalidTopKError("top_k must be greater than 0.")

    try:
        dataset = LocalDatasetStore().get(dataset_name)
    except ValidationError as exc:
        raise InvalidEvaluationDatasetError(f"Invalid dataset format: {exc}") from exc

    if not dataset.examples:
        raise EmptyDatasetError(f"Dataset is empty: {dataset_name}")

    results = []
    for index, example in enumerate(dataset.examples, start=1):
        question = example.inputs.get("question")
        reference_answer = example.outputs.get("answer")

        if not isinstance(question, str) or not question.strip():
            raise InvalidEvaluationDatasetError(f"Example {index} is missing inputs.question.")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise InvalidEvaluationDatasetError(f"Example {index} is missing outputs.answer.")

        rag_result = query_rag(
            question=question,
            model_name=model_name,
            top_k=top_k,
        )
        rag_answer = rag_result["answer"]
        retrieved_documents = rag_result["retrieved_documents"]

        correctness = evaluate_correctness(
            question=question,
            reference_answer=reference_answer,
            model_response=rag_answer,
            evaluator_model=evaluator_model,
        )
        groundedness = evaluate_groundedness(
            retrieved_documents=retrieved_documents,
            model_answer=rag_answer,
            evaluator_model=evaluator_model,
        )
        relevance = evaluate_answer_relevance(
            question=question,
            model_answer=rag_answer,
            evaluator_model=evaluator_model,
        )
        retrieval_relevance = evaluate_retrieval_relevance(
            question=question,
            retrieved_documents=retrieved_documents,
            evaluator_model=evaluator_model,
        )

        results.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "rag_answer": rag_answer,
                "correctness": correctness,
                "groundedness": groundedness,
                "relevance": relevance,
                "retrieval_relevance": retrieval_relevance,
                "retrieved_docs_count": len(retrieved_documents),
            }
        )

    response = {
        "mode": "rag",
        "dataset_name": dataset_name,
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "created_at": _utc_now(),
        "total_examples": len(results),
        "summary": {
            "correctness_score": _score(results, "correctness"),
            "groundedness_score": _score(results, "groundedness"),
            "relevance_score": _score(results, "relevance"),
            "retrieval_relevance_score": _score(results, "retrieval_relevance"),
        },
        "results": results,
        "saved_result_path": None,
        "saved_csv_path": None,
    }

    if save_result:
        prefix = f"rag_evaluation_{dataset_name}_{model_name}"
        bundle = LocalResultStore().save_result_bundle(prefix, response)
        response["saved_result_path"] = bundle["json_path"]
        response["saved_csv_path"] = bundle["csv_path"]

    return response


def _score(results: List[Dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return round(sum(1 for result in results if result[key]) / len(results), 2)


def _validated_response_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(RagEvaluationResponse, "model_validate"):
        validated = RagEvaluationResponse.model_validate(response)
        return validated.model_dump(mode="json")

    validated = RagEvaluationResponse(**response)
    return validated.dict()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
