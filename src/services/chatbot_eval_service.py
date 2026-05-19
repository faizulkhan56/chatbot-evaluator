from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import ValidationError

from src.llm.evaluators import evaluate_concision, evaluate_correctness
from src.llm.openai_client import generate_chat_response
from src.schemas.evaluation import ChatbotEvaluationResponse
from src.storage.dataset_store import DatasetNotFoundError, LocalDatasetStore
from src.storage.result_store import LocalResultStore


class ChatbotEvaluationError(Exception):
    """Base exception for chatbot evaluation failures."""


class EmptyDatasetError(ChatbotEvaluationError):
    """Raised when a dataset has no examples."""


class InvalidEvaluationDatasetError(ChatbotEvaluationError):
    """Raised when a dataset example does not match the expected evaluation shape."""


def evaluate_chatbot_dataset(
    dataset_name: str,
    model_name: str,
    evaluator_model: str,
    instructions: str,
    concision_threshold: int,
    save_result: bool,
) -> Dict[str, Any]:
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

        model_response = generate_chat_response(
            question=question,
            model_name=model_name,
            instructions=instructions,
            temperature=0,
        )
        correctness = evaluate_correctness(
            question=question,
            reference_answer=reference_answer,
            model_response=model_response,
            evaluator_model=evaluator_model,
        )
        concision = evaluate_concision(
            model_response=model_response,
            threshold=concision_threshold,
        )

        results.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "model_response": model_response,
                "correctness": correctness,
                "concision": concision,
            }
        )

    total_examples = len(results)
    summary = {
        "correctness_score": _score(results, "correctness"),
        "concision_score": _score(results, "concision"),
    }
    response = {
        "mode": "chatbot",
        "dataset_name": dataset_name,
        "model_name": model_name,
        "evaluator_model": evaluator_model,
        "created_at": _utc_now(),
        "total_examples": total_examples,
        "summary": summary,
        "results": results,
        "saved_result_path": None,
        "saved_csv_path": None,
    }

    if save_result:
        prefix = f"chatbot_evaluation_{dataset_name}_{model_name}"
        bundle = LocalResultStore().save_result_bundle(prefix, response)
        response["saved_result_path"] = bundle["json_path"]
        response["saved_csv_path"] = bundle["csv_path"]

    return response


def _score(results: List[Dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return round(sum(1 for result in results if result[key]) / len(results), 2)


def _validated_response(response: Dict[str, Any]) -> str:
    if hasattr(ChatbotEvaluationResponse, "model_validate"):
        validated = ChatbotEvaluationResponse.model_validate(response)
        return validated.model_dump_json(indent=2)

    validated = ChatbotEvaluationResponse(**response)
    return validated.json(indent=2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
