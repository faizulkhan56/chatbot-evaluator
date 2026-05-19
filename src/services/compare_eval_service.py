from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.schemas.evaluation import DEFAULT_INSTRUCTIONS
from src.services.chatbot_eval_service import evaluate_chatbot_dataset
from src.services.rag_eval_service import evaluate_rag_dataset
from src.storage.result_store import LocalResultStore


class CompareEvaluationError(ValueError):
    """Raised when a model comparison request is invalid."""


def compare_models(
    mode: str,
    dataset_name: str,
    models: List[str],
    evaluator_model: str,
    save_result: bool,
    instructions: Optional[str] = None,
    concision_threshold: int = 30,
    top_k: int = 6,
) -> Dict[str, Any]:
    _validate_request(mode, dataset_name, models, concision_threshold, top_k)

    summary_by_model = []
    model_results = []

    for model_name in models:
        if mode == "chatbot":
            evaluation = evaluate_chatbot_dataset(
                dataset_name=dataset_name,
                model_name=model_name,
                evaluator_model=evaluator_model,
                instructions=instructions or DEFAULT_INSTRUCTIONS,
                concision_threshold=concision_threshold,
                save_result=False,
            )
        else:
            evaluation = evaluate_rag_dataset(
                dataset_name=dataset_name,
                model_name=model_name,
                evaluator_model=evaluator_model,
                top_k=top_k,
                save_result=False,
            )

        summary_by_model.append(_summary_for_model(evaluation))
        model_results.append(
            {
                "model_name": model_name,
                "results": evaluation["results"],
            }
        )

    response = {
        "mode": f"compare_{mode}",
        "dataset_name": dataset_name,
        "evaluator_model": evaluator_model,
        "models": models,
        "created_at": _utc_now(),
        "summary_by_model": summary_by_model,
        "results": model_results,
        "saved_result_path": None,
        "saved_csv_path": None,
    }

    if save_result:
        prefix = f"compare_{mode}_{dataset_name}"
        bundle = LocalResultStore().save_result_bundle(prefix, response)
        response["saved_result_path"] = bundle["json_path"]
        response["saved_csv_path"] = bundle["csv_path"]

    return response


def _validate_request(
    mode: str,
    dataset_name: str,
    models: List[str],
    concision_threshold: int,
    top_k: int,
) -> None:
    if mode not in {"chatbot", "rag"}:
        raise CompareEvaluationError("mode must be either 'chatbot' or 'rag'.")
    if not dataset_name.strip():
        raise CompareEvaluationError("dataset_name must not be empty.")
    if not models:
        raise CompareEvaluationError("models must not be empty.")
    if any(not model.strip() for model in models):
        raise CompareEvaluationError("models must not contain empty values.")
    if concision_threshold <= 0:
        raise CompareEvaluationError("concision_threshold must be greater than 0.")
    if top_k <= 0:
        raise CompareEvaluationError("top_k must be greater than 0.")


def _summary_for_model(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model_name": evaluation["model_name"],
        "total_examples": evaluation["total_examples"],
    }
    summary.update(evaluation["summary"])
    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
