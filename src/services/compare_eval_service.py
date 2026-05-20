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
    models: List[Any],
    evaluator_model: str,
    save_result: bool,
    instructions: Optional[str] = None,
    concision_threshold: int = 30,
    top_k: int = 6,
    evaluator_provider: str = "openai",
    source_filter: str | None = None,
) -> Dict[str, Any]:
    normalized_models = _normalize_models(models)
    _validate_request(mode, dataset_name, normalized_models, concision_threshold, top_k)

    summary_by_model = []
    model_results = []

    for model_spec in normalized_models:
        provider = model_spec["provider"]
        model_name = model_spec["model"]
        if mode == "chatbot":
            evaluation = evaluate_chatbot_dataset(
                dataset_name=dataset_name,
                provider=provider,
                model_name=model_name,
                evaluator_provider=evaluator_provider,
                evaluator_model=evaluator_model,
                instructions=instructions or DEFAULT_INSTRUCTIONS,
                concision_threshold=concision_threshold,
                save_result=False,
            )
        else:
            evaluation = evaluate_rag_dataset(
                dataset_name=dataset_name,
                provider=provider,
                model_name=model_name,
                evaluator_provider=evaluator_provider,
                evaluator_model=evaluator_model,
                top_k=top_k,
                save_result=False,
                source_filter=source_filter,
            )

        summary_by_model.append(_summary_for_model(evaluation))
        model_results.append(
            {
                "provider": provider,
                "model_name": model_name,
                "results": evaluation["results"],
            }
        )

    response = {
        "mode": f"compare_{mode}",
        "dataset_name": dataset_name,
        "evaluator_provider": evaluator_provider,
        "evaluator_model": evaluator_model,
        "models": normalized_models,
        "source_filter": source_filter if mode == "rag" else None,
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
    models: List[Dict[str, str]],
    concision_threshold: int,
    top_k: int,
) -> None:
    if mode not in {"chatbot", "rag"}:
        raise CompareEvaluationError("mode must be either 'chatbot' or 'rag'.")
    if not dataset_name.strip():
        raise CompareEvaluationError("dataset_name must not be empty.")
    if not models:
        raise CompareEvaluationError("models must not be empty.")
    if any(not model["model"].strip() for model in models):
        raise CompareEvaluationError("models must not contain empty values.")
    if concision_threshold <= 0:
        raise CompareEvaluationError("concision_threshold must be greater than 0.")
    if top_k <= 0:
        raise CompareEvaluationError("top_k must be greater than 0.")


def _summary_for_model(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "provider": evaluation.get("provider", "openai"),
        "model_name": evaluation["model_name"],
        "total_examples": evaluation["total_examples"],
    }
    summary.update(evaluation["summary"])
    return summary


def _normalize_models(models: List[Any]) -> List[Dict[str, str]]:
    normalized = []
    for item in models:
        if isinstance(item, str):
            model_name = item.strip()
            if not model_name:
                raise CompareEvaluationError("models must not contain empty values.")
            normalized.append({"provider": "openai", "model": model_name})
            continue

        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif hasattr(item, "dict"):
            item = item.dict()

        if not isinstance(item, dict):
            raise CompareEvaluationError("models must be strings or provider/model objects.")

        provider = str(item.get("provider", "openai")).strip().lower()
        model_name = str(item.get("model", "")).strip()
        if provider not in {"openai", "huggingface"}:
            raise CompareEvaluationError("model provider must be either 'openai' or 'huggingface'.")
        if not model_name:
            raise CompareEvaluationError("model object must include a non-empty model value.")

        normalized.append({"provider": provider, "model": model_name})

    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
