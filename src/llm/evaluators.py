from openai import OpenAI, OpenAIError

from src.core.config import get_settings
from src.llm.openai_client import LLMRequestError, MissingOpenAIAPIKeyError


def _judge_with_openai(system_prompt: str, user_prompt: str, evaluator_model: str) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise MissingOpenAIAPIKeyError("OPENAI_API_KEY is missing. Add it to .env before running evaluation.")

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=evaluator_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as exc:
        raise LLMRequestError(f"OpenAI evaluator request failed: {exc}") from exc
    except Exception as exc:
        raise LLMRequestError(f"Evaluator request failed: {exc}") from exc

    verdict = response.choices[0].message.content or ""
    return verdict.strip().upper()


def _judge(system_prompt: str, user_prompt: str, evaluator_provider: str, evaluator_model: str) -> str:
    if evaluator_provider.lower() != "openai":
        raise LLMRequestError("Only OpenAI evaluator_provider is supported in this phase.")
    return _judge_with_openai(system_prompt, user_prompt, evaluator_model)


def evaluate_correctness(
    question: str,
    reference_answer: str,
    model_response: str,
    evaluator_model: str,
    evaluator_provider: str = "openai",
) -> bool:
    prompt = f"""
QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

MODEL RESPONSE:
{model_response}

Decide whether the MODEL RESPONSE is semantically correct compared to the REFERENCE ANSWER.

Return only one word:
CORRECT or INCORRECT
"""

    return _judge("You are an expert evaluator.", prompt, evaluator_provider, evaluator_model) == "CORRECT"


def evaluate_concision(model_response: str, threshold: int = 30) -> bool:
    return len(model_response.split()) <= threshold


def evaluate_groundedness(
    retrieved_documents: list[dict],
    model_answer: str,
    evaluator_model: str,
    evaluator_provider: str = "openai",
) -> bool:
    facts = _documents_to_text(retrieved_documents)
    prompt = f"""
FACTS FROM RETRIEVED DOCUMENTS:
{facts}

MODEL ANSWER:
{model_answer}

Decide whether the MODEL ANSWER is grounded in the FACTS.

Return only one word:
GROUNDED or NOT_GROUNDED
"""
    return _judge("You are an expert RAG evaluator.", prompt, evaluator_provider, evaluator_model) == "GROUNDED"


def evaluate_answer_relevance(
    question: str,
    model_answer: str,
    evaluator_model: str,
    evaluator_provider: str = "openai",
) -> bool:
    prompt = f"""
QUESTION:
{question}

MODEL ANSWER:
{model_answer}

Decide whether the MODEL ANSWER directly addresses the QUESTION.

Return only one word:
RELEVANT or NOT_RELEVANT
"""
    return _judge("You are an expert evaluator.", prompt, evaluator_provider, evaluator_model) == "RELEVANT"


def evaluate_retrieval_relevance(
    question: str,
    retrieved_documents: list[dict],
    evaluator_model: str,
    evaluator_provider: str = "openai",
) -> bool:
    docs = _documents_to_text(retrieved_documents)
    prompt = f"""
QUESTION:
{question}

RETRIEVED DOCUMENTS:
{docs}

Decide whether the retrieved documents contain information relevant to answering the QUESTION.

Return only one word:
RELEVANT or NOT_RELEVANT
"""
    return _judge("You are an expert retrieval evaluator.", prompt, evaluator_provider, evaluator_model) == "RELEVANT"


def _documents_to_text(retrieved_documents: list[dict]) -> str:
    if not retrieved_documents:
        return "No documents retrieved."

    formatted = []
    for index, document in enumerate(retrieved_documents, start=1):
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
        formatted.append(f"Document {index} source={source}\n{content}")
    return "\n\n".join(formatted)
