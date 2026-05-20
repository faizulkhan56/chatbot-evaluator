from src.llm.huggingface_client import generate_huggingface_chat_response
from src.llm.openai_client import generate_chat_response


SUPPORTED_PROVIDERS = {"openai", "huggingface"}


class UnsupportedProviderError(ValueError):
    """Raised when an LLM provider is not supported."""


def normalize_provider(provider: str | None) -> str:
    normalized = (provider or "openai").strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise UnsupportedProviderError("provider must be either 'openai' or 'huggingface'.")
    return normalized


def generate_response(
    provider: str,
    model_name: str,
    question: str,
    instructions: str,
    temperature: float = 0,
) -> str:
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "openai":
        return generate_chat_response(
            question=question,
            model_name=model_name,
            instructions=instructions,
            temperature=temperature,
        )
    if normalized_provider == "huggingface":
        return generate_huggingface_chat_response(
            question=question,
            model_name=model_name,
            instructions=instructions,
            temperature=temperature,
        )

    raise UnsupportedProviderError("provider must be either 'openai' or 'huggingface'.")
