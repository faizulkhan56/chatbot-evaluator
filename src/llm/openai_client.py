from openai import OpenAI, OpenAIError

from src.core.config import get_settings


class MissingLLMAPIKeyError(ValueError):
    """Raised when a provider API key is not configured."""


class MissingOpenAIAPIKeyError(MissingLLMAPIKeyError):
    """Raised when OPENAI_API_KEY is not configured."""


class LLMRequestError(RuntimeError):
    """Raised when an LLM provider request fails."""


def generate_chat_response(
    question: str,
    model_name: str,
    instructions: str,
    temperature: float = 0,
) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise MissingOpenAIAPIKeyError("OPENAI_API_KEY is missing. Add it to .env before running evaluation.")

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
        )
    except OpenAIError as exc:
        raise LLMRequestError(f"OpenAI request failed: {exc}") from exc
    except Exception as exc:
        raise LLMRequestError(f"LLM request failed: {exc}") from exc

    message = response.choices[0].message.content
    return message.strip() if message else ""
