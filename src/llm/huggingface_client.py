from openai import OpenAI, OpenAIError

from src.core.config import get_settings
from src.llm.openai_client import LLMRequestError, MissingLLMAPIKeyError


class MissingHuggingFaceAPIKeyError(MissingLLMAPIKeyError):
    """Raised when HUGGINGFACE_API_KEY is not configured."""


def generate_huggingface_chat_response(
    question: str,
    model_name: str,
    instructions: str,
    temperature: float = 0,
) -> str:
    settings = get_settings()
    if not settings.huggingface_api_key:
        raise MissingHuggingFaceAPIKeyError(
            "HUGGINGFACE_API_KEY is missing. Add it to .env before using Hugging Face models."
        )

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=settings.huggingface_api_key,
        )
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
        )
    except OpenAIError as exc:
        raise LLMRequestError(
            f"Hugging Face request failed for model '{model_name}'. "
            "Check your token, model access, and inference provider availability. "
            f"Details: {exc}"
        ) from exc
    except Exception as exc:
        raise LLMRequestError(f"Hugging Face request failed: {exc}") from exc

    message = response.choices[0].message.content
    return message.strip() if message else ""
