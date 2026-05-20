from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


DEFAULT_INSTRUCTIONS = "Respond to the user's question in a short, concise manner."
ProviderLiteral = Literal["openai", "huggingface"]


class ModelSpec(BaseModel):
    provider: ProviderLiteral = "openai"
    model: str = Field(..., min_length=1)


class ChatbotEvaluationRequest(BaseModel):
    dataset_name: str
    provider: ProviderLiteral = "openai"
    model_name: str = "gpt-4o-mini"
    evaluator_provider: Literal["openai"] = "openai"
    evaluator_model: str = "gpt-4o-mini"
    instructions: str = DEFAULT_INSTRUCTIONS
    concision_threshold: int = Field(default=30, ge=1)
    save_result: bool = True


class EvaluationSummary(BaseModel):
    correctness_score: float
    concision_score: float


class ChatbotEvaluationResultItem(BaseModel):
    question: str
    reference_answer: str
    model_response: str
    correctness: bool
    concision: bool


class ChatbotEvaluationResponse(BaseModel):
    mode: str
    dataset_name: str
    provider: Optional[str] = None
    model_name: str
    evaluator_provider: Optional[str] = None
    evaluator_model: str
    created_at: Optional[str] = None
    total_examples: int
    summary: EvaluationSummary
    results: List[ChatbotEvaluationResultItem]
    saved_result_path: Optional[str] = None
    saved_csv_path: Optional[str] = None


class RagEvaluationRequest(BaseModel):
    dataset_name: str = "rag_eval_sample"
    provider: ProviderLiteral = "openai"
    model_name: str = "gpt-4o-mini"
    evaluator_provider: Literal["openai"] = "openai"
    evaluator_model: str = "gpt-4o-mini"
    top_k: int = Field(default=6, gt=0)
    source_filter: Optional[str] = None
    save_result: bool = True


class RagEvaluationSummary(BaseModel):
    correctness_score: float
    groundedness_score: float
    relevance_score: float
    retrieval_relevance_score: float


class RagEvaluationResultItem(BaseModel):
    question: str
    reference_answer: str
    rag_answer: str
    correctness: bool
    groundedness: bool
    relevance: bool
    retrieval_relevance: bool
    retrieved_docs_count: int


class RagEvaluationResponse(BaseModel):
    mode: str
    dataset_name: str
    provider: Optional[str] = None
    model_name: str
    evaluator_provider: Optional[str] = None
    evaluator_model: str
    source_filter: Optional[str] = None
    created_at: Optional[str] = None
    total_examples: int
    summary: RagEvaluationSummary
    results: List[RagEvaluationResultItem]
    saved_result_path: Optional[str] = None
    saved_csv_path: Optional[str] = None


class CompareEvaluationRequest(BaseModel):
    mode: Literal["chatbot", "rag"]
    dataset_name: str = Field(..., min_length=1)
    models: List[Union[str, ModelSpec]] = Field(..., min_length=1)
    evaluator_provider: Literal["openai"] = "openai"
    evaluator_model: str = "gpt-4o-mini"
    instructions: Optional[str] = DEFAULT_INSTRUCTIONS
    concision_threshold: int = Field(default=30, gt=0)
    top_k: int = Field(default=6, gt=0)
    source_filter: Optional[str] = None
    save_result: bool = True


class CompareModelSummary(BaseModel):
    provider: Optional[str] = None
    model_name: str
    total_examples: int
    correctness_score: float
    concision_score: Optional[float] = None
    groundedness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    retrieval_relevance_score: Optional[float] = None


class CompareModelResult(BaseModel):
    provider: Optional[str] = None
    model_name: str
    results: List[Dict[str, Any]]


class CompareEvaluationResponse(BaseModel):
    mode: str
    dataset_name: str
    evaluator_provider: Optional[str] = None
    evaluator_model: str
    models: List[Any]
    source_filter: Optional[str] = None
    created_at: Optional[str] = None
    summary_by_model: List[CompareModelSummary]
    results: List[CompareModelResult]
    saved_result_path: Optional[str] = None
    saved_csv_path: Optional[str] = None
