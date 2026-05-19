from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetExample(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)


class DatasetIO(BaseModel):
    name: str
    examples: List[DatasetExample] = Field(default_factory=list)


class DatasetCreateRequest(BaseModel):
    name: str
    examples: List[DatasetExample] = Field(default_factory=list)
    overwrite: bool = False


class DatasetResponse(BaseModel):
    name: str
    examples: List[DatasetExample]
    example_count: int
    path: Optional[str] = None


class DatasetListResponse(BaseModel):
    datasets: List[DatasetResponse]
    count: int
