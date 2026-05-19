from typing import List

from pydantic import BaseModel


class ResultFileInfo(BaseModel):
    file_name: str
    path: str
    file_type: str
    size_bytes: int


class ResultListResponse(BaseModel):
    results: List[ResultFileInfo]
