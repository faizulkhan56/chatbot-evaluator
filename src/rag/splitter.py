from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class InvalidChunkSettingsError(ValueError):
    """Raised when chunking configuration is invalid."""


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    if chunk_size <= 0:
        raise InvalidChunkSettingsError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise InvalidChunkSettingsError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise InvalidChunkSettingsError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
