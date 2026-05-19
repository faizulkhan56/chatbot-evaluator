import csv
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".xlsx", ".xls"}


class SourceDirectoryNotFoundError(FileNotFoundError):
    """Raised when the requested document source directory does not exist."""


class NoSupportedDocumentsError(ValueError):
    """Raised when no supported local documents are found."""


class DocumentLoadError(RuntimeError):
    """Raised when a supported document cannot be loaded."""


def load_local_documents(source_dir: str | Path) -> List[Document]:
    source_path = Path(source_dir)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path

    if not source_path.exists() or not source_path.is_dir():
        raise SourceDirectoryNotFoundError(f"Source directory not found: {source_path}")

    files = [
        path
        for path in sorted(source_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise NoSupportedDocumentsError(
            f"No supported documents found in {source_path}. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    documents = []
    for path in files:
        try:
            loaded = _load_file(path)
        except Exception as exc:
            raise DocumentLoadError(f"Failed to load document {path}: {exc}") from exc

        documents.extend(loaded)

    if not documents:
        raise NoSupportedDocumentsError(f"Supported files were found in {source_path}, but all were empty.")

    return documents


def _load_file(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _load_text_file(path)
    if suffix == ".pdf":
        return _load_pdf_file(path)
    if suffix == ".csv":
        return _load_csv_file(path)
    if suffix in {".xlsx", ".xls"}:
        return _load_excel_file(path)
    return []


def _load_text_file(path: Path) -> List[Document]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_pdf_file(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    documents = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": index},
                )
            )
    return documents


def _load_csv_file(path: Path) -> List[Document]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = []
        for index, row in enumerate(reader, start=1):
            row_text = "; ".join(f"{key}: {value}" for key, value in row.items())
            if row_text.strip():
                rows.append(f"Row {index}: {row_text}")

    if not rows:
        return []
    return [Document(page_content="\n".join(rows), metadata={"source": str(path)})]


def _load_excel_file(path: Path) -> List[Document]:
    sheet_map = pd.read_excel(path, sheet_name=None)
    documents = []
    for sheet_name, dataframe in sheet_map.items():
        if dataframe.empty:
            continue
        text = dataframe.to_csv(index=False)
        if text.strip():
            documents.append(
                Document(
                    page_content=f"Sheet: {sheet_name}\n{text}",
                    metadata={"source": str(path), "sheet": sheet_name},
                )
            )
    return documents
