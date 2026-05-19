from pathlib import Path
from typing import List

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".txt", ".md"}


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
            f"No supported documents found in {source_path}. Supported extensions: .txt, .md"
        )

    documents = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        except Exception as exc:
            raise DocumentLoadError(f"Failed to load document {path}: {exc}") from exc

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path)},
                )
            )

    if not documents:
        raise NoSupportedDocumentsError(f"Supported files were found in {source_path}, but all were empty.")

    return documents
