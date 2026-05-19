from pathlib import Path
from typing import BinaryIO, Dict, List

from src.core.config import PROJECT_ROOT
from src.storage.paths import documents_dir


SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".xlsx", ".xls"}


class DocumentStoreError(Exception):
    """Base exception for document storage errors."""


class InvalidDocumentFileNameError(DocumentStoreError):
    """Raised when an uploaded document filename is unsafe."""


class UnsupportedDocumentTypeError(DocumentStoreError):
    """Raised when an uploaded document type is unsupported."""


class LocalDocumentStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or documents_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_name: str, file_obj: BinaryIO) -> Dict[str, object]:
        path = self._path_for(file_name)
        with path.open("wb") as destination:
            while True:
                chunk = file_obj.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)

        return {
            "file_name": path.name,
            "path": self._display_path(path),
            "size_bytes": path.stat().st_size,
        }

    def list_documents(self) -> List[Dict[str, object]]:
        documents = []
        for path in sorted(self.base_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                continue
            documents.append(
                {
                    "file_name": path.name,
                    "path": self._display_path(path),
                    "file_type": path.suffix.lower().lstrip("."),
                    "size_bytes": path.stat().st_size,
                }
            )
        return documents

    def _path_for(self, file_name: str) -> Path:
        if Path(file_name).name != file_name:
            raise InvalidDocumentFileNameError("Document file name must not include path separators.")

        suffix = Path(file_name).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
            raise UnsupportedDocumentTypeError(f"Unsupported document type: {suffix}. Allowed types: {allowed}")

        path = (self.base_dir / file_name).resolve()
        base_dir = self.base_dir.resolve()
        if not path.is_relative_to(base_dir):
            raise InvalidDocumentFileNameError("Invalid document file name.")
        return path

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)
