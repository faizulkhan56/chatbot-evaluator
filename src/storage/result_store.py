import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from src.core.config import PROJECT_ROOT
from src.storage.paths import results_dir


class ResultStoreError(Exception):
    """Base exception for result storage errors."""


class InvalidResultFileNameError(ResultStoreError):
    """Raised when a requested result filename is unsafe."""


class ResultFileNotFoundError(ResultStoreError):
    """Raised when a requested result file does not exist."""


class UnsupportedResultFileTypeError(ResultStoreError):
    """Raised when a result file type is not supported."""


class ResultCSVExportError(ResultStoreError):
    """Raised when CSV export fails."""


class LocalResultStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or results_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, result_type: str, payload: Dict[str, Any]) -> Path:
        path = self._result_path(result_type, "json")
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def save_csv(self, result_type: str, rows: Iterable[Dict[str, Any]]) -> Path:
        path = self._result_path(result_type, "csv")
        normalized_rows = list(rows)
        fieldnames = self._fieldnames(normalized_rows)

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized_rows)

        return path

    def save_result_json(self, prefix: str, payload: Any) -> str:
        path = self._path_with_prefix(prefix, "json")
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self._display_path(path)

    def save_result_csv(self, prefix: str, rows: List[Dict[str, Any]]) -> Optional[str]:
        if not rows:
            return None

        path = self._path_with_prefix(prefix, "csv")
        try:
            self._write_csv(path, rows)
        except Exception as exc:
            raise ResultCSVExportError(f"Failed to export CSV result: {exc}") from exc
        return self._display_path(path)

    def save_result_bundle(self, prefix: str, payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
        safe_prefix = self._safe_filename_part(prefix)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_file_path = self.base_dir / f"{safe_prefix}_{timestamp}.json"
        csv_file_path = self.base_dir / f"{safe_prefix}_{timestamp}.csv"

        json_path = self._display_path(json_file_path)
        csv_path = None

        rows = self._rows_from_payload(payload)
        if rows:
            try:
                self._write_csv(csv_file_path, rows)
                csv_path = self._display_path(csv_file_path)
            except ResultCSVExportError:
                csv_path = None
            except Exception:
                csv_path = None

        payload["saved_result_path"] = json_path
        payload["saved_csv_path"] = csv_path
        json_file_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        return {
            "json_path": json_path,
            "csv_path": csv_path,
        }

    def list_results(self) -> List[Dict[str, Any]]:
        results = []
        for path in sorted(self.base_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv"}:
                continue
            results.append(
                {
                    "file_name": path.name,
                    "path": self._display_path(path),
                    "file_type": path.suffix.lower().lstrip("."),
                    "size_bytes": path.stat().st_size,
                }
            )
        return results

    def load_result(self, file_name: str) -> Dict[str, Any] | str | List[Dict[str, str]]:
        path = self._safe_result_path(file_name)
        suffix = path.suffix.lower()

        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if suffix == ".csv":
            with path.open("r", newline="", encoding="utf-8") as file:
                return list(csv.DictReader(file))

        raise UnsupportedResultFileTypeError(f"Unsupported result file type: {suffix}")

    def _result_path(self, result_type: str, suffix: str) -> Path:
        safe_type = "".join(char if char.isalnum() or char in "-_" else "_" for char in result_type)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.base_dir / f"{safe_type}_{timestamp}_{uuid4().hex[:8]}.{suffix}"

    def _path_with_prefix(self, prefix: str, suffix: str) -> Path:
        safe_prefix = self._safe_filename_part(prefix)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self.base_dir / f"{safe_prefix}_{timestamp}.{suffix}"

    def _safe_result_path(self, file_name: str) -> Path:
        if Path(file_name).name != file_name:
            raise InvalidResultFileNameError("Result file name must not include path separators.")

        path = (self.base_dir / file_name).resolve()
        base_dir = self.base_dir.resolve()
        if not path.is_relative_to(base_dir):
            raise InvalidResultFileNameError("Invalid result file name.")

        if path.suffix.lower() not in {".json", ".csv"}:
            raise UnsupportedResultFileTypeError("Only .json and .csv result files are supported.")
        if not path.exists() or not path.is_file():
            raise ResultFileNotFoundError(f"Result file not found: {file_name}")

        return path

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        fieldnames = self._fieldnames(rows)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _rows_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        if results and all(isinstance(item, dict) and "model_name" in item and "results" in item for item in results):
            flattened = []
            for model_group in results:
                model_name = model_group.get("model_name")
                for row in model_group.get("results") or []:
                    if isinstance(row, dict):
                        flattened.append({"model_name": model_name, **self._flatten_row(row)})
            return flattened

        return [self._flatten_row(row) for row in results if isinstance(row, dict)]

    def _flatten_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        flattened = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flattened[key] = json.dumps(value, default=str)
            else:
                flattened[key] = value
        return flattened

    def _safe_filename_part(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def _fieldnames(self, rows: List[Dict[str, Any]]) -> List[str]:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        return fieldnames
