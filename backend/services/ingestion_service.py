from __future__ import annotations

from pathlib import Path

from backend.services.rag_service import process_pdf


class IngestionService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_pdf(self, filename: str, file_bytes: bytes) -> Path:
        safe_name = Path(filename).name.replace(" ", "_")
        target_path = self.data_dir / safe_name
        target_path.write_bytes(file_bytes)
        return target_path

    def process(self, file_path: Path) -> dict:
        return process_pdf(str(file_path))
