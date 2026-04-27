from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import DATA_DIR
from backend.services.ingestion_service import IngestionService

router = APIRouter(prefix="/upload", tags=["upload"])
service = IngestionService(data_dir=DATA_DIR)


@router.post("")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        file_bytes = await file.read()
        saved_path = service.save_pdf(file.filename, file_bytes)
        result = service.process(saved_path)
        return {
            "status": "success",
            "message": result.get("message", "Document processed successfully"),
            "file_path": str(saved_path),
            "active_pdf": result.get("active_pdf"),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process file: {exc}") from exc
