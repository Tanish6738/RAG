from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag.loader import ingest_pdf

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload and ingest a PDF document.
    Validates file type and size, then runs the ingestion pipeline.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB).")

    try:
        result = await ingest_pdf(raw, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
