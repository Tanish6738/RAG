from fastapi import APIRouter, HTTPException
from app.rag.schema import QueryRequest, QueryResponse
from app.rag.retrieval import rag_query

router = APIRouter(prefix="/query", tags=["RAG Query"])


@router.post("/", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    """
    Endpoint to perform semantic retrieval and generate a grounded answer using RAG.
    """
    try:
        result = await rag_query(
            question=body.question,
            top_k=body.top_k,
            filename_filter=body.filename_filter,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
