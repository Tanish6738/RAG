from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import ingest, query, agent
from app.rag.qdrant import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure Qdrant collection is ready
    await ensure_collection()
    yield
    # Shutdown: Cleanup operations if needed


app = FastAPI(title="RAG Agent API", version="0.1.0", lifespan=lifespan)

# Register endpoints
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(agent.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
