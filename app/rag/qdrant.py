from uuid import uuid4
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.core.config import settings
from app.rag.schema import Chunk

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    """
    Initializes and returns a singleton AsyncQdrantClient instance.
    If settings.qdrant_url starts with http:// or https://, connects to Qdrant server.
    Otherwise, runs in serverless, local client-only storage mode (sqlite-backed).
    """
    global _client
    if _client is None:
        url = settings.qdrant_url
        if url.startswith("http://") or url.startswith("https://"):
            _client = AsyncQdrantClient(
                url=url,
                api_key=settings.qdrant_api_key,
            )
        else:
            # Serverless, local persistent client-only mode
            _client = AsyncQdrantClient(
                path=url,
            )
    return _client


async def ensure_collection() -> None:
    """
    Checks if collection exists and creates it with the proper dimensions
    and distance metric if it doesn't.
    """
    qc = get_qdrant()
    collection_name = settings.qdrant_collection_name
    exists = await qc.collection_exists(collection_name)
    if not exists:
        await qc.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )


async def upsert_chunks(chunk_vector_pairs: list[tuple[Chunk, list[float]]]) -> int:
    """
    Upserts a batch of chunks and their embedding vectors into Qdrant.
    Returns the count of successfully upserted points.
    """
    qc = get_qdrant()
    collection_name = settings.qdrant_collection_name

    points = [
        PointStruct(
            id=str(uuid4()),  # Qdrant requires UUID string or integer IDs
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "source_filename": chunk.source_filename,
                **chunk.metadata,
            },
        )
        for chunk, vector in chunk_vector_pairs
    ]

    await qc.upsert(
        collection_name=collection_name,
        points=points,
    )
    return len(points)


async def search(
    query_vector: list[float],
    top_k: int = settings.top_k,
    filename_filter: str | None = None,
) -> list[dict]:
    """
    Vector similarity search in Qdrant.
    Returns a list of payload dicts matching the search, each including the match score.
    """
    qc = get_qdrant()
    collection_name = settings.qdrant_collection_name

    query_filter = None
    if filename_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source_filename",
                    match=MatchValue(value=filename_filter),
                )
            ]
        )

    results = await qc.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=settings.similarity_threshold,
    )

    return [
        {**hit.payload, "score": hit.score}
        for hit in results
        if hit.payload is not None
    ]
