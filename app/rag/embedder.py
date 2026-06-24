import asyncio
from openai import AsyncOpenAI
from app.core.config import settings
from app.rag.schema import Chunk

# Instantiate the AsyncOpenAI client using the configured key and base url
client = AsyncOpenAI(api_key=settings.OPENAI_API, base_url=settings.BASE_URL)

BATCH_SIZE = 100  # Safe batch size for OpenAI embeddings


async def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """
    Generates embedding vectors for a list of Chunks using OpenAI's embedding API.
    Returns a list of tuples containing (Chunk, embedding_vector).
    """
    results: list[tuple[Chunk, list[float]]] = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]

        # Call OpenAI embedding model asynchronously
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )

        for chunk, embedding_data in zip(batch, response.data):
            results.append((chunk, embedding_data.embedding))

        # Sleep briefly between batches to respect rate limits
        if i + BATCH_SIZE < len(chunks):
            await asyncio.sleep(0.1)

    return results
