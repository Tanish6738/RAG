from app.core.config import settings
from app.rag.embedder import client
from app.rag.qdrant import search

RAG_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer the user's question using ONLY 
the context chunks provided below. Each chunk is delimited by <chunk> tags 
and includes its source filename and page number.

Rules:
- If the answer is in the context, answer directly and cite the source as [filename, p.N].
- If the answer is NOT in the context, say "I don't have enough information to answer that."
- Never fabricate facts. Never answer from general knowledge.
- Keep answers concise unless the user asks for elaboration.
"""


async def _embed_query(query: str) -> list[float]:
    """
    Generates embedding vector for a textual search query.
    """
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=[query],
    )
    return response.data[0].embedding


async def retrieve(
    query: str,
    top_k: int = settings.top_k,
    filename_filter: str | None = None,
) -> list[dict]:
    """
    Embeds the user query and retrieves top_k semantically similar chunks from Qdrant.
    """
    query_vector = await _embed_query(query)
    return await search(query_vector, top_k=top_k, filename_filter=filename_filter)


def _build_context(chunks: list[dict]) -> str:
    """
    Formats the retrieved chunks into structured XML-like blocks for injection into the prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f'<chunk id="{i}" source="{chunk["source_filename"]}" page="{chunk["page_number"]}" score="{chunk["score"]:.2f}">\n'
            f'{chunk["text"]}\n'
            f'</chunk>'
        )
    return "\n\n".join(parts)


async def rag_query(
    question: str,
    top_k: int = settings.top_k,
    filename_filter: str | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Orchestrates the entire RAG pipeline:
    1. Retrieve relevant chunks from vector DB.
    2. Format chunks into context.
    3. Generate response with OpenAI Chat model using grounding constraints.
    """
    # 1. Retrieve candidate chunks
    chunks = await retrieve(question, top_k=top_k, filename_filter=filename_filter)
    if not chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "chunks": [],
            "model": settings.openai_chat_model,
            "usage": None,
        }

    # 2. Build context
    context = _build_context(chunks)

    # 3. Assemble messages
    messages: list[dict] = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    # Inject conversation history if available (limit to last 3 turns / 6 messages)
    if conversation_history:
        messages.extend(conversation_history[-6:])

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    # 4. Generate grounded completion
    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.1,  # Low temperature makes output predictable and faithful to context
        max_tokens=1024,
    )

    usage = response.usage
    usage_dict = None
    if usage:
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }

    return {
        "answer": response.choices[0].message.content,
        "chunks": chunks,
        "model": response.model,
        "usage": usage_dict,
    }
