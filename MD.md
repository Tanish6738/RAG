# Production RAG + Multi-Tool Agent System
## Complete Implementation Blueprint

> **Stack:** Python · FastAPI · OpenAI · Qdrant · Pydantic v2 · pypdf · pdfplumber · LangChain (optional utils)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Environment & Dependencies](#3-environment--dependencies)
4. [Core Configuration (Pydantic Settings)](#4-core-configuration-pydantic-settings)
5. [Document Ingestion Pipeline](#5-document-ingestion-pipeline)
6. [Embedding & Vector Storage (Qdrant)](#6-embedding--vector-storage-qdrant)
7. [Retrieval Pipeline](#7-retrieval-pipeline)
8. [RAG Pipeline](#8-rag-pipeline)
9. [Multi-Tool Agent](#9-multi-tool-agent)
10. [FastAPI Layer](#10-fastapi-layer)
11. [Concepts & Mental Models](#11-concepts--mental-models)

---

## 1. Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │          FastAPI Layer           │
                        │  /ingest  /query  /agent/chat   │
                        └────────────┬────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
    ┌─────────▼──────────┐                      ┌──────────▼────────┐
    │  Ingestion Pipeline │                      │   Agent Pipeline   │
    │                    │                      │                    │
    │ PDF → Extract       │                      │  Planner           │
    │     → Chunk         │                      │  → Tool Router     │
    │     → Embed         │                      │    → RAG Tool      │
    │     → Store         │                      │    → Search Tool   │
    └─────────┬──────────┘                      │    → Calc Tool     │
              │                                  │  → Critic          │
              ▼                                  │  → Responder       │
    ┌─────────────────────┐                      └──────────┬────────┘
    │      Qdrant DB       │◄─────────────────────────────-─┘
    │  (Vector Storage)    │         Retrieval
    └─────────────────────┘
```

**Key insight:** There are two separate phases — **offline ingestion** (documents go in) and **online inference** (queries come in). The agent uses tools, one of which happens to call the retrieval+RAG pipeline.

---

## 2. Project Structure

```
rag_agent/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI entry point
│   ├── core/
│   │   ├── config.py              # Pydantic Settings
│   │   ├── logging.py             # Structured logging
│   │   └── exceptions.py          # Custom exceptions
│   ├── models/
│   │   ├── documents.py           # Pydantic schemas for docs
│   │   ├── chunks.py              # Chunk schemas
│   │   ├── agent.py               # Agent message schemas
│   │   └── responses.py           # API response schemas
│   ├── services/
│   │   ├── ingestion/
│   │   │   ├── extractor.py       # pypdf + pdfplumber extraction
│   │   │   ├── chunker.py         # Text chunking strategies
│   │   │   ├── embedder.py        # OpenAI embeddings
│   │   │   └── pipeline.py        # Orchestrates extraction→chunk→embed→store
│   │   ├── retrieval/
│   │   │   ├── qdrant_client.py   # Qdrant wrapper
│   │   │   ├── retriever.py       # Hybrid search logic
│   │   │   └── reranker.py        # Optional reranking
│   │   ├── rag/
│   │   │   ├── context_builder.py # Formats retrieved chunks into context
│   │   │   ├── generator.py       # OpenAI completion with context
│   │   │   └── pipeline.py        # Full RAG pipeline
│   │   └── agent/
│   │       ├── tools.py           # Tool definitions
│   │       ├── executor.py        # Tool execution
│   │       ├── loop.py            # Agentic loop
│   │       └── memory.py          # Conversation memory
│   └── api/
│       ├── ingest.py              # /ingest endpoints
│       ├── query.py               # /query endpoints
│       └── agent.py               # /agent endpoints
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agent.py
├── .env
├── pyproject.toml
└── docker-compose.yml             # Qdrant local dev
```

---

## 3. Environment & Dependencies

### `pyproject.toml`

```toml
[project]
name = "rag-agent"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # API & Config
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "python-multipart>=0.0.9",      # file uploads

    # OpenAI
    "openai>=1.35.0",

    # PDF Processing
    "pypdf>=4.2.0",                 # fast text extraction, metadata
    "pdfplumber>=0.11.0",           # tables, layout-aware extraction

    # Vector DB
    "qdrant-client>=1.9.0",

    # Text Processing
    "tiktoken>=0.7.0",              # token counting for chunking
    "nltk>=3.8.0",                  # sentence tokenization
    "chardet>=5.2.0",               # encoding detection

    # Utilities
    "tenacity>=8.4.0",              # retry logic
    "structlog>=24.1.0",            # structured logging
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx"]
```

### `.env`

```env
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                        # leave empty for local
QDRANT_COLLECTION_NAME=knowledge_base

# Chunking
CHUNK_SIZE=512                          # tokens
CHUNK_OVERLAP=64                        # tokens

# Retrieval
TOP_K=5
SIMILARITY_THRESHOLD=0.7
```

### `docker-compose.yml` (local Qdrant)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.9.4
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Run: `docker compose up -d`

---

## 4. Core Configuration (Pydantic Settings)

**`app/core/config.py`**

```python
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic BaseSettings reads values from environment variables automatically.
    Fields with no default are REQUIRED — the app fails fast if they're missing.
    
    CONCEPT: Pydantic Settings gives you:
      - Type validation (CHUNK_SIZE="abc" → ValidationError, not a silent bug)
      - A single source of truth for all config
      - Easy injection into services via dependency injection
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
    embedding_dimensions: int = 1536   # text-embedding-3-small = 1536

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "knowledge_base"

    # Chunking
    chunk_size: int = 512          # in tokens
    chunk_overlap: int = 64        # token overlap between adjacent chunks

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.7

    @computed_field
    @property
    def qdrant_collection(self) -> str:
        return self.qdrant_collection_name


# Singleton — import this everywhere
settings = Settings()
```

---

## 5. Document Ingestion Pipeline

### 5.1 PDF Extraction — `app/services/ingestion/extractor.py`

```python
"""
WHY TWO PDF LIBRARIES?
- pypdf   : Fast, minimal. Great for text-only PDFs and metadata.
- pdfplumber: Slower but layout-aware. Handles tables, columns, multi-line cells.

Strategy: try pypdf first; fall back to pdfplumber if text is sparse or garbled.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


@dataclass
class PageContent:
    page_number: int        # 1-indexed
    text: str
    tables: list[list[list[str]]]   # [table][row][cell]
    has_images: bool
    char_count: int


@dataclass
class DocumentContent:
    filename: str
    total_pages: int
    pages: list[PageContent]
    metadata: dict[str, str]
    extraction_method: str          # "pypdf" | "pdfplumber"


class PDFExtractor:
    SPARSE_THRESHOLD = 50           # chars per page below this → use pdfplumber

    def extract(self, source: Path | bytes, filename: str = "doc.pdf") -> DocumentContent:
        raw = source if isinstance(source, bytes) else source.read_bytes()

        # Pass 1: quick pypdf extraction
        content = self._extract_pypdf(raw, filename)

        avg_chars = (
            sum(p.char_count for p in content.pages) / len(content.pages)
            if content.pages else 0
        )

        # Pass 2: if pypdf is sparse, upgrade to pdfplumber
        if avg_chars < self.SPARSE_THRESHOLD:
            content = self._extract_pdfplumber(raw, filename)

        return content

    def _extract_pypdf(self, raw: bytes, filename: str) -> DocumentContent:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(PageContent(
                page_number=i + 1,
                text=text.strip(),
                tables=[],
                has_images=bool(page.images),
                char_count=len(text),
            ))
        return DocumentContent(
            filename=filename,
            total_pages=len(reader.pages),
            pages=pages,
            metadata=dict(reader.metadata or {}),
            extraction_method="pypdf",
        )

    def _extract_pdfplumber(self, raw: bytes, filename: str) -> DocumentContent:
        pages = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                # Convert table cells: None → ""
                clean_tables = [
                    [[cell or "" for cell in row] for row in table]
                    for table in tables
                ]
                # Append table text after body text
                table_text = self._tables_to_text(clean_tables)
                full_text = f"{text}\n\n{table_text}".strip()
                pages.append(PageContent(
                    page_number=i + 1,
                    text=full_text,
                    tables=clean_tables,
                    has_images=bool(page.images),
                    char_count=len(full_text),
                ))
        return DocumentContent(
            filename=filename,
            total_pages=len(pages),
            pages=pages,
            metadata={},
            extraction_method="pdfplumber",
        )

    def _tables_to_text(self, tables: list[list[list[str]]]) -> str:
        lines = []
        for table in tables:
            for row in table:
                lines.append(" | ".join(cell.strip() for cell in row if cell.strip()))
        return "\n".join(lines)
```

---

### 5.2 Chunking — `app/services/ingestion/chunker.py`

```python
"""
CHUNKING CONCEPTS:

Why chunk at all?
  Embedding models have a token limit (e.g. 8192 for text-embedding-3-small).
  Also, smaller, focused chunks retrieve more precisely than full documents.

Overlap:
  Chunk N and Chunk N+1 share `overlap` tokens. This prevents a key sentence
  from being cut exactly at a boundary, losing its meaning.

Token-based vs character-based:
  Token-based is more precise because embeddings and LLMs operate in token space.
  We use tiktoken to count tokens consistently with OpenAI models.
"""

import re
from dataclasses import dataclass, field

import tiktoken

from app.core.config import settings
from app.services.ingestion.extractor import DocumentContent


@dataclass
class Chunk:
    chunk_id: str               # "{filename}::p{page}::c{index}"
    text: str
    token_count: int
    page_number: int
    chunk_index: int            # position within the document
    source_filename: str
    metadata: dict = field(default_factory=dict)


class TokenChunker:
    """
    Sliding window chunker in token space.
    Splits on sentence boundaries where possible so chunks don't break mid-sentence.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
        encoding_name: str = "cl100k_base",   # used by all OpenAI models
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, doc: DocumentContent) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        carry_tokens: list[int] = []   # overlap tokens carried from previous chunk

        for page in doc.pages:
            if not page.text.strip():
                continue

            page_tokens = self.enc.encode(page.text)
            window = carry_tokens + page_tokens

            while len(window) >= self.chunk_size:
                token_slice = window[: self.chunk_size]
                text = self.enc.decode(token_slice)

                chunks.append(Chunk(
                    chunk_id=f"{doc.filename}::p{page.page_number}::c{chunk_index}",
                    text=text,
                    token_count=len(token_slice),
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    source_filename=doc.filename,
                    metadata={
                        "extraction_method": doc.extraction_method,
                        "total_pages": doc.total_pages,
                    },
                ))
                chunk_index += 1
                window = window[self.chunk_size - self.overlap :]

            carry_tokens = window   # leftover becomes carry for next page

        # Flush remainder
        if carry_tokens:
            text = self.enc.decode(carry_tokens)
            chunks.append(Chunk(
                chunk_id=f"{doc.filename}::p{doc.pages[-1].page_number}::c{chunk_index}",
                text=text,
                token_count=len(carry_tokens),
                page_number=doc.pages[-1].page_number,
                chunk_index=chunk_index,
                source_filename=doc.filename,
                metadata={},
            ))

        return chunks
```

---

### 5.3 Embedder — `app/services/ingestion/embedder.py`

```python
"""
EMBEDDING CONCEPTS:

An embedding is a fixed-length vector (list of floats) that represents
the *semantic meaning* of a piece of text. Texts with similar meaning
will have vectors that are close in high-dimensional space (measured by
cosine similarity).

text-embedding-3-small returns 1536-dimensional vectors.
We batch requests to OpenAI's API to stay within rate limits.
"""

import asyncio
from app.core.config import settings
from app.services.ingestion.chunker import Chunk
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=settings.openai_api_key)

BATCH_SIZE = 100   # OpenAI allows up to 2048 inputs, 100 is safe


async def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """Returns (chunk, embedding_vector) pairs."""
    results: list[tuple[Chunk, list[float]]] = []
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]
        
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        # response.data[j].embedding is a list[float]
        for chunk, embedding_data in zip(batch, response.data):
            results.append((chunk, embedding_data.embedding))
        
        # Be polite to rate limits between batches
        if i + BATCH_SIZE < len(chunks):
            await asyncio.sleep(0.1)
    
    return results
```

---

### 5.4 Qdrant Storage — `app/services/retrieval/qdrant_client.py`

```python
"""
QDRANT CONCEPTS:

Collection : Like a table. Stores vectors + payloads (metadata dicts).
Point      : One entry — has an ID, a vector, and a payload.
Payload    : Arbitrary JSON stored alongside each vector (chunk text, filename, etc.)

Distance metric:
  Cosine similarity is standard for text embeddings.
  Returns scores in [-1, 1]; higher = more similar.
"""

from uuid import uuid4
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
from app.core.config import settings
from app.services.ingestion.chunker import Chunk

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _client


async def ensure_collection() -> None:
    """Create collection if it doesn't exist. Safe to call on every startup."""
    qc = get_qdrant()
    exists = await qc.collection_exists(settings.qdrant_collection_name)
    if not exists:
        await qc.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )


async def upsert_chunks(chunk_vector_pairs: list[tuple[Chunk, list[float]]]) -> int:
    """Insert or update chunks. Returns count of upserted points."""
    qc = get_qdrant()
    points = [
        PointStruct(
            id=str(uuid4()),       # Qdrant accepts UUID strings
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
        collection_name=settings.qdrant_collection_name,
        points=points,
    )
    return len(points)


async def search(
    query_vector: list[float],
    top_k: int = settings.top_k,
    filename_filter: str | None = None,
) -> list[dict]:
    """
    Vector similarity search. Returns list of payload dicts with score.
    
    CONCEPT: We send the query's embedding vector to Qdrant, which finds
    the stored points whose vectors are closest in cosine space. No keyword
    matching — pure semantic similarity.
    """
    qc = get_qdrant()
    
    query_filter = None
    if filename_filter:
        query_filter = Filter(
            must=[FieldCondition(
                key="source_filename",
                match=MatchValue(value=filename_filter)
            )]
        )
    
    results = await qc.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=settings.similarity_threshold,
    )
    
    return [
        {**hit.payload, "score": hit.score}
        for hit in results
    ]
```

---

### 5.5 Ingestion Pipeline Orchestrator — `app/services/ingestion/pipeline.py`

```python
"""
The pipeline ties together: Extract → Chunk → Embed → Store.
This is the only file the API layer needs to call.
"""

import structlog
from app.services.ingestion.extractor import PDFExtractor
from app.services.ingestion.chunker import TokenChunker
from app.services.ingestion.embedder import embed_chunks
from app.services.retrieval.qdrant_client import ensure_collection, upsert_chunks

log = structlog.get_logger()


async def ingest_pdf(raw_bytes: bytes, filename: str) -> dict:
    await ensure_collection()

    # 1. Extract
    log.info("ingestion.extract.start", filename=filename)
    extractor = PDFExtractor()
    doc = extractor.extract(raw_bytes, filename)
    log.info("ingestion.extract.done", pages=doc.total_pages, method=doc.extraction_method)

    # 2. Chunk
    chunker = TokenChunker()
    chunks = chunker.chunk_document(doc)
    log.info("ingestion.chunk.done", chunk_count=len(chunks))

    # 3. Embed
    chunk_vector_pairs = await embed_chunks(chunks)
    log.info("ingestion.embed.done")

    # 4. Store
    stored = await upsert_chunks(chunk_vector_pairs)
    log.info("ingestion.store.done", stored=stored)

    return {
        "filename": filename,
        "pages": doc.total_pages,
        "chunks": len(chunks),
        "stored": stored,
        "extraction_method": doc.extraction_method,
    }
```

---

## 6. Embedding & Vector Storage (Qdrant)

Already covered in §5.3 and §5.4. The key flow to internalize:

```
Query text
    │
    ▼  embed with same model used at ingestion time
Query vector [0.12, -0.84, ..., 0.03]  # 1536 floats
    │
    ▼  cosine similarity search in Qdrant
Top-K chunks  (chunk text + metadata + similarity score)
```

**Critical rule:** Always use the **same embedding model** for ingestion and retrieval. Switching models invalidates your entire vector store.

---

## 7. Retrieval Pipeline

**`app/services/retrieval/retriever.py`**

```python
"""
RETRIEVAL CONCEPTS:

Dense retrieval (what we do):
  - Embed the query, search Qdrant by cosine similarity.
  - Strength: finds semantically similar text even with different wording.
  - Weakness: misses exact keyword matches a user expects.

Hybrid retrieval (next level):
  - Combine dense + sparse (BM25) with Reciprocal Rank Fusion.
  - Qdrant supports sparse vectors natively since v1.7.

Reranking:
  - A cross-encoder model scores (query, chunk) pairs jointly — more accurate
    than a bi-encoder but too slow to run over the full collection.
  - Run reranker only on the top_k * 3 candidates, return top_k.
"""

from openai import AsyncOpenAI
from app.core.config import settings
from app.services.retrieval.qdrant_client import search

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def _embed_query(query: str) -> list[float]:
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
    Embed the query, retrieve top_k chunks from Qdrant.
    Returns list of chunk payloads with similarity scores.
    """
    query_vector = await _embed_query(query)
    return await search(query_vector, top_k=top_k, filename_filter=filename_filter)
```

---

## 8. RAG Pipeline

**`app/services/rag/pipeline.py`**

```python
"""
RAG CONCEPTS:

Retrieval-Augmented Generation — instead of asking the LLM to answer from 
parametric memory (baked-in training weights), we:

  1. Retrieve relevant text from our DB (grounded, up-to-date, citable)
  2. Inject it into the prompt as context
  3. Instruct the LLM to answer ONLY from that context

Why this matters:
  - Reduces hallucination (model can't invent — it either finds it or says "I don't know")
  - Makes answers auditable (you know which chunk answered the question)
  - No need to fine-tune for new knowledge — just update the DB

The system prompt is the most important lever you control.
"""

from openai import AsyncOpenAI
from app.core.config import settings
from app.services.retrieval.retriever import retrieve

client = AsyncOpenAI(api_key=settings.openai_api_key)

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


def _build_context(chunks: list[dict]) -> str:
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
    Full RAG pipeline: retrieve → build context → generate.
    Returns answer + retrieved chunks for transparency.
    """
    # Step 1: Retrieve
    chunks = await retrieve(question, top_k=top_k, filename_filter=filename_filter)
    if not chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "chunks": [],
            "model": settings.openai_chat_model,
        }

    # Step 2: Build context
    context = _build_context(chunks)

    # Step 3: Assemble messages
    messages: list[dict] = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    
    # Inject prior conversation (multi-turn RAG)
    if conversation_history:
        messages.extend(conversation_history[-6:])   # last 3 turns
    
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    # Step 4: Generate
    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.1,    # low temp = more faithful to context
        max_tokens=1024,
    )

    return {
        "answer": response.choices[0].message.content,
        "chunks": chunks,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
```

---

## 9. Multi-Tool Agent

### 9.1 Tool Definitions — `app/services/agent/tools.py`

```python
"""
TOOL / FUNCTION CALLING CONCEPTS:

OpenAI's function calling lets the model decide WHEN to call a tool and 
what arguments to pass. The model outputs a structured JSON "tool call" 
instead of text, your code runs the tool, you send the result back, 
and the model incorporates it into the final answer.

The agentic loop:
  
  User message
      │
      ▼
  LLM → maybe "call tool X with args Y"
      │
      ▼
  Your code runs X(Y) → result
      │
      ▼
  LLM receives result → maybe "call tool Z" or final answer
      │
      ▼
  Repeat until no more tool calls → final user-facing answer

Each tool definition is a JSON schema the model reads at inference time.
The name + description are critical — the model reads them to decide which 
tool to pick. Be explicit and specific.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the internal document knowledge base and answer questions "
                "about ingested PDFs. Use this for any question that might be answered "
                "by uploaded documents. Returns relevant text excerpts and a generated answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to look up in documents."
                    },
                    "filename_filter": {
                        "type": "string",
                        "description": "Optional: restrict search to a specific document filename."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5, max 10).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents that have been ingested into the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a safe mathematical expression. Use for arithmetic, "
                "percentages, or numerical reasoning that should not be left to the LLM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python math expression, e.g. '(1500 * 0.08) / 12'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Generate a structured summary of a specific ingested document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Exact filename of the document to summarize."
                    }
                },
                "required": ["filename"]
            }
        }
    }
]
```

---

### 9.2 Tool Executor — `app/services/agent/executor.py`

```python
"""
The executor maps tool names → actual Python async functions.
It is the bridge between the LLM's decision and real code.
"""

import ast
import math
import operator
import json

from app.services.rag.pipeline import rag_query
from app.services.retrieval.qdrant_client import get_qdrant
from app.core.config import settings

# Safe math operators only
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Evaluate arithmetic expression without exec/eval risks."""
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsafe expression: {ast.dump(node)}")
    return _eval(ast.parse(expr, mode="eval").body)


async def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Runs the named tool with given args, returns a string result.
    The string is sent back to the LLM as the tool result message.
    """
    try:
        if tool_name == "rag_search":
            result = await rag_query(
                question=tool_args["query"],
                top_k=min(tool_args.get("top_k", 5), 10),
                filename_filter=tool_args.get("filename_filter"),
            )
            # Return a compact summary; the agent will synthesize the final answer
            sources = [
                f"[{c['source_filename']}, p.{c['page_number']}, score={c['score']:.2f}]"
                for c in result["chunks"]
            ]
            return json.dumps({
                "answer": result["answer"],
                "sources": sources,
            })

        elif tool_name == "list_documents":
            qc = get_qdrant()
            # Scroll through unique filenames in Qdrant payloads
            results, _ = await qc.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=1000,
                with_payload=["source_filename"],
            )
            filenames = sorted({r.payload["source_filename"] for r in results})
            return json.dumps({"documents": filenames, "count": len(filenames)})

        elif tool_name == "calculate":
            value = _safe_eval(tool_args["expression"])
            return json.dumps({"result": value, "expression": tool_args["expression"]})

        elif tool_name == "summarize_document":
            result = await rag_query(
                question="Provide a comprehensive summary covering: main topics, key findings, important data, and conclusions.",
                top_k=10,
                filename_filter=tool_args["filename"],
            )
            return json.dumps({"summary": result["answer"], "filename": tool_args["filename"]})

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
```

---

### 9.3 Agentic Loop — `app/services/agent/loop.py`

```python
"""
THE AGENTIC LOOP:

This is the core of an AI agent. Instead of a single prompt → response,
we run a loop:

  while LLM wants to call tools:
      1. Send messages to LLM
      2. LLM returns tool_calls (or final answer)
      3. Execute each tool call
      4. Append tool results to message history
      5. Go back to step 1

This continues until the model returns a normal text message (no tool calls).

MAX_ITERATIONS prevents infinite loops (a model misbehaving or a tool 
returning bad results that confuse the model into looping forever).
"""

import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.agent.tools import TOOLS
from app.services.agent.executor import execute_tool

client = AsyncOpenAI(api_key=settings.openai_api_key)

MAX_ITERATIONS = 10

AGENT_SYSTEM_PROMPT = """\
You are an intelligent document assistant with access to a knowledge base of ingested PDFs.

Your capabilities (via tools):
- rag_search: search documents and get grounded answers
- list_documents: see what documents are available
- calculate: do precise arithmetic
- summarize_document: get a full summary of a document

Behavior:
- Always search the knowledge base before answering document-related questions.
- If a user asks about multiple topics, use tools for each one sequentially.
- Be concise but complete. Cite sources when answering from documents.
- If you don't know and the knowledge base has nothing, say so clearly.
"""


async def run_agent(
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Run the full agentic loop for a single user turn.
    Returns the final answer + tool call trace for observability.
    """
    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    
    if conversation_history:
        messages.extend(conversation_history[-10:])   # last 5 turns
    
    messages.append({"role": "user", "content": user_message})
    
    tool_trace: list[dict] = []
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",   # model decides: call a tool or answer directly
            temperature=0.2,
        )
        
        choice = response.choices[0]
        
        # Append assistant message to history (with or without tool calls)
        messages.append(choice.message.model_dump(exclude_none=True))
        
        # If no tool calls → final answer
        if not choice.message.tool_calls:
            return {
                "answer": choice.message.content,
                "tool_trace": tool_trace,
                "iterations": iterations,
                "model": response.model,
            }
        
        # Execute each tool call in this turn
        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            
            tool_trace.append({"tool": fn_name, "args": fn_args})
            
            result = await execute_tool(fn_name, fn_args)
            
            tool_trace[-1]["result_preview"] = result[:200]
            
            # Append tool result in the required format
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    
    # Safety: if we hit max iterations, return what we have
    return {
        "answer": "I reached the maximum reasoning steps without a final answer. Please try a more specific question.",
        "tool_trace": tool_trace,
        "iterations": iterations,
        "model": settings.openai_chat_model,
    }
```

---

### 9.4 Conversation Memory — `app/services/agent/memory.py`

```python
"""
MEMORY CONCEPTS:

LLMs have NO persistent memory — every API call is stateless.
"Memory" is simply injecting past messages into the new request.

Types:
  Buffer memory     : keep the last N messages (what we implement)
  Summary memory    : summarize old context to compress tokens
  Vector memory     : embed past exchanges and retrieve relevant ones
  
For production: store conversation_id → messages in Redis or PostgreSQL.
Here we keep it simple with an in-process dict (lost on restart).
"""

from collections import deque

# conversation_id → list of {"role": ..., "content": ...} dicts
_store: dict[str, deque] = {}

MAX_HISTORY = 20   # messages per conversation


def get_history(conversation_id: str) -> list[dict]:
    return list(_store.get(conversation_id, []))


def append(conversation_id: str, role: str, content: str) -> None:
    if conversation_id not in _store:
        _store[conversation_id] = deque(maxlen=MAX_HISTORY)
    _store[conversation_id].append({"role": role, "content": content})


def clear(conversation_id: str) -> None:
    _store.pop(conversation_id, None)
```

---

## 10. FastAPI Layer

### Pydantic Request/Response Models — `app/models/`

```python
# app/models/agent.py
from pydantic import BaseModel, Field
from typing import Optional


class IngestRequest(BaseModel):
    # File upload handled by FastAPI's UploadFile — no body model needed


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filename_filter: Optional[str] = None


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str = Field(..., description="UUID for conversation continuity")


class ToolTrace(BaseModel):
    tool: str
    args: dict
    result_preview: str = ""


class AgentChatResponse(BaseModel):
    answer: str
    conversation_id: str
    tool_trace: list[ToolTrace]
    iterations: int
    model: str
```

### API Routers

**`app/api/ingest.py`**

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion.pipeline import ingest_pdf

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:    # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB).")
    
    result = await ingest_pdf(raw, file.filename)
    return result
```

**`app/api/query.py`**

```python
from fastapi import APIRouter
from app.models.agent import QueryRequest
from app.services.rag.pipeline import rag_query

router = APIRouter(prefix="/query", tags=["RAG Query"])


@router.post("/")
async def query_endpoint(body: QueryRequest):
    return await rag_query(
        question=body.question,
        top_k=body.top_k,
        filename_filter=body.filename_filter,
    )
```

**`app/api/agent.py`**

```python
from fastapi import APIRouter
from app.models.agent import AgentChatRequest, AgentChatResponse
from app.services.agent.loop import run_agent
from app.services.agent import memory

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(body: AgentChatRequest):
    history = memory.get_history(body.conversation_id)
    
    result = await run_agent(
        user_message=body.message,
        conversation_history=history,
    )
    
    # Persist turn to memory
    memory.append(body.conversation_id, "user", body.message)
    memory.append(body.conversation_id, "assistant", result["answer"])
    
    return AgentChatResponse(
        answer=result["answer"],
        conversation_id=body.conversation_id,
        tool_trace=result["tool_trace"],
        iterations=result["iterations"],
        model=result["model"],
    )


@router.delete("/chat/{conversation_id}")
async def clear_conversation(conversation_id: str):
    memory.clear(conversation_id)
    return {"cleared": conversation_id}
```

**`app/main.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import ingest, query, agent
from app.services.retrieval.qdrant_client import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure Qdrant collection exists
    await ensure_collection()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(title="RAG Agent API", version="0.1.0", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(agent.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 11. Concepts & Mental Models

### The Two Phases

| Phase | When | Input | Output |
|-------|------|-------|--------|
| **Ingestion** | Offline / on upload | Raw PDF bytes | Vectors in Qdrant |
| **Inference** | On every user query | User question | Grounded answer |

### Why Chunking Size Matters

| Chunk size | Problem |
|-----------|---------|
| Too small (< 100 tokens) | Loses context; chunks are meaningless fragments |
| Too large (> 1000 tokens) | Pollutes retrieved context with irrelevant content; hits embedding model limits |
| 400–600 tokens | Sweet spot for most knowledge retrieval tasks |

### Similarity Score Threshold

Setting `similarity_threshold = 0.7` means: only return chunks where the query's vector and the chunk's vector are at least 70% similar (in cosine space). Too low → noisy irrelevant chunks pollute the context. Too high → miss relevant chunks. Start at 0.65–0.75 and tune empirically.

### Tool Calling vs RAG — When to Use Each

| Situation | Pattern |
|-----------|---------|
| User asks a document question | Agent calls `rag_search` → RAG pipeline |
| User asks "what docs do you have?" | Agent calls `list_documents` |
| User asks "what's 15% of £4,200?" | Agent calls `calculate` |
| User asks multi-part question | Agent calls multiple tools in sequence |
| Simple greetings / meta questions | Agent answers directly (no tool call) |

### Production Checklist

- [ ] Swap in-memory conversation store (memory.py) for Redis
- [ ] Add rate limiting (`slowapi` or API gateway)
- [ ] Add authentication (JWT on all endpoints)
- [ ] Enable Qdrant persistent storage (done via Docker volume mount)
- [ ] Add reranking (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` via HuggingFace)
- [ ] Add async task queue (Celery + Redis) for large PDF ingestion
- [ ] Store chunk hashes to detect and skip duplicate documents on re-upload
- [ ] Add evaluation: compare retrieved chunk relevance and answer faithfulness with `ragas`

---

*Generated as a learning-first production blueprint — every layer has inline explanations of the underlying AI engineering concepts.*