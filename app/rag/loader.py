import io
from pathlib import Path
import pdfplumber
from pypdf import PdfReader
import structlog

from app.rag.schema import PageContent, DocumentContent
from app.rag.chunker import TokenChunker
from app.rag.embedder import embed_chunks
from app.rag.qdrant import ensure_collection, upsert_chunks

log = structlog.get_logger()


class PDFExtractor:
    """
    Handles PDF text and table extraction.
    Uses pypdf for fast, text-only parsing and metadata retrieval.
    Falls back to pdfplumber for layout-aware table extraction if character count is sparse.
    """
    SPARSE_THRESHOLD = 50  # Average chars per page. Below this -> use pdfplumber.

    def extract(self, source: Path | bytes, filename: str = "doc.pdf") -> DocumentContent:
        raw = source if isinstance(source, bytes) else source.read_bytes()

        # Pass 1: Quick pypdf extraction
        content = self._extract_pypdf(raw, filename)

        avg_chars = (
            sum(p.char_count for p in content.pages) / len(content.pages)
            if content.pages
            else 0
        )

        # Pass 2: If pypdf is sparse, upgrade to pdfplumber
        if avg_chars < self.SPARSE_THRESHOLD:
            log.info("ingestion.extract.upgrade_to_pdfplumber", filename=filename, avg_chars=avg_chars)
            content = self._extract_pdfplumber(raw, filename)

        return content

    def _extract_pypdf(self, raw: bytes, filename: str) -> DocumentContent:
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(
                PageContent(
                    page_number=i + 1,
                    text=text.strip(),
                    tables=[],
                    has_images=bool(page.images),
                    char_count=len(text),
                )
            )

        # Clean metadata to string representation
        raw_metadata = reader.metadata or {}
        cleaned_metadata = {str(k): str(v) for k, v in raw_metadata.items()}

        return DocumentContent(
            filename=filename,
            total_pages=len(reader.pages),
            pages=pages,
            metadata=cleaned_metadata,
            extraction_method="pypdf",
        )

    def _extract_pdfplumber(self, raw: bytes, filename: str) -> DocumentContent:
        pages = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []

                # Convert table cells: None -> ""
                clean_tables = [
                    [[cell or "" for cell in row] for row in table]
                    for table in tables
                ]

                # Append table text after body text
                table_text = self._tables_to_text(clean_tables)
                full_text = f"{text}\n\n{table_text}".strip()

                pages.append(
                    PageContent(
                        page_number=i + 1,
                        text=full_text,
                        tables=clean_tables,
                        has_images=bool(page.images),
                        char_count=len(full_text),
                    )
                )

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


async def ingest_pdf(raw_bytes: bytes, filename: str) -> dict:
    """
    Orchestrates the entire ingestion pipeline:
    Extract text -> Chunk into tokens -> Generate embeddings -> Store in Qdrant.
    """
    # 0. Ensure collection exists
    await ensure_collection()

    # 1. Extract
    log.info("ingestion.extract.start", filename=filename)
    extractor = PDFExtractor()
    doc = extractor.extract(raw_bytes, filename)
    log.info(
        "ingestion.extract.done",
        pages=doc.total_pages,
        method=doc.extraction_method,
    )

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
