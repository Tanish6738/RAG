import tiktoken
from app.core.config import settings
from app.rag.schema import DocumentContent, Chunk


class TokenChunker:
    """
    Sliding window chunker in token space using tiktoken.
    Splits text from pages into chunks of length `chunk_size` with `overlap` tokens.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        overlap: int = settings.chunk_overlap,
        encoding_name: str = "cl100k_base",  # default for OpenAI v3/v4 models (cl100k_base / o200k_base)
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, doc: DocumentContent) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        carry_tokens: list[int] = []  # overlap tokens carried from previous page

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

            carry_tokens = window  # leftover becomes carry for next page

        # Flush any remaining tokens
        if carry_tokens and doc.pages:
            text = self.enc.decode(carry_tokens)
            chunks.append(Chunk(
                chunk_id=f"{doc.filename}::p{doc.pages[-1].page_number}::c{chunk_index}",
                text=text,
                token_count=len(carry_tokens),
                page_number=doc.pages[-1].page_number,
                chunk_index=chunk_index,
                source_filename=doc.filename,
                metadata={
                    "extraction_method": doc.extraction_method,
                    "total_pages": doc.total_pages,
                },
            ))

        return chunks
