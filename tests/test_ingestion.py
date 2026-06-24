from app.rag.chunker import TokenChunker
from app.rag.schema import DocumentContent, PageContent


def test_chunker_basic():
    """
    Test that the sliding window TokenChunker functions correctly on a mock document.
    """
    chunker = TokenChunker(chunk_size=100, overlap=10)
    doc = DocumentContent(
        filename="dummy.pdf",
        total_pages=1,
        pages=[
            PageContent(
                page_number=1,
                text="Hello world! This is a test of the ingestion pipeline's chunker.",
                tables=[],
                has_images=False,
                char_count=67,
            )
        ],
        metadata={},
        extraction_method="test",
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Hello world! This is a test of the ingestion pipeline's chunker."
