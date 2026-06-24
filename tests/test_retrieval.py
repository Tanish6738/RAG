from app.rag.retrieval import _build_context


def test_build_context():
    """
    Test that the context builder wraps candidate chunks in XML tags and formats them properly.
    """
    chunks = [
        {
            "source_filename": "company_report.pdf",
            "page_number": 3,
            "score": 0.8654,
            "text": "The Q4 revenue increased by 12% compared to last year.",
        }
    ]
    context = _build_context(chunks)
    assert '<chunk id="1" source="company_report.pdf" page="3" score="0.87">' in context
    assert "The Q4 revenue increased by 12%" in context
