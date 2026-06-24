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
