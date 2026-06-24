import ast
import json
import math
import operator

from app.rag.retrieval import rag_query
from app.rag.qdrant import get_qdrant
from app.core.config import settings

# Safe math operators only to avoid arbitrary code execution
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """
    Evaluate arithmetic expression securely without exec/eval risks by parsing the AST.
    """
    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        # Deprecated but kept for older Python versions backward compatibility
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsafe expression: {ast.dump(node)}")
        
    cleaned_expr = expr.strip()
    return _eval(ast.parse(cleaned_expr, mode="eval").body)


async def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Dispatches tool names to actual Python async functions and returns a JSON string.
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
                for c in result.get("chunks", [])
            ]
            return json.dumps({
                "answer": result.get("answer", ""),
                "sources": sources,
            })

        elif tool_name == "list_documents":
            qc = get_qdrant()
            # Scroll through up to 1000 points to collect unique source_filenames
            results, _ = await qc.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=1000,
                with_payload=["source_filename"],
            )
            filenames = sorted({
                r.payload.get("source_filename")
                for r in results
                if r.payload and "source_filename" in r.payload
            })
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
            return json.dumps({"summary": result.get("answer", ""), "filename": tool_args["filename"]})

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})
