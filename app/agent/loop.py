import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.agent.tools import TOOLS
from app.agent.executor import execute_tool

client = AsyncOpenAI(api_key=settings.OPENAI_API, base_url=settings.BASE_URL)

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
    Returns a dictionary containing the final answer, trace, iterations, and model.
    """
    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-10:])  # last 5 turns

    messages.append({"role": "user", "content": user_message})

    tool_trace: list[dict] = []
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # Model decides: call a tool or answer directly
            temperature=0.2,
        )

        choice = response.choices[0]

        # Append assistant message to history (with or without tool calls)
        messages.append(choice.message.model_dump(exclude_none=True))

        # If no tool calls -> final answer
        if not choice.message.tool_calls:
            return {
                "answer": choice.message.content or "",
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

    # Safety mechanism to break infinite tool execution loops
    return {
        "answer": "I reached the maximum reasoning steps without a final answer. Please try a more specific question.",
        "tool_trace": tool_trace,
        "iterations": iterations,
        "model": settings.openai_chat_model,
    }
