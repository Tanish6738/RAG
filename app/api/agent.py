from fastapi import APIRouter
from app.rag.schema import AgentChatRequest, AgentChatResponse
from app.agent.loop import run_agent
from app.agent import memory

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(body: AgentChatRequest):
    """
    POST endpoint to converse with the multi-tool agent.
    Retrieves history, processes the user message, updates conversation memory, and returns the response.
    """
    history = memory.get_history(body.conversation_id)

    result = await run_agent(
        user_message=body.message,
        conversation_history=history,
    )

    # Persist the exchange to conversation memory
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
    """
    DELETE endpoint to clear conversation memory for a specific conversation ID.
    """
    memory.clear(conversation_id)
    return {"cleared": conversation_id}
