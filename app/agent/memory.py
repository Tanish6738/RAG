from collections import deque

# In-memory dictionary mapping conversation_id -> deque of message dicts
_store: dict[str, deque] = {}

MAX_HISTORY = 20  # Keeps the last 20 messages (10 turns) per conversation


def get_history(conversation_id: str) -> list[dict]:
    """
    Retrieves the conversation history for a specific UUID as a list of dicts.
    """
    return list(_store.get(conversation_id, []))


def append(conversation_id: str, role: str, content: str) -> None:
    """
    Appends a user or assistant message to the conversation history.
    """
    if conversation_id not in _store:
        _store[conversation_id] = deque(maxlen=MAX_HISTORY)
    _store[conversation_id].append({"role": role, "content": content})


def clear(conversation_id: str) -> None:
    """
    Clears the history for a specific conversation.
    """
    _store.pop(conversation_id, None)
