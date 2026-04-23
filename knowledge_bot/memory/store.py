"""
memory/store.py
---------------
Session memory management for KnowledgeBot.
"""

from typing import Dict
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class SessionMemoryStore:
    """
    In-process session memory store.
    One ChatMessageHistory per session_id, stored in a plain dict.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ChatMessageHistory] = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve or create the ChatMessageHistory for session_id.
        This method signature matches what RunnableWithMessageHistory expects.
        """
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def clear_session(self, session_id: str) -> None:
        """Wipe history for a specific session."""
        self._store[session_id] = ChatMessageHistory()

    def clear_all(self) -> None:
        """Wipe ALL session histories."""
        self._store.clear()

    def session_exists(self, session_id: str) -> bool:
        """Return True if the session has any stored messages."""
        return (
            session_id in self._store
            and len(self._store[session_id].messages) > 0
        )

    def get_message_count(self, session_id: str) -> int:
        """Return the number of messages stored for a session."""
        if session_id not in self._store:
            return 0
        return len(self._store[session_id].messages)

    def get_all_messages(self, session_id: str) -> list:
        """Return all BaseMessage objects for a session."""
        if session_id not in self._store:
            return []
        return self._store[session_id].messages


# Module-level singleton — shared across the entire application
memory_store = SessionMemoryStore()