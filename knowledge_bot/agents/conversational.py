"""
memory/summary_store.py
-----------------------
ConversationSummaryMemory manager — version-safe imports.

COMPATIBILITY:
    ConversationSummaryMemory moved between LangChain versions.
    We try multiple import paths with fallbacks.
    The summary buffer attribute also changed names across versions.
"""

from langchain_groq import ChatGroq

# ── Version-safe import for ConversationSummaryMemory ─────────────────────────
try:
    from langchain.memory import ConversationSummaryMemory
except ImportError:
    try:
        from langchain_community.memory import ConversationSummaryMemory  # type: ignore
    except ImportError:
        ConversationSummaryMemory = None  # type: ignore


class SummaryMemoryManager:
    """
    Manages ConversationSummaryMemory instances per session.

    Falls back gracefully if ConversationSummaryMemory is unavailable
    in the installed LangChain version — chat continues without summary.
    """

    def __init__(self, llm: ChatGroq) -> None:
        self._llm   = llm
        self._store = {}
        self._available = ConversationSummaryMemory is not None

        if not self._available:
            print("[SummaryMemory] ConversationSummaryMemory not available — "
                  "summary memory disabled, using buffer only.")

    def _get_or_create(self, session_id: str):
        """Retrieve or create a ConversationSummaryMemory for session_id."""
        if not self._available:
            return None
        if session_id not in self._store:
            self._store[session_id] = ConversationSummaryMemory(
                llm=self._llm,
                memory_key="chat_history",
                return_messages=True,
                human_prefix="User",
                ai_prefix="KnowledgeBot",
            )
        return self._store[session_id]

    def get_summary(self, session_id: str) -> str:
        """
        Return the running summary text for a session.

        Tries multiple attribute names for cross-version compatibility:
            buffer               (LangChain >= 0.1)
            moving_summary_buffer (LangChain < 0.1)
        """
        if not self._available or session_id not in self._store:
            return ""
        memory = self._store[session_id]
        summary = (
            getattr(memory, "buffer", None)
            or getattr(memory, "moving_summary_buffer", None)
            or ""
        )
        return summary

    def get_message_count(self, session_id: str) -> int:
        """Return number of messages in the unsummarised buffer."""
        if not self._available or session_id not in self._store:
            return 0
        try:
            return len(self._store[session_id].chat_memory.messages)
        except Exception:
            return 0

    def save_context(
        self,
        session_id: str,
        human_message: str,
        ai_message: str,
    ) -> None:
        """Save a human + AI turn to summary memory."""
        if not self._available:
            return
        memory = self._get_or_create(session_id)
        if memory is None:
            return
        try:
            memory.save_context(
                {"input": human_message},
                {"output": ai_message},
            )
        except Exception as e:
            print(f"[SummaryMemory] save_context failed: {e}")

    def clear_session(self, session_id: str) -> None:
        """Wipe summary memory for a session."""
        if session_id in self._store:
            del self._store[session_id]

    def load_memory_variables(self, session_id: str) -> dict:
        """Load memory variables for prompt injection."""
        if not self._available:
            return {"chat_history": []}
        memory = self._get_or_create(session_id)
        if memory is None:
            return {"chat_history": []}
        try:
            return memory.load_memory_variables({})
        except Exception as e:
            print(f"[SummaryMemory] load failed: {e}")
            return {"chat_history": []}