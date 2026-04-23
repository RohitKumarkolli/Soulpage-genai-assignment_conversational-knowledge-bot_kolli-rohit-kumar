"""
memory/summary_store.py
-----------------------
ConversationSummaryMemory manager for KnowledgeBot.

HOW ConversationSummaryMemory WORKS:
    1. For the first few messages → stores them verbatim (like BufferMemory)
    2. When conversation grows beyond a token threshold:
         → LLM summarises the OLDER portion of the history
         → Keeps the summary as a running text buffer
         → Keeps the most recent messages verbatim
    3. On every new turn, the LLM sees:
         Summary: "User asked about X, bot explained Y..."  (compressed)
         + Most recent turns verbatim

WHY this matters:
    Without summary memory, a 50-turn conversation fills the LLM's
    8192-token context window. With summary memory, old turns compress
    to ~100 tokens — conversations can go on indefinitely.

COMPATIBILITY NOTE:
    LangChain renamed the summary buffer attribute across versions:
        < 0.1.x  → moving_summary_buffer
        >= 0.1.x → buffer
    We use getattr() with fallbacks so the code works on all versions.
"""

from langchain.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq


class SummaryMemoryManager:
    """
    Manages ConversationSummaryMemory instances per session.

    ConversationSummaryMemory uses an LLM internally to generate
    summaries — so it needs the same LLM instance as the agent.

    Usage:
        manager = SummaryMemoryManager(llm)
        manager.save_context("session_1", "Who is Musk?", "He is CEO of...")
        summary = manager.get_summary("session_1")
    """

    def __init__(self, llm: ChatGroq) -> None:
        """
        Args:
            llm: ChatGroq instance used to generate summaries.
                 Should be the same model as the main agent.
        """
        self._llm   = llm
        self._store = {}    # session_id → ConversationSummaryMemory

    # ── Private helper ─────────────────────────────────────────────────────────

    def _get_or_create(self, session_id: str) -> ConversationSummaryMemory:
        """
        Retrieve or create a ConversationSummaryMemory for session_id.

        Args:
            session_id: Unique session identifier.

        Returns:
            ConversationSummaryMemory for this session.
        """
        if session_id not in self._store:
            self._store[session_id] = ConversationSummaryMemory(
                llm=self._llm,
                memory_key="chat_history",
                return_messages=True,
                human_prefix="User",
                ai_prefix="KnowledgeBot",
            )
        return self._store[session_id]

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_summary(self, session_id: str) -> str:
        """
        Return the current running summary text for a session.

        COMPATIBILITY FIX:
            LangChain renamed `moving_summary_buffer` to `buffer` in v0.1+.
            We try both names with getattr() so this works across all versions.

        Args:
            session_id: The session to inspect.

        Returns:
            str: The summary string, or empty string if none exists yet.
        """
        if session_id not in self._store:
            return ""

        memory = self._store[session_id]

        # Try new name first (LangChain >= 0.1), fall back to old name
        summary = (
            getattr(memory, "buffer", None)
            or getattr(memory, "moving_summary_buffer", None)
            or ""
        )
        return summary

    def get_message_count(self, session_id: str) -> int:
        """
        Return how many messages are in the unsummarised buffer.

        Args:
            session_id: The session to inspect.

        Returns:
            int: Message count (0 if session doesn't exist).
        """
        if session_id not in self._store:
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
        """
        Save a human + AI turn to summary memory.

        ConversationSummaryMemory.save_context() internally decides whether
        to keep the turn verbatim or trigger an LLM summarisation pass.

        Args:
            session_id   : The session to update.
            human_message: The user's message text.
            ai_message   : The bot's response text.
        """
        memory = self._get_or_create(session_id)
        try:
            memory.save_context(
                {"input": human_message},
                {"output": ai_message},
            )
        except Exception as e:
            # Gracefully degrade — summary memory failure should never
            # break the main chat flow
            print(f"[SummaryMemory] save_context failed for {session_id}: {e}")

    def clear_session(self, session_id: str) -> None:
        """Wipe summary memory for a specific session."""
        if session_id in self._store:
            del self._store[session_id]

    def load_memory_variables(self, session_id: str) -> dict:
        """
        Load memory variables for prompt injection.

        Returns:
            dict: e.g. {"chat_history": [list of messages]}
        """
        memory = self._get_or_create(session_id)
        try:
            return memory.load_memory_variables({})
        except Exception as e:
            print(f"[SummaryMemory] load_memory_variables failed: {e}")
            return {"chat_history": []}