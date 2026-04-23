"""
ui/chat.py
----------
Chat area UI — updated for Milestone 9.

Changes:
    • process_user_input() accepts summary_manager and passes it to invoke_agent
    • Tool badge now shows knowledge_base with a distinct icon
"""

import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory

from ..config import APP_CONFIG
from ..agents import invoke_agent


# ── Tool badge icon mapping ────────────────────────────────────────────────────
_TOOL_ICONS = {
    "web_search"    : "🌐",
    "wikipedia"     : "📚",
    "knowledge_base": "🗂️",
}


def render_header() -> None:
    """Top banner with gradient title."""
    st.markdown(
        f"""
        <div class="kb-header">
            <div style="font-size:2.5rem">🤖</div>
            <div>
                <h1>{APP_CONFIG.bot_name}</h1>
                <p>Conversational AI · Memory · Web Search · Knowledge Base</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """Welcome card — updated for M9 to include knowledge base examples."""
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome to KnowledgeBot v2.0!</h3>
            <p>I remember our conversation, search the web in real time,<br>
               and answer from a private knowledge base.</p>
            <br>
            <p style="font-size:0.82rem;color:#475569">Try one of these:</p>
            <div class="suggestion">🗂️ What is KnowledgeBot and how does it work?</div>
            <div class="suggestion">🔍 Who is the current CEO of OpenAI?</div>
            <div class="suggestion">📚 What is quantum entanglement?</div>
            <div class="suggestion">📰 Latest AI news this week</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_badge(tools_used: list) -> None:
    """Render source badge with tool-specific icons."""
    if not tools_used:
        return
    parts = []
    for t in tools_used:
        icon = _TOOL_ICONS.get(t, "🔍")
        parts.append(f"{icon} {t}")
    tools_str = " · ".join(parts)
    st.markdown(
        f'<div class="tool-tag">Sources used: {tools_str}</div>',
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Re-render all messages from session_state on every rerun."""
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            render_tool_badge(msg.get("tools_used", []))


def process_user_input(
    agent: RunnableWithMessageHistory,
    user_input: str,
    summary_manager=None,
) -> None:
    """
    Handle one user → agent → response cycle.

    M9 update: passes summary_manager to invoke_agent so the turn
    is saved to ConversationSummaryMemory in addition to the buffer.

    Args:
        agent          : The RunnableWithMessageHistory agent.
        user_input     : The user's message string.
        summary_manager: SummaryMemoryManager | None
    """

    # ── User bubble ────────────────────────────────────────────────────────────
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "tools_used": []}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # ── Agent + assistant bubble ───────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            result = invoke_agent(
                agent=agent,
                user_input=user_input,
                session_id=st.session_state.session_id,
                summary_manager=summary_manager,   # M9: pass summary manager
            )

        st.markdown(result["answer"])
        render_tool_badge(result["tools_used"])

    # ── Persist ────────────────────────────────────────────────────────────────
    st.session_state.messages.append({
        "role"      : "assistant",
        "content"   : result["answer"],
        "tools_used": result["tools_used"],
    })
    st.session_state.total_queries += 1