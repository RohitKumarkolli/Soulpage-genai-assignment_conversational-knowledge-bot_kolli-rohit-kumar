"""
ui/sidebar.py
-------------
Sidebar UI — updated for Milestone 9.

New in M9:
    • Shows active memory strategy (Buffer vs Summary)
    • Displays live summary text when summary memory is active
    • Shows knowledge_base tool status badge
    • KB entry count shown in tool status
"""

import time
import streamlit as st

from ..config import APP_CONFIG, MEMORY_CONFIG
from ..memory import memory_store


def render_status_badges(agent_ready: bool) -> None:
    """Render coloured status indicator badges."""
    if agent_ready:
        st.markdown(
            '<span class="status-badge badge-green">● LLM Online</span>'
            '<span class="status-badge badge-blue">● Memory Active</span><br>'
            '<span class="status-badge badge-purple">● Web Search</span>'
            '<span class="status-badge badge-purple">● Wikipedia</span><br>'
            '<span class="status-badge badge-yellow">● Knowledge Base</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge badge-red">⚠ Agent Offline</span>',
            unsafe_allow_html=True,
        )


def render_memory_strategy() -> None:
    """
    Show which memory strategy is active and key stats.
    New in Milestone 9.
    """
    strategy = "📝 Summary Memory" if MEMORY_CONFIG.use_summary_memory else "📋 Buffer Memory"
    st.markdown(f"**Strategy:** {strategy}")

    if MEMORY_CONFIG.use_summary_memory:
        st.caption(
            f"Compresses turns older than "
            f"~{MEMORY_CONFIG.summary_token_limit} tokens. "
            f"Prevents context overflow in long sessions."
        )
    else:
        st.caption("Stores all messages verbatim. Best for short sessions.")


def render_session_stats() -> None:
    """Display message count, query count, and session ID."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.get("messages", [])))
    with col2:
        st.metric("Queries", st.session_state.get("total_queries", 0))
    session_id = st.session_state.get("session_id", "")
    st.caption(f"Session: `{session_id[:22]}…`")


def render_memory_inspector(summary_manager=None) -> None:
    """
    Display memory contents — buffer messages + optional summary.

    M9 update: shows the running summary from ConversationSummaryMemory
    alongside (or instead of) the raw message buffer, giving the user
    full visibility into what the LLM will receive as context.

    Args:
        summary_manager: SummaryMemoryManager | None
    """
    st.caption("What KnowledgeBot remembers")

    session_id = st.session_state.get("session_id", "default")

    # ── Summary section (M9) ──────────────────────────────────────────────────
    if summary_manager and MEMORY_CONFIG.use_summary_memory:
        summary_text = summary_manager.get_summary(session_id)
        if summary_text:
            with st.expander("📝 Running Summary (compressed history)", expanded=False):
                st.markdown(
                    f'<div style="color:#94a3b8;font-size:0.85rem;'
                    f'background:#1e293b;padding:10px;border-radius:8px;'
                    f'border-left:3px solid #a78bfa">{summary_text}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("*No summary yet — builds after first few turns.*")

    # ── Buffer messages ────────────────────────────────────────────────────────
    messages = memory_store.get_all_messages(session_id)

    if not messages:
        st.info("Memory is empty. Start chatting!", icon="💭")
        return

    with st.expander(
        f"View {len(messages)} recent messages (verbatim)",
        expanded=False,
    ):
        for i, msg in enumerate(messages, 1):
            role    = msg.__class__.__name__.replace("Message", "")
            preview = msg.content[:90] + ("…" if len(msg.content) > 90 else "")
            color   = "mem-human" if role == "Human" else "mem-ai"
            icon    = "👤" if role == "Human" else "🤖"
            st.markdown(
                f'<div class="mem-row">'
                f'<span class="{color}">{icon} [{i}] {role}</span><br>'
                f'<span style="color:#64748b">{preview}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_controls() -> None:
    """Clear Conversation and New Session buttons."""
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        session_id = st.session_state.session_id
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        memory_store.clear_session(session_id)
        st.success("Conversation cleared!")
        st.rerun()

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.session_state.session_id    = f"session_{int(time.time())}"
        st.success("New session started!")
        st.rerun()


def render_suggestions() -> None:
    """Example prompts — updated for M9 to test all three tools."""
    suggestions = [
        "What is KnowledgeBot?",           # → knowledge_base
        "How does ReAct agent work?",      # → knowledge_base
        "Who is Sam Altman?",              # → wikipedia
        "Latest AI news this week",        # → web_search
        "What memory types do you use?",   # → knowledge_base
        "Who won the last FIFA World Cup?", # → web_search
    ]
    for s in suggestions:
        st.caption(f"• {s}")


def render_sidebar(agent_ready: bool, summary_manager=None) -> None:
    """
    Render the complete sidebar.

    Args:
        agent_ready     : Whether the agent loaded without errors.
        summary_manager : SummaryMemoryManager | None (M9).
    """
    with st.sidebar:
        st.markdown(f"## 🤖 {APP_CONFIG.bot_name}")
        st.markdown("*Conversational AI with Memory & Tools*")
        st.divider()

        st.markdown("### ⚡ System Status")
        render_status_badges(agent_ready)
        st.divider()

        st.markdown("### 📊 Session Stats")
        render_session_stats()
        st.divider()

        st.markdown("### 🧠 Memory")
        render_memory_strategy()
        st.divider()

        st.markdown("### 🗂️ Memory Inspector")
        render_memory_inspector(summary_manager)
        st.divider()

        st.markdown("### 🎛️ Controls")
        render_controls()
        st.divider()

        st.markdown("### 💡 Try Asking")
        render_suggestions()
        st.divider()

        st.markdown(
            f'<div style="color:#475569;font-size:0.75rem;text-align:center">'
            f"{APP_CONFIG.bot_name} v{APP_CONFIG.bot_version}<br>"
            f"LangChain · Groq · Streamlit</div>",
            unsafe_allow_html=True,
        )