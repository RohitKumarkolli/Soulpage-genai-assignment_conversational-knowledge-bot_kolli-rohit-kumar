"""
app.py
------
KnowledgeBot v2.0 — Streamlit entry point (Milestone 9).

Changes from M7:
    • build_agent() now returns (agent, summary_manager) tuple
    • summary_manager is threaded through render_sidebar and process_user_input
    • Everything else stays thin — business logic lives in submodules

Run from the PROJECT ROOT:
    streamlit run run.py
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv

from .config     import APP_CONFIG
from .agents     import build_agent
from .ui.sidebar import render_sidebar
from .ui.chat    import (
    render_header,
    render_welcome,
    render_chat_history,
    process_user_input,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_CONFIG.page_title,
    page_icon=APP_CONFIG.page_icon,
    layout=APP_CONFIG.layout,
    initial_sidebar_state=APP_CONFIG.sidebar_state,
)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    .kb-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .kb-header h1 {
        margin: 0; font-size: 1.8rem;
        background: linear-gradient(90deg, #4fc3f7, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kb-header p { margin: 4px 0 0 0; color: #8892b0; font-size: 0.9rem; }

    .status-badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.75rem;
        font-weight: 600; margin: 2px 4px 2px 0;
    }
    .badge-green  { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
    .badge-blue   { background:#1a2a3a; color:#60a5fa; border:1px solid #60a5fa; }
    .badge-purple { background:#2a1a3a; color:#c084fc; border:1px solid #c084fc; }
    .badge-yellow { background:#2a2a1a; color:#facc15; border:1px solid #facc15; }
    .badge-red    { background:#3a1a1a; color:#f87171; border:1px solid #f87171; }

    .tool-tag {
        font-size: 0.72rem; color: #94a3b8;
        margin-top: 8px; padding: 4px 10px;
        background: #1e293b; border-radius: 6px;
        border-left: 3px solid #4fc3f7; display: inline-block;
    }

    .mem-row { padding:6px 0; border-bottom:1px solid #1e293b; font-size:0.82rem; }
    .mem-human { color:#4ade80; }
    .mem-ai    { color:#c084fc; }

    .welcome-card {
        background: #1a1f2e; border: 1px solid #2d3561;
        border-radius: 12px; padding: 28px;
        text-align: center; color: #8892b0;
        margin: 40px auto; max-width: 560px;
    }
    .welcome-card h3 { color:#e2e8f0; margin-bottom:8px; }
    .suggestion {
        background: #0f172a; border: 1px solid #334155;
        border-radius: 8px; padding: 8px 14px;
        margin: 6px 0; font-size: 0.88rem; text-align: left;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e293b;
    }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    """Initialise session_state keys — never overwrites existing values."""
    st.session_state.setdefault("messages",      [])
    st.session_state.setdefault("total_queries", 0)
    st.session_state.setdefault("session_id",    f"session_{int(time.time())}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Streamlit app entry point — reruns top-to-bottom on every user action.

    M9 change: build_agent() returns (agent, summary_manager).
    summary_manager is passed to render_sidebar and process_user_input
    so they can display / update summary memory state.
    """
    _init_session_state()

    # ── Environment check ──────────────────────────────────────────────────────
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "**GROQ_API_KEY not found.**\n\n"
            "Add it to your `.env` file:\n```\nGROQ_API_KEY=your_key_here\n```\n"
            "Get a free key at [console.groq.com](https://console.groq.com)",
            icon="🔑",
        )
        render_sidebar(agent_ready=False)
        st.stop()

    # ── Build agent + summary manager (cached) ─────────────────────────────────
    agent, summary_manager = build_agent()

    # ── UI ─────────────────────────────────────────────────────────────────────
    render_sidebar(agent_ready=True, summary_manager=summary_manager)
    render_header()

    if not st.session_state.messages:
        render_welcome()
    else:
        render_chat_history()

    # ── Input ──────────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask me anything…"):
        process_user_input(agent, user_input, summary_manager=summary_manager)
        st.rerun()


if __name__ == "__main__":
    main()