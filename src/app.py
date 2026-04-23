"""
app.py
------
Milestone 6: Streamlit Chat Interface (FIXED)

Key fixes applied:
    1. Removed output_messages_key from RunnableWithMessageHistory
       (only valid for chains returning messages, not AgentExecutor strings)
    2. Removed [Source:...] instruction from system prompt
       (was causing LLM to return ONLY the tag, dropping the actual answer)
    3. Tool detection via intermediate_steps (reliable, no text parsing)

Run:
    streamlit run src/app.py
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import os
import time
from typing import Dict

# ── Streamlit ──────────────────────────────────────────────────────────────────
import streamlit as st

# ── LangChain ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be FIRST Streamlit call
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KnowledgeBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
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
        margin: 0;
        font-size: 1.8rem;
        background: linear-gradient(90deg, #4fc3f7, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kb-header p { margin: 4px 0 0 0; color: #8892b0; font-size: 0.9rem; }

    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-green  { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
    .badge-blue   { background:#1a2a3a; color:#60a5fa; border:1px solid #60a5fa; }
    .badge-purple { background:#2a1a3a; color:#c084fc; border:1px solid #c084fc; }
    .badge-yellow { background:#2a2a1a; color:#facc15; border:1px solid #facc15; }

    .tool-tag {
        font-size: 0.72rem;
        color: #94a3b8;
        margin-top: 8px;
        padding: 4px 10px;
        background: #1e293b;
        border-radius: 6px;
        border-left: 3px solid #4fc3f7;
        display: inline-block;
    }

    .mem-row { padding:6px 0; border-bottom:1px solid #1e293b; font-size:0.82rem; }
    .mem-human { color:#4ade80; }
    .mem-ai    { color:#c084fc; }

    .welcome-card {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        color: #8892b0;
        margin: 40px auto;
        max-width: 560px;
    }
    .welcome-card h3 { color:#e2e8f0; margin-bottom:8px; }
    .suggestion {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        text-align: left;
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
# 1.  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_environment() -> bool:
    """Load .env and return True if GROQ_API_KEY exists."""
    load_dotenv()
    return bool(os.getenv("GROQ_API_KEY"))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SESSION MEMORY STORE  (module-level — survives Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════════

_session_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return or create a ChatMessageHistory for the given session_id."""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD AGENT  (cached — built once, reused forever)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_agent() -> RunnableWithMessageHistory:
    """
    Build and cache the full conversational agent.

    ROOT CAUSE OF BLANK RESPONSES — explained:

    Bug 1: output_messages_key="output"
        RunnableWithMessageHistory uses output_messages_key to save the AI
        response back into the LangChain memory store. When set to "output",
        it expects result["output"] to be a list of BaseMessage objects.
        AgentExecutor returns a plain string there instead.
        This type mismatch caused LangChain to silently swallow the answer.
        FIX → remove output_messages_key entirely. The wrapper auto-saves
        the HumanMessage + AIMessage to history correctly without it.

    Bug 2: "[Source: web_search | wikipedia | internal]" in system prompt
        This format instruction told the LLM to end every reply with a tag.
        The LLM interpreted "end with" as "output only", producing responses
        like: "[Source: wikipedia]" with nothing before it.
        FIX → remove the format instruction. Tool detection now uses
        intermediate_steps (reliable, no text parsing needed).
    """

    # ── Tools ──────────────────────────────────────────────────────────────────
    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the internet for current, recent, or real-time information. "
            "Use for: news, recent events, sports scores, prices, latest releases. "
            "Input: a concise search query string."
        ),
    )

    wiki_tool = WikipediaQueryRun(
        name="wikipedia",
        description=(
            "Look up encyclopedic or factual background information. "
            "Use for: biographies, history, science, geography, definitions. "
            "Input: a topic name or question."
        ),
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=1000,
        ),
    )

    tools = [search_tool, wiki_tool]

    # ── LLM ────────────────────────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
    )

    # ── Prompt ─────────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are KnowledgeBot, a helpful and intelligent AI assistant "
                "with access to web search and Wikipedia tools.\n\n"
                "TOOL SELECTION RULES:\n"
                "  - Use web_search for: current events, recent news, live scores, "
                "prices, anything that changes over time.\n"
                "  - Use wikipedia for: biographies, history, science concepts, "
                "geography, definitions, encyclopedic knowledge.\n"
                "  - Answer directly (no tool) for: math, logic, or general knowledge "
                "you are confident about.\n\n"
                "RESPONSE RULES:\n"
                "  - Always write a complete, helpful answer.\n"
                "  - Resolve pronouns (he/she/they/it) using the conversation history.\n"
                "  - Summarise tool results in your own words.\n"
                "  - Keep answers clear and concise.\n"
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),   # conversation memory
        ("human", "{input}"),                                # current user message
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # tool call working memory
    ])

    # ── Agent + Executor ───────────────────────────────────────────────────────
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,   # safe now — output_messages_key removed
    )

    # ── Wrap with memory ───────────────────────────────────────────────────────
    # output_messages_key is intentionally NOT set here.
    # When omitted, RunnableWithMessageHistory correctly:
    #   • reads result["input"]  → saves as HumanMessage to history
    #   • reads result["output"] → saves as AIMessage to history
    # No type mismatch, no silent data loss.

    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        # output_messages_key intentionally omitted ← KEY FIX
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """Initialise all session_state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(agent_ready: bool) -> None:
    """Render left sidebar: status, stats, memory inspector, controls."""
    with st.sidebar:
        st.markdown("## 🤖 KnowledgeBot")
        st.markdown("*Conversational AI with Memory & Tools*")
        st.divider()

        # Status
        st.markdown("### ⚡ System Status")
        if agent_ready:
            st.markdown(
                '<span class="status-badge badge-green">● LLM Online</span>'
                '<span class="status-badge badge-blue">● Memory Active</span><br>'
                '<span class="status-badge badge-purple">● Web Search</span>'
                '<span class="status-badge badge-purple">● Wikipedia</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge badge-yellow">⚠ Agent Offline</span>',
                unsafe_allow_html=True,
            )
        st.divider()

        # Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.get("messages", [])))
        with col2:
            st.metric("Queries", st.session_state.get("total_queries", 0))
        st.caption(f"Session: `{st.session_state.get('session_id','')[:20]}…`")
        st.divider()

        # Memory Inspector
        st.markdown("### 🧠 Memory Inspector")
        st.caption("What KnowledgeBot remembers")
        sid = st.session_state.get("session_id", "default")
        lang_msgs = get_session_history(sid).messages

        if not lang_msgs:
            st.info("Memory is empty. Start chatting!", icon="💭")
        else:
            with st.expander(f"View {len(lang_msgs)} stored messages", expanded=False):
                for i, msg in enumerate(lang_msgs, 1):
                    role = msg.__class__.__name__.replace("Message", "")
                    preview = msg.content[:90] + ("…" if len(msg.content) > 90 else "")
                    color = "mem-human" if role == "Human" else "mem-ai"
                    icon  = "👤" if role == "Human" else "🤖"
                    st.markdown(
                        f'<div class="mem-row">'
                        f'<span class="{color}">{icon} [{i}] {role}</span><br>'
                        f'<span style="color:#64748b">{preview}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        st.divider()

        # Controls
        st.markdown("### 🎛️ Controls")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            _session_store[st.session_state.session_id] = ChatMessageHistory()
            st.success("Conversation cleared!")
            st.rerun()

        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.session_id = f"session_{int(time.time())}"
            st.success("New session started!")
            st.rerun()

        st.divider()

        # Suggestions
        st.markdown("### 💡 Try Asking")
        for s in [
            "Who is the CEO of OpenAI?",
            "What is quantum entanglement?",
            "Latest AI news this week",
            "Tell me about the James Webb Telescope",
            "Who won the last FIFA World Cup?",
        ]:
            st.caption(f"• {s}")

        st.divider()
        st.markdown(
            '<div style="color:#475569;font-size:0.75rem;text-align:center">'
            "KnowledgeBot v1.0<br>LangChain · Groq · Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown("""
        <div class="kb-header">
            <div style="font-size:2.5rem">🤖</div>
            <div>
                <h1>KnowledgeBot</h1>
                <p>Conversational AI · Remembers context · Searches the web</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def render_welcome() -> None:
    st.markdown("""
        <div class="welcome-card">
            <h3>👋 Welcome to KnowledgeBot!</h3>
            <p>I answer questions, search the web in real time,<br>
               and remember our entire conversation.</p>
            <br>
            <p style="font-size:0.82rem;color:#475569">Try one of these:</p>
            <div class="suggestion">🔍 Who is the current CEO of OpenAI?</div>
            <div class="suggestion">📚 What is quantum entanglement?</div>
            <div class="suggestion">📰 What are the latest AI news this week?</div>
            <div class="suggestion">🌍 Tell me about the history of the internet</div>
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CHAT HISTORY RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_history() -> None:
    """Re-render all past messages from session_state."""
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                tools_str = " · ".join(f"🔍 {t}" for t in msg["tools_used"])
                st.markdown(
                    f'<div class="tool-tag">Sources used: {tools_str}</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PROCESS USER INPUT
# ══════════════════════════════════════════════════════════════════════════════

def process_user_input(agent: RunnableWithMessageHistory, user_input: str) -> None:
    """
    Handle one full user→agent→response cycle.

    Why this now works correctly:
        result["output"] is a clean string because output_messages_key
        was removed from RunnableWithMessageHistory. Previously that parameter
        caused LangChain to intercept and mishandle the string output, leaving
        result["output"] empty. Now it passes through untouched.

        Tool names come from result["intermediate_steps"] — a list of
        (AgentAction, tool_output) tuples. AgentAction.tool holds the name.
        This is more reliable than parsing the LLM's text output.
    """

    # Show user bubble
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "tools_used": []}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Invoke agent + render response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = agent.invoke(
                    {"input": user_input},
                    config={"configurable": {
                        "session_id": st.session_state.session_id
                    }},
                )

                # Clean string answer — now reliable after the fix
                answer: str = result.get("output", "").strip()

                # Safety net for debugging
                if not answer:
                    answer = (
                        "⚠️ Empty response received.\n\n"
                        f"**Debug — full result dict:**\n```python\n{result}\n```"
                    )

                # Extract tool names from intermediate steps
                tools_used: list = []
                for action, _ in result.get("intermediate_steps", []):
                    tool_name = getattr(action, "tool", None)
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)

            except Exception as e:
                answer = f"⚠️ Error:\n```\n{str(e)}\n```"
                tools_used = []

        # Render answer
        st.markdown(answer)

        # Render tool badge
        if tools_used:
            tools_str = " · ".join(f"🔍 {t}" for t in tools_used)
            st.markdown(
                f'<div class="tool-tag">Sources used: {tools_str}</div>',
                unsafe_allow_html=True,
            )

    # Persist to session_state
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools_used": tools_used}
    )
    st.session_state.total_queries += 1


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Streamlit entry point."""
    init_session_state()

    if not load_environment():
        st.error(
            "**GROQ_API_KEY not found.**\n\n"
            "Add it to your `.env` file:\n```\nGROQ_API_KEY=your_key_here\n```",
            icon="🔑",
        )
        render_sidebar(agent_ready=False)
        st.stop()

    agent = build_agent()
    render_sidebar(agent_ready=True)
    render_header()

    if not st.session_state.messages:
        render_welcome()
    else:
        render_chat_history()

    if user_input := st.chat_input("Ask me anything…"):
        process_user_input(agent, user_input)
        st.rerun()


if __name__ == "__main__":
    main()
