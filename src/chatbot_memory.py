"""
chatbot_memory.py
-----------------
Milestone 3: Chatbot with Conversational Memory

Concepts covered:
    - ChatMessageHistory          : in-memory store of chat messages
    - RunnableWithMessageHistory  : LCEL wrapper that auto-injects history
    - MessagesPlaceholder         : slot in the prompt that receives history
    - Session IDs                 : isolate memory per user/session

How memory works (the full picture):
    1. User sends message N
    2. RunnableWithMessageHistory fetches messages 1..N-1 from history store
    3. It injects them into the prompt via MessagesPlaceholder
    4. The full prompt (history + new message) is sent to the LLM
    5. The LLM response is appended back to the history store
    6. Repeat for message N+1

Run:
    python src/chatbot_memory.py
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import os
import sys
from typing import Dict                        # For type-hinting the session store

# ── Third-party ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# MessagesPlaceholder ↑ is a special prompt slot that accepts a LIST of messages
# (the full conversation history) rather than a single string variable.

from langchain_core.output_parsers import StrOutputParser

from langchain_core.chat_history import BaseChatMessageHistory
# BaseChatMessageHistory ↑ is the abstract base class for all history stores.
# We use the in-memory implementation below, but the interface is identical
# for Redis, DynamoDB, SQL, etc. — making it easy to swap later.

from langchain_community.chat_message_histories import ChatMessageHistory
# ChatMessageHistory ↑ is the simplest concrete implementation:
# it stores messages in a plain Python list in RAM.

from langchain_core.runnables.history import RunnableWithMessageHistory
# RunnableWithMessageHistory ↑ wraps any LCEL chain and automatically:
#   • fetches history before each invoke
#   • appends new human + AI messages after each invoke

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.table import Table

# ── Initialise rich console ────────────────────────────────────────────────────
console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_environment() -> None:
    """Load .env and validate required API keys."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]❌  GROQ_API_KEY missing — check your .env file.[/bold red]")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SESSION STORE
# ══════════════════════════════════════════════════════════════════════════════

# This dict acts as an in-memory database of chat sessions.
# Key   → session_id (a string like "user_123" or "default")
# Value → ChatMessageHistory object (the list of messages for that session)
#
# In production you would replace this dict with a Redis or DynamoDB backend,
# but the interface RunnableWithMessageHistory uses is identical either way.

session_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or create the ChatMessageHistory for a given session ID.

    RunnableWithMessageHistory calls this function automatically before
    every .invoke() to fetch the right history store.

    Args:
        session_id: A unique string identifying the conversation session.
                    Allows multiple isolated conversations in the same process.

    Returns:
        BaseChatMessageHistory: The history object for this session.
    """
    if session_id not in session_store:
        # First message in this session — create a fresh empty history
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD THE MEMORY-AWARE CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_chain() -> RunnableWithMessageHistory:
    """
    Build and return a memory-aware LCEL chain.

    Chain anatomy:
    ┌──────────────────────────────────────────────────────────────────┐
    │  ChatPromptTemplate                                               │
    │   • SystemMessage    : fixed persona                             │
    │   • MessagesPlaceholder("history") : injected by memory wrapper  │
    │   • HumanMessage     : current user turn  {user_input}           │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                           ChatGroq (LLM)
                                 │
                                 ▼
                         StrOutputParser
                                 │
                                 ▼
                    RunnableWithMessageHistory
                    (wraps the chain above,
                     auto-manages history r/w)

    Returns:
        RunnableWithMessageHistory: ready to invoke with session config
    """

    # --- 3a. Prompt Template with MessagesPlaceholder -------------------------
    #
    # MessagesPlaceholder("history") is the KEY difference from Milestone 2.
    # When the chain runs, RunnableWithMessageHistory:
    #   1. Calls get_session_history(session_id) to fetch past messages
    #   2. Passes them in as the "history" variable
    #   3. MessagesPlaceholder expands them inline in the prompt
    #
    # Result sent to LLM on turn 3 of a conversation:
    #   SystemMessage:    "You are KnowledgeBot..."
    #   HumanMessage:     "Who is Elon Musk?"          ← from history
    #   AIMessage:        "Elon Musk is a CEO..."       ← from history
    #   HumanMessage:     "Where was he born?"          ← from history
    #   AIMessage:        "He was born in Pretoria..."  ← from history
    #   HumanMessage:     "What is he famous for?"      ← current input

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are KnowledgeBot, a helpful and concise AI assistant. "
                    "You remember everything said in this conversation. "
                    "Use the conversation history to give contextual, accurate answers. "
                    "If a question uses pronouns like 'he', 'she', 'they', 'it', "
                    "resolve them from the conversation history."
                ),
            ),
            MessagesPlaceholder(variable_name="history"),  # ← history injected HERE
            (
                "human",
                "{user_input}",                            # ← current user message
            ),
        ]
    )

    # --- 3b. LLM ---------------------------------------------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=512,
    )

    # --- 3c. Base chain (same LCEL pattern as Milestone 2) --------------------
    base_chain = prompt | llm | StrOutputParser()

    # --- 3d. Wrap with RunnableWithMessageHistory ------------------------------
    #
    # This wrapper intercepts every .invoke() call:
    #   BEFORE:  loads history → injects into "history" variable
    #   AFTER:   appends HumanMessage + AIMessage to the history store
    #
    # input_messages_key  → which key in the invoke dict is the user message
    # history_messages_key → which prompt variable receives the history list

    memory_chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,              # our session factory function
        input_messages_key="user_input",  # matches {user_input} in prompt
        history_messages_key="history",   # matches MessagesPlaceholder name
    )

    return memory_chain


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DEBUG HELPER — show what is stored in memory
# ══════════════════════════════════════════════════════════════════════════════

def print_memory_state(session_id: str) -> None:
    """
    Pretty-print the current contents of the session's message history.

    This is purely for educational purposes — it lets you SEE exactly
    what is being sent to the LLM as 'history' on each turn.

    Args:
        session_id: The session whose history to display.
    """
    history = get_session_history(session_id)
    messages = history.messages   # plain list of BaseMessage objects

    if not messages:
        console.print("[dim]  (memory is empty)[/dim]")
        return

    table = Table(
        title=f"🧠 Memory State — session: '{session_id}'",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", style="bold", width=12)
    table.add_column("Content (truncated to 80 chars)")

    for i, msg in enumerate(messages, start=1):
        role = msg.__class__.__name__.replace("Message", "")  # "Human", "AI"
        content_preview = msg.content[:80] + ("…" if len(msg.content) > 80 else "")
        role_color = "green" if role == "Human" else "magenta"
        table.add_row(str(i), f"[{role_color}]{role}[/{role_color}]", content_preview)

    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_chat_loop(memory_chain: RunnableWithMessageHistory) -> None:
    """
    Interactive CLI loop that invokes the memory-aware chain each turn.

    Key difference from Milestone 2:
        chain.invoke() now takes a config dict with a session_id.
        RunnableWithMessageHistory uses this to look up / update the
        correct history store automatically.

    Args:
        memory_chain: The RunnableWithMessageHistory chain.
    """

    # Use a fixed session ID for this CLI session.
    # In a multi-user app you'd generate a unique ID per user/tab.
    SESSION_ID = "cli_session_1"

    # Welcome banner
    console.print(
        Panel(
            "[bold cyan]KnowledgeBot v0.2[/bold cyan]  —  Memory-Enabled Chatbot\n"
            "[dim]I now remember everything you say in this session.\n"
            "Type [bold]memory[/bold] to inspect what I remember.\n"
            "Type [bold]clear[/bold] to wipe my memory.\n"
            "Type [bold]exit[/bold] to quit.[/dim]",
            border_style="cyan",
        )
    )
    console.print(
        "[bold green]✅  Memory is ON.[/bold green] "
        "Try: ask about a person, then ask a follow-up using 'he' or 'she'.\n"
    )

    while True:
        # ── Get user input ─────────────────────────────────────────────────────
        try:
            console.print("[bold green]You ▶[/bold green] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        # ── Special commands ───────────────────────────────────────────────────
        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            console.print(
                Panel("[bold cyan]👋  Goodbye! Memory cleared on exit.[/bold cyan]",
                      border_style="cyan")
            )
            break

        if user_input.lower() == "memory":
            # Show exactly what is stored — great for learning
            print_memory_state(SESSION_ID)
            continue

        if user_input.lower() == "clear":
            # Wipe session history
            session_store[SESSION_ID] = ChatMessageHistory()
            console.print("[bold yellow]🧹  Memory cleared.[/bold yellow]")
            continue

        # ── Invoke the memory-aware chain ──────────────────────────────────────
        #
        # The config dict is how we pass the session_id to
        # RunnableWithMessageHistory.  It calls get_session_history(session_id)
        # before the chain runs, and updates it after.
        #
        # Everything else (history injection, history saving) is automatic.

        try:
            console.print("[dim]  thinking...[/dim]", end="\r")
            response: str = memory_chain.invoke(
                {"user_input": user_input},           # chain input
                config={"configurable": {"session_id": SESSION_ID}},  # memory config
            )
        except Exception as e:
            console.print(f"[bold red]❌  Error:[/bold red] {e}")
            continue

        # ── Print response ─────────────────────────────────────────────────────
        console.print(Rule(style="dim"))
        console.print(
            Panel(
                Text(response, style="white"),
                title="[bold magenta]🤖 KnowledgeBot[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Orchestrate Milestone 3."""
    load_environment()
    memory_chain = build_memory_chain()
    run_chat_loop(memory_chain)


if __name__ == "__main__":
    main()