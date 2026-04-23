"""
chatbot_conversational.py
--------------------------
Milestone 5: Conversational Agent — Memory + Tools Combined

Concepts covered:
    - Wrapping AgentExecutor with RunnableWithMessageHistory
    - chat_history placeholder   : injects past conversation into agent prompt
    - agent_scratchpad           : agent's per-query working memory (tool calls)
    - session-based memory store : isolates conversations per user
    - Full ReAct loop with context awareness

Architecture:
    User
     │
     ▼
    RunnableWithMessageHistory      ← auto-loads & saves conversation history
     │
     ▼
    AgentExecutor                   ← Reason → Act → Observe loop
     │
     ├── ChatPromptTemplate
     │     • system message
     │     • MessagesPlaceholder("chat_history")     ← conversation memory
     │     • HumanMessage {input}
     │     • MessagesPlaceholder("agent_scratchpad") ← tool call working memory
     │
     ├── ChatGroq (llama-3.1-8b-instant)
     │
     └── Tools: [web_search, wikipedia]

Run:
    python src/chatbot_conversational.py
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import os
import sys
from typing import Dict

# ── Third-party ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain.agents import create_tool_calling_agent, AgentExecutor

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.table import Table
from rich import box

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
    console.print("[bold green]✅  Environment loaded[/bold green]")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SESSION MEMORY STORE
# ══════════════════════════════════════════════════════════════════════════════

# In-memory dict: session_id → ChatMessageHistory
# This is identical to Milestone 3 — same pattern, same interface.
# In production: replace with Redis / DynamoDB / PostgreSQL backend.

session_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or create a ChatMessageHistory for the given session.

    Called automatically by RunnableWithMessageHistory before and
    after every agent invocation.

    Args:
        session_id: Unique identifier for this conversation session.

    Returns:
        BaseChatMessageHistory: The message history for this session.
    """
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def build_tools() -> list:
    """
    Build and return [DuckDuckGoSearchRun, WikipediaQueryRun].

    Same tools as Milestone 4. Descriptions are carefully written
    because the LLM reads them to decide WHEN to use each tool.

    Returns:
        list: configured Tool objects
    """

    # Tool 1: DuckDuckGo — live web search, no API key
    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the internet for current, recent, or real-time information. "
            "Use for: news, recent events, current prices, sports results, "
            "latest product releases, or anything that changes over time. "
            "Input: a concise search query string."
        ),
    )

    # Tool 2: Wikipedia — encyclopedic knowledge, free API
    wiki_tool = WikipediaQueryRun(
        name="wikipedia",
        description=(
            "Look up factual, encyclopedic, or background information. "
            "Use for: biographies, history, science, geography, definitions. "
            "Input: a topic name or descriptive question."
        ),
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=1000,  # slightly more context for follow-ups
        ),
    )

    console.print("[bold green]✅  Tools loaded:[/bold green] web_search, wikipedia")
    return [search_tool, wiki_tool]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BUILD THE CONVERSATIONAL AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_conversational_agent(tools: list) -> RunnableWithMessageHistory:
    """
    Build a memory-aware, tool-enabled conversational agent.

    This is the core of Milestone 5. Two placeholder types work together:

    ┌─────────────────────────────────────────────────────────────────┐
    │  MessagesPlaceholder("chat_history")                             │
    │  ─────────────────────────────────                              │
    │  • Populated by RunnableWithMessageHistory                      │
    │  • Contains ALL previous Human + AI messages in the session     │
    │  • Gives agent context for pronoun resolution & follow-ups      │
    │  • Persists ACROSS queries                                       │
    └─────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────┐
    │  MessagesPlaceholder("agent_scratchpad")                         │
    │  ──────────────────────────────────────                         │
    │  • Populated by AgentExecutor during tool-calling               │
    │  • Contains tool calls + results for the CURRENT query only     │
    │  • Gives agent visibility into what it has already searched     │
    │  • Cleared after each final answer                              │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        tools: List of Tool objects from build_tools()

    Returns:
        RunnableWithMessageHistory: The fully wired conversational agent.
    """

    # --- 4a. LLM --------------------------------------------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,      # deterministic = better tool selection
        max_tokens=1024,
    )

    # --- 4b. Prompt Template --------------------------------------------------
    #
    # Prompt slot order matters:
    #   1. system           → sets persona & rules (always first)
    #   2. chat_history     → past conversation (so agent has full context)
    #   3. human {input}    → current user message
    #   4. agent_scratchpad → tool calls for THIS query (must be last)
    #
    # Why chat_history BEFORE the current input?
    #   The LLM reads left-to-right. Seeing history first lets it resolve
    #   "he/she/it" in the current message before deciding on a tool.

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are KnowledgeBot, an intelligent conversational AI "
                    "assistant with access to web search and Wikipedia.\n\n"

                    "CAPABILITIES:\n"
                    "  • You remember the entire conversation history\n"
                    "  • You can search the web for current information\n"
                    "  • You can look up encyclopedic facts on Wikipedia\n\n"

                    "TOOL USAGE RULES:\n"
                    "  1. Current events / recent data / live info  → web_search\n"
                    "  2. Biographies / history / science / facts   → wikipedia\n"
                    "  3. Follow-up questions about a known topic   → use history first,\n"
                    "     only search if history lacks the detail\n"
                    "  4. Math / logic / general knowledge you have → no tool needed\n\n"

                    "CONVERSATION RULES:\n"
                    "  • Always resolve pronouns (he/she/they/it) using chat history\n"
                    "  • Never ask 'who do you mean?' if history makes it clear\n"
                    "  • Be concise — summarise tool results, don't paste them raw\n"
                    "  • Always mention which source you used (web or Wikipedia)\n"
                ),
            ),
            # ── Conversation memory injected here ──────────────────────────────
            MessagesPlaceholder(variable_name="chat_history"),

            # ── Current user message ───────────────────────────────────────────
            ("human", "{input}"),

            # ── Agent's tool-call working memory (must be last) ────────────────
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # --- 4c. Create the base agent --------------------------------------------
    agent = create_tool_calling_agent(llm, tools, prompt)

    # --- 4d. Wrap in AgentExecutor --------------------------------------------
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,             # show reasoning steps
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=False,  # keep final output clean
    )

    # --- 4e. Wrap AgentExecutor with RunnableWithMessageHistory ---------------
    #
    # This is the integration point between memory and the agent.
    #
    # How it works:
    #   BEFORE invoke:
    #     • Calls get_session_history(session_id)
    #     • Injects history messages into "chat_history" placeholder
    #   AFTER invoke:
    #     • Reads the "input" (human message) from invoke dict
    #     • Reads the "output" (AI final answer) from agent result
    #     • Appends both as HumanMessage + AIMessage to the history store
    #
    # input_messages_key  → key in invoke() dict that holds the user message
    # output_messages_key → key in agent result dict that holds the AI response
    # history_messages_key → which prompt placeholder receives the history

    conversational_agent = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",          # matches {input} in prompt
        history_messages_key="chat_history", # matches MessagesPlaceholder name
        output_messages_key="output",        # AgentExecutor returns {"output": "..."}
    )

    console.print("[bold green]✅  Conversational agent ready[/bold green]")
    return conversational_agent


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MEMORY INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

def print_memory_state(session_id: str) -> None:
    """
    Display the current conversation history in a formatted table.

    Useful for verifying that memory is being saved correctly after
    each agent invocation.

    Args:
        session_id: The session whose history to display.
    """
    history = get_session_history(session_id)
    messages = history.messages

    if not messages:
        console.print("[dim]  Memory is empty.[/dim]")
        return

    table = Table(
        title=f"🧠 Conversation Memory  —  session: '{session_id}'",
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Role", width=12)
    table.add_column("Content Preview (first 100 chars)")

    for i, msg in enumerate(messages, start=1):
        role = msg.__class__.__name__.replace("Message", "")
        preview = msg.content[:100] + ("…" if len(msg.content) > 100 else "")
        color = "green" if role == "Human" else "magenta"
        table.add_row(str(i), f"[{color}]{role}[/{color}]", preview)

    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DEMO SCENARIO (automated test of memory + tools)
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(agent: RunnableWithMessageHistory, session_id: str) -> None:
    """
    Run a scripted two-turn demo that proves memory + tools work together.

    Turn 1: Ask about a person → agent uses web_search
    Turn 2: Ask follow-up with pronoun → agent resolves from memory, 
            uses wikipedia for detail

    Args:
        agent: The conversational agent.
        session_id: Session identifier for memory isolation.
    """
    demo_queries = [
        "Who is the CEO of OpenAI?",
        "Where did he study and what did he study?",
    ]

    console.print(
        Panel(
            "[bold yellow]🎬  Running Demo Scenario[/bold yellow]\n"
            "[dim]This shows memory + tools working together.\n"
            "Turn 1: searches web for CEO info\n"
            "Turn 2: resolves 'he' from memory, searches for education[/dim]",
            border_style="yellow",
        )
    )

    config = {"configurable": {"session_id": session_id}}

    for i, query in enumerate(demo_queries, start=1):
        console.print(f"\n[bold green]Demo Turn {i}[/bold green]")
        console.print(f"[bold green]You ▶[/bold green]  {query}")
        console.print(Rule("[dim]Agent Reasoning[/dim]", style="dim"))

        try:
            result = agent.invoke({"input": query}, config=config)
            answer = result["output"]
        except Exception as e:
            console.print(f"[bold red]❌  Error:[/bold red] {e}")
            continue

        console.print(Rule("[dim]Answer[/dim]", style="dim"))
        console.print(
            Panel(
                Text(answer, style="white"),
                title="[bold magenta]🤖 KnowledgeBot[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    console.print("\n[bold cyan]📋 Memory after demo:[/bold cyan]")
    print_memory_state(session_id)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  INTERACTIVE CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_chat_loop(agent: RunnableWithMessageHistory) -> None:
    """
    Interactive CLI loop for the full conversational agent.

    Commands available during chat:
        memory → inspect the current conversation history
        clear  → wipe memory for this session
        demo   → run the automated 2-turn demo
        exit   → quit

    Args:
        agent: The RunnableWithMessageHistory conversational agent.
    """
    SESSION_ID = "cli_session_1"
    config = {"configurable": {"session_id": SESSION_ID}}

    console.print(
        Panel(
            "[bold cyan]KnowledgeBot v1.0[/bold cyan]  —  Conversational Agent\n"
            "[dim]I remember our conversation AND can search the web.\n\n"
            "Commands:\n"
            "  [bold]memory[/bold]  → show what I remember\n"
            "  [bold]clear[/bold]   → wipe memory\n"
            "  [bold]demo[/bold]    → run automated demo scenario\n"
            "  [bold]exit[/bold]    → quit[/dim]",
            border_style="cyan",
        )
    )

    console.print(
        "[bold green]✅  Memory ON  |  "
        "[bold blue]🔍 Tools ON[/bold blue]  (web_search + wikipedia)[/bold green]\n"
    )

    while True:
        # ── User input ─────────────────────────────────────────────────────────
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
                Panel("[bold cyan]👋  Goodbye![/bold cyan]", border_style="cyan")
            )
            break

        if user_input.lower() == "memory":
            print_memory_state(SESSION_ID)
            continue

        if user_input.lower() == "clear":
            session_store[SESSION_ID] = ChatMessageHistory()
            console.print("[bold yellow]🧹  Memory cleared.[/bold yellow]")
            continue

        if user_input.lower() == "demo":
            # Run demo in a fresh isolated session
            run_demo(agent, session_id="demo_session")
            continue

        # ── Invoke the agent ───────────────────────────────────────────────────
        #
        # What happens under the hood:
        #   1. RunnableWithMessageHistory fetches chat_history for SESSION_ID
        #   2. AgentExecutor runs the Reason → Act loop:
        #        a. LLM sees: system + chat_history + user input + empty scratchpad
        #        b. LLM decides: tool needed? → runs tool → appends to scratchpad
        #        c. LLM sees scratchpad result → decides: more tools or final answer?
        #        d. Returns final answer
        #   3. RunnableWithMessageHistory saves HumanMessage + AIMessage to store

        console.print(Rule("[dim]Agent Reasoning[/dim]", style="dim"))

        try:
            result = agent.invoke(
                {"input": user_input},
                config=config,
            )
            answer: str = result["output"]
        except Exception as e:
            console.print(f"[bold red]❌  Agent error:[/bold red] {e}")
            continue

        # ── Print response ─────────────────────────────────────────────────────
        console.print(Rule("[dim]Answer[/dim]", style="dim"))
        console.print(
            Panel(
                Text(answer, style="white"),
                title="[bold magenta]🤖 KnowledgeBot[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Orchestrate Milestone 5."""
    load_environment()
    tools = build_tools()
    agent = build_conversational_agent(tools)
    run_chat_loop(agent)


if __name__ == "__main__":
    main()