"""
chatbot_basic.py
----------------
Milestone 2: Basic Stateless CLI Chatbot

Concepts covered:
    - ChatPromptTemplate  : structures the prompt sent to the LLM
    - LCEL pipe (|)       : chains prompt → LLM → parser into one pipeline
    - StrOutputParser     : converts AIMessage → plain string
    - Stateless loop      : each message is independent (no memory)

Run:
    python src/chatbot_basic.py
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import os
import sys

# ── Third-party ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate   # Structures prompt layout
from langchain_core.output_parsers import StrOutputParser  # Extracts text from AIMessage

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# ── Initialise rich console ────────────────────────────────────────────────────
console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_environment() -> None:
    """Load .env and validate that the required API key exists."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]❌  GROQ_API_KEY missing — check your .env file.[/bold red]")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD THE CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_chain():
    """
    Construct and return a stateless LangChain LCEL chain.

    The chain has three components joined by the pipe (|) operator:

        prompt_template  →  llm  →  output_parser

    ┌─────────────────────────────────────────────────────────┐
    │  ChatPromptTemplate                                      │
    │   • Holds a system message (bot persona)                │
    │   • Holds a human message placeholder ({user_input})    │
    │   • .format_messages() fills placeholders at runtime    │
    └─────────────────────┬───────────────────────────────────┘
                          │  list of BaseMessage objects
                          ▼
    ┌─────────────────────────────────────────────────────────┐
    │  ChatGroq (LLM)                                          │
    │   • Receives the formatted messages                     │
    │   • Returns an AIMessage object                         │
    └─────────────────────┬───────────────────────────────────┘
                          │  AIMessage(content="...")
                          ▼
    ┌─────────────────────────────────────────────────────────┐
    │  StrOutputParser                                         │
    │   • Pulls .content out of AIMessage                     │
    │   • Returns a plain Python str                          │
    └─────────────────────────────────────────────────────────┘

    Returns:
        Runnable: an LCEL chain ready to call with .invoke()
    """

    # --- 2a. Prompt Template ---------------------------------------------------
    # ChatPromptTemplate.from_messages() accepts a list of (role, content) tuples.
    #
    # "system"  → sets the AI's behaviour for the ENTIRE conversation
    # "human"   → the actual user message; {user_input} is a runtime variable
    #
    # Why a template instead of a raw string?
    #   ✔ Reusable across every turn
    #   ✔ Easy to tweak the persona in ONE place
    #   ✔ LangChain can validate missing variables before sending to the LLM

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are KnowledgeBot, a helpful and concise AI assistant. "
                    "Answer questions clearly and accurately. "
                    "If you are unsure about something, say so honestly. "
                    "Keep responses focused and avoid unnecessary padding."
                ),
            ),
            (
                "human",
                "{user_input}",   # ← placeholder filled at runtime
            ),
        ]
    )

    # --- 2b. LLM ---------------------------------------------------------------
    # Same Groq model as Milestone 1.
    # temperature=0.7 keeps answers natural but not too random.

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=512,
    )

    # --- 2c. Output Parser -----------------------------------------------------
    # Without this, .invoke() returns an AIMessage object.
    # With StrOutputParser, we get a clean Python string — easier to print.

    output_parser = StrOutputParser()

    # --- 2d. Assemble the Chain (LCEL) -----------------------------------------
    # The | operator is syntactic sugar for:
    #   RunnableSequence(prompt_template, llm, output_parser)
    #
    # When we call chain.invoke({"user_input": "..."}) later:
    #   1. prompt_template.invoke({"user_input": "..."}) → list of messages
    #   2. llm.invoke([messages])                        → AIMessage
    #   3. output_parser.invoke(AIMessage)               → str

    chain = prompt_template | llm | output_parser

    return chain


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_chat_loop(chain) -> None:
    """
    Run an interactive CLI chat loop.

    Each iteration:
        1. Read user input from stdin
        2. Pass it to chain.invoke()  (stateless — no history)
        3. Print the bot's response
        4. Repeat until user types 'exit' or 'quit'

    Args:
        chain: The LCEL chain returned by build_chain()
    """

    # Welcome banner
    console.print(
        Panel(
            "[bold cyan]KnowledgeBot v0.1[/bold cyan]  —  Basic Chatbot\n"
            "[dim]Type your question and press Enter.\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.[/dim]",
            border_style="cyan",
        )
    )
    console.print(
        "[bold yellow]⚠  Note:[/bold yellow] This bot has [bold red]NO MEMORY[/bold red]. "
        "Each question is answered independently.\n"
    )

    while True:
        # ── Get user input ─────────────────────────────────────────────────────
        try:
            # rich doesn't have a styled input(), so we print the prompt then read
            console.print("[bold green]You ▶[/bold green] ", end="")
            user_input = input().strip()   # .strip() removes accidental whitespace
        except (EOFError, KeyboardInterrupt):
            # Handles Ctrl+C or piped input ending
            console.print("\n[dim]Session ended.[/dim]")
            break

        # ── Handle empty input ─────────────────────────────────────────────────
        if not user_input:
            console.print("[dim]  (empty input — please type something)[/dim]")
            continue

        # ── Handle exit commands ───────────────────────────────────────────────
        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            console.print(
                Panel("[bold cyan]👋  Goodbye! See you next time.[/bold cyan]",
                      border_style="cyan")
            )
            break

        # ── Invoke the chain ───────────────────────────────────────────────────
        # chain.invoke() expects a dict whose keys match the template variables.
        # Here we have one variable: {user_input}
        #
        # STATELESS PROOF: We pass ONLY the current message.
        # There is no conversation history passed in — the LLM sees just this one
        # human turn plus the fixed system message.  It has zero knowledge of
        # what was said before.

        try:
            console.print("[dim]  thinking...[/dim]", end="\r")  # inline spinner text
            response: str = chain.invoke({"user_input": user_input})
        except Exception as e:
            console.print(f"[bold red]❌  Error:[/bold red] {e}")
            continue

        # ── Print the response ─────────────────────────────────────────────────
        console.print(Rule(style="dim"))   # visual separator
        console.print(
            Panel(
                Text(response, style="white"),
                title="[bold magenta]🤖 KnowledgeBot[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print()   # blank line for readability


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orchestrate Milestone 2:
        1. Load environment
        2. Build the LCEL chain
        3. Start the chat loop
    """
    load_environment()
    chain = build_chain()
    run_chat_loop(chain)


if __name__ == "__main__":
    main()