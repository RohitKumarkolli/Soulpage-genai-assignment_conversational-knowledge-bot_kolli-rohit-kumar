"""
test_llm.py
-----------
Milestone 1: Basic LLM Connection Test

Purpose:
    Verify that our LangChain + Groq setup works correctly
    before building anything on top of it.

Run:
    python src/test_llm.py
"""

# --- Standard Library ---
import os                          # To access environment variables
import sys                         # To exit gracefully on errors

# --- Third-party ---
from dotenv import load_dotenv     # Loads KEY=VALUE pairs from .env into os.environ

from langchain_groq import ChatGroq            # Groq-backed LangChain chat model
from langchain_core.messages import HumanMessage, SystemMessage  # Message types
from rich.console import Console               # Colored terminal output
from rich.panel import Panel                   # Boxed display in terminal

# ── Initialise rich console for pretty output ──────────────────────────────────
console = Console()


def load_environment() -> None:
    """
    Load environment variables from the .env file.

    load_dotenv() searches for a .env file starting from the current directory
    and walks up the directory tree.  It silently does nothing if no file is
    found, so we add an explicit check to catch misconfiguration early.
    """
    load_dotenv()  # Reads .env and injects variables into os.environ

    if not os.getenv("GROQ_API_KEY"):
        console.print(
            "[bold red]❌ GROQ_API_KEY not found![/bold red]\n"
            "Create a [cyan].env[/cyan] file and add:\n"
            "  [green]GROQ_API_KEY=your_key_here[/green]\n\n"
            "Get a free key at: [link]https://console.groq.com[/link]"
        )
        sys.exit(1)  # Exit with a non-zero code to signal failure

    console.print("[bold green]✅ Environment loaded successfully[/bold green]")


def build_llm() -> ChatGroq:
    """
    Instantiate and return a ChatGroq LLM object.

    ChatGroq wraps Groq's REST API and conforms to LangChain's
    BaseChatModel interface — meaning we can swap it for any other
    LangChain-compatible LLM (OpenAI, Anthropic, etc.) later with
    zero changes to the rest of our code.

    Model choice — llama-3.1-8b-instant:
        • Free tier on Groq
        • 8-billion parameter Llama 3.1 — strong instruction following
        • Very fast inference (Groq's speciality)

    Returns:
        ChatGroq: configured LLM instance
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # Model identifier on Groq
        temperature=0.7,                 # 0 = deterministic, 1 = creative
        max_tokens=512,                  # Cap response length
        # api_key is auto-read from GROQ_API_KEY env var
    )
    console.print("[bold green]✅ LLM initialised:[/bold green] llama-3.1-8b-instant via Groq")
    return llm


def run_smoke_test(llm: ChatGroq) -> None:
    """
    Send a minimal two-message conversation to the LLM and print the response.

    LangChain chat models accept a list of Message objects:
        SystemMessage  — sets the AI's persona/behaviour
        HumanMessage   — the user's input
        AIMessage      — the model's previous replies (used in multi-turn)

    Under the hood, llm.invoke() serialises these to the provider's API
    format and deserialises the response back into an AIMessage object.
    """
    console.print("\n[bold cyan]📡 Sending test prompt to LLM...[/bold cyan]")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant called KnowledgeBot. "
                "Be concise and clear in your responses."
            )
        ),
        HumanMessage(
            content="Hello! In one sentence, what can you help me with?"
        ),
    ]

    # .invoke() is synchronous — it blocks until the full response is received.
    # We'll switch to streaming in later milestones.
    response = llm.invoke(messages)

    # response is an AIMessage; its text content lives in response.content
    console.print(
        Panel(
            f"[bold white]{response.content}[/bold white]",
            title="[bold green]🤖 KnowledgeBot Response[/bold green]",
            border_style="green",
        )
    )

    # Metadata Groq returns (tokens used, latency, model)
    console.print(f"\n[dim]Model    : {response.response_metadata.get('model_name')}[/dim]")
    console.print(f"[dim]Tokens   : {response.response_metadata.get('token_usage')}[/dim]")


def main() -> None:
    """
    Entry point — orchestrates setup and smoke test.

    Keeping main() thin (just calling other functions) is a
    best practice: it makes each step independently testable.
    """
    console.print(
        Panel(
            "[bold yellow]Milestone 1 — LLM Smoke Test[/bold yellow]",
            subtitle="Conversational Knowledge Bot",
            border_style="yellow",
        )
    )

    load_environment()   # Step 1: load .env
    llm = build_llm()    # Step 2: create LLM
    run_smoke_test(llm)  # Step 3: ping the model

    console.print("\n[bold green]🎉 Milestone 1 Complete! LangChain + Groq is working.[/bold green]")


# This guard ensures main() only runs when the file is executed directly,
# NOT when it is imported as a module (important for testing later).
if __name__ == "__main__":
    main()