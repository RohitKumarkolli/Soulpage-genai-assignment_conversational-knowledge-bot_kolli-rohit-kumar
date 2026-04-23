"""
chatbot_tools.py
----------------
Milestone 4: Chatbot with Tools (DuckDuckGo + Wikipedia)

Concepts covered:
    - Tool                      : a named callable the LLM can invoke
    - create_tool_calling_agent : builds a ReAct-style agent
    - AgentExecutor             : the Reason → Act → Observe loop
    - DuckDuckGoSearchRun       : live web search (no API key needed)
    - WikipediaQueryRun         : Wikipedia article summaries
    - Tool selection logic      : how the LLM decides which tool to use

NOTE:
    This milestone is intentionally stateless (no memory) so you can
    clearly see how tools work in isolation.
    Milestone 5 combines memory + tools together.

Run:
    python src/chatbot_tools.py
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import os
import sys

# ── Third-party ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# MessagesPlaceholder ↑ is needed by the agent to hold its intermediate
# reasoning steps (tool calls + results) between iterations.

# --- Tool imports --------------------------------------------------------------
from langchain_community.tools import DuckDuckGoSearchRun
# DuckDuckGoSearchRun ↑ wraps DuckDuckGo's HTML search endpoint.
# Returns a plain-text snippet of the top results.
# Requires: pip install duckduckgo-search

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# WikipediaQueryRun ↑ calls the Wikipedia API and returns a summary.
# WikipediaAPIWrapper ↑ configures how many sentences to retrieve.

# --- Agent imports -------------------------------------------------------------
from langchain.agents import create_tool_calling_agent, AgentExecutor
# create_tool_calling_agent ↑ builds an agent that uses the LLM's native
#   function/tool-calling capability (supported by llama-3.1 on Groq).
#
# AgentExecutor ↑ is the runtime loop:
#   while not done:
#       call LLM → if tool needed: run tool, feed result back → else: return answer

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


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DEFINE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def build_tools() -> list:
    """
    Create and return the list of tools available to the agent.

    Each tool has three things the LLM uses to decide when to call it:
        name        → short identifier  (e.g. "duckduckgo_search")
        description → plain-English explanation of what the tool does
                      and WHEN to use it  ← the LLM reads this!
        func        → the Python callable that actually runs when chosen

    Tool selection logic (how the LLM decides):
        The agent prompt tells the LLM: "You have these tools: [tool list].
        For each tool, here is its name and description."
        The LLM reads those descriptions and picks the best tool for the query.

        Example reasoning (internal, not shown to user):
            Query: "Who won the Champions League final yesterday?"
            → Needs live data → use DuckDuckGo (current events)

            Query: "What is the history of the Roman Empire?"
            → Encyclopedic knowledge → use Wikipedia

            Query: "What is 15% of 340?"
            → Pure math, no tool needed → answer directly

    Returns:
        list: [DuckDuckGoSearchRun, WikipediaQueryRun]
    """

    # --- Tool 1: DuckDuckGo Search --------------------------------------------
    # Best for: current events, recent news, live data, specific URLs
    # No API key required — uses DuckDuckGo's public search endpoint

    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Use this tool to search the internet for current, recent, or "
            "real-time information. "
            "Best for: news, recent events, current prices, sports scores, "
            "latest releases, anything that may have changed recently. "
            "Input should be a concise search query string."
        ),
    )

    # --- Tool 2: Wikipedia ----------------------------------------------------
    # Best for: encyclopedic knowledge, biographies, historical facts,
    #           scientific concepts, definitions
    # Uses the free Wikipedia API — no key required

    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=2,          # Fetch summaries from top 2 matching articles
        doc_content_chars_max=800 # Truncate each article to 800 chars to save tokens
    )

    wiki_tool = WikipediaQueryRun(
        name="wikipedia",
        description=(
            "Use this tool to look up factual, encyclopedic information. "
            "Best for: biographies, historical events, scientific concepts, "
            "geography, definitions, background knowledge. "
            "Input should be a topic name or question."
        ),
        api_wrapper=wiki_wrapper,
    )

    tools = [search_tool, wiki_tool]

    # Print tool summary for transparency
    table = Table(
        title="🛠️  Available Tools",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
    )
    table.add_column("Tool Name", style="bold yellow", width=16)
    table.add_column("Description", style="white")

    for tool in tools:
        table.add_row(tool.name, tool.description[:80] + "…")

    console.print(table)
    return tools


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD THE AGENT
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(tools: list) -> AgentExecutor:
    """
    Construct the ReAct agent with the given tools.

    Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │  AgentExecutor  (the loop)                                        │
    │  ┌────────────────────────────────────────────────────────────┐  │
    │  │  Agent  (the decision maker)                               │  │
    │  │  ┌─────────────────────────────────────────────────────┐  │  │
    │  │  │  ChatPromptTemplate                                  │  │  │
    │  │  │   • system message (persona + tool instructions)    │  │  │
    │  │  │   • MessagesPlaceholder("agent_scratchpad")         │  │  │
    │  │  │     ↑ stores tool calls + results between steps     │  │  │
    │  │  │   • human message  {input}                          │  │  │
    │  │  └─────────────────────────────────────────────────────┘  │  │
    │  │                        │                                   │  │
    │  │                   ChatGroq LLM                             │  │
    │  │            (with tool schemas bound to it)                 │  │
    │  └────────────────────────────────────────────────────────────┘  │
    │                                                                   │
    │  Loop:                                                            │
    │    1. LLM decides: answer directly OR call a tool                │
    │    2. If tool: AgentExecutor calls it, appends result to         │
    │       agent_scratchpad, loops back to LLM                        │
    │    3. If answer: return final response                           │
    └──────────────────────────────────────────────────────────────────┘

    Args:
        tools: List of Tool objects from build_tools()

    Returns:
        AgentExecutor: ready to run with .invoke()
    """

    # --- 3a. LLM with tool calling support ------------------------------------
    # We bind tools to the LLM here.
    # "bind_tools" sends the tool schemas (name, description, parameters)
    # to Groq alongside the prompt, so the LLM knows what tools are available
    # and how to call them in a structured way.

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,    # 0 for agents = more reliable tool-call decisions
        max_tokens=1024,  # agents need more tokens for reasoning steps
    )

    # --- 3b. Agent Prompt -----------------------------------------------------
    # This prompt is specifically structured for tool-calling agents.
    #
    # MessagesPlaceholder("agent_scratchpad") is CRITICAL:
    #   It holds the agent's "working memory" for the current query —
    #   the sequence of tool calls made and their results.
    #   This is different from conversation history (Milestone 3).
    #   It's reset after each final answer.
    #
    #   Example scratchpad contents mid-reasoning:
    #     ToolCallMessage(tool="web_search", input="Champions League 2024 winner")
    #     ToolResultMessage("Real Madrid won the 2024 Champions League...")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are KnowledgeBot, a helpful AI assistant with access "
                    "to web search and Wikipedia tools.\n\n"
                    "RULES:\n"
                    "1. For current events, news, or recent data → use web_search\n"
                    "2. For encyclopedic facts, biographies, history → use wikipedia\n"
                    "3. For math, logic, or things you know confidently → answer directly\n"
                    "4. Always cite which tool you used at the end of your answer.\n"
                    "5. Be concise. Do not repeat the tool output verbatim — summarise it."
                ),
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # tool call history
            (
                "human",
                "{input}",   # note: agent uses "input" not "user_input"
            ),
        ]
    )

    # --- 3c. Create the agent -------------------------------------------------
    # create_tool_calling_agent wires together:
    #   • the prompt
    #   • the LLM (with tools bound)
    #   • an output parser that handles tool-call vs final-answer routing

    agent = create_tool_calling_agent(llm, tools, prompt)

    # --- 3d. Wrap in AgentExecutor --------------------------------------------
    # AgentExecutor is the runtime that actually runs the Reason-Act loop.
    #
    # verbose=True  → prints each reasoning step to the console
    #                 invaluable for learning what the agent is doing
    # max_iterations → safety limit: prevent infinite tool-call loops
    # handle_parsing_errors → if LLM output is malformed, retry gracefully

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,            # ← shows agent's internal steps
        max_iterations=5,        # ← max tool calls before forcing an answer
        handle_parsing_errors=True,
    )

    return agent_executor


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_chat_loop(agent_executor: AgentExecutor) -> None:
    """
    Interactive CLI loop for the tool-enabled agent.

    Differences from previous milestones:
        • invoke() uses "input" key (agent standard) instead of "user_input"
        • verbose=True in AgentExecutor prints live reasoning steps
        • Response is in result["output"] instead of being a plain string

    Args:
        agent_executor: The AgentExecutor from build_agent()
    """

    console.print(
        Panel(
            "[bold cyan]KnowledgeBot v0.3[/bold cyan]  —  Tool-Enabled Agent\n"
            "[dim]I can now search the web and Wikipedia in real time!\n"
            "Type [bold]exit[/bold] to quit.[/dim]\n\n"
            "[dim]Try asking:\n"
            "  • 'Who is the current CEO of OpenAI?'\n"
            "  • 'What is quantum entanglement?'\n"
            "  • 'What were the latest AI news this week?'[/dim]",
            border_style="cyan",
        )
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

        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            console.print(
                Panel("[bold cyan]👋  Goodbye![/bold cyan]", border_style="cyan")
            )
            break

        # ── Invoke the agent ───────────────────────────────────────────────────
        # The agent executor will:
        #   1. Send user input + tool schemas to LLM
        #   2. LLM decides: tool needed? Which one?
        #   3. If yes: call the tool, append result to scratchpad
        #   4. Send scratchpad back to LLM for next decision
        #   5. Repeat until LLM returns a final answer (no more tool calls)
        #
        # verbose=True means steps 2-4 are printed live to console — watch them!

        console.print(Rule("[dim]Agent Reasoning[/dim]", style="dim"))

        try:
            result = agent_executor.invoke({"input": user_input})
            # result is a dict: {"input": ..., "output": "final answer string"}
            final_answer: str = result["output"]
        except Exception as e:
            console.print(f"[bold red]❌  Agent error:[/bold red] {e}")
            continue

        # ── Print final answer ─────────────────────────────────────────────────
        console.print(Rule("[dim]Final Answer[/dim]", style="dim"))
        console.print(
            Panel(
                Text(final_answer, style="white"),
                title="[bold magenta]🤖 KnowledgeBot[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Orchestrate Milestone 4."""
    load_environment()
    tools = build_tools()
    agent_executor = build_agent(tools)
    run_chat_loop(agent_executor)


if __name__ == "__main__":
    main()