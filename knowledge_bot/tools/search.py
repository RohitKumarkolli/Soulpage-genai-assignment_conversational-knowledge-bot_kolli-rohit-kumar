"""
tools/search.py
---------------
All tool definitions for KnowledgeBot.

Tool priority (agent reads descriptions to decide):
    1. knowledge_base  → project-specific facts (no API call, instant)
    2. wikipedia       → encyclopedic background knowledge
    3. web_search      → current events, live data

Adding a new tool:
    1. Create build_X_tool() below
    2. Add it to build_all_tools() list
    3. The agent picks it up automatically — no other changes needed
"""

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from ..config import WIKIPEDIA_CONFIG              # relative import
from .knowledge_base import build_knowledge_base_tool  # sibling module


def build_web_search_tool() -> DuckDuckGoSearchRun:
    """
    DuckDuckGo web search — real-time, no API key required.
    Best for: current events, news, live scores, recent releases.
    """
    return DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the internet for current, recent, or real-time information. "
            "Use for: breaking news, recent events, sports scores, stock prices, "
            "latest product releases, current office holders, or anything that "
            "changes over time. "
            "Input: a concise search query string."
        ),
    )


def build_wikipedia_tool() -> WikipediaQueryRun:
    """
    Wikipedia — free encyclopedic knowledge, no API key required.
    Best for: biographies, history, science, geography, definitions.
    """
    wrapper = WikipediaAPIWrapper(
        top_k_results=WIKIPEDIA_CONFIG.top_k_results,
        doc_content_chars_max=WIKIPEDIA_CONFIG.doc_content_chars_max,
    )
    return WikipediaQueryRun(
        name="wikipedia",
        description=(
            "Look up encyclopedic, factual, or background information. "
            "Use for: biographies, historical events, scientific concepts, "
            "geography, definitions, or stable knowledge. "
            "Input: a topic name or descriptive question."
        ),
        api_wrapper=wrapper,
    )


def build_all_tools() -> list:
    """
    Build and return all tools available to the agent.

    ORDER MATTERS for the agent's tool selection:
    knowledge_base is listed first so the agent considers it first
    for project-specific queries before hitting the web.

    Returns:
        list: All configured Tool instances.
    """
    return [
        build_knowledge_base_tool(),   # ← check local KB first (M9)
        build_wikipedia_tool(),        # ← encyclopedic fallback
        build_web_search_tool(),       # ← live web fallback
    ]