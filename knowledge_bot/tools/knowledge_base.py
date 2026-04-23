"""
tools/knowledge_base.py
-----------------------
Local knowledge base tool for KnowledgeBot.

WHY a local knowledge base?
    Web search and Wikipedia are great for general queries, but for
    project-specific, company-specific, or private knowledge you need
    a local source the agent can search instantly — no API call needed.

    Examples of what goes in the knowledge base:
        • Your project's own documentation
        • Company FAQs
        • Product specifications
        • Internal policies
        • Any facts you want the bot to know with certainty

HOW it works:
    1. Load knowledge_base.json at startup
    2. Expose a search function as a LangChain Tool
    3. The agent calls it like any other tool — it sees the description
       and decides when to use it
    4. Search is keyword-based (fast, no embedding needed)

SEARCH STRATEGY:
    Simple keyword matching — checks if any of the entry's keywords
    appear in the user's query. Production systems would use
    vector embeddings (FAISS, Chroma) for semantic search,
    but keyword search works well for structured FAQs.
"""

import json
import os
from typing import Optional
from langchain.tools import Tool


# ── Path resolution ────────────────────────────────────────────────────────────
# __file__ = knowledge_bot/tools/knowledge_base.py
# We go up two levels to reach the project root, then into data/

_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
_KB_PATH     = os.path.join(_PROJECT_ROOT, "data", "knowledge_base.json")


def _load_knowledge_base() -> list[dict]:
    """
    Load and parse the knowledge base JSON file.

    Returns:
        list[dict]: List of knowledge base entries.
                    Empty list if file not found or malformed.
    """
    if not os.path.exists(_KB_PATH):
        print(f"[KnowledgeBase] File not found: {_KB_PATH}")
        return []

    try:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        print(f"[KnowledgeBase] Loaded {len(entries)} entries from {_KB_PATH}")
        return entries
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[KnowledgeBase] Parse error: {e}")
        return []


# Load once at module import time — not on every tool call
_KB_ENTRIES: list[dict] = _load_knowledge_base()


def search_knowledge_base(query: str) -> str:
    """
    Search the local knowledge base for entries matching the query.

    Search algorithm:
        1. Normalise query to lowercase
        2. For each KB entry, check if any keyword appears in the query
        3. If match found, return the entry's content
        4. If multiple matches, return all concatenated
        5. If no match, return a "not found" message

    This function is wrapped as a LangChain Tool — the agent
    calls it with a plain string query.

    Args:
        query: The user's question or search string.

    Returns:
        str: Matching knowledge base content, or "not found" message.
    """
    if not _KB_ENTRIES:
        return "Knowledge base is empty or could not be loaded."

    query_lower = query.lower().strip()
    matches     = []

    for entry in _KB_ENTRIES:
        keywords = entry.get("keywords", [])
        # Check if any keyword from this entry appears in the query
        if any(kw.lower() in query_lower for kw in keywords):
            topic   = entry.get("topic", "Unknown")
            content = entry.get("content", "")
            matches.append(f"[{topic}]\n{content}")

    if matches:
        return "\n\n".join(matches)

    return (
        "No matching information found in the local knowledge base. "
        "Try web_search or wikipedia for this query."
    )


def build_knowledge_base_tool() -> Tool:
    """
    Build and return the knowledge base LangChain Tool.

    WHY Tool() instead of @tool decorator?
        We need to pass a named function (search_knowledge_base)
        so the agent can inspect its signature. Tool() is the
        explicit, production-safe way to wrap any callable.

    Returns:
        Tool: configured knowledge base search tool.
    """
    return Tool(
        name="knowledge_base",
        func=search_knowledge_base,
        description=(
            "Search the local knowledge base for project-specific, "
            "internal, or pre-loaded information. "
            "Use this tool FIRST for questions about: "
            "KnowledgeBot itself, its tech stack, how it was built, "
            "its features, LangChain concepts, Groq, setup instructions, "
            "memory types, or milestones. "
            "Only use web_search or wikipedia if this tool returns no results. "
            "Input: a question or topic string."
        ),
    )