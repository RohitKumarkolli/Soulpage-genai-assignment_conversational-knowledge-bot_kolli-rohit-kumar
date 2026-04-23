"""
config.py
---------
Central configuration for KnowledgeBot.
Single source of truth for all settings — edit here, change everywhere.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """LLM inference settings."""
    model: str       = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int  = 1024


@dataclass(frozen=True)
class WikipediaConfig:
    """Wikipedia tool settings."""
    top_k_results: int          = 2
    doc_content_chars_max: int  = 1000


@dataclass(frozen=True)
class MemoryConfig:
    """
    Memory strategy settings.

    use_summary_memory:
        True  → ConversationSummaryMemory (compresses old turns, good for long sessions)
        False → ChatMessageHistory buffer (stores verbatim, good for short sessions)

    summary_token_limit:
        When the buffer exceeds this many tokens, older turns are summarised.
        Lower = more aggressive summarisation.
        Higher = more verbatim history kept.
    """
    use_summary_memory: bool = True
    summary_token_limit: int = 1000     # tokens before summarisation triggers


@dataclass(frozen=True)
class AppConfig:
    """Streamlit app settings."""
    page_title: str    = "KnowledgeBot"
    page_icon: str     = "🤖"
    layout: str        = "wide"
    sidebar_state: str = "expanded"
    bot_name: str      = "KnowledgeBot"
    bot_version: str   = "2.0"          # bumped for M9


def get_groq_api_key() -> str:
    """
    Retrieve GROQ_API_KEY from environment.

    Raises:
        EnvironmentError: if not set.
    """
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Add it to your .env file: GROQ_API_KEY=your_key_here\n"
            "Get a free key at: https://console.groq.com"
        )
    return key


# ── Singleton instances ────────────────────────────────────────────────────────
LLM_CONFIG       = LLMConfig()
WIKIPEDIA_CONFIG = WikipediaConfig()
MEMORY_CONFIG    = MemoryConfig()
APP_CONFIG       = AppConfig()