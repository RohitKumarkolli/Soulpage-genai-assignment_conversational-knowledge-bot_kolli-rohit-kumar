"""
streamlit_app.py
----------------
Streamlit Cloud entry point for KnowledgeBot.

WHY this file instead of run.py?
    Streamlit Cloud executes a single file directly as __main__.
    run.py used sys.path manipulation which doesn't work reliably
    on cloud. This file adds the project root to sys.path explicitly
    BEFORE any knowledge_bot imports, making the package resolvable
    in all environments (local, cloud, Docker).

Streamlit Cloud looks for:
    1. streamlit_app.py  (root level) ← this file
    2. app.py            (root level)
    3. src/app.py

Deploy command (Streamlit Cloud dashboard):
    Main file path: streamlit_app.py
"""

import sys
import os

# ── Path setup ─────────────────────────────────────────────────────────────────
# Add the project root to sys.path so `knowledge_bot` is importable
# as a package regardless of the working directory.
# __file__ = /path/to/project/streamlit_app.py
# dirname  = /path/to/project/  ← this is what we add
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import and run ──────────────────────────────────────────────────────────────
from knowledge_bot.app import main  # noqa: E402

main()