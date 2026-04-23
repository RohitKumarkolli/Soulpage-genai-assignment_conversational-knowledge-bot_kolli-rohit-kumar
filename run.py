"""
run.py
------
Root-level launcher for KnowledgeBot.

WHY this file?
    When Streamlit runs knowledge_bot/app.py directly, app.py uses
    relative imports (from .config import ...) which only work when
    the file is executed AS PART OF A PACKAGE — not as a standalone script.

    Streamlit runs files as __main__, which breaks relative imports.

    SOLUTION: run this file from the project root instead.
    It adds the project root to sys.path and then calls Streamlit
    programmatically, so knowledge_bot is a proper importable package.

Usage:
    # From the project root (the folder containing this file):
    streamlit run run.py
"""

import sys
import os

# Ensure the project root is on sys.path so `knowledge_bot` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the app
from knowledge_bot import app   # noqa: E402

app.main()