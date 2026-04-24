"""
agents/conversational.py
------------------------
Conversational agent — Milestone 9.

IMPORT FIX (latest LangChain):
    create_tool_calling_agent moved — now imported from langchain.agents
    with a fallback to langchain_core.agents for newer versions.
    AgentExecutor stays in langchain.agents.
"""

import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── Agent imports with version-safe fallback ───────────────────────────────────
# In LangChain 0.3.x, create_tool_calling_agent lives in langchain.agents.
# We wrap in try/except so if a newer version moves it, we fall back gracefully.
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain_core.agents import create_tool_calling_agent  # type: ignore
    from langchain.agents import AgentExecutor

from ..config import LLM_CONFIG, MEMORY_CONFIG
from ..memory import memory_store, SummaryMemoryManager
from ..tools  import build_all_tools


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm() -> ChatGroq:
    """Instantiate ChatGroq from config."""
    return ChatGroq(
        model=LLM_CONFIG.model,
        temperature=LLM_CONFIG.temperature,
        max_tokens=LLM_CONFIG.max_tokens,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt() -> ChatPromptTemplate:
    """
    Build the agent prompt template with chain-of-thought + structured output.

    Prompt engineering techniques:
        1. Explicit tool priority order (KB → Wikipedia → Web)
        2. Chain-of-thought reasoning hints
        3. Structured output format rules
        4. Negative constraints (what NOT to do)
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are KnowledgeBot v2.0, an intelligent conversational AI \
with access to three tools: a local knowledge base, Wikipedia, and web search.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL SELECTION RULES (follow this priority order):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. knowledge_base  → Check THIS FIRST for questions about:
   KnowledgeBot, its tech stack, LangChain, Groq, how it was built,
   setup instructions, memory types, milestones, or project-specific info.

2. wikipedia       → Use for encyclopedic, factual, or background knowledge:
   biographies, history, science, geography, definitions, stable facts.

3. web_search      → Use ONLY for real-time or recent information:
   today's news, live scores, current prices, recent events, latest releases.

4. No tool needed  → Answer directly for:
   math, logic, greetings, or things you know with high confidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING (think before acting):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before selecting a tool, ask yourself:
  • Is this about KnowledgeBot or its internals? → knowledge_base
  • Does the question use pronouns (he/she/they/it)?
    → Resolve from conversation history FIRST
  • Is this a follow-up to a previous answer?
    → Use existing context before calling a tool
  • Is real-time data needed?
    → Only then use web_search

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Lead with the direct answer to the question
  • Add 2-3 sentences of supporting context
  • Keep total response under 200 words unless detail is requested
  • For multi-part questions, use a brief numbered list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Always write a complete, helpful answer
  ✓ Resolve pronouns from conversation history before using any tool
  ✓ Summarise tool results — never paste raw output verbatim
  ✗ Never say "I don't know" — always try a tool first
  ✗ Never use web_search for questions the knowledge_base can answer""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# AGENT EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

def _build_agent_executor(llm: ChatGroq, prompt: ChatPromptTemplate) -> AgentExecutor:
    """Build AgentExecutor with all three tools."""
    tools = build_all_tools()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CACHED BUILD
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_agent() -> tuple:
    """
    Build and cache the agent + summary memory manager.

    Returns:
        tuple: (RunnableWithMessageHistory, SummaryMemoryManager | None)
    """
    llm            = _build_llm()
    prompt         = _build_prompt()
    agent_executor = _build_agent_executor(llm, prompt)

    agent = RunnableWithMessageHistory(
        agent_executor,
        memory_store.get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        # output_messages_key intentionally omitted — prevents blank responses
    )

    summary_manager = SummaryMemoryManager(llm) if MEMORY_CONFIG.use_summary_memory else None

    return agent, summary_manager


# ══════════════════════════════════════════════════════════════════════════════
# INVOKE
# ══════════════════════════════════════════════════════════════════════════════

def invoke_agent(
    agent: RunnableWithMessageHistory,
    user_input: str,
    session_id: str,
    summary_manager: "SummaryMemoryManager | None" = None,
) -> dict:
    """
    Invoke the agent and return a clean result dict.

    Returns:
        dict: {answer: str, tools_used: list, error: str|None}
    """
    try:
        result = agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        answer: str = result.get("output", "").strip()

        if not answer:
            answer = (
                "⚠️ The agent returned an empty response.\n\n"
                f"**Debug — raw result:**\n```python\n{result}\n```"
            )

        # Extract tool names from intermediate steps
        tools_used: list = []
        for action, _ in result.get("intermediate_steps", []):
            tool_name = getattr(action, "tool", None)
            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)

        # Save to summary memory if enabled
        if summary_manager:
            summary_manager.save_context(
                session_id=session_id,
                human_message=user_input,
                ai_message=answer,
            )

        return {"answer": answer, "tools_used": tools_used, "error": None}

    except Exception as e:
        return {
            "answer"    : f"⚠️ An error occurred:\n```\n{str(e)}\n```",
            "tools_used": [],
            "error"     : str(e),
        }