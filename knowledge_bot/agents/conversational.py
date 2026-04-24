"""
agents/conversational.py
------------------------
Conversational agent — Milestone 9 upgrade.

Changes from M7:
    1. Improved system prompt with chain-of-thought + structured output rules
    2. ConversationSummaryMemory support alongside buffer memory
    3. Three tools: knowledge_base (new) + wikipedia + web_search
    4. invoke_agent() now also saves to SummaryMemoryManager if enabled
"""

import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain_core.agents import create_tool_calling_agent
    from langchain.agents import AgentExecutor

from ..config  import LLM_CONFIG, MEMORY_CONFIG

try:
    from ..memory  import memory_store, SummaryMemoryManager
except ImportError:
    None
from ..tools   import build_all_tools


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
# PROMPT ENGINEERING  (Milestone 9 upgrade)
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt() -> ChatPromptTemplate:
    """
    Build the M9 agent prompt — structured output + chain-of-thought hints.

    PROMPT ENGINEERING TECHNIQUES USED:

    1. ROLE DEFINITION (line 1)
       Clear persona with explicit capability list. The LLM performs better
       when it knows exactly what it is and what it can do.

    2. TOOL PRIORITY ORDERING (TOOL RULES section)
       We explicitly tell the LLM: try knowledge_base FIRST, then wikipedia,
       then web_search. Without this, the agent might default to web_search
       for everything, wasting tokens and time on queries the KB can answer.

    3. CHAIN-OF-THOUGHT HINT (REASONING section)
       "Before answering, think about..." encourages the LLM to reason
       step-by-step internally, producing more accurate tool selections
       and better-structured answers.

    4. STRUCTURED OUTPUT FORMAT (RESPONSE FORMAT section)
       Telling the LLM exactly how to format responses ensures consistency.
       Confidence indicators help users calibrate trust in answers.

    5. PRONOUN RESOLUTION (CONTEXT AWARENESS section)
       Explicit instruction to resolve "he/she/it/they" from chat history
       before deciding on a tool — prevents unnecessary web searches.

    6. NEGATIVE EXAMPLES (what NOT to do)
       "Never paste raw tool output" and "Never say 'I don't know'"
       are negative constraints that prevent common failure modes.
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are KnowledgeBot v2.0, an intelligent conversational AI assistant \
with access to three tools: a local knowledge base, Wikipedia, and web search.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL SELECTION RULES (follow this priority order):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. knowledge_base  → Check THIS FIRST for questions about:
   KnowledgeBot, its tech stack, LangChain, Groq, how it was built,
   setup instructions, memory types, milestones, or any project-specific info.

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
Structure your answers as follows:
  • Lead with the direct answer to the question
  • Add supporting context (2-3 sentences max)
  • If uncertain, say: "Based on [source], ..." or "As of my last data, ..."
  • For multi-part questions, use a brief numbered list
  • Keep total response under 200 words unless detail is specifically requested

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Always write a complete, helpful answer
  ✓ Resolve pronouns from conversation history before using any tool
  ✓ Summarise tool results — never paste raw output verbatim
  ✗ Never say "I don't know" — always try a tool first
  ✗ Never use web_search for questions the knowledge_base can answer
  ✗ Never repeat the user's question back to them""",
        ),

        # ── Conversation history ───────────────────────────────────────────────
        # Injected by RunnableWithMessageHistory (buffer) or
        # manually from SummaryMemoryManager (summary).
        # Either way, this slot holds all prior context.
        MessagesPlaceholder(variable_name="chat_history"),

        # ── Current user message ───────────────────────────────────────────────
        ("human", "{input}"),

        # ── Agent working memory (tool calls for this query only) ──────────────
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# AGENT EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

def _build_agent_executor(llm: ChatGroq, prompt: ChatPromptTemplate) -> AgentExecutor:
    """Build AgentExecutor with all three tools."""
    tools = build_all_tools()   # [knowledge_base, wikipedia, web_search]
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
# CACHED AGENT BUILD
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_agent() -> tuple:
    """
    Build and cache the agent + summary memory manager.

    Returns a TUPLE so both can be cached together:
        (RunnableWithMessageHistory, SummaryMemoryManager | None)

    WHY return SummaryMemoryManager from here?
        build_agent() is cached with @st.cache_resource.
        SummaryMemoryManager needs the LLM instance, which is created here.
        Returning both ensures they share the same LLM object without
        creating a second one.

    Returns:
        tuple: (agent, summary_manager | None)
    """
    llm            = _build_llm()
    prompt         = _build_prompt()
    agent_executor = _build_agent_executor(llm, prompt)

    # ── Memory wrapper ─────────────────────────────────────────────────────────
    agent = RunnableWithMessageHistory(
        agent_executor,
        memory_store.get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        # output_messages_key intentionally omitted — prevents blank responses
    )

    # ── Summary memory manager ─────────────────────────────────────────────────
    if MEMORY_CONFIG.use_summary_memory:
        summary_manager = SummaryMemoryManager(llm)
    else:
        summary_manager = None

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

    If summary_manager is provided, also saves the turn to
    ConversationSummaryMemory (in addition to the buffer store).

    Args:
        agent          : The RunnableWithMessageHistory agent.
        user_input     : The user's message string.
        session_id     : Session ID for memory routing.
        summary_manager: Optional SummaryMemoryManager for M9 summary memory.

    Returns:
        dict:
            "answer"     : str   — the final response text
            "tools_used" : list  — tool names used (empty if no tool)
            "error"      : str | None — error message if failed
    """
    try:
        result = agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        # Extract final answer string
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

        # ── Also save to summary memory if enabled ─────────────────────────────
        # This runs AFTER the agent responds so the summary reflects
        # the completed turn, not a partial one.
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