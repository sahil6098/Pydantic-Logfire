import logfire
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_tavily import TavilySearch

from .prompts import _EXPLAINER_PROMPT, _ANALYST_PROMPT, _CREATOR_PROMPT , _FORMATTER_PROMPT , _PLANNER_PROMPT
from .state import AgentState
from .llms import groq_llm, gemini_llm



_groq = groq_llm()
_tavily = TavilySearch(max_results = 4)

# ── Nodes ─────────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    with logfire.span("planner",
                      question=state["question"],
                      session_id=state.get("session_id", "")):
        response = _groq.invoke([
            SystemMessage(content=_PLANNER_PROMPT),
            HumanMessage(content=state["question"]),
        ])
        intent = response.content.strip().lower().split()[0]
        if intent not in ("explain", "analyze", "create", "search"):
            intent = "explain"

        logfire.info("intent_classified",
                     intent=intent,
                     question=state["question"])

        return {
            "intent": intent,
            "node_path": state.get("node_path", []) + ["planner"],
        }


def explainer(state: AgentState) -> dict:
    with logfire.span("explainer",
                      question=state["question"],
                      model="openai/gpt-oss-20b"):
        response = _groq.invoke([
            SystemMessage(content=_EXPLAINER_PROMPT),
            HumanMessage(content=state["question"]),
        ])
        logfire.info("explanation_ready",
                     answer_length=len(response.content))
        return {
            "specialist_output": response.content,
            "model_used": "groq/openai/gpt-oss-20b",
            "node_path": state.get("node_path", []) + ["explainer"],
        }
        

def analyst(state: AgentState) -> dict:
    with logfire.span("analyst",
                      question=state["question"],
                      model="openai/gpt-oss-20b"):
        response = _groq.invoke([
            SystemMessage(content=_ANALYST_PROMPT),
            HumanMessage(content=state["question"]),
        ])
        logfire.info("analysis_ready",
                     answer_length=len(response.content))
        return {
            "specialist_output": response.content,
            "model_used": "groq/openai/gpt-oss-20b",
            "node_path": state.get("node_path", []) + ["analyst"],
        }


def creator(state: AgentState) -> dict:
    with logfire.span("creator",
                      question=state["question"],
                      model="openai/gpt-oss-20b"):
        response = _groq.invoke([
            SystemMessage(content=_CREATOR_PROMPT),
            HumanMessage(content=state["question"]),
        ])
        logfire.info("creation_ready",
                     answer_length=len(response.content))
        return {
            "specialist_output": response.content,
            "model_used": "groq/openai/gpt-oss-20b",
            "node_path": state.get("node_path", []) + ["creator"],
        }


def web_searcher(state: AgentState) -> dict:
    with logfire.span("web_searcher",
                      query=state["question"]):

        # Step 1 — Tavily search
        with logfire.span("tavily_search", query=state["question"]):
            _raw = _tavily.invoke(state["question"])
            # TavilySearch may return a list[dict] or a plain string
            if isinstance(_raw, list):
                results = _raw
                raw = "\n\n".join(
                    f"[{i+1}] {r.get('url', '')}\n{r.get('content', '')}"
                    for i, r in enumerate(results)
                )
                sources = [r.get("url", "") for r in results]
            else:
                results = []
                raw = str(_raw)
                sources = []
            logfire.info("search_complete",
                         num_results=len(results),
                         sources=sources)

        # Step 2 — Groq synthesises the results
        with logfire.span("synthesize_results",
                          model="openai/gpt-oss-20b"):
            summary = _groq.invoke([
                SystemMessage(content=(
                    "Synthesize the search results into a clear, concise answer. "
                    "Include key facts and mention sources where relevant."
                )),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Search Results:\n{raw}"
                )),
            ])
            logfire.info("synthesis_ready",
                         answer_length=len(summary.content))

        return {
            "specialist_output": summary.content,
            "search_results": raw,
            "model_used": "tavily + groq/llama-3.3-70b-versatile",
            "node_path": state.get("node_path", []) + ["web_searcher"],
        }



def formatter(state: AgentState) -> dict:
    with logfire.span("formatter",
                      intent=state["intent"],
                      model="llama-3.3-70b-versatile"):
        response = _groq.invoke([
            SystemMessage(content=_FORMATTER_PROMPT),
            HumanMessage(content=state["specialist_output"]),
        ])
        final     = response.content
        model_tag = state.get("model_used", "") + " → groq/llama-3.3-70b-versatile (formatter)"

        logfire.info("formatter_done",
                     input_length=len(state["specialist_output"]),
                     output_length=len(final))

        return {
            "final_answer": final,
            "model_used": model_tag,
            "node_path": state.get("node_path", []) + ["formatter"],
        }
