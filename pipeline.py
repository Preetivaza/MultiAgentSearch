import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from langgraph.graph import StateGraph, START, END

from agents import (
    critic_chain,
    evidence_collector_node,
    planner_node,
    research_orchestrator_node,
    writer_chain,
    claim_extraction_node,
    evidence_verification_node,
    verdict_node,
    report_writer_node,
    citation_validation_node,
    reviewer_node,
    targeted_revision_node,
    route_after_review,
    evaluation_node,
)
from schemas import ResearchPlan, ResearchState
import time



def is_valid_url(url: str) -> bool:
    """Validate if string is a properly formatted HTTP/HTTPS URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def extract_valid_urls(text: str) -> List[str]:
    """Extract and validate HTTP/HTTPS URLs from any text string."""
    urls = re.findall(r'https?://[^\s)\]}>,]+', text)
    valid_urls = []
    for url in urls:
        url = url.rstrip(".,;:)]}")
        if is_valid_url(url):
            valid_urls.append(url)
    return valid_urls


# ── LangGraph Workflow Nodes ──────────────────────────────────────────────────

def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Writer Node:
    Synthesizes research from evidence collector and verified claims into a structured report.
    """
    return report_writer_node(state)


def critic_node(state: ResearchState) -> Dict[str, Any]:
    """
    Critic Node:
    Reviews and evaluates the generated research report.
    """
    report = state.get("report", "")
    if not report:
        return {
            "feedback": "Critic Review: No report found to review.",
            "errors": ["Critic skipped: Report was empty."]
        }

    try:
        feedback = critic_chain.invoke({"report": report})
        return {
            "feedback": feedback,
            "errors": [],
            "success": True
        }
    except Exception as e:
        error_msg = f"Critic Node error: {str(e)}"
        return {
            "feedback": "Critic review unavailable.",
            "errors": [error_msg],
            "success": True
        }


# ── Conditional Routing ───────────────────────────────────────────────────────

def route_after_planning(state: ResearchState) -> str:
    """
    Validates plan before proceeding to research execution.
    If planning produced errors or no subquestions, terminates early.
    """
    plan = state.get("plan")
    errors = state.get("errors", [])

    if not plan or not getattr(plan, "subquestions", None):
        return "end"

    return "orchestrator"


def route_after_revision_step(state: ResearchState) -> str:
    """
    Routes from targeted revision node back into the pipeline:
    - If researcher was targeted: re-extracts claims and re-verifies evidence.
    - If rewriter was targeted: rewrites the report directly with guidance.
    """
    assessment = state.get("review_assessment", {}) or {}
    if assessment.get("revision_target") == "researcher":
        return "re_extract"
    return "re_write"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_research_graph() -> Any:
    """
    Builds and compiles the full multi-agent LangGraph workflow:
    1. Planner: Analyzes query & generates validated subquestions
    2. Orchestrator: Dispatches concurrent research tasks for each subquestion
    3. Evidence Collector: Combines results, deduplicates sources, and preserves metadata
    4. Claim Extractor: Splits findings into individual factual claims
    5. Verifier & Verdict: Verifies claims against source passages, catches contradictions & rejects invalid URLs
    6. Report Writer: Synthesizes structured research report with verification insights
    7. Citation Validator: Checks source IDs and citation integrity
    8. Reviewer Node: Comprehensive evaluation of coverage, evidence, citations, and contradictions
    9. Conditional Revision Loop: Re-routes for targeted rewriting or research, strictly terminating at limit
    10. Critic: Final peer review and scoring
    """
    workflow = StateGraph(ResearchState)

    # Register all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("orchestrator", research_orchestrator_node)
    workflow.add_node("evidence_collector", evidence_collector_node)
    workflow.add_node("claim_extractor", claim_extraction_node)
    workflow.add_node("verifier", evidence_verification_node)
    workflow.add_node("verdict", verdict_node)
    workflow.add_node("report_writer", report_writer_node)
    workflow.add_node("citation_validator", citation_validation_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("targeted_revision", targeted_revision_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("evaluator", evaluation_node)

    # 1. Start with planner
    workflow.add_edge(START, "planner")

    # 2. Conditional edge after planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "orchestrator": "orchestrator",
            "end": END,
        }
    )

    # 3. Main forward path
    workflow.add_edge("orchestrator", "evidence_collector")
    workflow.add_edge("evidence_collector", "claim_extractor")
    workflow.add_edge("claim_extractor", "verifier")
    workflow.add_edge("verifier", "verdict")
    workflow.add_edge("verdict", "report_writer")
    workflow.add_edge("report_writer", "citation_validator")
    workflow.add_edge("citation_validator", "reviewer")

    # 4. Conditional Edge after Reviewer: Pass -> Publish (Critic) or Needs Revision -> Targeted Revision
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "publish": "critic",
            "targeted_rewrite": "targeted_revision",
            "targeted_research": "targeted_revision",
        }
    )

    # 5. Targeted Revision feeds back into pipeline (Revalidating claims and citations!)
    workflow.add_conditional_edges(
        "targeted_revision",
        route_after_revision_step,
        {
            "re_extract": "claim_extractor",
            "re_write": "report_writer",
        }
    )

    # 6. Critic routes to Evaluator for observability & metric calculation
    workflow.add_edge("critic", "evaluator")
    workflow.add_edge("evaluator", END)

    return workflow.compile()


# ── Pipeline Runner ───────────────────────────────────────────────────────────

def _safe_print(text: str = ""):
    """Safely print text on Windows consoles without charmap UnicodeEncodeErrors."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            clean_text = text.encode(encoding, errors="replace").decode(encoding)
            print(clean_text)
        except Exception:
            print(text.encode("ascii", errors="replace").decode("ascii"))


def run_research_pipeline(topic: str, max_revisions: int = 2) -> dict:
    """
    Runs the multi-agent research pipeline using the compiled LangGraph.
    Returns backward-compatible dictionary for console and Streamlit UI.
    """
    app = build_research_graph()

    initial_state: ResearchState = {
        "query": topic,
        "plan": None,
        "errors": [],
        "research_results": [],
        "evidence": None,
        "claims": [],
        "verification_report": {},
        "report_metadata": {},
        "review_assessment": {},
        "revision_count": 0,
        "max_revisions": max_revisions,
        "actionable_guidance": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "start_time": time.time(),
        "run_trace": [],
        "node_latencies": {},
        "token_usage": {},
        "evaluation_result": None,
        "success": False
    }

    _safe_print("\n" + "=" * 60)
    _safe_print(f"[ResearchMind] Launching Multi-Agent System for: '{topic}'")
    _safe_print("=" * 60)

    final_state = app.invoke(initial_state)

    # Extract all valid URLs from collected evidence for backwards compatibility
    evidence = final_state.get("evidence") or {}
    unique_sources = evidence.get("unique_sources", []) if isinstance(evidence, dict) else []
    valid_urls = [s.get("url") for s in unique_sources if isinstance(s, dict) and s.get("url")]

    plan_data = final_state.get("plan")
    search_context = evidence.get("compiled_research_context", "") if isinstance(evidence, dict) else ""

    result = {
        "success": final_state.get("success", False),
        "plan": plan_data.model_dump() if hasattr(plan_data, "model_dump") else plan_data,
        "search_results": search_context,
        "valid_urls": valid_urls,
        "evidence": evidence,
        "research_results": final_state.get("research_results", []),
        "claims": final_state.get("claims", []),
        "verification_report": final_state.get("verification_report", {}),
        "report_metadata": final_state.get("report_metadata", {}),
        "review_assessment": final_state.get("review_assessment", {}),
        "revision_count": final_state.get("revision_count", 0),
        "evaluation_result": final_state.get("evaluation_result", {}),
        "markdown_table": final_state.get("evaluation_result", {}).get("markdown_table", ""),
        "scraped_content": search_context,
        "report": final_state.get("report", ""),
        "feedback": final_state.get("feedback", ""),
        "errors": final_state.get("errors", [])
    }


    if not result["success"] and not result["report"]:
        result["error"] = "; ".join(result["errors"]) if result["errors"] else "Pipeline terminated unexpectedly."

    return result




if __name__ == "__main__":
    import sys

    topic_input = input("\nEnter a research topic: ") if len(sys.argv) < 2 else sys.argv[1]

    output = run_research_pipeline(topic_input)

    if output.get("plan"):
        _safe_print("\n[PLAN] VALIDATED RESEARCH PLAN:")
        _safe_print(f"Topic: {output['plan'].get('topic')}")
        _safe_print(f"Objective: {output['plan'].get('objective')}")
        for sq in output['plan'].get('subquestions', []):
            _safe_print(f"  * [{sq.get('focus_area')}] {sq.get('question')} (Search: '{sq.get('search_query')}')")

    if output.get("evidence"):
        ev = output["evidence"]
        _safe_print(f"\n[EVIDENCE] COLLECTED: {ev.get('successful_tasks')}/{ev.get('total_tasks')} tasks successful, {ev.get('unique_sources_count')} unique sources.")

    if output.get("report"):
        _safe_print("\n[REPORT] FINAL REPORT:\n")
        _safe_print(output["report"])

    if output.get("feedback"):
        _safe_print("\n[CRITIC] CRITIC FEEDBACK:\n")
        _safe_print(output["feedback"])

    if output.get("markdown_table"):
        _safe_print("\n" + "=" * 60)
        _safe_print("[EVALUATION] OBSERVABILITY REPORT:")
        _safe_print("=" * 60)
        _safe_print(output["markdown_table"])

    if output.get("errors"):
        _safe_print("\n[WARNINGS/ERRORS]:")
        for err in output["errors"]:
            _safe_print(f"  - {err}")