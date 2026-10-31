import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set
from schemas import (
    EvidenceCollection,
    ResearchPlan,
    ResearchState,
    SourceMetadata,
    SubQuestion,
    SubQuestionResult,
)
from tools import web_search

logger = logging.getLogger(__name__)


def parse_search_output(raw_output: Any) -> List[SourceMetadata]:
    """
    Parses various outputs from web_search tools (JSON string, dict, or list)
    and extracts a clean list of SourceMetadata objects with url, title, snippet.
    """
    sources: List[SourceMetadata] = []

    if isinstance(raw_output, str):
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Handle plain text output by creating a single snippet if URL is present
            return sources
    elif isinstance(raw_output, dict):
        data = raw_output
    elif isinstance(raw_output, list):
        data = {"results": raw_output}
    else:
        return sources

    # Check for error in JSON response
    if isinstance(data, dict) and not data.get("success", True) and "error" in data:
        raise RuntimeError(f"Search tool returned error: {data.get('error')}")

    results = data.get("results", []) if isinstance(data, dict) else []

    for item in results:
        if isinstance(item, dict):
            url = item.get("url", "").strip()
            title = item.get("title", "").strip()
            snippet = item.get("snippet", item.get("content", "")).strip()

            if url and url.startswith(("http://", "https://")):
                sources.append(
                    SourceMetadata(
                        url=url,
                        title=title or url,
                        snippet=snippet
                    )
                )

    return sources


def execute_subquestion_research(
    subquestion: SubQuestion,
    search_func: Callable[[str], Any] = web_search
) -> SubQuestionResult:
    """
    Executes research for an individual subquestion:
    - Runs the search query
    - Extracts source metadata (URLs, titles, excerpts)
    - Resilient: if search fails, records the error without raising an unhandled exception.
    """
    try:
        logger.info(f"Researching Subquestion {subquestion.id}: {subquestion.question}")
        # LangChain tools expect structured input through invoke(); plain callables
        # (used by tests and custom integrations) continue to receive the query.
        if hasattr(search_func, "invoke"):
            raw_result = search_func.invoke({"query": subquestion.search_query})
        else:
            raw_result = search_func(subquestion.search_query)

        if hasattr(raw_result, "content"):
            raw_result = raw_result.content

        sources = parse_search_output(raw_result)

        # Build a concise excerpt summary
        summary_snippets = [f"[{s.title}] ({s.url}): {s.snippet}" for s in sources if s.snippet]
        summary = "\n\n".join(summary_snippets) if summary_snippets else "No relevant excerpts retrieved."

        return SubQuestionResult(
            subquestion_id=subquestion.id,
            question=subquestion.question,
            search_query=subquestion.search_query,
            sources=sources,
            summary=summary,
            error=None
        )

    except Exception as e:
        error_msg = f"Failed to execute research for subquestion {subquestion.id} ('{subquestion.question}'): {str(e)}"
        logger.error(error_msg)
        return SubQuestionResult(
            subquestion_id=subquestion.id,
            question=subquestion.question,
            search_query=subquestion.search_query,
            sources=[],
            summary="",
            error=error_msg
        )


def orchestrate_research_tasks(
    subquestions: List[SubQuestion],
    search_func: Callable[[str], Any] = web_search,
    max_workers: int = 4
) -> List[SubQuestionResult]:
    """
    Executes multiple independent research tasks concurrently using ThreadPoolExecutor.
    Maintains deterministic ordering by subquestion ID in the output list.
    """
    results_map: Dict[int, SubQuestionResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sq = {
            executor.submit(execute_subquestion_research, sq, search_func): sq
            for sq in subquestions
        }

        for future in as_completed(future_to_sq):
            sq = future_to_sq[future]
            try:
                res = future.result()
                results_map[sq.id] = res
            except Exception as e:
                # Fallback if future execution itself crashed
                results_map[sq.id] = SubQuestionResult(
                    subquestion_id=sq.id,
                    question=sq.question,
                    search_query=sq.search_query,
                    sources=[],
                    summary="",
                    error=f"Task execution crash: {str(e)}"
                )

    # Return results sorted by subquestion ID
    return [results_map[sq.id] for sq in subquestions if sq.id in results_map]


def collect_evidence(
    results: List[SubQuestionResult]
) -> EvidenceCollection:
    """
    Evidence Collector:
    - Combines results across all subquestions
    - Preserves source URLs and relevant excerpts
    - Deduplicates sources by URL across subquestions
    - Preserves successful results even if other tasks failed
    - Formats a unified compiled research context for downstream drafting
    """
    unique_sources_map: Dict[str, SourceMetadata] = {}
    compiled_blocks: List[str] = []
    successful_count = 0
    failed_count = 0

    for res in results:
        if res.error:
            failed_count += 1
            compiled_blocks.append(
                f"### Subquestion {res.subquestion_id}: {res.question}\n"
                f"*Status*: Failed to retrieve sources ({res.error})\n"
            )
            continue

        successful_count += 1
        block_lines = [
            f"### Subquestion {res.subquestion_id}: {res.question}",
            f"*Search Query*: `{res.search_query}`",
            "*Findings*:",
            res.summary or "No findings recorded.",
            "*Sources Cited*:"
        ]

        if res.sources:
            for s in res.sources:
                block_lines.append(f"- [{s.title}]({s.url})")
                # Deduplicate sources globally across subquestions
                if s.url not in unique_sources_map:
                    unique_sources_map[s.url] = s
                elif not unique_sources_map[s.url].snippet and s.snippet:
                    # Enrich snippet if previous duplicate had empty snippet
                    unique_sources_map[s.url].snippet = s.snippet
        else:
            block_lines.append("- (No direct URLs returned)")

        compiled_blocks.append("\n".join(block_lines))

    return EvidenceCollection(
        total_tasks=len(results),
        successful_tasks=successful_count,
        failed_tasks=failed_count,
        unique_sources_count=len(unique_sources_map),
        unique_sources=list(unique_sources_map.values()),
        results_by_subquestion=results,
        compiled_research_context="\n\n---\n\n".join(compiled_blocks)
    )


# ── LangGraph Node Functions ──────────────────────────────────────────────────

def research_orchestrator_node(
    state: ResearchState,
    search_func: Callable[[str], Any] = web_search
) -> Dict[str, Any]:
    """
    LangGraph Research Orchestrator Node.
    Dispatches research tasks for each planned subquestion concurrently.
    """
    plan: Optional[ResearchPlan] = state.get("plan")

    if not plan or not plan.subquestions:
        error_msg = "Research Orchestrator Error: No validated research plan or subquestions found in state."
        logger.error(error_msg)
        return {
            "research_results": [],
            "errors": [error_msg]
        }

    results = orchestrate_research_tasks(
        subquestions=plan.subquestions,
        search_func=search_func
    )

    # Collect errors from any failed subquestions without dropping successful results
    new_errors = [r.error for r in results if r.error]

    return {
        "research_results": results,
        "errors": new_errors
    }


def evidence_collector_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph Evidence Collector Node.
    Combines research results, deduplicates sources, and preserves source metadata.
    """
    results: List[SubQuestionResult] = state.get("research_results", [])

    if not results:
        warning_msg = "Evidence Collector: No subquestion research results available to collect."
        logger.warning(warning_msg)
        return {
            "evidence": EvidenceCollection().model_dump(),
            "errors": [warning_msg]
        }

    evidence = collect_evidence(results)

    logger.info(
        f"Collected evidence: {evidence.successful_tasks}/{evidence.total_tasks} tasks successful, "
        f"{evidence.unique_sources_count} unique sources found."
    )

    return {
        "evidence": evidence.model_dump(),
        "errors": []
    }
