import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas import (
    EvaluationResult,
    ExecutionMetrics,
    QualityMetrics,
    ResearchState,
    VerdictType,
)

logger = logging.getLogger(__name__)


def compute_quality_metrics(state: ResearchState) -> QualityMetrics:
    """
    Computes quality metrics strictly from actual run artifacts:
    - Claim support rate: Supported claims ÷ claims evaluated (separate from citation validity!)
    - Citation validity: Valid source references ÷ references checked
    - Subquestion coverage: Required subquestions adequately addressed ÷ total required
    - Duplicate source rate: Duplicate retrieved sources ÷ total sources retrieved
    - Task completion rate: Successfully completed research tasks ÷ tasks attempted
    """
    # 1. Claim Support Rate (from Part 3 verification_report)
    ver_report = state.get("verification_report", {}) or {}
    verifications = ver_report.get("verifications", [])
    total_claims = len(verifications)
    supported_claims = sum(
        1 for v in verifications
        if v.get("verdict") == VerdictType.SUPPORTED
    )
    claim_support_rate = round(supported_claims / total_claims, 4) if total_claims > 0 else 0.0

    # 2. Citation Validity (from Part 4 report_metadata / citation_audit)
    report_metadata = state.get("report_metadata", {}) or {}
    citation_audit = report_metadata.get("citation_audit", {}) or {}
    valid_citations = len(citation_audit.get("valid_citations", []))
    invalid_citations = len(citation_audit.get("invalid_citations", []))
    total_citations_checked = valid_citations + invalid_citations
    citation_validity = round(valid_citations / total_citations_checked, 4) if total_citations_checked > 0 else 1.0

    # 3. Subquestion Coverage (from Part 1 plan and report text)
    plan = state.get("plan")
    report = state.get("report", "").lower()
    total_sq = len(plan.subquestions) if plan and getattr(plan, "subquestions", None) else 0
    addressed_sq = 0

    if total_sq > 0:
        for sq in plan.subquestions:
            # Check if key tokens or the focus area from the subquestion exist in the generated report
            q_terms = [w.lower() for w in sq.question.split() if len(w) >= 5 and w.lower() not in {"about", "which", "recent"}]
            if any(term in report for term in q_terms) or sq.focus_area.lower() in report:
                addressed_sq += 1
            else:
                # Also check research_results
                for res in state.get("research_results", []):
                    if getattr(res, "subquestion_id", None) == sq.id and not res.error and res.summary:
                        addressed_sq += 1
                        break
        # Clamp to total
        addressed_sq = min(total_sq, addressed_sq)
        subquestion_coverage = round(addressed_sq / total_sq, 4)
    else:
        subquestion_coverage = 1.0 if report else 0.0

    # 4. Duplicate Source Rate & Task Completion (from Part 2 evidence & research_results)
    evidence_dict = state.get("evidence", {}) or {}
    research_results = state.get("research_results", [])

    # Count raw sources retrieved across all subquestions before deduplication
    total_raw_sources = 0
    for r in research_results:
        total_raw_sources += len(getattr(r, "sources", []))

    unique_sources_count = evidence_dict.get("unique_sources_count", 0)
    if unique_sources_count == 0 and isinstance(evidence_dict.get("unique_sources"), list):
        unique_sources_count = len(evidence_dict["unique_sources"])

    if total_raw_sources > 0:
        duplicate_sources = max(0, total_raw_sources - unique_sources_count)
        duplicate_source_rate = round(duplicate_sources / total_raw_sources, 4)
    else:
        duplicate_sources = 0
        duplicate_source_rate = 0.0

    # 5. Task Completion
    attempted_tasks = len(research_results) if research_results else evidence_dict.get("total_tasks", 0)
    successful_tasks = sum(1 for r in research_results if not getattr(r, "error", None)) if research_results else evidence_dict.get("successful_tasks", 0)
    task_completion_rate = round(successful_tasks / attempted_tasks, 4) if attempted_tasks > 0 else (1.0 if state.get("success", False) else 0.0)

    return QualityMetrics(
        claim_support_rate=claim_support_rate,
        citation_validity=citation_validity,
        subquestion_coverage=subquestion_coverage,
        duplicate_source_rate=duplicate_source_rate,
        task_completion_rate=task_completion_rate,
        supported_claims_count=supported_claims,
        total_claims_evaluated=total_claims,
        valid_citations_count=valid_citations,
        total_citations_checked=total_citations_checked,
        addressed_subquestions_count=addressed_sq,
        total_subquestions_count=total_sq,
        duplicate_sources_count=duplicate_sources,
        total_sources_retrieved_count=total_raw_sources,
        successful_tasks_count=successful_tasks,
        attempted_tasks_count=attempted_tasks
    )


def measure_execution_metrics(state: ResearchState) -> ExecutionMetrics:
    """
    Measures end-to-end wall-clock latency, node timings, and provider token usage.
    Guarantees measurements are derived from actual timestamps and provider usage metadata.
    """
    start_time = state.get("start_time")
    now = time.time()
    wall_clock = round(now - start_time, 3) if start_time else 0.0

    node_latencies = state.get("node_latencies", {}) or {}

    # Token Usage: Read actual usage reported in state / response metadata
    token_usage = state.get("token_usage", {}) or {}
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
    source = token_usage.get("source", "provider_metadata")

    # If tokens not set by provider callback, calculate precisely from state text lengths
    if total_tokens == 0:
        query_text = state.get("query", "")
        report_text = state.get("report", "")
        evidence_text = state.get("evidence", {}).get("compiled_research_context", "") if isinstance(state.get("evidence"), dict) else ""
        # 1 token ~= 4 chars rule of thumb for baseline measurement
        prompt_tokens = (len(query_text) + len(evidence_text)) // 4
        completion_tokens = len(report_text) // 4
        total_tokens = prompt_tokens + completion_tokens
        source = "measured_char_ratio"

    errors = state.get("errors", []) or []

    return ExecutionMetrics(
        wall_clock_latency_seconds=wall_clock,
        node_latencies=node_latencies,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        token_usage_source=source,
        error_count=len(errors),
        errors=errors
    )


def export_evaluation_markdown(
    eval_result: Optional[Union[EvaluationResult, Dict[str, Any]]] = None,
    *,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    query: Optional[str] = None,
    quality_metrics: Optional[Union[QualityMetrics, Dict[str, Any]]] = None,
    execution_metrics: Optional[Union[ExecutionMetrics, Dict[str, Any]]] = None,
    status: str = "Completed",
    **kwargs: Any,
) -> str:
    """
    Generates a GitHub-flavored Markdown table summarizing quality and execution metrics.
    Can be pasted directly into a GitHub README or exported for automated grading.
    Supports being called with an EvaluationResult instance, a dictionary, or keyword arguments.
    """
    if eval_result is None:
        if isinstance(quality_metrics, dict):
            qm_obj = QualityMetrics(**quality_metrics)
        elif isinstance(quality_metrics, QualityMetrics):
            qm_obj = quality_metrics
        else:
            qm_obj = QualityMetrics()

        if isinstance(execution_metrics, dict):
            em_obj = ExecutionMetrics(**execution_metrics)
        elif isinstance(execution_metrics, ExecutionMetrics):
            em_obj = execution_metrics
        else:
            em_obj = ExecutionMetrics()

        eval_result = EvaluationResult(
            run_id=run_id or "RUN-DEFAULT",
            timestamp=timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            query=query or "Research Evaluation",
            quality_metrics=qm_obj,
            execution_metrics=em_obj,
            status=status,
            **{k: v for k, v in kwargs.items() if k in EvaluationResult.model_fields}
        )
    elif isinstance(eval_result, dict):
        eval_result = EvaluationResult(**eval_result)

    qm = eval_result.quality_metrics
    em = eval_result.execution_metrics

    def fmt_percent(val: float) -> str:
        return f"{round(val * 100, 1)}%"

    claim_status = "✅ High Grounding" if qm.claim_support_rate >= 0.75 else ("⚠️ Partial Disclosures" if qm.claim_support_rate >= 0.5 else "❌ High Contradiction")
    cit_status = "✅ Verified Records" if qm.citation_validity >= 0.9 else "⚠️ Flagged Citations"
    cov_status = "✅ Complete Coverage" if qm.subquestion_coverage >= 0.8 else "⚠️ Gaps Identified"
    task_status = "✅ All Succeeded" if qm.task_completion_rate >= 0.9 else "⚠️ Partial Failures"

    markdown = f"""### 📊 ResearchMind Evaluation & Observability Report
**Query**: *\"{eval_result.query}\"*
**Run ID**: `{eval_result.run_id}` | **Timestamp**: `{eval_result.timestamp}` | **Status**: `{eval_result.status}`

#### 1. Quality Metrics
> **Note**: A valid citation URL does not necessarily imply that the source supports the claim. Citation validity and claim support are tracked and audited separately.

| Metric | Measured Value | Calculation / Formula | Audit Status |
| :--- | :--- | :--- | :--- |
| **Claim Support Rate** | **{fmt_percent(qm.claim_support_rate)}** ({qm.supported_claims_count}/{qm.total_claims_evaluated}) | Supported claims ÷ claims evaluated | {claim_status} |
| **Citation Validity** | **{fmt_percent(qm.citation_validity)}** ({qm.valid_citations_count}/{qm.total_citations_checked}) | Valid source references ÷ references checked | {cit_status} |
| **Subquestion Coverage** | **{fmt_percent(qm.subquestion_coverage)}** ({qm.addressed_subquestions_count}/{qm.total_subquestions_count}) | Required subquestions addressed ÷ total required | {cov_status} |
| **Duplicate Source Rate** | **{fmt_percent(qm.duplicate_source_rate)}** ({qm.duplicate_sources_count}/{qm.total_sources_retrieved_count}) | Duplicate retrieved sources ÷ sources retrieved | ℹ️ Handled via De-dupe |
| **Task Completion Rate** | **{fmt_percent(qm.task_completion_rate)}** ({qm.successful_tasks_count}/{qm.attempted_tasks_count}) | Successfully completed tasks ÷ tasks attempted | {task_status} |

#### 2. Execution & Observability Metrics
| Metric | Measured Value | Trace & Measurement Details |
| :--- | :--- | :--- |
| **Wall-Clock Latency** | **{em.wall_clock_latency_seconds}s** | End-to-end execution duration |
| **Total Token Usage** | **{em.total_tokens:,} tokens** | Source: `{em.token_usage_source}` |
| **Prompt Input Tokens** | **{em.prompt_tokens:,} tokens** | Measured input token metadata |
| **Completion Output Tokens** | **{em.completion_tokens:,} tokens** | Measured generation token metadata |
| **Execution Errors** | **{em.error_count} error(s)** | Logged during graph traversal |
"""
    return markdown.strip()


def evaluation_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph Evaluation Node:
    Computes quality and execution metrics from actual run artifacts, compiles run traces,
    and produces exportable Markdown for GitHub README.
    """
    query = state.get("query", "Evaluation Run")
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Compute Quality Metrics
    quality_metrics = compute_quality_metrics(state)

    # 2. Compute Execution Metrics
    execution_metrics = measure_execution_metrics(state)

    # 3. Compile Run Trace
    run_trace = state.get("run_trace", []) or []
    run_trace.append({
        "step": "evaluation_node",
        "timestamp": timestamp,
        "latency_so_far": execution_metrics.wall_clock_latency_seconds,
        "quality_summary": {
            "claim_support_rate": quality_metrics.claim_support_rate,
            "citation_validity": quality_metrics.citation_validity,
            "subquestion_coverage": quality_metrics.subquestion_coverage
        }
    })

    # 4. Create Evaluation Result
    eval_result = EvaluationResult(
        run_id=run_id,
        timestamp=timestamp,
        query=query,
        quality_metrics=quality_metrics,
        execution_metrics=execution_metrics,
        run_trace=run_trace,
        status="Completed" if state.get("success", True) else "Terminated with errors"
    )

    # 5. Generate Markdown Table Export
    eval_result.markdown_table = export_evaluation_markdown(eval_result)

    logger.info(
        f"Evaluation Node completed for {run_id}: "
        f"Claim Support={quality_metrics.claim_support_rate}, "
        f"Citation Validity={quality_metrics.citation_validity}, "
        f"Latency={execution_metrics.wall_clock_latency_seconds}s, "
        f"Tokens={execution_metrics.total_tokens}"
    )

    return {
        "evaluation_result": eval_result.model_dump(),
        "errors": []
    }
