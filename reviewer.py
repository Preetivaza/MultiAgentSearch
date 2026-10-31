import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from schemas import (
    ResearchPlan,
    ResearchState,
    ReviewAssessment,
    VerdictType,
)
from tools import web_search

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a rigorous Scientific Editor and Senior Peer Reviewer.
Evaluate the draft research report on four strict dimensions:
1. Coverage: Are the research query and planned subquestions adequately addressed?
2. Evidence Quality: Is the text grounded in empirical facts and citations rather than unsupported generalizations?
3. Citation Integrity: Are citations accurate and free of invalid or unverified references?
4. Contradictions: Are conflicting findings and contradictory evidence explicitly disclosed?

Assessment Rules:
- If there are fixable issues (e.g. invalid citations, undisclosed contradictions, missing major sections), set status to 'Needs revision'.
- Choose revision_target:
  - 'rewriter' if issues can be fixed by text rephrasing, proper disclosures, or citation cleanup.
  - 'researcher' if essential empirical evidence is missing and requires new searches.
- If the report meets high publication standards, set status to 'Pass' and revision_target to 'none'.
- Provide concrete, actionable_guidance for the revision.
"""


def evaluate_report_heuristics(
    report: str,
    query: str,
    plan: Optional[ResearchPlan],
    verification_report: Dict[str, Any],
    report_metadata: Dict[str, Any]
) -> ReviewAssessment:
    """
    Deterministic rule-based reviewer evaluation for offline test suites and LLM fallback.
    Strictly audits Coverage, Evidence, Citations, and Contradictions.
    """
    issues: List[str] = []
    coverage_score = 9.0
    evidence_score = 9.0
    citation_score = 9.0
    contradiction_score = 9.0

    report_lower = report.lower() if report else ""

    # 1. Coverage Check
    required_sections = ["executive summary", "key findings", "limitations", "references"]
    for sec in required_sections:
        if sec not in report_lower:
            issues.append(f"Missing required report section: '{sec.title()}'.")
            coverage_score -= 2.0

    if plan and hasattr(plan, "subquestions") and plan.subquestions:
        unaddressed_sq = 0
        for sq in plan.subquestions:
            # Check if key words from subquestion appear in report
            kw = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', sq.question.lower()) if w not in {"what", "which", "recent", "about"}]
            if kw and not any(k in report_lower for k in kw):
                unaddressed_sq += 1
        if unaddressed_sq > 0:
            issues.append(f"{unaddressed_sq} planned subquestion(s) lack substantive coverage.")
            coverage_score -= 1.5 * unaddressed_sq

    # 2. Evidence Check
    if len(report.split()) < 50:
        issues.append("Report is excessively brief and lacks depth.")
        evidence_score -= 3.0

    # 3. Citation Integrity Check
    citation_audit = report_metadata.get("citation_audit", {}) if isinstance(report_metadata, dict) else {}
    if citation_audit:
        invalid_citations = citation_audit.get("invalid_citations", [])
        integrity_score = citation_audit.get("citation_integrity_score", 1.0)
        if invalid_citations or integrity_score < 0.85:
            issues.append(f"Citation audit flagged {len(invalid_citations)} unverified citation(s) (Integrity score: {integrity_score}).")
            citation_score -= 3.5

    # Check for unverified citation markers in text
    if "⚠️ unverified" in report_lower:
        issues.append("Report text still contains unverified citation warnings that must be excised or corrected.")
        citation_score -= 2.0

    # 4. Contradiction Check
    verifications = verification_report.get("verifications", []) if isinstance(verification_report, dict) else []
    contradicted_claims = [v for v in verifications if v.get("verdict") == VerdictType.CONTRADICTED]
    if contradicted_claims:
        # Report MUST disclose contradictions
        if "contradict" not in report_lower and "dispute" not in report_lower:
            issues.append(f"Report failed to disclose {len(contradicted_claims)} known contradictory claim(s) identified in verification.")
            contradiction_score -= 4.0

    coverage_score = max(0.0, min(10.0, coverage_score))
    evidence_score = max(0.0, min(10.0, evidence_score))
    citation_score = max(0.0, min(10.0, citation_score))
    contradiction_score = max(0.0, min(10.0, contradiction_score))

    overall_score = round((coverage_score + evidence_score + citation_score + contradiction_score) / 4.0, 1)

    # Decision logic
    if issues or overall_score < 7.5:
        status = "Needs revision"
        # Determine target
        has_citation_or_contradiction_issue = any("citation" in i.lower() or "contradict" in i.lower() for i in issues)
        if any("unaddressed" in i.lower() or "missing" in i.lower() for i in issues) and len(report.split()) < 40 and not has_citation_or_contradiction_issue:
            target = "researcher"
            guidance = "Conduct targeted research on unaddressed subquestions and expand empirical findings."
        else:
            target = "rewriter"
            guidance = "Address the following issues: " + "; ".join(issues)
        feedback = f"Report requires revisions before publication. Issues: {'; '.join(issues)}"
    else:
        status = "Pass"
        target = "none"
        guidance = ""
        feedback = "The report meets rigorous academic coverage, citation integrity, and evidence standards."

    return ReviewAssessment(
        status=status,
        overall_score=overall_score,
        coverage_score=coverage_score,
        evidence_score=evidence_score,
        citation_score=citation_score,
        contradiction_score=contradiction_score,
        revision_target=target,
        feedback=feedback,
        actionable_guidance=guidance,
        issues_identified=issues
    )


def build_reviewer_chain(llm: Any):
    """Builds structured LangChain reviewer chain using Google Gemini."""
    try:
        structured_llm = llm.with_structured_output(ReviewAssessment)
        prompt = ChatPromptTemplate.from_messages([
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", """Research Topic: {topic}

Report Text:
{report}

Verification Findings:
{verification_summary}

Citation Audit:
{citation_audit}

Provide your structured assessment with status ('Pass' or 'Needs revision'), scores, and actionable guidance:""")
        ])
        return prompt | structured_llm
    except Exception:
        return None


def reviewer_node(
    state: ResearchState,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    LangGraph Reviewer Node:
    Conducts comprehensive review of coverage, evidence, citations, and contradictions.
    Returns a structured ReviewAssessment.
    """
    report = state.get("report", "")
    query = state.get("query", "")
    plan = state.get("plan")
    ver_report = state.get("verification_report", {}) or {}
    report_metadata = state.get("report_metadata", {}) or {}

    assessment: Optional[ReviewAssessment] = None

    if llm is None:
        try:
            from agents import llm as default_llm
            llm = default_llm
        except Exception:
            llm = None

    if llm is not None:
        try:
            chain = build_reviewer_chain(llm)
            if chain:
                output = chain.invoke({
                    "topic": query,
                    "report": report[:6000],
                    "verification_summary": ver_report.get("summary", ""),
                    "citation_audit": json.dumps(report_metadata.get("citation_audit", {}))
                })
                if isinstance(output, ReviewAssessment):
                    assessment = output
                elif isinstance(output, dict):
                    assessment = ReviewAssessment.model_validate(output)
        except Exception as e:
            logger.warning(f"LLM reviewer chain failed ({e}), using heuristic evaluation.")

    if assessment is None:
        assessment = evaluate_report_heuristics(
            report=report,
            query=query,
            plan=plan,
            verification_report=ver_report,
            report_metadata=report_metadata
        )

    logger.info(
        f"Reviewer Node completed: status='{assessment.status}', "
        f"score={assessment.overall_score}/10, target='{assessment.revision_target}'"
    )

    return {
        "review_assessment": assessment.model_dump(),
        "errors": []
    }


def targeted_revision_node(
    state: ResearchState,
    search_func: Callable[[str], Any] = web_search
) -> Dict[str, Any]:
    """
    LangGraph Targeted Revision Node:
    - Increments revision_count.
    - If revision_target is 'researcher', conducts targeted searches to fill evidence gaps.
    - Supplies actionable_guidance to writer and revalidates state.
    """
    current_count = state.get("revision_count", 0)
    new_count = current_count + 1
    assessment_dict = state.get("review_assessment", {}) or {}
    target = assessment_dict.get("revision_target", "rewriter")
    guidance = assessment_dict.get("actionable_guidance", "")
    query = state.get("query", "")

    logger.info(f"Executing targeted revision iteration {new_count} (target='{target}').")

    new_errors: List[str] = []

    # If targeted research is requested, run search on the missing areas
    if target == "researcher" and guidance:
        try:
            targeted_search_query = f"{query} {guidance[:40]}"
            logger.info(f"Running targeted research query: '{targeted_search_query}'")
            raw_res = search_func(targeted_search_query)
            # Update research context with new findings
            evidence = state.get("evidence", {}) or {}
            compiled = evidence.get("compiled_research_context", "")
            compiled += f"\n\n### Targeted Revision Research:\n{str(raw_res)[:1500]}"
            evidence["compiled_research_context"] = compiled
        except Exception as e:
            new_errors.append(f"Targeted research failed: {str(e)}")

    return {
        "revision_count": new_count,
        "actionable_guidance": guidance,
        "errors": new_errors
    }


def route_after_review(state: ResearchState) -> str:
    """
    Conditional Routing after Reviewer Node:
    - If status == 'Pass' OR revision_count >= max_revisions:
      -> routes to 'publish' (stops loop unconditionally).
    - If status == 'Needs revision':
      -> routes to 'targeted_revision' to fix issues and revalidate.
    """
    assessment_dict = state.get("review_assessment", {}) or {}
    status = assessment_dict.get("status", "Pass")
    target = assessment_dict.get("revision_target", "rewriter")

    rev_count = state.get("revision_count", 0)
    max_rev = state.get("max_revisions", 2)

    logger.info(f"Routing after review: status='{status}', iteration={rev_count}/{max_rev}")

    # Strict termination check: pass or limit reached
    if status == "Pass" or rev_count >= max_rev:
        return "publish"

    # Route to targeted revision (which feeds back to rewriter or researcher)
    if target == "researcher":
        return "targeted_research"
    else:
        return "targeted_rewrite"
