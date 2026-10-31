import logging
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas import (
    CitationValidationResult,
    ReportMetadata,
    ResearchState,
    SourceMetadata,
    VerdictType,
)

logger = logging.getLogger(__name__)


REPORT_SYSTEM_PROMPT = """You are an elite Technology Strategist and Research Director.
Your task is to deliver an authoritative, polished, and direct executive research report synthesizing real-world evidence from verified sources.

POLISHED WRITING & EDITORIAL STANDARDS:
- **Clean & Direct**: Eliminate unnecessary filler, repetitive bullet points, or raw web-scraping artifacts. Every sentence must deliver high information density.
- **Synthesized Analysis**: Integrate key insights into an elegant, coherent narrative. Connect empirical data to strategic takeaways and real-world impact.
- **Natural Caveat Integration**: Discuss limitations, disputes, and uncertainties seamlessly in clear, natural prose without raw internal verification jargon or debugging notes.
- **Accurate Grounding**: Back every substantive claim with its verified source citation tag [S1], [S2].

MANDATORY SECTIONS (Level 2 ## headings in exact order):
# Research Report: <topic>
## Executive Summary
## Key Findings & Evidence Synthesis
## Verified Claims & Disclosures
## Limitations, Gaps & Uncertainties
## References
"""


def prepare_sources_and_citations(
    raw_sources: List[Any]
) -> Tuple[Dict[str, SourceMetadata], Dict[str, str], str]:
    """
    Assigns standardized source IDs [S1], [S2] to verified source records.
    Returns:
    - id_to_source: Dict mapping 'S1' -> SourceMetadata
    - url_to_id: Dict mapping normalized URL -> 'S1'
    - references_text: Formatted Markdown reference list
    """
    id_to_source: Dict[str, SourceMetadata] = {}
    url_to_id: Dict[str, str] = {}
    ref_lines: List[str] = []

    idx = 1
    for s in raw_sources:
        if isinstance(s, dict):
            url = s.get("url", "").strip()
            title = s.get("title", "") or url
            snippet = s.get("snippet", "")
        elif isinstance(s, SourceMetadata):
            url = s.url.strip()
            title = s.title or url
            snippet = s.snippet
        else:
            continue

        if not url:
            continue

        source_id = f"S{idx}"
        meta = SourceMetadata(source_id=source_id, url=url, title=title, snippet=snippet)
        id_to_source[source_id] = meta
        url_to_id[url.lower().rstrip("/")] = source_id

        ref_lines.append(f"- **[{source_id}]** [{title}]({url})")
        idx += 1

    references_text = "\n".join(ref_lines) if ref_lines else "No verified source records available."
    return id_to_source, url_to_id, references_text


def build_fallback_markdown_report(
    topic: str,
    evidence_text: str,
    verification_summary: str,
    verifications: List[Dict[str, Any]],
    references_text: str,
    id_to_source: Dict[str, SourceMetadata]
) -> str:
    """
    Generates a deterministic, publication-grade Markdown report when LLM is offline or in mock test mode.
    Guarantees inclusion of actual research evidence, citation tags, and contradiction disclosures.
    """
    contradictions = [v for v in verifications if v.get("verdict") == VerdictType.CONTRADICTED]
    insufficient = [v for v in verifications if v.get("verdict") == VerdictType.INSUFFICIENT_EVIDENCE]
    supported = [v for v in verifications if v.get("verdict") in (VerdictType.SUPPORTED, VerdictType.PARTIALLY_SUPPORTED)]

    # Pick available citation tag
    cite_tag = f"[{list(id_to_source.keys())[0]}]" if id_to_source else "[Unreferenced]"

    report_parts = [
        f"# Research Report: {topic}\n",
        "## Executive Summary",
        f"This report presents an evidence-backed synthesis investigating **{topic}**. Findings were gathered across multiple research tasks, evaluated against primary source passages, and audited for factual integrity {cite_tag}.\n",
        "## Key Findings & Evidence Synthesis",
        evidence_text if evidence_text else "No empirical research findings were recorded.",
        "\n## Verified Claims & Disclosures",
        f"### Summary of Verification Audit\n{verification_summary if verification_summary else 'Claims were audited against primary source passages.'}\n"
    ]

    if supported:
        report_parts.append("### Corroborated Findings")
        for s in supported:
            # Attach source citations if matched
            matched = s.get("matched_sources", [])
            s_tags = [f"[{m.get('source_id') or m.get('url')}]" for m in matched if isinstance(m, dict)]
            cite_str = " ".join(s_tags) if s_tags else cite_tag
            report_parts.append(f"- **{s.get('claim_text')}** {cite_str}\n  *Verification Note*: {s.get('reasoning')}")

    if contradictions:
        report_parts.append("\n### Contradictions & Disputed Evidence")
        for c in contradictions:
            report_parts.append(
                f"- ⚠️ **CONTRADICTED CLAIM**: *\"{c.get('claim_text')}\"*\n"
                f"  *Evidence Disclosure*: {c.get('reasoning')}"
            )
    else:
        report_parts.append("\n### Contradictions & Disputed Evidence\nNo explicit factual contradictions were identified among the audited sources.")

    if insufficient:
        report_parts.append("\n### Insufficient Evidence & Unverified Claims")
        for u in insufficient:
            rejected = u.get("rejected_urls", [])
            rejected_note = f" (Rejected invalid references: {', '.join(rejected)})" if rejected else ""
            report_parts.append(f"- ❓ *\"{u.get('claim_text')}\"*{rejected_note}: {u.get('reasoning')}")

    report_parts.extend([
        "\n## Limitations, Gaps & Uncertainties",
        "- Scope constraints: Search queries were targeted to current open web records and academic repositories.",
        "- Incomplete coverage: Claims marked as insufficient evidence require supplementary experimental data.",
        "- Dynamic domain: Rapid development in this topic means findings reflect records available at the time of research.",
        "\n## References",
        references_text
    ])

    return "\n\n".join(report_parts)


def _report_meets_minimum_requirements(
    report_text: str,
    verifications: List[Dict[str, Any]],
) -> bool:
    """Reject LLM drafts that omit required structure or audited negative claims."""
    required_headings = (
        "# Research Report",
        "## Executive Summary",
        "## Key Findings & Evidence Synthesis",
        "## Verified Claims & Disclosures",
        "## Limitations, Gaps & Uncertainties",
        "## References",
    )
    if any(heading not in report_text for heading in required_headings):
        return False

    level_one_headings = []
    level_two_headings = []
    for hashes, title in re.findall(r"^(#{1,2})\s+(.+?)\s*$", report_text, flags=re.MULTILINE):
        if len(hashes) == 1:
            level_one_headings.append(title.strip())
        else:
            level_two_headings.append(title.strip())

    allowed_level_two = {heading.removeprefix("## ") for heading in required_headings[1:]}
    if len(level_one_headings) != 1 or not level_one_headings[0].startswith("Research Report:"):
        return False
    if any(heading not in allowed_level_two for heading in level_two_headings):
        return False
    if any(level_two_headings.count(heading) != 1 for heading in allowed_level_two):
        return False
    heading_positions = [report_text.find(heading) for heading in required_headings]
    if heading_positions != sorted(heading_positions):
        return False

    report_lower = report_text.casefold()
    for verification in verifications:
        if hasattr(verification, "model_dump"):
            verification = verification.model_dump()
        if not isinstance(verification, dict):
            continue

        verdict = str(verification.get("verdict", "")).casefold()
        claim_text = str(verification.get("claim_text", "")).strip()
        if not claim_text:
            continue

        if verdict == VerdictType.CONTRADICTED.casefold():
            if claim_text.casefold() not in report_lower or "contradict" not in report_lower:
                return False
        elif verdict == VerdictType.INSUFFICIENT_EVIDENCE.casefold():
            has_gap_disclosure = any(
                marker in report_lower
                for marker in ("insufficient evidence", "unsupported", "unverified")
            )
            if claim_text.casefold() not in report_lower or not has_gap_disclosure:
                return False

    return True


def _format_verification_records(verifications: List[Dict[str, Any]]) -> str:
    """Pass concise per-claim audit details to the writer model."""
    records = []
    for verification in verifications:
        if hasattr(verification, "model_dump"):
            verification = verification.model_dump()
        if not isinstance(verification, dict):
            continue

        matched_sources = []
        for source in verification.get("matched_sources", []):
            if hasattr(source, "model_dump"):
                source = source.model_dump()
            if isinstance(source, dict):
                matched_sources.append({
                    "source_id": source.get("source_id"),
                    "title": source.get("title"),
                    "url": source.get("url"),
                })

        records.append({
            "claim_id": verification.get("claim_id"),
            "claim_text": verification.get("claim_text", ""),
            "verdict": verification.get("verdict", ""),
            "reasoning": str(verification.get("reasoning", ""))[:500],
            "matched_sources": matched_sources,
            "rejected_urls": verification.get("rejected_urls", []),
        })

    return json.dumps(records, ensure_ascii=False, default=str)[:6000]


def _compact_fallback_evidence(
    evidence_dict: Dict[str, Any],
    compiled_context: str,
    max_chars: int = 10000,
) -> str:
    """Keep deterministic fallback reports readable when search context is large."""
    def escape_source_headings(text: str) -> str:
        return re.sub(
            r"(?m)^(#{1,6})(?=\s)",
            lambda match: "\\" + match.group(1),
            text,
        )

    task_results = evidence_dict.get("results_by_subquestion", [])
    if isinstance(task_results, list) and task_results:
        per_task_limit = max(600, max_chars // len(task_results))
        sections = []
        for result in task_results:
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            if not isinstance(result, dict):
                continue

            heading = f"### Subquestion {result.get('subquestion_id', '')}: {result.get('question', '')}".strip()
            summary = escape_source_headings(
                str(result.get("summary") or result.get("error") or "No findings recorded.").strip()
            )
            section = f"{heading}\n\n{summary}"
            if len(section) > per_task_limit:
                section = section[:per_task_limit].rsplit(" ", 1)[0].rstrip() + " … [excerpt shortened]"
            sections.append(section)

        compact_context = "\n\n---\n\n".join(sections)
        if compact_context:
            return compact_context[:max_chars]

    compiled_context = escape_source_headings(compiled_context)
    if len(compiled_context) <= max_chars:
        return compiled_context
    marker = " … [research context shortened]"
    excerpt_limit = max(0, max_chars - len(marker))
    return compiled_context[:excerpt_limit].rsplit(" ", 1)[0].rstrip() + marker


def _normalize_report_headings(report_text: str) -> str:
    """Keep embedded source headings from becoming report-level sections."""
    allowed_sections = {
        "Executive Summary",
        "Key Findings & Evidence Synthesis",
        "Verified Claims & Disclosures",
        "Limitations, Gaps & Uncertainties",
        "References",
    }
    lines = report_text.splitlines()
    normalized = []
    for line_number, line in enumerate(lines):
        match = re.match(r"^(#{1,2})\s+(.+)$", line)
        if not match:
            normalized.append(line)
            continue

        hashes, title = match.groups()
        is_title = (
            line_number == 0
            and hashes == "#"
            and title.startswith("Research Report:")
        )
        is_allowed_section = hashes == "##" and title.strip() in allowed_sections
        normalized.append(line if is_title or is_allowed_section else f"\\{line}")

    return "\n".join(normalized)


def report_writer_node(
    state: ResearchState,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    LangGraph Report Writer Node:
    - Synthesizes research results and verified claims into structured Markdown sections.
    - Uses standardized source IDs [S1], [S2].
    - Explicitly discloses contradictions and unverified claims.
    """
    topic = state.get("query", "Comprehensive Research Topic").strip()
    evidence_dict = state.get("evidence", {}) or {}
    ver_report = state.get("verification_report", {}) or {}

    raw_sources = []
    if isinstance(evidence_dict, dict):
        raw_sources = evidence_dict.get("unique_sources", [])
    if not raw_sources:
        for res in state.get("research_results", []):
            raw_sources.extend(getattr(res, "sources", []))

    # 1. Format sources with standardized IDs
    id_to_source, url_to_id, references_text = prepare_sources_and_citations(raw_sources)

    compiled_context = evidence_dict.get("compiled_research_context", "") if isinstance(evidence_dict, dict) else ""
    if not compiled_context:
        # Fall back to research results
        results = state.get("research_results", [])
        compiled_context = "\n\n".join([r.summary for r in results if r.summary])

    ver_summary = ver_report.get("summary", "")
    verifications = ver_report.get("verifications", [])

    # Handle missing evidence gracefully
    if not compiled_context and not raw_sources:
        warning_msg = "Report Writer Warning: No research evidence available to synthesize."
        logger.warning(warning_msg)
        fallback_report = (
            f"# Research Report: {topic}\n\n"
            "## Executive Summary\n"
            "Unable to generate empirical findings due to a complete absence of retrieved evidence or search results.\n\n"
            "## Key Findings & Evidence Synthesis\n"
            "*No research findings available.*\n\n"
            "## Limitations, Gaps & Uncertainties\n"
            "All planned research tasks returned insufficient or unretrievable sources. No empirical conclusions can be drawn.\n\n"
            "## References\n"
            "No valid source records retrieved."
        )
        return {
            "report": fallback_report,
            "errors": [warning_msg]
        }

    guidance = state.get("actionable_guidance", "")
    rev_count = state.get("revision_count", 0)

    # 2. Generate report using LLM if available, else deterministic template
    report_text = ""
    if llm is None:
        try:
            from agents import llm as default_llm
            llm = default_llm
        except Exception:
            llm = None

    if llm is not None:
        try:
            rev_prompt_addon = f"\n\nReviewer Feedback & Targeted Guidance (Revision iteration {rev_count}):\n{guidance}\nEnsure all cited issues are resolved." if guidance else ""
            prompt = ChatPromptTemplate.from_messages([
                ("system", REPORT_SYSTEM_PROMPT + """
MANDATORY HEADINGS IN EXACT ORDER:
# Research Report: <topic>
## Executive Summary
## Key Findings & Evidence Synthesis
## Verified Claims & Disclosures
## Limitations, Gaps & Uncertainties
## References

IMPORTANT: Do NOT alter or add any other level 1 (#) or level 2 (##) headings. Use level 3 (###) for any sub-headings."""),
                ("human", f"""Research Topic: {{topic}}

Verified Sources Reference List:
{{references_text}}

Empirical Research Findings:
{{compiled_context}}

Factual Verification Audit:
{{verification_summary}}

Claim-Level Verification Records (preserve contradicted and insufficient-evidence claim text verbatim):
{{verification_records}}
{rev_prompt_addon}

Draft the complete, structured Markdown research report with standardized citations [S1], [S2] and explicit contradiction disclosures:""")
            ])
            chain = prompt | llm | StrOutputParser()
            output = chain.invoke({
                "topic": topic,
                "references_text": references_text,
                "compiled_context": compiled_context[:6000],
                "verification_summary": ver_summary or "Verification audit completed.",
                "verification_records": _format_verification_records(verifications),
            })
            if output and len(output.strip()) > 100:
                report_text = output.strip()
                report_text = re.sub(
                    r"^# Executive Summary\s*$",
                    "## Executive Summary",
                    report_text,
                    count=1,
                    flags=re.MULTILINE,
                )
                if not re.search(r"^# Research Report\b", report_text, flags=re.MULTILINE):
                    report_text = f"# Research Report: {topic}\n\n{report_text}"
                
                # If the LLM omitted the References section, append it
                if "## References" not in report_text and references_text:
                    report_text = f"{report_text}\n\n## References\n{references_text}"

                # Normalize any extra ## subheadings produced by the LLM into ### subheadings
                # so the report maintains valid canonical sections while keeping its rich depth
                canonical_l2 = {
                    "Executive Summary",
                    "Key Findings & Evidence Synthesis",
                    "Verified Claims & Disclosures",
                    "Limitations, Gaps & Uncertainties",
                    "References"
                }
                cleaned_lines = []
                for line in report_text.splitlines():
                    m = re.match(r"^##\s+(.+)$", line)
                    if m and m.group(1).strip() not in canonical_l2:
                        cleaned_lines.append(f"### {m.group(1).strip()}")
                    else:
                        cleaned_lines.append(line)
                report_text = "\n".join(cleaned_lines)

                # Ensure required headings exist in correct order
                required_headings = (
                    "## Executive Summary",
                    "## Key Findings & Evidence Synthesis",
                    "## Verified Claims & Disclosures",
                    "## Limitations, Gaps & Uncertainties",
                    "## References",
                )
                for req_h in required_headings:
                    if req_h not in report_text:
                        report_text = f"{report_text}\n\n{req_h}\nNo findings recorded."

                # Enrich with any missing audited negative or disputed claims to guarantee audit integrity without discarding prose
                missing_audit_lines = []
                report_lower = report_text.casefold()
                for v in verifications:
                    ver_d = v.model_dump() if hasattr(v, "model_dump") else (v if isinstance(v, dict) else {})
                    c_text = str(ver_d.get("claim_text", "")).strip()
                    verdict = str(ver_d.get("verdict", "")).casefold()
                    if not c_text:
                        continue
                    if verdict == VerdictType.CONTRADICTED.casefold() and (c_text.casefold() not in report_lower or "contradict" not in report_lower):
                        missing_audit_lines.append(f"- **Contradicted finding**: *\"{c_text}\"* -- {ver_d.get('reasoning')}")
                    elif verdict == VerdictType.INSUFFICIENT_EVIDENCE.casefold() and (c_text.casefold() not in report_lower or not any(m in report_lower for m in ("insufficient evidence", "unsupported", "unverified"))):
                        missing_audit_lines.append(f"- **Insufficient evidence**: *\"{c_text}\"* -- {ver_d.get('reasoning')}")

                if missing_audit_lines and "## Verified Claims & Disclosures" in report_text:
                    parts = report_text.split("## Verified Claims & Disclosures")
                    injected_block = "\n\n" + "\n".join(missing_audit_lines) + "\n\n"
                    report_text = f"{parts[0]}## Verified Claims & Disclosures{injected_block}{parts[1]}"

                if not _report_meets_minimum_requirements(report_text, verifications):
                    logger.warning(
                        "LLM report omitted required structure or audited negative claims; "
                        "using deterministic synthesis."
                    )
                    report_text = ""
        except Exception as e:
            logger.warning(f"LLM report writing failed ({e}), falling back to structured synthesis.")


    if not report_text:
        fallback_context = _compact_fallback_evidence(evidence_dict, compiled_context)
        report_text = build_fallback_markdown_report(
            topic=topic,
            evidence_text=fallback_context,
            verification_summary=ver_summary,
            verifications=verifications,
            references_text=references_text,
            id_to_source=id_to_source
        )

    report_text = _normalize_report_headings(report_text)
    logger.info(f"Report writer generated Markdown report ({len(report_text)} chars).")
    return {
        "report": report_text,
        "errors": []
    }


def citation_validation_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph Citation Validation Node:
    - Scans the draft report for source citation IDs (e.g. [S1], [S2]) and URLs.
    - Checks every cited source against verified source records from Part 2.
    - Rejects / flags invalid or hallucinated citations.
    - Discloses citation audit results in ReportMetadata.
    """
    report = state.get("report", "")
    evidence_dict = state.get("evidence", {}) or {}

    raw_sources = []
    if isinstance(evidence_dict, dict):
        raw_sources = evidence_dict.get("unique_sources", [])
    if not raw_sources:
        for res in state.get("research_results", []):
            raw_sources.extend(getattr(res, "sources", []))

    id_to_source, url_to_id, _ = prepare_sources_and_citations(raw_sources)
    valid_ids = set(id_to_source.keys())
    valid_urls = set(url_to_id.keys())

    # Find all citation patterns in the report: e.g. [S1], [S2], [S99]
    cited_id_matches = re.findall(r'\[S(\d+)\]', report)
    cited_ids = [f"S{num}" for num in cited_id_matches]

    # Find any Markdown URL links in the body (excluding the References section)
    body_text = report.split("## References")[0] if "## References" in report else report
    raw_urls = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', body_text)

    valid_citations: List[str] = []
    invalid_citations: List[str] = []
    audit_notes: List[str] = []

    # 1. Audit Source ID citations
    for cid in set(cited_ids):
        if cid in valid_ids:
            valid_citations.append(cid)
        else:
            invalid_citations.append(cid)
            audit_notes.append(f"Rejected invalid citation ID [{cid}]: not found in verified source records.")

    # 2. Audit directly cited URLs
    for anchor, url in raw_urls:
        norm = url.lower().rstrip("/")
        if any(norm == v or norm in v for v in valid_urls):
            valid_citations.append(url)
        else:
            invalid_citations.append(url)
            audit_notes.append(f"Rejected unverified citation URL: {url}")

    # 3. Identify uncited sources
    uncited_sources = [sid for sid in valid_ids if sid not in valid_citations]

    total_citations = len(valid_citations) + len(invalid_citations)
    if total_citations > 0:
        score = round(len(valid_citations) / total_citations, 3)
    else:
        score = 1.0 if not invalid_citations else 0.0

    is_valid = len(invalid_citations) == 0

    audit_result = CitationValidationResult(
        total_citations_found=total_citations,
        valid_citations=list(dict.fromkeys(valid_citations)),
        invalid_citations=list(dict.fromkeys(invalid_citations)),
        uncited_sources=uncited_sources,
        citation_integrity_score=score,
        is_valid=is_valid,
        audit_notes=audit_notes
    )

    # 4. If invalid citations exist, replace or annotate them in the report text
    validated_report = report
    if invalid_citations:
        for bad_id in invalid_citations:
            if bad_id.startswith("S"):
                validated_report = re.sub(
                    rf'\[{bad_id}\]',
                    f'[⚠️ Unverified Citation: {bad_id}]',
                    validated_report
                )
            else:
                validated_report = validated_report.replace(bad_id, f"{bad_id} (⚠️ Unverified)")

    # 5. Extract present section headings
    section_headings = re.findall(r'^#{1,3}\s+(.+)$', validated_report, flags=re.MULTILINE)

    # Check for disclosed contradictions and unsupported claims
    disclosed_contradictions = len(re.findall(r'contradict', validated_report, flags=re.IGNORECASE))
    disclosed_unsupported = len(re.findall(r'insufficient evidence|unsupported|unverified', validated_report, flags=re.IGNORECASE))

    report_meta = ReportMetadata(
        topic=state.get("query", ""),
        word_count=len(validated_report.split()),
        sections_present=section_headings,
        disclosed_contradictions_count=disclosed_contradictions,
        disclosed_unsupported_count=disclosed_unsupported,
        citation_audit=audit_result
    )

    logger.info(
        f"Citation validation complete: score={score}, {len(valid_citations)} valid, "
        f"{len(invalid_citations)} invalid."
    )

    return {
        "report": validated_report,
        "report_metadata": report_meta.model_dump(),
        "errors": [note for note in audit_notes] if audit_notes else []
    }
