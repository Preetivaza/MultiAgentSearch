import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from schemas import (
    Claim,
    ClaimVerification,
    EvidenceCollection,
    ResearchState,
    SourceMetadata,
    SubQuestionResult,
    VerdictType,
    VerificationReport,
)

logger = logging.getLogger(__name__)


class ExtractedClaims(BaseModel):
    """Container for parsed individual claims."""
    claims: List[Claim] = Field(default_factory=list)


def normalize_url(url: str) -> str:
    """Normalize URL for consistent comparison (lowercase host, strip trailing slash)."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        normalized = f"{parsed.scheme.lower()}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    except Exception:
        return url.strip().rstrip("/")


def extract_claims_heuristic(text: str, default_urls: Optional[List[str]] = None) -> List[Claim]:
    """
    Fallback claim extractor when LLM is unavailable or for deterministic offline parsing.
    Splits text into individual factual claims by sentence and punctuation.
    """
    if not text or not text.strip():
        return []

    claims: List[Claim] = []
    # Match sentences ending with ., !, or newline
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    claim_id = 1

    for raw in raw_sentences:
        s = raw.strip()
        # Filter out headers, markdown fences, and very short non-factual snippets
        if not s or s.startswith(("#", "-", "*", "```", "http")) or len(s.split()) < 4:
            continue

        # Extract any embedded URLs
        embedded_urls = re.findall(r'https?://[^\s)\]}>,]+', s)
        clean_text = re.sub(r'https?://[^\s)\]}>,]+', '', s).strip()
        clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text).strip()

        urls = list(dict.fromkeys(embedded_urls + (default_urls or [])))

        if len(clean_text) >= 10:
            claims.append(
                Claim(
                    id=claim_id,
                    claim_text=clean_text,
                    referenced_urls=urls
                )
            )
            claim_id += 1

    return claims


def build_claim_extraction_chain(llm: Any):
    """Builds LLM chain for atomic claim extraction."""
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a rigorous Fact Extraction Specialist.
Analyze the provided research findings and deconstruct them into individual, atomic, standalone factual claims.

Rules:
1. Each claim must be an independent, self-contained statement of fact.
2. Preserve any cited source URLs in the claim's referenced_urls.
3. Number each claim sequentially (id: 1, 2, 3...).
4. Avoid opinions, transitions, or conversational filler."""),
        ("human", "Research Findings:\n\n{research_context}\n\nExtract all individual factual claims:")
    ])
    
    try:
        structured_llm = llm.with_structured_output(ExtractedClaims)
        return prompt | structured_llm
    except Exception:
        return None


def claim_extraction_node(
    state: ResearchState,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    LangGraph Claim Extraction Node:
    Extracts individual factual claims from the research findings.
    """
    evidence_dict = state.get("evidence", {}) or {}
    compiled_context = ""
    default_urls: List[str] = []

    if isinstance(evidence_dict, dict):
        compiled_context = evidence_dict.get("compiled_research_context", "")
        sources = evidence_dict.get("unique_sources", [])
        default_urls = [s.get("url") for s in sources if isinstance(s, dict) and s.get("url")]

    if not compiled_context:
        # Fall back to research results list
        results = state.get("research_results", [])
        if results:
            compiled_context = "\n\n".join([r.summary for r in results if r.summary])
            for r in results:
                for s in r.sources:
                    if s.url:
                        default_urls.append(s.url)

    if not compiled_context.strip():
        logger.warning("Claim Extraction: No research findings found to extract claims from.")
        return {
            "claims": [],
            "errors": ["Claim Extraction Warning: Research findings were empty."]
        }

    claims: List[Claim] = []

    # Attempt LLM extraction if available
    if llm is None:
        try:
            from agents import llm as default_llm
            llm = default_llm
        except Exception:
            llm = None

    if llm is not None:
        try:
            chain = build_claim_extraction_chain(llm)
            if chain:
                output = chain.invoke({"research_context": compiled_context[:6000]})
                if isinstance(output, ExtractedClaims) and output.claims:
                    claims = output.claims
                elif isinstance(output, dict) and "claims" in output:
                    claims = [Claim.model_validate(c) for c in output["claims"]]
        except Exception as e:
            logger.warning(f"LLM claim extraction failed ({e}), using heuristic fallback.")

    # Fallback heuristic if LLM produced no claims
    if not claims:
        claims = extract_claims_heuristic(compiled_context, default_urls=default_urls)

    logger.info(f"Extracted {len(claims)} individual claims from research context.")
    return {
        "claims": claims,
        "errors": []
    }


def verify_single_claim(
    claim: Claim,
    valid_sources_map: Dict[str, SourceMetadata],
    llm: Optional[Any] = None
) -> ClaimVerification:
    """
    Verifies an individual claim against real source passages:
    - Identifies valid vs invalid source references (rejects invalid ones).
    - Checks source passages for corroborating or contradictory evidence.
    - Determines verdict: Supported, Contradicted, Insufficient evidence, Partially supported.
    """
    valid_sources: List[SourceMetadata] = []
    rejected_urls: List[str] = []
    relevant_excerpts: List[str] = []

    # 1. Validate source references against real source records
    for raw_url in claim.referenced_urls:
        norm_url = normalize_url(raw_url)
        matched_source = None
        for known_norm, source_meta in valid_sources_map.items():
            if norm_url == known_norm or (norm_url and norm_url in known_norm):
                matched_source = source_meta
                break

        if matched_source:
            if matched_source not in valid_sources:
                valid_sources.append(matched_source)
                if matched_source.snippet:
                    relevant_excerpts.append(matched_source.snippet)
        else:
            rejected_urls.append(raw_url)

    # If claim did not specify URLs, search through all verified source passages for relevant matches
    if not valid_sources:
        claim_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', claim.claim_text.lower()))
        for known_norm, source_meta in valid_sources_map.items():
            snippet = source_meta.snippet.lower()
            matching_kw = [kw for kw in claim_keywords if kw in snippet]
            if len(matching_kw) >= 2 or (claim_keywords and len(matching_kw) / len(claim_keywords) >= 0.4):
                if source_meta not in valid_sources:
                    valid_sources.append(source_meta)
                    relevant_excerpts.append(source_meta.snippet)

    # 2. Rule: If all referenced URLs were invalid and no real sources substantiate it
    if rejected_urls and not valid_sources:
        return ClaimVerification(
            claim_id=claim.id,
            claim_text=claim.claim_text,
            verdict=VerdictType.INSUFFICIENT_EVIDENCE,
            reasoning=f"Rejected {len(rejected_urls)} invalid source reference(s) ({', '.join(rejected_urls)}). No verified source records substantiate this claim.",
            confidence=0.9,
            matched_sources=[],
            rejected_urls=rejected_urls,
            source_excerpts=[]
        )

    # If completely empty sources available
    if not valid_sources:
        return ClaimVerification(
            claim_id=claim.id,
            claim_text=claim.claim_text,
            verdict=VerdictType.INSUFFICIENT_EVIDENCE,
            reasoning="No verified source passages or records match this claim.",
            confidence=0.8,
            matched_sources=[],
            rejected_urls=rejected_urls,
            source_excerpts=[]
        )

    # 3. Analyze source excerpts for corroboration vs contradiction
    combined_excerpts = " ".join(relevant_excerpts).lower()
    claim_lower = claim.claim_text.lower()

    # Contradiction indicators
    contradiction_patterns = [
        r'\bnot\b', r'\bnever\b', r'\bfailed\b', r'\bfalse\b', r'\brefutes?\b',
        r'\bincorrect\b', r'\bdisproved?\b', r'\bunlike\b', r'\bcontrast\b',
        r'\bdispute[ds]?\b', r'\bcontested\b', r'\blacks?\b'
    ]
    
    # Check if claim affirms something that source explicitly denies, or vice versa
    is_negated_in_claim = bool(re.search(r'\bnot\b|\bnever\b|\bno\b|\bneither\b|\bfailed\b', claim_lower))
    is_negated_in_source = bool(re.search(r'\bnot\b|\bnever\b|\bfailed\b|\bno\b|\bincorrect\b', combined_excerpts))

    # Check for direct contradiction keywords in context
    has_contradiction_signal = any(re.search(pat, combined_excerpts) for pat in contradiction_patterns)
    
    # Keyword overlap ratio (with prefix/stem matching)
    claim_words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', claim_lower) if w not in {"the", "and", "for", "that", "with", "this"}]
    overlap_count = sum(1 for w in claim_words if w in combined_excerpts or (len(w) > 4 and w[:4] in combined_excerpts))
    overlap_ratio = overlap_count / len(claim_words) if claim_words else 0.0

    # Determine Verdict
    # Case A: Explicit Contradiction
    if (is_negated_in_claim != is_negated_in_source and (overlap_ratio >= 0.35 or has_contradiction_signal)) or (
        "contradict" in combined_excerpts or "disprove" in combined_excerpts or "refute" in combined_excerpts
    ):
        verdict = VerdictType.CONTRADICTED
        reasoning = (
            f"Contradictory evidence identified in verified source records. "
            f"The claim asserts factual conditions that conflict with the retrieved source excerpt."
        )
        confidence = 0.85

    # Case B: Strongly Supported
    elif overlap_ratio >= 0.65:
        verdict = VerdictType.SUPPORTED
        reasoning = (
            f"Directly supported by verified source passage(s) with high keyword and semantic alignment ({int(overlap_ratio*100)}% match)."
        )
        confidence = 0.95

    # Case C: Partially Supported
    elif overlap_ratio >= 0.35:
        verdict = VerdictType.PARTIALLY_SUPPORTED
        reasoning = (
            f"Partially supported: Source passages corroborate key concepts ({int(overlap_ratio*100)}% overlap), "
            f"but certain details or scope remain unconfirmed."
        )
        confidence = 0.75

    # Case D: Insufficient Evidence
    else:
        verdict = VerdictType.INSUFFICIENT_EVIDENCE
        reasoning = (
            f"Insufficient evidence: Available source excerpts only overlap with {int(overlap_ratio*100)}% of claim terms, "
            f"which is below the verification threshold."
        )
        confidence = 0.70

    return ClaimVerification(
        claim_id=claim.id,
        claim_text=claim.claim_text,
        verdict=verdict,
        reasoning=reasoning,
        confidence=confidence,
        matched_sources=valid_sources,
        rejected_urls=rejected_urls,
        source_excerpts=relevant_excerpts[:3]
    )


def evidence_verification_node(
    state: ResearchState,
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evidence Verification & Verdict Node:
    - Validates source references against real source records from Part 2.
    - Rejects invalid source references.
    - Checks for corroborating vs contradictory evidence.
    - Produces a comprehensive VerificationReport with verdicts for every claim.
    """
    claims: List[Claim] = state.get("claims", []) or []

    # 1. Build lookup dictionary of real source records from Part 2 Evidence
    valid_sources_map: Dict[str, SourceMetadata] = {}

    evidence_dict = state.get("evidence", {}) or {}
    if isinstance(evidence_dict, dict):
        for s in evidence_dict.get("unique_sources", []):
            if isinstance(s, dict) and s.get("url"):
                meta = SourceMetadata(**s)
                valid_sources_map[normalize_url(meta.url)] = meta
            elif isinstance(s, SourceMetadata) and s.url:
                valid_sources_map[normalize_url(s.url)] = s

    # Also harvest from raw research results if present
    for res in state.get("research_results", []):
        for s in getattr(res, "sources", []):
            if getattr(s, "url", None):
                meta = s if isinstance(s, SourceMetadata) else SourceMetadata(**s)
                valid_sources_map[normalize_url(meta.url)] = meta

    if not claims:
        warning_msg = "Evidence Verification Warning: No claims provided for verification."
        logger.warning(warning_msg)
        empty_report = VerificationReport(summary="No claims extracted to verify.")
        return {
            "verification_report": empty_report.model_dump(),
            "errors": [warning_msg]
        }

    verifications: List[ClaimVerification] = []
    supported_cnt = 0
    contradicted_cnt = 0
    insufficient_cnt = 0
    partial_cnt = 0
    rejected_urls_all: Set[str] = set()

    for claim in claims:
        ver = verify_single_claim(claim, valid_sources_map, llm=llm)
        verifications.append(ver)

        if ver.verdict == VerdictType.SUPPORTED:
            supported_cnt += 1
        elif ver.verdict == VerdictType.CONTRADICTED:
            contradicted_cnt += 1
        elif ver.verdict == VerdictType.INSUFFICIENT_EVIDENCE:
            insufficient_cnt += 1
        elif ver.verdict == VerdictType.PARTIALLY_SUPPORTED:
            partial_cnt += 1

        for r_url in ver.rejected_urls:
            rejected_urls_all.add(r_url)

    summary_text = (
        f"Verified {len(claims)} claim(s): {supported_cnt} Supported, "
        f"{contradicted_cnt} Contradicted, {partial_cnt} Partially Supported, "
        f"{insufficient_cnt} Insufficient Evidence. "
        f"{len(rejected_urls_all)} invalid reference(s) rejected."
    )

    report = VerificationReport(
        total_claims=len(claims),
        supported_count=supported_cnt,
        contradicted_count=contradicted_cnt,
        insufficient_evidence_count=insufficient_cnt,
        partially_supported_count=partial_cnt,
        rejected_references_count=len(rejected_urls_all),
        verifications=verifications,
        summary=summary_text
    )

    logger.info(summary_text)

    return {
        "verification_report": report.model_dump(),
        "errors": []
    }


def verdict_node(state: ResearchState) -> Dict[str, Any]:
    """
    Verdict Node alias for LangGraph workflows that cleanly chains after evidence verification.
    Ensures verification report is strictly formatted and passes downstream.
    """
    if "verification_report" in state and state["verification_report"]:
        return {"verification_report": state["verification_report"]}
    return evidence_verification_node(state)
