import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class SubQuestion(BaseModel):
    """Represents a focused research subquestion derived from the user's query."""
    id: int = Field(
        ...,
        description="Unique identifier for the subquestion (e.g. 1, 2, 3)"
    )
    question: str = Field(
        ...,
        min_length=5,
        description="Clear, focused, and deep research subquestion"
    )
    search_query: str = Field(
        ...,
        min_length=3,
        description="Targeted keyword search query optimized for search engines"
    )
    focus_area: str = Field(
        default="General",
        description="Categorical dimension (e.g., Background, Architecture, Performance, Limitations, Future Outlook)"
    )

class ResearchPlan(BaseModel):
    """Structured and validated research plan containing subquestions ready for dispatch."""
    topic: str = Field(
        ...,
        description="Refined summary or title of the research topic"
    )
    objective: str = Field(
        ...,
        description="Main objective and scope of this research plan"
    )
    subquestions: List[SubQuestion] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="List of 1 to 6 focused subquestions covering complementary aspects"
    )
    suggested_sources: List[str] = Field(
        default_factory=list,
        description="Recommended domains or source types (e.g., arXiv, GitHub, documentation, industry reports)"
    )

class SourceMetadata(BaseModel):
    """Metadata for a web source discovered during research."""
    source_id: Optional[str] = Field(default=None, description="Standardized reference ID (e.g. S1, S2)")
    url: str = Field(..., description="Canonical HTTP/HTTPS URL of the source")
    title: str = Field(default="", description="Web page or document title")
    snippet: str = Field(default="", description="Relevant excerpt or content snippet")



class SubQuestionResult(BaseModel):
    """Research findings collected for an individual subquestion."""
    subquestion_id: int = Field(..., description="ID matching the SubQuestion")
    question: str = Field(..., description="The subquestion investigated")
    search_query: str = Field(..., description="The query used to search")
    sources: List[SourceMetadata] = Field(
        default_factory=list,
        description="List of sources discovered for this subquestion"
    )
    summary: str = Field(
        default="",
        description="Synthesized summary of findings for this subquestion"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the search or extraction failed"
    )


class EvidenceCollection(BaseModel):
    """Aggregated evidence produced by the Evidence Collector."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    unique_sources_count: int = 0
    unique_sources: List[SourceMetadata] = Field(default_factory=list)
    results_by_subquestion: List[SubQuestionResult] = Field(default_factory=list)
    compiled_research_context: str = ""


class VerdictType:
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT_EVIDENCE = "Insufficient evidence"
    PARTIALLY_SUPPORTED = "Partially supported"

    ALL_VERDICTS = [
        SUPPORTED,
        CONTRADICTED,
        INSUFFICIENT_EVIDENCE,
        PARTIALLY_SUPPORTED
    ]


class Claim(BaseModel):
    """An individual atomic factual claim extracted from research findings."""
    id: int = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., min_length=5, description="Factual statement extracted from research")
    subquestion_id: Optional[int] = Field(default=None, description="Associated subquestion ID")
    referenced_urls: List[str] = Field(
        default_factory=list,
        description="Source URLs cited or associated with this claim"
    )


class ClaimVerification(BaseModel):
    """Verification analysis and verdict for an individual factual claim."""
    claim_id: int = Field(..., description="ID matching the Claim")
    claim_text: str = Field(..., description="The factual claim investigated")
    verdict: str = Field(
        ...,
        description="Verdict: 'Supported', 'Contradicted', 'Insufficient evidence', or 'Partially supported'"
    )
    reasoning: str = Field(
        ...,
        description="Justification explaining why this verdict was reached based on source passages"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this verification verdict"
    )
    matched_sources: List[SourceMetadata] = Field(
        default_factory=list,
        description="Real, validated source records that substantiate or refute the claim"
    )
    rejected_urls: List[str] = Field(
        default_factory=list,
        description="Invalid, unknown, or unverified source URLs that were rejected"
    )
    source_excerpts: List[str] = Field(
        default_factory=list,
        description="Direct quotes or passages from sources used for verification"
    )


class VerificationReport(BaseModel):
    """Structured report produced by the Verification and Verdict nodes."""
    total_claims: int = 0
    supported_count: int = 0
    contradicted_count: int = 0
    insufficient_evidence_count: int = 0
    partially_supported_count: int = 0
    rejected_references_count: int = 0
    verifications: List[ClaimVerification] = Field(default_factory=list)
    summary: str = ""


class CitationValidationResult(BaseModel):
    """Integrity audit result for report citations."""
    total_citations_found: int = 0
    valid_citations: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    uncited_sources: List[str] = Field(default_factory=list)
    citation_integrity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    is_valid: bool = True
    audit_notes: List[str] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    """Metadata generated alongside the final research report."""
    topic: str
    word_count: int = 0
    sections_present: List[str] = Field(default_factory=list)
    disclosed_contradictions_count: int = 0
    disclosed_unsupported_count: int = 0
    citation_audit: Optional[CitationValidationResult] = None


class ReviewAssessment(BaseModel):
    """Structured assessment produced by the Reviewer Node."""
    status: str = Field(
        ...,
        description="'Pass' if the report meets all standards, 'Needs revision' if fixable issues exist"
    )
    overall_score: float = Field(default=8.0, ge=0.0, le=10.0)
    coverage_score: float = Field(default=8.0, ge=0.0, le=10.0)
    evidence_score: float = Field(default=8.0, ge=0.0, le=10.0)
    citation_score: float = Field(default=8.0, ge=0.0, le=10.0)
    contradiction_score: float = Field(default=8.0, ge=0.0, le=10.0)
    revision_target: str = Field(
        default="none",
        description="'rewriter', 'researcher', or 'none'"
    )
    feedback: str = Field(default="", description="Detailed qualitative feedback")
    actionable_guidance: str = Field(
        default="",
        description="Targeted instructions for the rewriter or researcher to resolve issues"
    )
    issues_identified: List[str] = Field(default_factory=list)


class ExecutionMetrics(BaseModel):
    """Observability execution metrics measuring time, cost, and errors."""
    wall_clock_latency_seconds: float = Field(default=0.0, description="Total wall-clock duration in seconds")
    node_latencies: Dict[str, float] = Field(default_factory=dict, description="Execution time per node in seconds")
    prompt_tokens: int = Field(default=0, description="Actual input prompt tokens reported by LLM provider")
    completion_tokens: int = Field(default=0, description="Actual output tokens reported by LLM provider")
    total_tokens: int = Field(default=0, description="Sum of prompt and completion tokens")
    token_usage_source: str = Field(default="provider_metadata", description="Origin of token metrics")
    error_count: int = Field(default=0, description="Total errors encountered across execution")
    errors: List[str] = Field(default_factory=list, description="Logged error messages")


class QualityMetrics(BaseModel):
    """Quality metrics evaluating evidence, citations, coverage, and claim validity."""
    claim_support_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Supported claims ÷ claims evaluated (Tracked separately from citation validity)"
    )
    citation_validity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Valid source references ÷ references checked"
    )
    subquestion_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Required subquestions adequately addressed ÷ total required"
    )
    duplicate_source_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Duplicate retrieved sources ÷ total sources retrieved"
    )
    task_completion_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Successfully completed research tasks ÷ tasks attempted"
    )
    # Raw counts for auditing
    supported_claims_count: int = 0
    total_claims_evaluated: int = 0
    valid_citations_count: int = 0
    total_citations_checked: int = 0
    addressed_subquestions_count: int = 0
    total_subquestions_count: int = 0
    duplicate_sources_count: int = 0
    total_sources_retrieved_count: int = 0
    successful_tasks_count: int = 0
    attempted_tasks_count: int = 0


class EvaluationResult(BaseModel):
    """Comprehensive evaluation artifact combining quality and execution metrics."""
    run_id: str
    timestamp: str
    query: str
    quality_metrics: QualityMetrics
    execution_metrics: ExecutionMetrics
    run_trace: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "Completed"
    markdown_table: str = ""


class ResearchState(TypedDict, total=False):
    """
    Central LangGraph State for the Multi-Agent Research System.
    Stores query, plan, errors, research results, claims, verification report, and downstream artifacts.
    """
    query: str
    plan: Optional[ResearchPlan]
    errors: Annotated[List[str], operator.add]
    research_results: Annotated[List[SubQuestionResult], operator.add]
    evidence: Optional[Dict[str, Any]]
    claims: Optional[List[Claim]]
    verification_report: Optional[Dict[str, Any]]
    report_metadata: Optional[Dict[str, Any]]
    review_assessment: Optional[Dict[str, Any]]
    revision_count: int
    max_revisions: int
    actionable_guidance: Optional[str]
    scraped_content: Optional[str]
    report: Optional[str]
    feedback: Optional[str]
    start_time: float
    node_latencies: Dict[str, float]
    token_usage: Dict[str, int]
    run_trace: Annotated[List[Dict[str, Any]], operator.add]
    evaluation_result: Optional[Dict[str, Any]]
    success: bool




