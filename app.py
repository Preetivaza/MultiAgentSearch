import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from api import run_manager, ResearchSettings
from evaluation import export_evaluation_markdown
from pipeline import build_research_graph
from schemas import (
    Claim,
    ClaimVerification,
    CitationValidationResult,
    EvaluationResult,
    EvidenceCollection,
    ExecutionMetrics,
    QualityMetrics,
    ReportMetadata,
    ResearchPlan,
    ResearchState,
    ReviewAssessment,
    SourceMetadata,
    SubQuestion,
    SubQuestionResult,
    VerdictType,
)
import re


def generate_pdf(report_md: str, topic: str = "Research Report") -> bytes:
    """Generate a clean, high-grade binary PDF document from report Markdown."""
    from fpdf import FPDF

    class ReportPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "ResearchMind · Multi-Agent Executive Intelligence Report", align="R")
            self.ln(10)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Clean non-latin-1 characters safely
    def sanitize(text: str) -> str:
        # replace smart quotes, dashes, bullets
        replacements = {
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "—": "-", "–": "-", "•": "*", "…": "...",
            "®": "(R)", "©": "(C)", "™": "(TM)",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "replace").decode("latin-1")

    lines = report_md.splitlines()
    for raw_line in lines:
        line = sanitize(raw_line.strip())
        if not line:
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(20, 30, 50)
            pdf.multi_cell(0, 9, line[2:].strip())
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(40, 60, 100)
            pdf.multi_cell(0, 7, line[3:].strip())
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 70, 90)
            pdf.multi_cell(0, 6, line[4:].strip())
            pdf.ln(1)
        elif line.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(40, 40, 40)
            clean_item = re.sub(r"\*\*([^*]+)\*\*", r"\1", line[2:])
            pdf.multi_cell(0, 5, f"  •  {clean_item}")
            pdf.ln(1)
        else:
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(40, 40, 40)
            clean_body = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            pdf.multi_cell(0, 5, clean_body)
            pdf.ln(1.5)

    return bytes(pdf.output())


# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · Multi-Agent Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Design System & Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,300&display=swap');

/* Base page styling */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #090a0f;
    background-image:
        radial-gradient(ellipse 90% 60% at 20% -10%, rgba(255, 140, 50, 0.12) 0%, transparent 65%),
        radial-gradient(ellipse 70% 50% at 85% 105%, rgba(255, 70, 30, 0.08) 0%, transparent 60%);
    background-attachment: fixed;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.8rem 2.5rem 4rem; max-width: 1350px; }

/* Hero Header */
.hero {
    text-align: left;
    padding: 1.5rem 0 1.2rem;
    border-bottom: 1px solid rgba(255, 140, 50, 0.15);
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: #f5f2eb;
    margin: 0 0 0.5rem;
}
.hero h1 span {
    color: #ff8c32;
    background: linear-gradient(135deg, #ff9d42 0%, #ff5e28 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.0rem;
    font-weight: 300;
    color: #a49e96;
    max-width: 780px;
    line-height: 1.55;
    margin: 0;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(255, 140, 50, 0.25);
}

/* Section Header */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f5f2eb;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.2rem 0 1rem;
}
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    background: rgba(255, 140, 50, 0.12);
    color: #ff8c32;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-weight: 600;
}

/* Pipeline Stage Cards */
.stage-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.25s ease;
}
.stage-card.running {
    border-color: rgba(255, 140, 50, 0.5);
    background: rgba(255, 140, 50, 0.06);
    box-shadow: 0 0 15px rgba(255, 140, 50, 0.1);
}
.stage-card.done {
    border-color: rgba(80, 200, 120, 0.35);
    background: rgba(80, 200, 120, 0.035);
}
.stage-card.failed {
    border-color: rgba(255, 80, 80, 0.4);
    background: rgba(255, 80, 80, 0.05);
}
.stage-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #ff8c32;
    font-weight: 600;
    margin-right: 0.6rem;
}
.stage-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: #f0ebe0;
}
.stage-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
}
.badge-waiting { color: #666; background: rgba(255,255,255,0.03); }
.badge-running { color: #ff8c32; background: rgba(255,140,50,0.15); animation: pulse 1.5s infinite; }
.badge-done    { color: #50c878; background: rgba(80,200,120,0.15); }
.badge-failed  { color: #ff5050; background: rgba(255,80,80,0.15); }

@keyframes pulse {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}

/* Verdict Badges */
.verdict-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    letter-spacing: 0.06em;
}
.pill-supported {
    background: rgba(80, 200, 120, 0.15);
    color: #50c878;
    border: 1px solid rgba(80, 200, 120, 0.3);
}
.pill-contradicted {
    background: rgba(255, 80, 80, 0.15);
    color: #ff5050;
    border: 1px solid rgba(255, 80, 80, 0.3);
}
.pill-partial {
    background: rgba(255, 180, 50, 0.15);
    color: #ffb432;
    border: 1px solid rgba(255, 180, 50, 0.3);
}
.pill-insufficient {
    background: rgba(160, 155, 150, 0.15);
    color: #a09890;
    border: 1px solid rgba(160, 155, 150, 0.3);
}

/* Source Box */
.source-box {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.7rem;
}
.source-box:hover {
    border-color: rgba(255, 140, 50, 0.25);
}
.source-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #ff8c32;
    font-weight: 700;
}
.source-domain {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #8c857b;
    background: rgba(255,255,255,0.05);
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-left: 0.5rem;
}

/* Final Report Container */
.report-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 140, 50, 0.2);
    border-radius: 16px;
    padding: 2.2rem 2.6rem;
    line-height: 1.75;
    color: #e5e0d8;
}

/* Metric tile */
.metric-tile {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: left;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    color: #9c9489;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #f5f2eb;
}
.metric-sub {
    font-size: 0.75rem;
    color: #7d766d;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load Benchmarks Helper ───────────────────────────────────────────────────
@st.cache_data
def load_benchmark_questions() -> List[Dict[str, Any]]:
    bench_file = os.path.join(os.path.dirname(__file__), "benchmarks", "benchmark_questions.json")
    if os.path.exists(bench_file):
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# ── Simulation Generator Helper (Offline Mode) ────────────────────────────────
def generate_simulated_run_state(query: str, depth: str = "standard", max_revisions: int = 2) -> Dict[str, Any]:
    """Generates complete, realistic multi-agent research artifacts for offline/demo inspection."""
    run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    sq1 = SubQuestion(
        id=1,
        question=f"What are the core technical mechanisms and recent architectural breakthroughs in {query[:45]}?",
        search_query=f"{query[:35]} technical architecture breakthroughs 2025",
        focus_area="Architecture"
    )
    sq2 = SubQuestion(
        id=2,
        question=f"How do empirical benchmarks and performance metrics compare to established baseline models?",
        search_query=f"{query[:35]} empirical benchmarks performance metrics",
        focus_area="Benchmarks"
    )
    sq3 = SubQuestion(
        id=3,
        question=f"What are the primary scaling bottlenecks, compute trade-offs, and future outlook?",
        search_query=f"{query[:35]} limitations scaling bottlenecks future outlook",
        focus_area="Limitations"
    )

    plan = ResearchPlan(
        topic=query[:60],
        objective=f"Evaluate empirical literature and synthesize verifiable findings for: {query}",
        subquestions=[sq1, sq2, sq3]
    )

    s1 = SourceMetadata(
        source_id="S1",
        url="https://arxiv.org/abs/2401.00001",
        title=f"Empirical Scaling Laws and Architecture Advances in AI Research",
        snippet="Primary benchmark evaluations demonstrate substantial efficiency improvements and test-time reasoning scaling across standardized academic datasets."
    )
    s2 = SourceMetadata(
        source_id="S2",
        url="https://nature.com/articles/computational-methods-2025",
        title="Hardware Thermal Constraints and Latency Trade-offs in Scaled Compute",
        snippet="Empirical measurements reveal that while algorithmic efficiency gains are significant, hardware thermal dissipation and multi-turn inference latency impose practical bottlenecks."
    )
    s3 = SourceMetadata(
        source_id="S3",
        url="https://github.com/google-deepmind/research-frontiers",
        title="Open Reproducibility Benchmarks and Multi-Agent Systems",
        snippet="Independent reproduction audits indicate that unconstrained environments experience up to 25% lower throughput than idealized synthetic benchmarks."
    )

    rr1 = SubQuestionResult(
        subquestion_id=1,
        question=sq1.question,
        search_query=sq1.search_query,
        sources=[s1],
        summary="Novel architectures demonstrate notable gains in empirical benchmark reasoning tests."
    )
    rr2 = SubQuestionResult(
        subquestion_id=2,
        question=sq2.question,
        search_query=sq2.search_query,
        sources=[s2],
        summary="Inference latency and hardware compute overhead present measurable constraints."
    )
    rr3 = SubQuestionResult(
        subquestion_id=3,
        question=sq3.question,
        search_query=sq3.search_query,
        sources=[s3],
        summary="Independent reproducibility studies show performance variance in production environments."
    )

    ev_collection = {
        "total_tasks": 3,
        "successful_tasks": 3,
        "failed_tasks": 0,
        "unique_sources_count": 3,
        "unique_sources": [s1.model_dump(), s2.model_dump(), s3.model_dump()],
        "results_by_subquestion": [rr1.model_dump(), rr2.model_dump(), rr3.model_dump()],
        "compiled_research_context": f"Empirical findings on {query}: Architecture advances [S1], hardware latency bottlenecks [S2], and reproducibility bounds [S3]."
    }

    c1 = Claim(id=1, claim_text="Specialized model architectures scale reasoning accuracy predictably on benchmark tests.", referenced_urls=[s1.url])
    c2 = Claim(id=2, claim_text="Hardware thermal dissipation and compute overhead remain primary scaling constraints.", referenced_urls=[s2.url])
    c3 = Claim(id=3, claim_text="New architectures achieve zero latency overhead under unconstrained inference workloads.", referenced_urls=[s2.url, s3.url])
    c4 = Claim(id=4, claim_text="Room-temperature quantum-assisted inference achieved 100x efficiency gains in consumer laptops.", referenced_urls=["https://unverified-hallucinated.org/fake"])

    v1 = ClaimVerification(
        claim_id=1,
        claim_text=c1.claim_text,
        verdict=VerdictType.SUPPORTED,
        reasoning="Corroborated by primary benchmark evaluations in verified arXiv source record.",
        confidence=0.95,
        matched_sources=[s1],
        rejected_urls=[]
    )
    v2 = ClaimVerification(
        claim_id=2,
        claim_text=c2.claim_text,
        verdict=VerdictType.SUPPORTED,
        reasoning="Directly corroborated by Nature computational methods analysis.",
        confidence=0.92,
        matched_sources=[s2],
        rejected_urls=[]
    )
    v3 = ClaimVerification(
        claim_id=3,
        claim_text=c3.claim_text,
        verdict=VerdictType.CONTRADICTED,
        reasoning="Directly contradicted by empirical benchmarks showing non-zero latency overhead.",
        confidence=0.88,
        matched_sources=[s2, s3],
        rejected_urls=[]
    )
    v4 = ClaimVerification(
        claim_id=4,
        claim_text=c4.claim_text,
        verdict=VerdictType.INSUFFICIENT_EVIDENCE,
        reasoning="Rejected hallucinated citation. No verified source records corroborate consumer laptop deployment.",
        confidence=0.85,
        matched_sources=[],
        rejected_urls=["https://unverified-hallucinated.org/fake"]
    )

    vr_dict = {
        "total_claims": 4,
        "supported_count": 2,
        "contradicted_count": 1,
        "insufficient_evidence_count": 1,
        "partially_supported_count": 0,
        "rejected_references_count": 1,
        "verifications": [v1.model_dump(), v2.model_dump(), v3.model_dump(), v4.model_dump()],
        "summary": "2 Supported, 1 Contradicted, 1 Insufficient Evidence, 1 Rejected Invalid Reference."
    }

    report_content = f"""# Research Report: {query}

## Executive Summary
This report investigates state-of-the-art advances and empirical scaling dynamics regarding **{query}** [S1]. Multi-agent evaluation reveals substantial capability gains alongside hardware and latency trade-offs [S2].

## Key Findings & Evidence Synthesis
1. **Empirical Architecture Scaling**: Specialized architectures demonstrate predictable accuracy scaling over previous generation baselines across multiple benchmark evaluations [S1].
2. **Computational Overhead & Bottlenecks**: Hardware thermal constraints and multi-turn inference latency impose measurable practical deployment hurdles [S2].
3. **Reproducibility Discrepancies**: Independent reproduction audits indicate that unconstrained real-world environments experience up to 25% lower throughput than synthetic benchmarks [S3].

## Verified Claims & Disclosures
### Summary of Verification Audit
Audit completed across 4 discrete atomic claims extracted from primary source records.

### Corroborated Findings
- **Claim 1 (Supported)**: Model reasoning accuracy scales predictably on standardized test suites [S1].
- **Claim 2 (Supported)**: Thermal dissipation and compute overhead remain primary operational constraints [S2].

### Contradictions & Disputed Evidence
- **Claim 3 (Contradicted)**: Commercial assertions claiming zero latency overhead under heavy workloads are refuted by empirical benchmarks [S2, S3].

### Insufficient Evidence
- **Claim 4 (Insufficient Evidence)**: Unverified assertions of 100x efficiency gains in consumer hardware remain unsubstantiated.

## Limitations, Gaps & Uncertainties
Hardware costs, memory bandwidth limits, and multi-turn verification overhead remain primary bottlenecks requiring further algorithmic and hardware co-design.

## References
- **[S1]** [Empirical Scaling Laws and Architecture Advances](https://arxiv.org/abs/2401.00001)
- **[S2]** [Hardware Thermal Constraints and Latency Trade-offs in Scaled Compute](https://nature.com/articles/computational-methods-2025)
- **[S3]** [Open Reproducibility Benchmarks and Multi-Agent Systems](https://github.com/google-deepmind/research-frontiers)
"""

    rep_meta = {
        "topic": query,
        "word_count": len(report_content.split()),
        "sections_present": ["Executive Summary", "Key Findings", "Verified Claims", "Limitations", "References"],
        "disclosed_contradictions_count": 1,
        "disclosed_unsupported_count": 1,
        "citation_audit": {
            "total_citations_found": 4,
            "valid_citations": ["S1", "S2", "S3"],
            "invalid_citations": [],
            "uncited_sources": [],
            "citation_integrity_score": 1.0,
            "is_valid": True,
            "audit_notes": ["All cited source references [S1], [S2], [S3] map to verified primary source records."]
        }
    }

    rev_assessment = {
        "status": "Pass",
        "overall_score": 8.8,
        "coverage_score": 9.0,
        "evidence_score": 8.5,
        "citation_score": 9.5,
        "contradiction_score": 8.5,
        "revision_target": "none",
        "feedback": "The report meets rigorous academic coverage, citation integrity, and transparent contradiction disclosure standards.",
        "actionable_guidance": "",
        "issues_identified": []
    }

    critic_feedback = """Score: 9/10

Strengths:
- Clear, empirical grounding in arXiv and Nature source citations.
- Explicit and transparent disclosure of contradicted latency assertions.
- Balanced assessment of limitations and thermal bottlenecks.

Areas to Improve:
- Further quantify hardware bandwidth thresholds across specific GPU/TPU architectures.

One line verdict:
An exceptionally well-grounded, publication-ready multi-agent research synthesis."""

    qm = QualityMetrics(
        claim_support_rate=0.75,
        citation_validity=1.0,
        subquestion_coverage=1.0,
        duplicate_source_rate=0.25,
        task_completion_rate=1.0,
        supported_claims_count=2,
        total_claims_evaluated=4,
        valid_citations_count=3,
        total_citations_checked=3,
        addressed_subquestions_count=3,
        total_subquestions_count=3,
        duplicate_sources_count=1,
        total_sources_retrieved_count=4,
        successful_tasks_count=3,
        attempted_tasks_count=3
    )

    em = ExecutionMetrics(
        wall_clock_latency_seconds=3.42,
        node_latencies={"planner": 0.45, "orchestrator": 1.15, "verifier": 0.65, "writer": 0.75, "evaluator": 0.42},
        prompt_tokens=2450,
        completion_tokens=920,
        total_tokens=3370,
        token_usage_source="provider_metadata",
        error_count=0,
        errors=[]
    )

    eval_result_obj = EvaluationResult(
        run_id=run_id,
        timestamp=now_str,
        query=query,
        quality_metrics=qm,
        execution_metrics=em,
        run_trace=[
            {"step": "planner", "timestamp": now_str, "status": "completed"},
            {"step": "orchestrator", "timestamp": now_str, "status": "completed"},
            {"step": "verifier", "timestamp": now_str, "status": "completed"},
            {"step": "writer", "timestamp": now_str, "status": "completed"},
            {"step": "evaluator", "timestamp": now_str, "status": "completed"},
        ],
        status="Completed"
    )
    eval_result_obj.markdown_table = export_evaluation_markdown(eval_result_obj)
    eval_result = eval_result_obj.model_dump()

    return {
        "plan": plan,
        "research_results": [rr1, rr2, rr3],
        "evidence": ev_collection,
        "claims": [c1, c2, c3, c4],
        "verification_report": vr_dict,
        "report": report_content,
        "report_metadata": rep_meta,
        "review_assessment": rev_assessment,
        "feedback": critic_feedback,
        "evaluation_result": eval_result,
        "errors": []
    }


# ── Session State Initialization ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}
if "running" not in st.session_state:
    st.session_state.running = False
if "done" not in st.session_state:
    st.session_state.done = False
if "run_trace" not in st.session_state:
    st.session_state.run_trace = []
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "current_stage" not in st.session_state:
    st.session_state.current_stage = None
if "completed_stages" not in st.session_state:
    st.session_state.completed_stages = []
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""


# ── Header / Hero ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">LangGraph Multi-Agent Architecture · Parts 1–6 Integration</div>
    <h1>Research<span>Mind</span> Dashboard</h1>
    <p class="hero-sub">
        End-to-end multi-agent research engine. Deconstructs queries into validated subquestions,
        coordinates concurrent web retrieval, extracts atomic claims, verifies verdicts against primary passages,
        audits citation integrity, executes editorial peer review loops, and computes rigorous quality benchmarks.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Workspace & Research Settings ────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Research Settings")

    # Depth profiles
    depth_option = st.selectbox(
        "Research Depth Profile",
        options=["Standard Analysis", "Quick Scan", "Deep Empirical Research"],
        index=0,
        help="Controls number of subquestions, retrieval breadth, and allowed revision cycles."
    )

    depth_map = {
        "Quick Scan": {"depth": "quick", "subquestions": 2, "max_revisions": 1},
        "Standard Analysis": {"depth": "standard", "subquestions": 3, "max_revisions": 2},
        "Deep Empirical Research": {"depth": "deep", "subquestions": 5, "max_revisions": 3},
    }
    selected_depth_cfg = depth_map[depth_option]

    max_rev_slider = st.slider(
        "Max Revision Loops",
        min_value=0,
        max_value=3,
        value=selected_depth_cfg["max_revisions"],
        help="Strict ceiling on reviewer-driven rewrite and research cycles."
    )

    focus_dimensions = st.multiselect(
        "Focus Dimensions",
        options=["Architecture", "Benchmarks", "Limitations", "Future Outlook", "Commercial Landscape", "Security"],
        default=["Architecture", "Benchmarks", "Limitations"],
        help="Categorical areas guiding the planner agent during subquestion generation."
    )

    execution_backend = st.radio(
        "Execution Engine",
        options=["Compiled LangGraph Workflow", "FastAPI Run Manager"],
        index=0,
        help="Direct LangGraph stream or background FastAPI run management."
    )

    st.markdown("---")
    st.markdown("### 📚 Benchmark Library")
    benchmarks = load_benchmark_questions()
    if benchmarks:
        bench_titles = [f"{b.get('id')}: {b.get('query')[:45]}..." for b in benchmarks]
        selected_bench = st.selectbox("Load Standard Benchmark", options=["Select a benchmark..."] + bench_titles)
        if selected_bench != "Select a benchmark...":
            idx = bench_titles.index(selected_bench)
            st.session_state.selected_query = benchmarks[idx]["query"]

    st.markdown("---")
    # API Key Configuration in Sidebar
    st.markdown("### 🔑 API Configuration")
    raw_gemini_key = os.getenv("GOOGLE_API_KEY", "").strip()
    is_placeholder_gemini = (
        not raw_gemini_key
        or "your_gemini_api_key" in raw_gemini_key
        or raw_gemini_key == "dummy_key_to_prevent_startup_crash"
    )

    user_gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value="" if is_placeholder_gemini else raw_gemini_key,
        placeholder="AIzaSy...",
        help="Get a free Gemini API key from https://aistudio.google.com/app/apikey"
    )

    if user_gemini_key and user_gemini_key != raw_gemini_key:
        os.environ["GOOGLE_API_KEY"] = user_gemini_key
        # Update .env file
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        try:
            if os.path.exists(env_file):
                with open(env_file, "r", encoding="utf-8") as f:
                    content = f.read()
                new_content = re.sub(r"GOOGLE_API_KEY=.*", f"GOOGLE_API_KEY={user_gemini_key}", content)
                with open(env_file, "w", encoding="utf-8") as f:
                    f.write(new_content)
        except Exception:
            pass
        st.success("✅ Gemini API Key updated in environment!")
        st.rerun()

    demo_mode = st.checkbox(
        "🧪 Offline Simulation Mode",
        value=is_placeholder_gemini,
        help="Run full multi-agent pipeline using benchmark mock data without external API calls."
    )


# Notice if Gemini API Key is missing / placeholder
if is_placeholder_gemini and not demo_mode:
    st.warning("""
    ⚠️ **Google Gemini API Key is not configured yet!**
    Your `.env` file currently contains the placeholder: `GOOGLE_API_KEY=your_gemini_api_key_here`.

    👉 **Two ways to proceed:**
    1. Paste your Gemini API key into the sidebar (free key available at [Google AI Studio](https://aistudio.google.com/app/apikey)).
    2. Or leave **🧪 Offline Simulation Mode** checked in the sidebar to explore all dashboard tabs and features immediately!
    """)


# ── Part 1: Research Workspace ────────────────────────────────────────────────
col_query, col_actions = st.columns([4, 1.2])

with col_query:
    default_text = st.session_state.selected_query or "How does test-time compute scaling improve reasoning capabilities in modern LLMs compared to pre-training scaling laws?"
    research_query = st.text_area(
        "Research Question or Topic",
        value=default_text,
        height=95,
        placeholder="Enter a deep research topic or technical question...",
        help="The Research Planner will deconstruct this query into focused, multi-dimensional subquestions."
    )

with col_actions:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    start_btn = st.button("⚡ Start Research Run", use_container_width=True, type="primary")
    reset_btn = st.button("🔄 Reset Workspace", use_container_width=True)

if reset_btn:
    st.session_state.results = {}
    st.session_state.running = False
    st.session_state.done = False
    st.session_state.pipeline_error = None
    st.session_state.run_trace = []
    st.session_state.completed_stages = []
    st.session_state.current_stage = None
    st.session_state.selected_query = ""
    st.rerun()


# ── Execution Logic (Invoking the compiled LangGraph workflow) ────────────────
PIPELINE_ORDER = [
    ("planner", "01. Research Planner", "Deconstructs query into validated subquestions"),
    ("orchestrator", "02. Orchestrator", "Dispatches concurrent search tasks"),
    ("evidence_collector", "03. Evidence Collector", "Combines & deduplicates source metadata"),
    ("claim_extractor", "04. Claim Extractor", "Isolates atomic factual statements"),
    ("verifier", "05. Evidence Verifier", "Cross-checks claims vs source passages"),
    ("verdict", "06. Verdict Issuer", "Assigns Supported, Contradicted, or Insufficient"),
    ("report_writer", "07. Report Writer", "Synthesizes structured Markdown report"),
    ("citation_validator", "08. Citation Validator", "Audits source IDs [S1] & flags fabrications"),
    ("reviewer", "09. Reviewer & Revision", "Assesses coverage, citations & triggers revisions"),
    ("critic", "10. Critic Node", "Final qualitative peer review and scoring"),
    ("evaluator", "11. Observability & Evaluation", "Computes quality metrics, latency & tokens"),
]

if start_btn:
    if not research_query.strip():
        st.warning("Please provide a research question first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.pipeline_error = None
        st.session_state.completed_stages = []
        st.session_state.current_stage = "planner"
        st.rerun()


# Live LangGraph Streaming Engine
if st.session_state.running and not st.session_state.done:
    st.markdown('<div class="section-title"><span class="section-tag">ACTIVE RUN</span> Live Multi-Agent Execution Stream</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        # Simulation Mode
        if demo_mode:
            sim_state = generate_simulated_run_state(
                query=research_query.strip(),
                depth=selected_depth_cfg["depth"],
                max_revisions=max_rev_slider
            )
            total_steps = len(PIPELINE_ORDER)

            for idx, (node_name, label, desc) in enumerate(PIPELINE_ORDER):
                st.session_state.current_stage = node_name
                if node_name not in st.session_state.completed_stages:
                    st.session_state.completed_stages.append(node_name)
                progress_bar.progress(min(0.98, (idx + 1) / total_steps))
                status_placeholder.info(f"⚡ **Simulating LangGraph Node:** `{label}` ... ({idx + 1}/{total_steps} stages)")
                time.sleep(0.3)

            progress_bar.progress(1.0)
            status_placeholder.success("✓ **Simulation run complete.** All 11 multi-agent nodes finished.")
            st.session_state.results = sim_state
            st.session_state.running = False
            st.session_state.done = True
            st.rerun()

        # Live LangGraph Graph Execution Mode
        else:
            results_accumulator: Dict[str, Any] = {}
            initial_state: ResearchState = {
                "query": research_query.strip(),
                "plan": None,
                "errors": [],
                "research_results": [],
                "evidence": None,
                "claims": [],
                "verification_report": {},
                "report_metadata": {},
                "review_assessment": {},
                "revision_count": 0,
                "max_revisions": max_rev_slider,
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

            # Compile and stream from LangGraph workflow
            compiled_workflow = build_research_graph()
            step_count = 0
            total_steps = len(PIPELINE_ORDER)

            for step_update in compiled_workflow.stream(initial_state, stream_mode="updates"):
                step_count += 1
                progress_ratio = min(0.95, step_count / total_steps)
                progress_bar.progress(progress_ratio)

                for node_name, node_output in step_update.items():
                    st.session_state.current_stage = node_name
                    if node_name not in st.session_state.completed_stages:
                        st.session_state.completed_stages.append(node_name)

                    status_placeholder.info(f"⚡ **Executing LangGraph Node:** `{node_name}` ... Completed {len(st.session_state.completed_stages)} stage(s)")

                    if isinstance(node_output, dict):
                        results_accumulator.update(node_output)

            progress_bar.progress(1.0)
            status_placeholder.success("✓ **LangGraph execution complete.** All multi-agent nodes finished successfully.")

            st.session_state.results = dict(results_accumulator)
            st.session_state.running = False
            st.session_state.done = True
            st.rerun()

    except Exception as e:
        error_str = str(e)
        if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
            st.session_state.pipeline_error = """
            🚨 **Google Gemini API Key Rejected (INVALID_ARGUMENT):**
            Google's API rejected the provided key (`API_KEY_INVALID`).

            **To resolve:**
            1. Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
            2. Paste it into the **Google Gemini API Key** field in the left sidebar.
            3. Alternatively, check **🧪 Offline Simulation Mode** in the sidebar to test the complete dashboard immediately with benchmark data!
            """
        else:
            st.session_state.pipeline_error = f"Pipeline execution error: {error_str}\n\n```\n{traceback.format_exc()}\n```"
        st.session_state.running = False
        st.session_state.done = False
        st.rerun()


# ── Part 2: Agent Execution Tracker ───────────────────────────────────────────
res = st.session_state.results

with st.expander("📡 Part 2: Agent Execution Tracker & Stage Progression", expanded=st.session_state.running or not res):
    col_stages_left, col_stages_right = st.columns(2)

    for i, (node_key, label, desc) in enumerate(PIPELINE_ORDER):
        is_completed = node_key in st.session_state.completed_stages or node_key in res
        is_active = st.session_state.running and st.session_state.current_stage == node_key

        card_class = "stage-card"
        if is_active:
            card_class += " running"
            badge_html = '<span class="stage-badge badge-running">● RUNNING</span>'
        elif is_completed:
            card_class += " done"
            badge_html = '<span class="stage-badge badge-done">✓ DONE</span>'
        else:
            badge_html = '<span class="stage-badge badge-waiting">WAITING</span>'

        target_col = col_stages_left if i < 6 else col_stages_right
        with target_col:
            st.markdown(f"""
            <div class="{card_class}">
                <div>
                    <span class="stage-num">{label.split('.')[0]}</span>
                    <span class="stage-name">{label.split('. ')[1]}</span>
                    <div style="font-size:0.75rem;color:#857e74;margin-top:2px;">{desc}</div>
                </div>
                <div>{badge_html}</div>
            </div>
            """, unsafe_allow_html=True)

    # Subquestion task breakdown table if available
    research_results = res.get("research_results", [])
    if research_results:
        st.markdown("#### Subquestion Execution Matrix")
        tasks_data = []
        for r in research_results:
            sq_id = getattr(r, "subquestion_id", None) if hasattr(r, "subquestion_id") else r.get("subquestion_id", "?")
            q_text = getattr(r, "question", "") if hasattr(r, "question") else r.get("question", "")
            s_query = getattr(r, "search_query", "") if hasattr(r, "search_query") else r.get("search_query", "")
            sources = getattr(r, "sources", []) if hasattr(r, "sources") else r.get("sources", [])
            err = getattr(r, "error", None) if hasattr(r, "error") else r.get("error")

            tasks_data.append({
                "ID": f"SQ-{sq_id}",
                "Focus Subquestion": q_text,
                "Engine Query": s_query,
                "Sources Retrieved": len(sources),
                "Status": "⚠️ Failed" if err else "✅ Succeeded",
                "Error": err or "None"
            })
        st.dataframe(tasks_data, use_container_width=True)

    # Display logged warnings / errors clearly
    logged_errors = res.get("errors", [])
    if logged_errors:
        st.warning(f"⚠️ **Logged Execution Notices & Warnings ({len(logged_errors)}):**")
        for err in logged_errors:
            st.markdown(f"- `{err}`")


# Display pipeline fatal errors if any
if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)


# ── Results Sections (Visible when results are ready) ────────────────────────
if res:
    st.markdown('<div class="section-title"><span class="section-tag">AUDITED RESULTS</span> Research Findings & Synthesized Artifacts</div>', unsafe_allow_html=True)

    # Tabbed Workspace
    tab_report, tab_evidence, tab_claims, tab_evaluation = st.tabs([
        "📝 Final Report & Export",
        "📦 Evidence Inspector & Sources",
        "🛡️ Claim Verification & Contradictions",
        "📊 Evaluation & Observability Panel"
    ])

    # ── TAB 1: Final Report & Export (Part 4) ──────────────────────────────────
    with tab_report:
        report_text = res.get("report", "")
        plan = res.get("plan")
        review = res.get("review_assessment", {})
        critic = res.get("feedback", "")

        col_rep_main, col_rep_meta = st.columns([3.5, 1.2])

        with col_rep_main:
            if report_text:
                st.markdown(f'<div class="report-container">{report_text}</div>', unsafe_allow_html=True)
            else:
                st.info("No report generated yet. Run the pipeline to synthesize research.")

        with col_rep_meta:
            st.markdown("#### 📥 Export Tools")
            if report_text:
                # Prepare a self-contained, publication-grade styled HTML document ready for 1-click Print to PDF
                safe_title = research_query or "Research Report"
                styled_html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.65;
    color: #1a1a1a;
    background: #ffffff;
    max-width: 850px;
    margin: 40px auto;
    padding: 0 30px;
}}
h1 {{
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 12px;
    margin-bottom: 24px;
}}
h2 {{
    font-size: 1.35rem;
    font-weight: 700;
    color: #1e293b;
    margin-top: 32px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 6px;
}}
h3 {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #334155;
    margin-top: 20px;
}}
p, li {{
    font-size: 0.96rem;
    color: #334155;
}}
ul {{
    padding-left: 20px;
}}
li {{
    margin-bottom: 8px;
}}
a {{
    color: #2563eb;
    text-decoration: none;
}}
a:hover {{
    text-decoration: underline;
}}
strong {{
    color: #0f172a;
}}
code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88em;
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
}}
.badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 16px;
}}
@media print {{
    body {{
        margin: 20px;
        padding: 0;
        max-width: 100%;
    }}
    .no-print {{
        display: none !important;
    }}
}}
</style>
</head>
<body>
<div class="no-print" style="margin-bottom: 24px; padding: 12px 18px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
    <div><strong>📄 Printable Executive Report</strong> &bull; Press <kbd>Ctrl+P</kbd> or click to Save as PDF</div>
    <button onclick="window.print()" style="background: #2563eb; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer;">🖨️ Print / Save as PDF</button>
</div>
<div class="badge">ResearchMind &bull; Multi-Agent Intelligence Report</div>
{report_text.replace(chr(10), '<br>')}
<script>
// Automatically convert basic markdown formatting
document.body.innerHTML = document.body.innerHTML
    .replace(/^# (.*?)<br>/gm, '<h1>$1</h1>')
    .replace(/^## (.*?)<br>/gm, '<h2>$1</h2>')
    .replace(/^### (.*?)<br>/gm, '<h3>$1</h3>')
    .replace(/\\[S(\\d+)\\]/g, '<strong>[S$1]</strong>')
    .replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^\\)]+)\\)/g, '<a href="$2" target="_blank">$1</a>')
    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/\\*(.*?)\\*/g, '<em>$1</em>');
</script>
</body>
</html>"""

                # 1. Direct PDF file download
                try:
                    pdf_bytes = generate_pdf(report_text, topic=research_query)
                    st.download_button(
                        label="📕 Download Report (.pdf)",
                        data=pdf_bytes,
                        file_name=f"research_report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.caption(f"Direct PDF conversion note: {pdf_err}")

                # 2. Browser Print / HTML view
                st.download_button(
                    label="🖨️ Open Print-to-PDF (.html)",
                    data=styled_html_report,
                    file_name=f"research_report_{int(time.time())}.html",
                    mime="text/html",
                    use_container_width=True
                )

                # 3. Markdown source
                st.download_button(
                    label="⬇ Download Report (.md)",
                    data=report_text,
                    file_name=f"researchmind_report_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

                # Export full JSON bundle
                export_bundle = {
                    "query": research_query,
                    "report": report_text,
                    "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
                    "verification": res.get("verification_report"),
                    "evaluation": res.get("evaluation_result"),
                    "completed_stages": st.session_state.completed_stages,
                }
                st.download_button(
                    label="⬇ Download Full Bundle (.json)",
                    data=json.dumps(export_bundle, indent=2, default=str),
                    file_name=f"research_bundle_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("#### 🧐 Editorial Peer Review")
            if review:
                status = review.get("status", "Pass")
                score = review.get("overall_score", 8.0)
                status_color = "#50c878" if status == "Pass" else "#ffb432"
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:0.9rem 1rem;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#8c857b;">EDITORIAL DECISION</div>
                    <div style="font-size:1.15rem;font-weight:800;color:{status_color};margin:0.2rem 0;">{status.upper()} ({score}/10)</div>
                    <div style="font-size:0.78rem;color:#b0a89d;">Target: <code>{review.get('revision_target', 'none')}</code></div>
                </div>
                """, unsafe_allow_html=True)
                if review.get("actionable_guidance"):
                    st.caption(f"**Guidance Applied:** {review.get('actionable_guidance')}")

            if critic:
                st.markdown("---")
                st.markdown("#### ⚖️ Critic Feedback")
                st.markdown(f"<div style='font-size:0.82rem;color:#cdc8bf;line-height:1.5;'>{critic}</div>", unsafe_allow_html=True)

    # ── TAB 2: Evidence Inspector (Part 2 & Part 4) ───────────────────────────
    with tab_evidence:
        ev = res.get("evidence", {}) or {}
        unique_sources = ev.get("unique_sources", [])
        total_tasks = ev.get("total_tasks", 0)
        succeeded_tasks = ev.get("successful_tasks", 0)
        failed_tasks = ev.get("failed_tasks", 0)

        # Overview counters
        c_e1, c_e2, c_e3, c_e4 = st.columns(4)
        with c_e1:
            st.metric("Unique Verified Sources", f"{len(unique_sources)}")
        with c_e2:
            st.metric("Total Search Tasks", f"{total_tasks}")
        with c_e3:
            st.metric("Successful Tasks", f"{succeeded_tasks}")
        with c_e4:
            st.metric("Failed Tasks", f"{failed_tasks}", delta_color="inverse")

        st.markdown("---")
        st.markdown("#### Real Source Records Linked with Standard IDs [S1], [S2]...")

        if unique_sources:
            for s in unique_sources:
                sid = s.get("source_id", "S?")
                url = s.get("url", "#")
                title = s.get("title") or "Web Document"
                snippet = s.get("snippet", "")
                domain = url.split("/")[2] if "://" in url else "web"

                st.markdown(f"""
                <div class="source-box">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.4rem;">
                        <div>
                            <span class="source-id">[{sid}]</span>
                            <a href="{url}" target="_blank" style="color:#f5f2eb;font-weight:600;font-size:0.92rem;text-decoration:none;margin-left:0.4rem;">{title}</a>
                            <span class="source-domain">{domain}</span>
                        </div>
                    </div>
                    <div style="font-size:0.82rem;color:#b5afa6;line-height:1.5;">{snippet}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sources retrieved yet.")

        # Citation Integrity Audit details
        report_meta = res.get("report_metadata", {})
        audit = report_meta.get("citation_audit", {}) if isinstance(report_meta, dict) else {}
        if audit:
            st.markdown("---")
            st.markdown("#### 📑 Citation Integrity Audit")
            score = audit.get("citation_integrity_score", 1.0)
            score_pct = int(score * 100)
            audit_col1, audit_col2, audit_col3 = st.columns(3)
            with audit_col1:
                st.metric("Citation Integrity Score", f"{score_pct}%")
            with audit_col2:
                st.metric("Valid Citations", len(audit.get("valid_citations", [])))
            with audit_col3:
                st.metric("Invalid / Flagged", len(audit.get("invalid_citations", [])))

            if audit.get("invalid_citations"):
                st.error(f"Flagged Unverified Citations: {', '.join(audit.get('invalid_citations'))}")

    # ── TAB 3: Claim Verification & Contradictions (Part 3) ───────────────────
    with tab_claims:
        vr = res.get("verification_report", {}) or {}
        verifications = vr.get("verifications", [])

        # Verdict Summary Counters
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            st.markdown(f'<span class="verdict-pill pill-supported">✓ Supported: {vr.get("supported_count", 0)}</span>', unsafe_allow_html=True)
        with col_v2:
            st.markdown(f'<span class="verdict-pill pill-contradicted">✗ Contradicted: {vr.get("contradicted_count", 0)}</span>', unsafe_allow_html=True)
        with col_v3:
            st.markdown(f'<span class="verdict-pill pill-partial">◐ Partially Supported: {vr.get("partially_supported_count", 0)}</span>', unsafe_allow_html=True)
        with col_v4:
            st.markdown(f'<span class="verdict-pill pill-insufficient">? Insufficient Evidence: {vr.get("insufficient_evidence_count", 0)}</span>', unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Contradiction Alert Box
        contradicted_items = [v for v in verifications if v.get("verdict") == VerdictType.CONTRADICTED]
        if contradicted_items:
            st.error(f"🚨 **Disputed / Contradictory Evidence Detected ({len(contradicted_items)}):** The system flagged claims conflicting directly with empirical primary sources.")

        # Verdict filter
        verdict_filter = st.radio(
            "Filter Claims by Verdict",
            options=["All", "Supported", "Contradicted", "Partially supported", "Insufficient evidence"],
            horizontal=True
        )

        filtered_verifications = [
            v for v in verifications
            if verdict_filter == "All" or v.get("verdict") == verdict_filter
        ]

        st.markdown("---")
        for v in filtered_verifications:
            verdict = v.get("verdict", "")
            if verdict == VerdictType.SUPPORTED:
                pill_cls = "pill-supported"
            elif verdict == VerdictType.CONTRADICTED:
                pill_cls = "pill-contradicted"
            elif verdict == VerdictType.PARTIALLY_SUPPORTED:
                pill_cls = "pill-partial"
            else:
                pill_cls = "pill-insufficient"

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:0.9rem;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
                    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#ff8c32;font-weight:600;">CLAIM {v.get('claim_id')}</span>
                    <span class="verdict-pill {pill_cls}">{verdict.upper()}</span>
                </div>
                <div style="font-size:0.98rem;font-weight:600;color:#f5f2eb;margin-bottom:0.5rem;">{v.get('claim_text')}</div>
                <div style="font-size:0.85rem;color:#c0bab1;line-height:1.5;"><strong>Reasoning:</strong> {v.get('reasoning')}</div>
            </div>
            """, unsafe_allow_html=True)

            matched = v.get("matched_sources", [])
            if matched:
                with st.expander(f"🔗 Linked Real Sources ({len(matched)})", expanded=False):
                    for s in matched:
                        st.markdown(f"- [{s.get('title') or s.get('url')}]({s.get('url')})")

            rejected = v.get("rejected_urls", [])
            if rejected:
                st.caption(f"⚠️ **Rejected Invalid / Fabricated URLs:** {', '.join(rejected)}")

    # ── TAB 4: Evaluation & Observability Panel (Part 6) ──────────────────────
    with tab_evaluation:
        ev_res = res.get("evaluation_result", {})
        if ev_res:
            qm = ev_res.get("quality_metrics", {})
            em = ev_res.get("execution_metrics", {})
            md_table = ev_res.get("markdown_table", "")

            st.info("ℹ️ **Metric Independence Guarantee**: A valid citation reference does NOT imply the source supports the claim. Citation validity and claim support are audited, evaluated, and presented as distinct dimensions.")

            # Quality Metrics Grid
            st.markdown("#### Quality Metrics")
            q1, q2, q3, q4, q5 = st.columns(5)
            with q1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Claim Support Rate</div>
                    <div class="metric-val" style="color:#50c878;">{round(qm.get('claim_support_rate', 0.0) * 100, 1)}%</div>
                    <div class="metric-sub">{qm.get('supported_claims_count', 0)}/{qm.get('total_claims_evaluated', 0)} claims</div>
                </div>
                """, unsafe_allow_html=True)
            with q2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Citation Validity</div>
                    <div class="metric-val" style="color:#ff8c32;">{round(qm.get('citation_validity', 0.0) * 100, 1)}%</div>
                    <div class="metric-sub">{qm.get('valid_citations_count', 0)}/{qm.get('total_citations_checked', 0)} citations</div>
                </div>
                """, unsafe_allow_html=True)
            with q3:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Subquestion Coverage</div>
                    <div class="metric-val">{round(qm.get('subquestion_coverage', 0.0) * 100, 1)}%</div>
                    <div class="metric-sub">{qm.get('addressed_subquestions_count', 0)}/{qm.get('total_subquestions_count', 0)} questions</div>
                </div>
                """, unsafe_allow_html=True)
            with q4:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Duplicate Source Rate</div>
                    <div class="metric-val">{round(qm.get('duplicate_source_rate', 0.0) * 100, 1)}%</div>
                    <div class="metric-sub">{qm.get('duplicate_sources_count', 0)} duplicates</div>
                </div>
                """, unsafe_allow_html=True)
            with q5:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Task Completion</div>
                    <div class="metric-val" style="color:#50c878;">{round(qm.get('task_completion_rate', 0.0) * 100, 1)}%</div>
                    <div class="metric-sub">{qm.get('successful_tasks_count', 0)}/{qm.get('attempted_tasks_count', 0)} tasks</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            # Execution Telemetry
            st.markdown("#### Execution & Telemetry Metrics")
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Wall-Clock Latency</div>
                    <div class="metric-val">{em.get('wall_clock_latency_seconds', 0.0)}s</div>
                    <div class="metric-sub">Start-to-finish execution</div>
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Total Token Usage</div>
                    <div class="metric-val">{em.get('total_tokens', 0):,}</div>
                    <div class="metric-sub">Source: {em.get('token_usage_source', 'measured')}</div>
                </div>
                """, unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Prompt / Completion</div>
                    <div class="metric-val" style="font-size:1.4rem;">{em.get('prompt_tokens', 0):,} / {em.get('completion_tokens', 0):,}</div>
                    <div class="metric-sub">Tokens telemetred</div>
                </div>
                """, unsafe_allow_html=True)
            with e4:
                err_count = em.get("error_count", 0)
                err_color = "#50c878" if err_count == 0 else "#ffb432"
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-label">Logged Errors</div>
                    <div class="metric-val" style="color:{err_color};">{err_count}</div>
                    <div class="metric-sub">{'Zero fatal errors' if err_count == 0 else 'Warnings recorded'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Exportable GitHub README Evaluation Table
            if md_table:
                st.markdown("---")
                st.markdown("#### 📋 Exportable Evaluation Table (for GitHub README)")
                st.code(md_table, language="markdown")
                st.download_button(
                    label="⬇ Download Evaluation Table (.md)",
                    data=md_table,
                    file_name=f"evaluation_table_{ev_res.get('run_id', 'run')}.md",
                    mime="text/markdown"
                )

            # Full run trace json
            if ev_res.get("run_trace"):
                with st.expander("🔍 Inspect Full Run Trace & Event Telemetry", expanded=False):
                    st.json(ev_res["run_trace"])
        else:
            st.info("Evaluation metrics will appear once pipeline completes.")