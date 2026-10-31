# ResearchMind - Multi-Agent AI Research System 🔬

ResearchMind is an advanced, production-grade multi-agent research system built with **LangGraph**, **LangChain**, and powered by **Google Gemini** models with **Streamlit**. It deconstructs complex user research queries, coordinates concurrent search tasks, deduplicates evidence, verifies factual claims against source passages, synthesizes rigorous research reports, audits citations, performs editorial peer reviews with an automated revision loop, and conducts final critical evaluation.

---

## 🏗️ Architecture & Core Components

```
User Research Query
        │
        ▼
[Part 1: Research Planner Node] ──(Validates Plan & Subquestions)
        │
        ▼
[Part 2: LangGraph Orchestrator]
        ├── Dispatch Task 1 (Subquestion 1) ──┐
        ├── Dispatch Task 2 (Subquestion 2) ──┼── Concurrent Execution
        └── Dispatch Task N (Subquestion N) ──┘
        │
        ▼
[Evidence Collector Node] ──(Deduplicates sources, preserves URLs & excerpts)
        │
        ▼
[Part 3: Claim Extraction Node] ──(Splits findings into individual atomic claims)
        │
        ▼
[Part 3: Evidence Verification & Verdict Node] ──(Compares claims vs source passages)
        ├── Supported: Corroborated by verified source passages
        ├── Contradicted: Conflicting or negated by source evidence
        ├── Insufficient evidence: Missing evidence or invalid source reference rejected
        └── Partially supported: Partial overlap, unconfirmed sub-claims
        │
        ▼
[Part 4: Report Writer Node] ◄─────────────────────────────────────────────┐
        │                                                                  │
        ▼                                                                  │
[Part 4: Citation Validation Node] ──(Audits source IDs [S1], [S2])        │ Revalidation
        │                                                                  │ Path
        ▼                                                                  │
[Part 5: Reviewer Node] ──(Audits coverage, evidence, citations, etc.)     │
        │                                                                  │
        ├── Conditional Routing:                                           │
        │   ├── "Pass" OR iteration >= max_revisions ──> [Critic Node]     │
        │   └── "Needs revision" ──────────────────────> [Targeted Revision]
        │
        ▼
[Part 6: Evaluation & Observability Node] ──(Audits quality, traces, latency & tokens)
        │
        ▼
   Evaluation Results & Run Trace (Exportable Markdown Table) ──> [END]
```

### Part 1: Research Planner
- **LangGraph State (`ResearchState`)**: Typed state tracking `query`, `plan`, `errors`, `research_results`, `evidence`, `claims`, `verification_report`, `report_metadata`, `review_assessment`, `revision_count`, `max_revisions`, `report`, `feedback`, `start_time`, `run_trace`, `node_latencies`, `token_usage`, and `evaluation_result`.
- **Planner Node (`planner_node`)**: Uses Google Gemini with structured Pydantic output (`ResearchPlan`) to deconstruct broad queries into 2–5 focused subquestions across multiple dimensions (Architecture, Benchmarks, Limitations, Future Outlook).
- **Validated Research Plan**: Validates that every subquestion contains an actionable search query, ID, and focus area.

### Part 2: Research Execution Orchestrator & Evidence Collector
- **Concurrent Task Dispatch**: Dispatches independent research workers concurrently across subquestions using `ThreadPoolExecutor`.
- **Source Preservation**: Every subquestion preserves full source metadata (`url`, `title`, and `snippet`).
- **Deduplication**: The Evidence Collector automatically deduplicates sources that appear across multiple subquestions.
- **Fault-Tolerant Execution**: A failed search on one subquestion is reported transparently without discarding successful results from others.

### Part 3: Claim Verification & Verdicts
- **Claim Extraction Node (`claim_extraction_node`)**: Deconstructs research text into individual, atomic factual claims.
- **Evidence Verification Node (`evidence_verification_node`)**: Compares claims against real source passages extracted in Part 2.
- **Invalid Source Rejection**: Rejects hallucinated, unknown, or fabricated source URLs that are not part of verified source records.
- **Contradiction Detection**: Explicitly identifies conflicting or refuted factual statements.
- **Verdict Node (`verdict_node`)**: Issues one of 4 strict verdicts per claim (`Supported`, `Contradicted`, `Insufficient evidence`, `Partially supported`).

### Part 4: Report Writer & Citation Validation
- **Report Writer Node (`report_writer_node`)**: Synthesizes actual research evidence and verified claims into structured Markdown sections (`# Executive Summary`, `## Key Findings`, `## Verified Claims & Disclosures`, `## Limitations`, `## References`).
- **Citation Validation Node (`citation_validation_node`)**: Audits source citations (`[S1]`, `[S2]`) against verified sources, flags and annotates hallucinated citations (`[⚠️ Unverified Citation]`), and computes a `citation_integrity_score`.

### Part 5: Reviewer Node & Revision Loop
- **Reviewer Node (`reviewer_node`)**: Evaluates the draft across four strict criteria:
  - *Coverage*: Checks if all planned subquestions and report sections are addressed.
  - *Evidence Quality*: Verifies grounding in empirical evidence rather than vague claims.
  - *Citation Integrity*: Checks that no unverified citations or audit flags remain.
  - *Contradictions*: Ensures known contradictions are transparently disclosed.
- **Structured Assessment (`ReviewAssessment`)**: Returns `status` (`'Pass'` or `'Needs revision'`), scores (Coverage, Evidence, Citations, Contradictions, Overall), `revision_target` (`'rewriter'` or `'researcher'`), and `actionable_guidance`.
- **Conditional Routing (`route_after_review`)**:
  - `status == "Pass"` ➔ routes directly to publish (`critic`).
  - `status == "Needs revision"` and `revision_count < max_revisions` ➔ routes to `targeted_revision`.
- **Targeted Revision & Revalidation**: Revises specific sections with actionable guidance, revalidates citations and claims, and re-reviews.
- **Strict Iteration Limit**: Enforces a strict ceiling (`revision_count >= max_revisions`) to guarantee termination without infinite loops.

### Part 6: Evaluation & Observability
- **Evaluation Node (`evaluation_node`)**: Computes comprehensive quality and execution metrics from actual run artifacts:
  - **Claim Support Rate**: Supported claims ÷ claims evaluated (Audited separately from citation validity).
  - **Citation Validity**: Valid source references ÷ references checked.
  - **Subquestion Coverage**: Required subquestions addressed ÷ total required.
  - **Duplicate-Source Rate**: Duplicate retrieved sources ÷ total sources retrieved.
  - **Task Completion Rate**: Successfully completed research tasks ÷ tasks attempted.
  - **Wall-Clock Latency**: Measured from start-to-finish execution duration timestamps.
  - **Token Usage Telemetry**: Measured directly from provider API response metadata (or exact character length ratios, never guessed).
- **Metric Independence**: Crucial design rule — a valid citation URL does not imply the source supports the claim. Both are measured and reported as distinct dimensions.
- **Version-Controlled Benchmarks**: Evaluated against standardized queries in `benchmarks/benchmark_questions.json`.
- **Exportable Markdown Report**: Generates GitHub-flavored Markdown evaluation tables ready for README inclusion.

#### Sample Evaluation & Observability Report
```markdown
### 📊 ResearchMind Evaluation & Observability Report
**Query**: *"Evaluating Quantum Computing Error Correction Thresholds in 2025"*
**Run ID**: `RUN-9B4F12C0` | **Status**: `Completed`

#### 1. Quality Metrics
| Metric | Measured Value | Calculation / Formula | Audit Status |
| :--- | :--- | :--- | :--- |
| **Claim Support Rate** | **85.7%** (6/7) | Supported claims ÷ claims evaluated | ✅ High Grounding |
| **Citation Validity** | **100.0%** (5/5) | Valid source references ÷ references checked | ✅ Verified Records |
| **Subquestion Coverage** | **100.0%** (3/3) | Required subquestions addressed ÷ total required | ✅ Complete Coverage |
| **Duplicate Source Rate** | **25.0%** (1/4) | Duplicate retrieved sources ÷ sources retrieved | ℹ️ Handled via De-dupe |
| **Task Completion Rate** | **100.0%** (3/3) | Successfully completed tasks ÷ tasks attempted | ✅ All Succeeded |

#### 2. Execution & Observability Metrics
| Metric | Measured Value | Trace & Measurement Details |
| :--- | :--- | :--- |
| **Wall-Clock Latency** | **3.42s** | End-to-end execution duration |
| **Total Token Usage** | **2,460 tokens** | Source: `gemini_provider_response_metadata` |
| **Prompt Input Tokens** | **1,820 tokens** | Measured input token metadata |
| **Completion Output Tokens** | **640 tokens** | Measured generation token metadata |
| **Execution Errors** | **0 error(s)** | Logged during graph traversal |
```

---

## 📂 Project Structure

* [schemas.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/schemas.py): Pydantic data models (`ResearchPlan`, `SubQuestion`, `SourceMetadata`, `EvidenceCollection`, `Claim`, `ClaimVerification`, `VerificationReport`, `CitationValidationResult`, `ReportMetadata`, `ReviewAssessment`, `QualityMetrics`, `ExecutionMetrics`, `EvaluationResult`) and `ResearchState`.
* [planner.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/planner.py): Planner node and structured prompt/output parser.
* [orchestrator.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/orchestrator.py): Concurrent task execution, search parsing, and evidence collector.
* [verification.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/verification.py): Claim extraction, source verification, contradiction detection, and verdict generation.
* [writer.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/writer.py): Report writer, fallback synthesis, and citation validation node.
* [reviewer.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/reviewer.py): Reviewer node, heuristic evaluations, targeted revision node, and conditional routing.
* [evaluation.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/evaluation.py): Evaluation node, quality metric computations, token & latency telemetry, and markdown exporter.
* [benchmarks/benchmark_questions.json](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/benchmarks/benchmark_questions.json): Version-controlled benchmark queries across AI, Quantum, Biotech, and Clean Energy.
* [pipeline.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/pipeline.py): Compiled LangGraph `StateGraph` linking all nodes: Planner ➔ Orchestrator ➔ Collector ➔ Verifier ➔ Writer ➔ Validator ➔ Reviewer (Revision Loop) ➔ Critic ➔ Evaluator.
* [agents.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/agents.py): Gemini model setup and re-exports.
* [tools.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tools.py): `web_search` (Tavily) and `scrape_url` (BeautifulSoup).
* [api.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/api.py): FastAPI backend API with Pydantic validation, asynchronous run management, streaming LangGraph invocation, and report export.
* [app.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/app.py): ResearchMind Dashboard integrating Parts 1–6 with live LangGraph execution stream, evidence inspector, and evaluation panel.
* [tests/test_research_system.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tests/test_research_system.py): Automated tests for Part 1 & Part 2.
* [tests/test_claim_verification.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tests/test_claim_verification.py): Automated tests for Part 3.
* [tests/test_report_writer.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tests/test_report_writer.py): Automated tests for Part 4.
* [tests/test_reviewer_loop.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tests/test_reviewer_loop.py): Automated tests for Part 5.
* [tests/test_evaluation.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tests/test_evaluation.py): Automated tests for Part 6 (Evaluation & Observability).
* [tests/test_api.py](file:///f:/PLACEMENT/Projects/AI_ML/Multi-agent-research-system/tests/test_api.py): Automated tests for Part 7 (Backend API & Run Management).

---

### Part 7: ResearchMind Dashboard & Backend API Integration

- **Research Workspace**: Submit queries, load version-controlled benchmark templates, configure research depth profiles (`quick`, `standard`, `deep`), ceiling revision loops, and select focus dimensions.
- **Agent Execution Tracker**: Displays real-time task statuses, subquestion execution matrices, retry counts, latency per node, and completed research stages.
- **Evidence Inspector**: Interactive browser for deduplicated sources with standard IDs (`[S1]`, `[S2]`), domain tags, citation audits, and source passage excerpts.
- **Claim Verification Matrix**: Filter claims by strict verdict (`Supported`, `Contradicted`, `Partially supported`, `Insufficient evidence`), review reasoning, confidence, linked primary sources, and flagged unverified or rejected references.
- **Final Report & Export**: Formatted Markdown synthesis with download options (`.md`, `.json`), copy tools, and peer review assessments.
- **Evaluation Panel**: Interactive observability dashboard visualizing Claim Support Rate, Citation Validity (tracked independently!), Subquestion Coverage, Task Completion, and Token Telemetry, with exportable GitHub README evaluation tables.
- **FastAPI Backend (`api.py`)**: Asynchronous run management (`POST /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/status`, `GET /api/runs/{id}/export`).

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Keys
Create or update `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run Automated Tests
All 45 test suites across Parts 1 through 7 run 100% offline without requiring external API credentials:
```bash
pytest -v
```

### 4. Launch ResearchMind Dashboard (Streamlit)
```bash
streamlit run app.py
```

### 5. Launch FastAPI Backend API
```bash
uvicorn api:app --reload --port 8000
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### 6. Run Terminal Pipeline Directly
```bash
python pipeline.py "Quantum computing error correction 2025"
```
