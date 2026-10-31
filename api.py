"""
FastAPI Backend API for ResearchMind Multi-Agent Research System.
Provides endpoints for:
- Submitting research queries with custom research settings
- Managing and tracking asynchronous runs
- Real-time status updates across LangGraph pipeline stages
- Exporting generated reports and evaluation metrics
"""

import asyncio
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# pyrefly: ignore[missing-import]
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore[missing-import]
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from pipeline import build_research_graph
from schemas import ResearchPlan, ResearchState

app = FastAPI(
    title="ResearchMind Multi-Agent API",
    description="Backend API for managing and observing multi-agent LangGraph research runs.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ResearchSettings(BaseModel):
    research_depth: str = Field(
        default="standard",
        description="Depth profile: 'quick' (1 revision), 'standard' (2 revisions), 'deep' (3 revisions)"
    )
    max_revisions: int = Field(default=2, ge=0, le=4, description="Maximum editorial revision iterations")
    focus_areas: List[str] = Field(
        default_factory=lambda: ["Architecture", "Benchmarks", "Limitations"],
        description="Priority focus categories for subquestion deconstruction"
    )

class CreateRunRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Research topic or question")
    settings: Optional[ResearchSettings] = Field(
        default_factory=ResearchSettings,
        description="Custom research configuration"
    )


class RunStatusResponse(BaseModel):
    run_id: str
    query: str
    status: str = Field(..., description="'queued', 'running', 'completed', or 'failed'")
    current_stage: Optional[str] = None
    progress_percent: int = 0
    completed_stages: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    created_at: str


class RunDetailResponse(BaseModel):
    run_id: str
    query: str
    status: str
    settings: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    research_results: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    verification_report: Optional[Dict[str, Any]] = None
    report_metadata: Optional[Dict[str, Any]] = None
    review_assessment: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    feedback: Optional[str] = None
    evaluation_result: Optional[Dict[str, Any]] = None
    markdown_table: Optional[str] = None
    run_trace: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: str
    completed_at: Optional[str] = None


# ── In-Memory Run Store & Manager ─────────────────────────────────────────────

PIPELINE_STAGES = [
    "planner",
    "orchestrator",
    "evidence_collector",
    "claim_extractor",
    "verifier",
    "verdict",
    "report_writer",
    "citation_validator",
    "reviewer",
    "critic",
    "evaluator"
]


class RunManager:
    """Thread-safe state manager for research runs."""

    def __init__(self):
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_run(self, query: str, settings: ResearchSettings) -> str:
        run_id = f"RUN-{uuid.uuid4().hex[:8].upper()}"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "query": query,
                "status": "queued",
                "settings": settings.model_dump(),
                "current_stage": "initialized",
                "progress_percent": 0,
                "completed_stages": [],
                "error": None,
                "errors": [],
                "start_time": time.time(),
                "created_at": now_str,
                "completed_at": None,
                "final_state": {},
            }
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run else None

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            summaries = []
            for r in sorted(self._runs.values(), key=lambda x: x["start_time"], reverse=True):
                elapsed = round(time.time() - r["start_time"], 2) if r["status"] == "running" else r.get("elapsed_seconds", 0.0)
                summaries.append({
                    "run_id": r["run_id"],
                    "query": r["query"],
                    "status": r["status"],
                    "progress_percent": r.get("progress_percent", 0),
                    "created_at": r["created_at"],
                    "elapsed_seconds": elapsed,
                    "error": r.get("error")
                })
            return summaries

    def execute_run(self, run_id: str):
        """Runs the compiled LangGraph workflow with streaming updates."""
        with self._lock:
            if run_id not in self._runs:
                return
            run = self._runs[run_id]
            run["status"] = "running"
            run["start_time"] = time.time()
            query = run["query"]
            settings = run["settings"]

        max_revisions = settings.get("max_revisions", 2)
        if settings.get("research_depth") == "quick":
            max_revisions = min(max_revisions, 1)

        initial_state: ResearchState = {
            "query": query,
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

        accumulated_state = dict(initial_state)
        try:
            workflow = build_research_graph()

            # Stream updates from the compiled LangGraph workflow
            for step_update in workflow.stream(initial_state, stream_mode="updates"):
                for node_name, node_output in step_update.items():
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)

                    with self._lock:
                        if run_id in self._runs:
                            r = self._runs[run_id]
                            r["current_stage"] = node_name
                            if node_name not in r["completed_stages"]:
                                r["completed_stages"].append(node_name)

                            # Estimate progress
                            stage_idx = PIPELINE_STAGES.index(node_name) if node_name in PIPELINE_STAGES else len(r["completed_stages"])
                            r["progress_percent"] = min(99, int((stage_idx + 1) / len(PIPELINE_STAGES) * 100))

                            # Capture errors if any
                            errs = accumulated_state.get("errors", [])
                            if errs:
                                r["errors"] = list(errs)

            elapsed = round(time.time() - accumulated_state.get("start_time", time.time()), 2)
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            run_succeeded = bool(accumulated_state.get("success")) and bool(accumulated_state.get("report"))
            run_errors = list(accumulated_state.get("errors") or [])

            with self._lock:
                if run_id in self._runs:
                    r = self._runs[run_id]
                    r["status"] = "completed" if run_succeeded else "failed"
                    if run_succeeded:
                        r["progress_percent"] = 100
                        r["current_stage"] = "completed"
                    elif not run_errors:
                        run_errors.append("Research pipeline terminated before producing a final report.")
                    r["error"] = None if run_succeeded else "; ".join(run_errors)
                    r["errors"] = run_errors
                    r["elapsed_seconds"] = elapsed
                    r["completed_at"] = now_str
                    r["final_state"] = accumulated_state

        except Exception as e:
            elapsed = round(time.time() - initial_state["start_time"], 2)
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            run_errors = list(accumulated_state.get("errors") or [])
            if str(e) not in run_errors:
                run_errors.append(str(e))
            accumulated_state["errors"] = run_errors
            with self._lock:
                if run_id in self._runs:
                    r = self._runs[run_id]
                    r["status"] = "failed"
                    r["error"] = str(e)
                    r["errors"] = run_errors
                    r["elapsed_seconds"] = elapsed
                    r["completed_at"] = now_str
                    r["final_state"] = accumulated_state


# Global run manager singleton
run_manager = RunManager()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ResearchMind Multi-Agent API",
        "graph_compiled": True,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    """Retrieve version-controlled benchmark questions."""
    bench_path = os.path.join(os.path.dirname(__file__), "benchmarks", "benchmark_questions.json")
    if os.path.exists(bench_path):
        import json
        with open(bench_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/api/runs", response_model=RunStatusResponse, status_code=202)
def create_run(request: CreateRunRequest, background_tasks: BackgroundTasks):
    """
    Submits a research query, validates settings, creates a run ID,
    and dispatches execution against the compiled LangGraph workflow.
    """
    settings = request.settings or ResearchSettings()
    run_id = run_manager.create_run(request.query, settings)

    # Dispatch LangGraph execution in background
    background_tasks.add_task(run_manager.execute_run, run_id)

    run = run_manager.get_run(run_id)
    return RunStatusResponse(
        run_id=run_id,
        query=request.query,
        status="queued",
        current_stage="initialized",
        progress_percent=0,
        completed_stages=[],
        elapsed_seconds=0.0,
        created_at=run["created_at"]
    )


@app.get("/api/runs", response_model=List[Dict[str, Any]])
def list_runs():
    """List all recent runs and their statuses."""
    return run_manager.list_runs()


@app.get("/api/runs/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(run_id: str):
    """Real-time status polling for long-running runs."""
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    elapsed = round(time.time() - run["start_time"], 2) if run["status"] == "running" else run.get("elapsed_seconds", 0.0)

    return RunStatusResponse(
        run_id=run["run_id"],
        query=run["query"],
        status=run["status"],
        current_stage=run.get("current_stage"),
        progress_percent=run.get("progress_percent", 0),
        completed_stages=run.get("completed_stages", []),
        error=run.get("error"),
        elapsed_seconds=elapsed,
        created_at=run["created_at"]
    )


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
def get_run_details(run_id: str):
    """Inspect full research run artifacts, claims, report, and evaluation."""
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    final = run.get("final_state", {})
    evaluation_result = final.get("evaluation_result") or {}
    plan = final.get("plan")
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else (dict(plan) if isinstance(plan, dict) else None)

    # Format claims
    claims_list = []
    for c in final.get("claims", []):
        if hasattr(c, "model_dump"):
            claims_list.append(c.model_dump())
        elif isinstance(c, dict):
            claims_list.append(c)

    # Format research results
    rr_list = []
    for r in final.get("research_results", []):
        if hasattr(r, "model_dump"):
            rr_list.append(r.model_dump())
        elif isinstance(r, dict):
            rr_list.append(r)

    elapsed = round(time.time() - run["start_time"], 2) if run["status"] == "running" else run.get("elapsed_seconds", 0.0)
    errors = list(run.get("errors") or [])
    if run.get("error") and run["error"] not in errors:
        errors.append(run["error"])
    for error in final.get("errors", []):
        if error not in errors:
            errors.append(error)

    return RunDetailResponse(
        run_id=run["run_id"],
        query=run["query"],
        status=run["status"],
        settings=run.get("settings"),
        plan=plan_dict,
        evidence=final.get("evidence"),
        research_results=rr_list,
        claims=claims_list,
        verification_report=final.get("verification_report"),
        report_metadata=final.get("report_metadata"),
        review_assessment=final.get("review_assessment"),
        report=final.get("report"),
        feedback=final.get("feedback"),
        evaluation_result=evaluation_result or None,
        markdown_table=evaluation_result.get("markdown_table", ""),
        run_trace=final.get("run_trace", []),
        errors=errors,
        elapsed_seconds=elapsed,
        created_at=run["created_at"],
        completed_at=run.get("completed_at")
    )


@app.get("/api/runs/{run_id}/export")
def export_report(run_id: str, format: str = Query("markdown", pattern="^(markdown|json)$")):
    """Export the synthesized report or full artifact package."""
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    final = run.get("final_state", {})
    report_text = final.get("report", "")

    if format == "markdown":
        if not report_text:
            report_text = f"# Research Report: {run['query']}\n\n*Status: {run['status']}*\n\nNo report generated yet."
        return PlainTextResponse(content=report_text, media_type="text/markdown")
    else:
        return get_run_details(run_id)
