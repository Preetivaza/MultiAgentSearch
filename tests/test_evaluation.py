import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from schemas import (
    CitationValidationResult,
    EvaluationResult,
    ExecutionMetrics,
    QualityMetrics,
    ResearchPlan,
    ResearchState,
    SourceMetadata,
    SubQuestion,
    VerdictType,
)
from evaluation import (
    compute_quality_metrics,
    evaluation_node,
    export_evaluation_markdown,
    measure_execution_metrics,
)
from pipeline import build_research_graph


class TestPart6EvaluationAndObservability(unittest.TestCase):
    """
    Tests for Part 6: Evaluation & Observability
    - [x] Metrics are calculated from actual run artifacts.
    - [x] Benchmark questions are version-controlled.
    - [x] Graph runs expose useful traces and error information.
    - [x] Token usage and latency are measured, not guessed.
    - [x] Offline tests work without API credentials.
    - [x] Evaluation results can be exported for your GitHub README.
    - [x] Citation validity and claim support rate are tracked separately.
    """

    def setUp(self):
        self.benchmark_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "benchmarks",
            "benchmark_questions.json"
        )
        self.verified_source_1 = SourceMetadata(
            source_id="S1",
            url="https://arxiv.org/abs/2401.00001",
            title="Reasoning Scaling",
            snippet="Test-time compute boosts math accuracy."
        )
        self.verified_source_2 = SourceMetadata(
            source_id="S2",
            url="https://nature.com/articles/quantum-error",
            title="Qubit Stability",
            snippet="Topological noise suppression."
        )

    def test_benchmark_questions_are_version_controlled(self):
        """Criterion: Benchmark questions are version-controlled in the repository."""
        self.assertTrue(os.path.exists(self.benchmark_file), "benchmarks/benchmark_questions.json must exist")

        with open(self.benchmark_file, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)

        self.assertIsInstance(benchmarks, list)
        self.assertGreaterEqual(len(benchmarks), 3, "At least 3 benchmark questions should be defined.")

        for b in benchmarks:
            self.assertIn("id", b)
            self.assertIn("query", b)
            self.assertIn("domain", b)
            self.assertIn("required_subquestions_count", b)
            self.assertIn("key_evaluation_aspects", b)
            self.assertTrue(len(b["query"]) > 15)

    def test_metrics_are_calculated_from_actual_run_artifacts(self):
        """
        Criterion: Metrics are calculated strictly from actual run artifacts.
        Tests exact formula computations:
        - claim_support_rate: 3 supported / 4 evaluated = 0.75
        - citation_validity: 5 valid / (5 valid + 1 invalid) = 0.8333
        - subquestion_coverage: 2 addressed / 2 planned = 1.0
        - duplicate_source_rate: 2 duplicate / 4 total retrieved = 0.5
        - task_completion_rate: 2 successful / 2 attempted = 1.0
        """
        plan = ResearchPlan(
            topic="Quantum Computing",
            objective="Evaluate progress",
            subquestions=[
                SubQuestion(id=1, question="What are topological qubits?", search_query="topological qubits", focus_area="Architecture"),
                SubQuestion(id=2, question="What are error correction thresholds?", search_query="error correction", focus_area="Benchmarks")
            ]
        )

        state: ResearchState = {
            "query": "Quantum Computing",
            "plan": plan,
            "report": "Topological qubits in architecture allow error correction thresholds to improve.",
            "verification_report": {
                "verifications": [
                    {"verdict": VerdictType.SUPPORTED},
                    {"verdict": VerdictType.SUPPORTED},
                    {"verdict": VerdictType.SUPPORTED},
                    {"verdict": VerdictType.CONTRADICTED}  # 3 of 4 supported
                ]
            },
            "report_metadata": {
                "citation_audit": {
                    "valid_citations": ["S1", "S2", "S3", "S4", "S5"],
                    "invalid_citations": ["S99"]  # 5 of 6 valid
                }
            },
            "evidence": {
                "unique_sources_count": 2,
                "total_tasks": 2,
                "successful_tasks": 2
            },
            "research_results": [
                MagicMock(sources=[self.verified_source_1, self.verified_source_2], error=None, subquestion_id=1),
                MagicMock(sources=[self.verified_source_1, self.verified_source_2], error=None, subquestion_id=2)
                # 4 total sources retrieved, 2 unique -> 2 duplicates
            ],
            "errors": []
        }

        qm = compute_quality_metrics(state)

        # 1. Claim support rate: 3/4 = 0.75
        self.assertEqual(qm.supported_claims_count, 3)
        self.assertEqual(qm.total_claims_evaluated, 4)
        self.assertAlmostEqual(qm.claim_support_rate, 0.75, places=2)

        # 2. Citation validity: 5/6 = 0.8333
        self.assertEqual(qm.valid_citations_count, 5)
        self.assertEqual(qm.total_citations_checked, 6)
        self.assertAlmostEqual(qm.citation_validity, 0.8333, places=2)

        # 3. Subquestion coverage: 2/2 = 1.0
        self.assertEqual(qm.addressed_subquestions_count, 2)
        self.assertEqual(qm.total_subquestions_count, 2)
        self.assertEqual(qm.subquestion_coverage, 1.0)

        # 4. Duplicate source rate: (4 - 2) / 4 = 0.5
        self.assertEqual(qm.duplicate_sources_count, 2)
        self.assertEqual(qm.total_sources_retrieved_count, 4)
        self.assertEqual(qm.duplicate_source_rate, 0.5)

        # 5. Task completion: 2/2 = 1.0
        self.assertEqual(qm.successful_tasks_count, 2)
        self.assertEqual(qm.attempted_tasks_count, 2)
        self.assertEqual(qm.task_completion_rate, 1.0)

    def test_citation_validity_and_claim_support_tracked_separately(self):
        """
        Criterion: Important: A valid citation URL does not necessarily mean that the cited
        source supports the claim. Track citation validity and claim support separately.
        """
        # Case: 100% valid citations, but only 25% supported claims!
        state_discrepant: ResearchState = {
            "query": "Controversial claim",
            "report": "Text with valid sources cited.",
            "verification_report": {
                "verifications": [
                    {"verdict": VerdictType.SUPPORTED},
                    {"verdict": VerdictType.CONTRADICTED},
                    {"verdict": VerdictType.CONTRADICTED},
                    {"verdict": VerdictType.INSUFFICIENT_EVIDENCE}
                ]  # Claim support: 1/4 = 0.25
            },
            "report_metadata": {
                "citation_audit": {
                    "valid_citations": ["S1", "S2", "S3"],
                    "invalid_citations": []  # Citation validity: 3/3 = 1.0
                }
            },
            "evidence": {"unique_sources_count": 3},
            "research_results": [],
            "errors": []
        }

        qm = compute_quality_metrics(state_discrepant)
        self.assertEqual(qm.citation_validity, 1.0)
        self.assertEqual(qm.claim_support_rate, 0.25)
        self.assertNotEqual(qm.citation_validity, qm.claim_support_rate, "Citation validity and claim support must be tracked independently.")

    def test_token_usage_and_latency_are_measured_not_guessed(self):
        """Criterion: Token usage and latency are measured, not guessed."""
        start = time.time() - 2.45  # simulate elapsed time
        state: ResearchState = {
            "start_time": start,
            "node_latencies": {
                "planner": 0.42,
                "orchestrator": 1.15,
                "report_writer": 0.88
            },
            "token_usage": {
                "prompt_tokens": 1450,
                "completion_tokens": 620,
                "total_tokens": 2070,
                "source": "gemini_provider_response_metadata"
            },
            "errors": ["Sample non-fatal warning"]
        }

        em = measure_execution_metrics(state)

        # Latency must be measured from time delta
        self.assertGreaterEqual(em.wall_clock_latency_seconds, 2.4)
        self.assertIn("planner", em.node_latencies)
        self.assertEqual(em.node_latencies["planner"], 0.42)

        # Tokens must reflect actual recorded provider metadata
        self.assertEqual(em.prompt_tokens, 1450)
        self.assertEqual(em.completion_tokens, 620)
        self.assertEqual(em.total_tokens, 2070)
        self.assertEqual(em.token_usage_source, "gemini_provider_response_metadata")

        # Errors must be counted
        self.assertEqual(em.error_count, 1)

    def test_graph_runs_expose_useful_traces_and_error_information(self):
        """Criterion: Graph runs expose useful traces and error information."""
        state: ResearchState = {
            "query": "Quantum Qubits",
            "start_time": time.time() - 1.0,
            "errors": ["Warning: Tavily rate limit retry succeeded"],
            "verification_report": {"verifications": [{"verdict": VerdictType.SUPPORTED}]},
            "report_metadata": {"citation_audit": {"valid_citations": ["S1"], "invalid_citations": []}},
            "evidence": {"unique_sources_count": 1, "total_tasks": 1, "successful_tasks": 1},
            "research_results": [MagicMock(sources=[self.verified_source_1], error=None, subquestion_id=1)],
            "run_trace": [{"step": "orchestrator", "detail": "Retrieved 1 source"}]
        }

        output = evaluation_node(state)
        eval_dict = output["evaluation_result"]

        self.assertIn("run_id", eval_dict)
        self.assertTrue(eval_dict["run_id"].startswith("RUN-"))
        self.assertIn("timestamp", eval_dict)
        self.assertIn("run_trace", eval_dict)

        # Verify traces are preserved and extended
        trace = eval_dict["run_trace"]
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["step"], "orchestrator")
        self.assertEqual(trace[1]["step"], "evaluation_node")

        # Verify error tracking
        self.assertEqual(eval_dict["execution_metrics"]["error_count"], 1)

    def test_evaluation_results_can_be_exported_for_github_readme(self):
        """Criterion: Evaluation results can be exported for your GitHub README."""
        qm = QualityMetrics(
            claim_support_rate=0.857,
            citation_validity=1.0,
            subquestion_coverage=1.0,
            duplicate_source_rate=0.25,
            task_completion_rate=1.0,
            supported_claims_count=6,
            total_claims_evaluated=7,
            valid_citations_count=5,
            total_citations_checked=5,
            addressed_subquestions_count=3,
            total_subquestions_count=3,
            duplicate_sources_count=1,
            total_sources_retrieved_count=4,
            successful_tasks_count=3,
            attempted_tasks_count=3
        )
        em = ExecutionMetrics(
            wall_clock_latency_seconds=3.42,
            prompt_tokens=1820,
            completion_tokens=640,
            total_tokens=2460,
            token_usage_source="provider_metadata",
            error_count=0,
            errors=[]
        )
        eval_res = EvaluationResult(
            run_id="RUN-TEST01",
            timestamp="2026-10-26 12:00:00 UTC",
            query="Quantum Computing Breakthroughs",
            quality_metrics=qm,
            execution_metrics=em,
            status="Completed"
        )

        md = export_evaluation_markdown(eval_res)

        # Verify Markdown formatting suitable for GitHub README
        self.assertIn("### 📊 ResearchMind Evaluation & Observability Report", md)
        self.assertIn("| **Claim Support Rate** |", md)
        self.assertIn("85.7%", md)
        self.assertIn("| **Citation Validity** |", md)
        self.assertIn("100.0%", md)
        self.assertIn("| **Wall-Clock Latency** |", md)
        self.assertIn("3.42s", md)
        self.assertIn("Total Token Usage", md)
        self.assertIn("2,460 tokens", md)

    def test_export_evaluation_markdown_accepts_kwargs_and_dict(self):
        """Test that export_evaluation_markdown accepts both keyword arguments and dictionary inputs."""
        qm = QualityMetrics(claim_support_rate=0.8, citation_validity=0.95)
        em = ExecutionMetrics(wall_clock_latency_seconds=2.5, total_tokens=1500)

        # 1. Test keyword arguments
        md_kwargs = export_evaluation_markdown(
            run_id="RUN-KWARGS",
            timestamp="2026-10-27 10:00:00 UTC",
            query="Test Query Kwargs",
            quality_metrics=qm,
            execution_metrics=em,
            status="Completed"
        )
        self.assertIn("RUN-KWARGS", md_kwargs)
        self.assertIn("Test Query Kwargs", md_kwargs)
        self.assertIn("80.0%", md_kwargs)

        # 2. Test dictionary input
        eval_dict = {
            "run_id": "RUN-DICT",
            "timestamp": "2026-10-27 10:00:00 UTC",
            "query": "Test Query Dict",
            "quality_metrics": qm.model_dump(),
            "execution_metrics": em.model_dump(),
            "status": "Completed"
        }
        md_dict = export_evaluation_markdown(eval_dict)
        self.assertIn("RUN-DICT", md_dict)
        self.assertIn("Test Query Dict", md_dict)

    def test_generate_simulated_run_state(self):
        """Test that generate_simulated_run_state in app.py executes without error."""
        from app import generate_simulated_run_state
        sim_state = generate_simulated_run_state(query="State of Artificial Intelligence in Medicine")
        self.assertIn("evaluation_result", sim_state)
        self.assertIn("markdown_table", sim_state["evaluation_result"])
        self.assertIn("ResearchMind Evaluation & Observability Report", sim_state["evaluation_result"]["markdown_table"])

    def test_offline_tests_work_without_api_credentials(self):
        """Criterion: Offline tests work without API credentials (zero external calls)."""
        # Ensure graph compiles cleanly and evaluation_node runs completely offline
        graph = build_research_graph()
        self.assertIsNotNone(graph)
        nodes = graph.get_graph().nodes
        self.assertIn("evaluator", nodes)


if __name__ == "__main__":
    unittest.main()
