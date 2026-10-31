import unittest
from unittest.mock import MagicMock, patch
# pyrefly: ignore[missing-import]
from fastapi.testclient import TestClient

from api import app, run_manager, ResearchSettings
from schemas import ResearchPlan, SubQuestion, SourceMetadata


class TestBackendAPI(unittest.TestCase):
    """
    Tests for Prompt 7.1 Backend API integration:
    - [x] Pydantic input validation (Query, settings)
    - [x] Run creation and ID generation
    - [x] Real-time status tracking and polling
    - [x] Full run artifact inspection (claims, evidence, report, evaluation)
    - [x] Report export in Markdown and JSON formats
    - [x] Benchmarks retrieval
    """

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["graph_compiled"])

    def test_get_benchmarks(self):
        response = self.client.get("/api/benchmarks")
        self.assertEqual(response.status_code, 200)
        benchmarks = response.json()
        self.assertIsInstance(benchmarks, list)
        self.assertGreater(len(benchmarks), 0)
        self.assertIn("query", benchmarks[0])

    def test_create_run_validation(self):
        # Query too short should fail with 422
        bad_response = self.client.post("/api/runs", json={"query": "ab"})
        self.assertEqual(bad_response.status_code, 422)

        # Valid query should return 202 Accepted with run status
        valid_payload = {
            "query": "Quantum Computing Scalability in 2025",
            "settings": {
                "research_depth": "quick",
                "max_revisions": 1,
                "focus_areas": ["Architecture", "Benchmarks"]
            }
        }
        with patch.object(run_manager, "execute_run", return_value=None):
            response = self.client.post("/api/runs", json=valid_payload)
            self.assertEqual(response.status_code, 202)
            data = response.json()
            self.assertTrue(data["run_id"].startswith("RUN-"))
            self.assertEqual(data["status"], "queued")
            self.assertEqual(data["query"], valid_payload["query"])

    def test_run_status_and_detail_tracking(self):
        # Create a simulated completed run in run_manager
        run_id = run_manager.create_run(
            "Test AI Scaling",
            ResearchSettings(research_depth="standard", max_revisions=2)
        )
        # Update run data
        with run_manager._lock:
            run = run_manager._runs[run_id]
            run["status"] = "completed"
            run["progress_percent"] = 100
            run["current_stage"] = "evaluator"
            run["completed_stages"] = ["planner", "orchestrator", "evaluator"]
            run["final_state"] = {
                "report": "# Test Report\n\nEmpirical AI scaling insights.",
                "claims": [{"id": 1, "claim_text": "Compute scales test-time accuracy."}],
                "evidence": {"unique_sources_count": 2},
                "evaluation_result": {"run_id": run_id, "status": "Completed"}
            }

        # Test GET /api/runs/{run_id}/status
        status_res = self.client.get(f"/api/runs/{run_id}/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["status"], "completed")
        self.assertEqual(status_data["progress_percent"], 100)
        self.assertIn("planner", status_data["completed_stages"])

        # Test GET /api/runs/{run_id}
        detail_res = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(detail_res.status_code, 200)
        detail_data = detail_res.json()
        self.assertEqual(detail_data["report"], "# Test Report\n\nEmpirical AI scaling insights.")
        self.assertEqual(len(detail_data["claims"]), 1)

        # Test GET /api/runs/{run_id}/export (Markdown)
        export_md = self.client.get(f"/api/runs/{run_id}/export?format=markdown")
        self.assertEqual(export_md.status_code, 200)
        self.assertIn("# Test Report", export_md.text)

        # Test GET /api/runs/{run_id}/export (JSON)
        export_json = self.client.get(f"/api/runs/{run_id}/export?format=json")
        self.assertEqual(export_json.status_code, 200)
        self.assertEqual(export_json.json()["run_id"], run_id)

    def test_run_not_found(self):
        res = self.client.get("/api/runs/NON_EXISTENT_RUN")
        self.assertEqual(res.status_code, 404)

    def test_early_pipeline_failure_is_reported_and_details_stay_available(self):
        class PlannerFailureGraph:
            def stream(self, initial_state, stream_mode):
                yield {
                    "planner": {
                        "plan": None,
                        "errors": ["Planner could not generate a plan."],
                    }
                }

        run_id = run_manager.create_run(
            "Test a failed research plan",
            ResearchSettings(research_depth="quick", max_revisions=0),
        )

        with patch("api.build_research_graph", return_value=PlannerFailureGraph()):
            run_manager.execute_run(run_id)

        status_res = self.client.get(f"/api/runs/{run_id}/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertEqual(status_res.json()["status"], "failed")
        self.assertIn("Planner could not generate a plan.", status_res.json()["error"])

        detail_res = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(detail_res.status_code, 200)
        detail = detail_res.json()
        self.assertEqual(detail["status"], "failed")
        self.assertIsNone(detail["evaluation_result"])
        self.assertIn("Planner could not generate a plan.", detail["errors"])

        export_json = self.client.get(f"/api/runs/{run_id}/export?format=json")
        self.assertEqual(export_json.status_code, 200)
        self.assertEqual(export_json.json()["status"], "failed")

    def test_unexpected_graph_exception_preserves_failure_details(self):
        class CrashingGraph:
            def stream(self, initial_state, stream_mode):
                raise RuntimeError("simulated graph crash")

        run_id = run_manager.create_run(
            "Test a graph execution error",
            ResearchSettings(research_depth="quick", max_revisions=0),
        )

        with patch("api.build_research_graph", return_value=CrashingGraph()):
            run_manager.execute_run(run_id)

        detail_res = self.client.get(f"/api/runs/{run_id}")
        self.assertEqual(detail_res.status_code, 200)
        detail = detail_res.json()
        self.assertEqual(detail["status"], "failed")
        self.assertIn("simulated graph crash", detail["errors"])

        export_md = self.client.get(f"/api/runs/{run_id}/export?format=markdown")
        self.assertEqual(export_md.status_code, 200)
        self.assertIn("Status: failed", export_md.text)


if __name__ == "__main__":
    unittest.main()
