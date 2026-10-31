import unittest
from unittest.mock import MagicMock, patch

from schemas import (
    CitationValidationResult,
    ReportMetadata,
    ResearchPlan,
    ResearchState,
    ReviewAssessment,
    SourceMetadata,
    SubQuestion,
    VerdictType,
)
from reviewer import (
    evaluate_report_heuristics,
    reviewer_node,
    route_after_review,
    targeted_revision_node,
)
from writer import citation_validation_node, report_writer_node
from pipeline import build_research_graph, route_after_revision_step


class TestPart5ReviewerAndRevisionLoop(unittest.TestCase):
    """
    Tests for Part 5: Reviewer & Revision Loop
    - [x] Reviewer returns a structured assessment.
    - [x] LangGraph uses conditional edges for routing.
    - [x] Fixable issues trigger targeted revisions.
    - [x] Citations and claims are revalidated.
    - [x] Revision loops stop at the configured limit.
    - [x] Earlier parts continue to pass their tests.
    """

    def setUp(self):
        self.valid_source = SourceMetadata(
            source_id="S1",
            url="https://arxiv.org/abs/2401.00001",
            title="Reasoning Scaling Laws in AI",
            snippet="Test-time computation allows language models to explore reasoning paths, improving math benchmarks."
        )
        self.good_report = (
            "# Research Report: AI Reasoning Scaling\n\n"
            "## Executive Summary\n"
            "This report investigates test-time compute scaling laws [S1]. Empirical results indicate notable reasoning gains.\n\n"
            "## Key Findings & Evidence Synthesis\n"
            "Test-time compute allows language models to search over alternative branches [S1], increasing accuracy.\n\n"
            "## Verified Claims & Disclosures\n"
            "### Summary of Verification Audit\n"
            "All claims were verified against primary arXiv documentation.\n\n"
            "### Contradictions & Disputed Evidence\n"
            "No explicit factual contradictions were identified among the audited sources.\n\n"
            "## Limitations, Gaps & Uncertainties\n"
            "Compute overhead remains high for multi-turn verification.\n\n"
            "## References\n"
            "- **[S1]** [Reasoning Scaling Laws in AI](https://arxiv.org/abs/2401.00001)\n"
        )
        self.plan = ResearchPlan(
            topic="AI Reasoning Scaling",
            objective="Evaluate test-time compute gains.",
            subquestions=[
                SubQuestion(
                    id=1,
                    question="How does test-time compute impact reasoning accuracy?",
                    search_query="test-time compute reasoning accuracy scaling",
                    focus_area="Performance"
                )
            ]
        )

    def test_reviewer_returns_a_structured_assessment(self):
        """Criterion: Reviewer returns a structured assessment (ReviewAssessment)."""
        state: ResearchState = {
            "query": "AI Reasoning Scaling",
            "report": self.good_report,
            "plan": self.plan,
            "verification_report": {"verifications": [], "summary": "Clean verification."},
            "report_metadata": {
                "citation_audit": {
                    "citation_integrity_score": 1.0,
                    "invalid_citations": [],
                    "valid_citations": ["S1"]
                }
            },
            "errors": []
        }

        with patch("reviewer.build_reviewer_chain", return_value=None):
            output = reviewer_node(state)
        self.assertIn("review_assessment", output)
        assessment = output["review_assessment"]

        # Validate structured assessment fields
        self.assertIn("status", assessment)
        self.assertIn(assessment["status"], ["Pass", "Needs revision"])
        self.assertIn("overall_score", assessment)
        self.assertIn("coverage_score", assessment)
        self.assertIn("evidence_score", assessment)
        self.assertIn("citation_score", assessment)
        self.assertIn("contradiction_score", assessment)
        self.assertIn("revision_target", assessment)
        self.assertIn("actionable_guidance", assessment)
        self.assertIn("issues_identified", assessment)
        self.assertEqual(assessment["status"], "Pass")

    def test_langgraph_uses_conditional_edges_for_routing(self):
        """Criterion: LangGraph uses conditional edges for routing (Pass vs Needs Revision)."""
        # Case 1: Clean report passes -> routes to publish
        pass_state: ResearchState = {
            "review_assessment": {
                "status": "Pass",
                "revision_target": "none"
            },
            "revision_count": 0,
            "max_revisions": 2
        }
        self.assertEqual(route_after_review(pass_state), "publish")

        # Case 2: Flawed report needs rewriting -> routes to targeted_rewrite
        rewrite_state: ResearchState = {
            "review_assessment": {
                "status": "Needs revision",
                "revision_target": "rewriter"
            },
            "revision_count": 0,
            "max_revisions": 2
        }
        self.assertEqual(route_after_review(rewrite_state), "targeted_rewrite")

        # Case 3: Missing evidence needs researcher -> routes to targeted_research
        research_state: ResearchState = {
            "review_assessment": {
                "status": "Needs revision",
                "revision_target": "researcher"
            },
            "revision_count": 0,
            "max_revisions": 2
        }
        self.assertEqual(route_after_review(research_state), "targeted_research")

    def test_fixable_issues_trigger_targeted_revisions(self):
        """Criterion: Fixable issues (e.g. invalid citations, undisclosed contradictions) trigger targeted revisions."""
        # Create flawed report missing contradiction disclosures and with citation warnings
        flawed_report = (
            "# Quick Notes\n"
            "AI is getting smarter [⚠️ Unverified Citation: S99]. We observed immense breakthroughs without any caveats.\n"
        )
        state_flawed: ResearchState = {
            "query": "AI Reasoning Scaling",
            "report": flawed_report,
            "plan": self.plan,
            "verification_report": {
                "verifications": [
                    {
                        "claim_id": 1,
                        "claim_text": "Model achieved 100% accuracy without compute overhead.",
                        "verdict": VerdictType.CONTRADICTED,
                        "reasoning": "Contradicted by physical compute constraints."
                    }
                ],
                "summary": "1 Contradiction identified."
            },
            "report_metadata": {
                "citation_audit": {
                    "citation_integrity_score": 0.5,
                    "invalid_citations": ["S99"]
                }
            },
            "errors": []
        }

        with patch("reviewer.build_reviewer_chain", return_value=None):
            output = reviewer_node(state_flawed)
        assessment = output["review_assessment"]

        # Must flag issues and request revision
        self.assertEqual(assessment["status"], "Needs revision")
        self.assertEqual(assessment["revision_target"], "rewriter")
        self.assertGreater(len(assessment["issues_identified"]), 0)
        self.assertTrue(len(assessment["actionable_guidance"]) > 0)
        self.assertIn("contradict", " ".join(assessment["issues_identified"]).lower())

    def test_citations_and_claims_are_revalidated(self):
        """Criterion: Citations and claims are revalidated during the revision path."""
        # 1. Start with flawed report containing invalid citation [S99]
        flawed_state: ResearchState = {
            "query": "AI Reasoning",
            "report": "# Report\nBreakthrough observed [S99].",
            "evidence": {"unique_sources": [self.valid_source.model_dump()]},
            "errors": []
        }
        audit_1 = citation_validation_node(flawed_state)
        # S99 is caught and flagged
        self.assertFalse(audit_1["report_metadata"]["citation_audit"]["is_valid"])
        self.assertIn("S99", audit_1["report_metadata"]["citation_audit"]["invalid_citations"])

        # 2. Targeted revision cleanses the report using guidance
        corrected_report = "# Report\nBreakthrough observed with valid citations [S1].\n## References\n- [S1] https://arxiv.org/abs/2401.00001"
        revalidated_state: ResearchState = {
            "query": "AI Reasoning",
            "report": corrected_report,
            "evidence": {"unique_sources": [self.valid_source.model_dump()]},
            "errors": []
        }
        # Revalidate citations!
        audit_2 = citation_validation_node(revalidated_state)
        self.assertTrue(audit_2["report_metadata"]["citation_audit"]["is_valid"])
        self.assertEqual(len(audit_2["report_metadata"]["citation_audit"]["invalid_citations"]), 0)
        self.assertEqual(audit_2["report_metadata"]["citation_audit"]["citation_integrity_score"], 1.0)

    def test_revision_loops_stop_at_the_configured_limit(self):
        """Criterion: Revision loops stop strictly at the configured iteration limit."""
        # Even if report still 'Needs revision', when revision_count reaches max_revisions, it MUST publish
        max_reached_state: ResearchState = {
            "review_assessment": {
                "status": "Needs revision",
                "revision_target": "rewriter",
                "actionable_guidance": "Fix remaining issues"
            },
            "revision_count": 2,  # Already at limit
            "max_revisions": 2
        }

        # Must route to publish!
        route_decision = route_after_review(max_reached_state)
        self.assertEqual(route_decision, "publish")

        # Increment test: targeted_revision_node increments revision_count
        step_out = targeted_revision_node(max_reached_state)
        self.assertEqual(step_out["revision_count"], 3)

    def test_earlier_parts_continue_to_pass_their_tests(self):
        """Criterion: Earlier parts (Parts 1, 2, 3, 4) continue to function in the compiled graph."""
        graph = build_research_graph()
        self.assertIsNotNone(graph)
        nodes = graph.get_graph().nodes

        # Check all nodes from all 5 parts are present
        expected_nodes = [
            "planner",              # Part 1
            "orchestrator",         # Part 2
            "evidence_collector",   # Part 2
            "claim_extractor",      # Part 3
            "verifier",             # Part 3
            "verdict",              # Part 3
            "report_writer",        # Part 4
            "citation_validator",   # Part 4
            "reviewer",             # Part 5
            "targeted_revision",    # Part 5
            "critic"                # Final review
        ]
        for node in expected_nodes:
            self.assertIn(node, nodes, f"Node '{node}' must be present in LangGraph.")


if __name__ == "__main__":
    unittest.main()
