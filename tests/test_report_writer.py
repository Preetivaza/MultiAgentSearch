import unittest
from unittest.mock import MagicMock, patch

from schemas import (
    Claim,
    ClaimVerification,
    EvidenceCollection,
    ResearchPlan,
    ResearchState,
    SourceMetadata,
    SubQuestion,
    VerdictType,
    VerificationReport,
)
from writer import (
    _compact_fallback_evidence,
    _normalize_report_headings,
    _report_meets_minimum_requirements,
    build_fallback_markdown_report,
    citation_validation_node,
    prepare_sources_and_citations,
    report_writer_node,
)
from pipeline import build_research_graph


class TestPart4ReportWriter(unittest.TestCase):
    """
    Tests for Part 4: Report Writer & Citation Validation
    - [x] Report uses the actual research evidence.
    - [x] Citations reference valid source records.
    - [x] Unsupported claims and contradictions are disclosed.
    - [x] Output is a well-structured Markdown report.
    - [x] Tests cover missing evidence and invalid citations.
    - [x] Previous LangGraph functionality still works.
    """

    def setUp(self):
        self.verified_source_1 = SourceMetadata(
            source_id="S1",
            url="https://arxiv.org/abs/2401.12345",
            title="Reasoning Scaling Laws in Large Models",
            snippet="Test-time compute scaling laws show a 14% improvement in mathematical olympiad benchmarks."
        )
        self.verified_source_2 = SourceMetadata(
            source_id="S2",
            url="https://nature.com/articles/qubit-benchmarks-2025",
            title="Quantum Fault-Tolerance Assessment",
            snippet="Topological surface codes reduced physical noise but failed to reach total fault-tolerance."
        )
        self.unique_sources = [
            self.verified_source_1.model_dump(),
            self.verified_source_2.model_dump()
        ]
        self.compiled_evidence = (
            "### Subquestion 1: Test-time compute\n"
            "Findings: Test-time compute scaling laws show a 14% improvement in mathematical olympiad benchmarks.\n\n"
            "### Subquestion 2: Quantum error rates\n"
            "Findings: Topological surface codes reduced physical noise but failed to reach total fault-tolerance."
        )
        self.sample_verifications = [
            {
                "claim_id": 1,
                "claim_text": "Test-time compute scaling shows a 14% improvement in math benchmarks.",
                "verdict": VerdictType.SUPPORTED,
                "reasoning": "Supported by arXiv study.",
                "matched_sources": [self.verified_source_1.model_dump()],
                "rejected_urls": []
            },
            {
                "claim_id": 2,
                "claim_text": "Quantum computers reached complete fault-tolerance with zero error.",
                "verdict": VerdictType.CONTRADICTED,
                "reasoning": "Contradicted: Nature article notes physical codes failed to reach total fault-tolerance.",
                "matched_sources": [self.verified_source_2.model_dump()],
                "rejected_urls": []
            },
            {
                "claim_id": 3,
                "claim_text": "Qubits achieved room-temperature coherence without vacuum chambers.",
                "verdict": VerdictType.INSUFFICIENT_EVIDENCE,
                "reasoning": "Rejected 1 invalid reference (https://fake-science-blog.com/room-temp). No verified data.",
                "matched_sources": [],
                "rejected_urls": ["https://fake-science-blog.com/room-temp"]
            }
        ]

    def write_report_offline(self, state):
        """Exercise deterministic report synthesis without calling Gemini."""
        with patch("agents.llm", None):
            return report_writer_node(state)

    def test_report_uses_the_actual_research_evidence(self):
        """Criterion: Report uses the actual research evidence."""
        state: ResearchState = {
            "query": "Quantum computing and AI reasoning breakthroughs",
            "evidence": {
                "compiled_research_context": self.compiled_evidence,
                "unique_sources": self.unique_sources
            },
            "verification_report": {
                "summary": "1 Supported, 1 Contradicted, 1 Insufficient Evidence",
                "verifications": self.sample_verifications
            },
            "errors": []
        }

        output = self.write_report_offline(state)
        report = output["report"]

        self.assertIn("Test-time compute scaling laws", report)
        self.assertIn("Topological surface codes", report)
        self.assertIn("14% improvement", report)

    def test_citations_reference_valid_source_records(self):
        """Criterion: Citations reference valid source records."""
        state: ResearchState = {
            "query": "Reasoning scaling",
            "evidence": {
                "compiled_research_context": self.compiled_evidence,
                "unique_sources": self.unique_sources
            },
            "verification_report": {
                "summary": "Verified",
                "verifications": self.sample_verifications
            },
            "errors": []
        }

        # Step 1: Writer generates report
        writer_out = self.write_report_offline(state)
        report_text = writer_out["report"]

        # Step 2: Validate citations
        val_state: ResearchState = {
            **state,
            "report": report_text
        }
        val_out = citation_validation_node(val_state)
        audit = val_out["report_metadata"]["citation_audit"]

        self.assertTrue(audit["is_valid"])
        self.assertEqual(len(audit["invalid_citations"]), 0)
        self.assertGreaterEqual(len(audit["valid_citations"]), 1)
        self.assertEqual(audit["citation_integrity_score"], 1.0)

    def test_unsupported_claims_and_contradictions_are_disclosed(self):
        """Criterion: Unsupported claims and contradictions are explicitly disclosed."""
        state: ResearchState = {
            "query": "Quantum and AI breakthroughs",
            "evidence": {
                "compiled_research_context": self.compiled_evidence,
                "unique_sources": self.unique_sources
            },
            "verification_report": {
                "summary": "Audit summary",
                "verifications": self.sample_verifications
            },
            "errors": []
        }

        output = self.write_report_offline(state)
        report = output["report"]

        # Disclose contradicted claim
        self.assertIn("CONTRADICTED", report.upper())
        self.assertIn("Quantum computers reached complete fault-tolerance", report)

        # Disclose unsupported claim / insufficient evidence
        self.assertIn("Insufficient Evidence", report)
        self.assertIn("room-temperature coherence", report)

    def test_output_is_a_well_structured_markdown_report(self):
        """Criterion: Output is a well-structured Markdown report."""
        state: ResearchState = {
            "query": "Quantum Computing",
            "evidence": {
                "compiled_research_context": self.compiled_evidence,
                "unique_sources": self.unique_sources
            },
            "verification_report": {
                "summary": "Summary",
                "verifications": self.sample_verifications
            },
            "errors": []
        }

        output = self.write_report_offline(state)
        report = output["report"]

        # Check required Markdown sections
        self.assertIn("# Research Report", report)
        self.assertIn("## Executive Summary", report)
        self.assertIn("## Key Findings & Evidence Synthesis", report)
        self.assertIn("## Verified Claims & Disclosures", report)
        self.assertIn("## Limitations, Gaps & Uncertainties", report)
        self.assertIn("## References", report)

        # Check references list format
        self.assertIn("[S1]", report)
        self.assertIn("https://arxiv.org/abs/2401.12345", report)

    def test_missing_evidence_handled_gracefully(self):
        """Criterion: Tests cover missing evidence."""
        empty_state: ResearchState = {
            "query": "Obscure Query with Zero Results",
            "evidence": None,
            "research_results": [],
            "verification_report": None,
            "errors": []
        }

        output = report_writer_node(empty_state)
        report = output["report"]

        self.assertIn("Executive Summary", report)
        self.assertIn("Unable to generate empirical findings", report)
        self.assertIn("Limitations", report)
        self.assertEqual(len(output["errors"]), 1)
        self.assertIn("warning", output["errors"][0].lower())

    def test_llm_draft_with_extra_sections_is_rejected(self):
        draft = """# Research Report: Test Topic

## Executive Summary
Summary.

## Key Findings & Evidence Synthesis
Findings.

## Evals
Unexpected section.

## Verified Claims & Disclosures
Audit.

## Limitations, Gaps & Uncertainties
Limitations.

## References
Sources.
"""

        self.assertFalse(_report_meets_minimum_requirements(draft, []))

    def test_fallback_context_is_bounded(self):
        long_context = "Evidence with useful content. " * 1000

        compact = _compact_fallback_evidence({}, long_context)

        self.assertLessEqual(len(compact), 10000)
        self.assertIn("research context shortened", compact)

    def test_fallback_source_headings_are_not_report_sections(self):
        source_context = "### Publisher heading\nEvidence excerpt."

        compact = _compact_fallback_evidence({}, source_context)

        self.assertIn("\\### Publisher heading", compact)

    def test_noncanonical_headings_are_escaped_in_final_report(self):
        draft = "# Research Report: Test Topic\n## Executive Summary\nSummary.\n# Source page title"

        normalized = _normalize_report_headings(draft)

        self.assertIn("# Research Report: Test Topic", normalized)
        self.assertIn("## Executive Summary", normalized)
        self.assertIn("\\# Source page title", normalized)

    def test_invalid_citations_are_detected_and_rejected(self):
        """Criterion: Tests cover invalid citations (citation validator flags invalid citations)."""
        # Create a report with a hallucinated citation [S99] and unverified URL
        fake_url = "https://unverified-hallucinated-source.com/leak"
        draft_with_fake_citations = (
            "# Research Report: Quantum Computing\n\n"
            "## Executive Summary\n"
            "Recent progress has accelerated [S1]. However, secret leaks allege [S99] that "
            f"quantum computers are already decrypting RSA [unverified document]({fake_url}).\n\n"
            "## Key Findings\n"
            "Test-time compute shows promise [S1].\n\n"
            "## References\n"
            "- [S1] https://arxiv.org/abs/2401.12345\n"
        )

        state: ResearchState = {
            "query": "Quantum Computing",
            "report": draft_with_fake_citations,
            "evidence": {
                "unique_sources": [self.verified_source_1.model_dump()]  # Only S1 is real
            },
            "errors": []
        }

        output = citation_validation_node(state)
        audit = output["report_metadata"]["citation_audit"]

        self.assertFalse(audit["is_valid"])
        self.assertIn("S99", audit["invalid_citations"])
        self.assertIn(fake_url, audit["invalid_citations"])
        self.assertLess(audit["citation_integrity_score"], 1.0)

        # Verify the report text was annotated to protect reader from hallucinated citations
        validated_report = output["report"]
        self.assertIn("⚠️ Unverified Citation: S99", validated_report)
        self.assertIn("⚠️ Unverified", validated_report)

    def test_previous_langgraph_functionality_still_works(self):
        """Criterion: Previous LangGraph functionality still works across all 4 parts."""
        workflow = build_research_graph()
        self.assertIsNotNone(workflow)

        # Check nodes are registered in the compiled graph
        nodes = workflow.get_graph().nodes
        self.assertIn("planner", nodes)
        self.assertIn("orchestrator", nodes)
        self.assertIn("evidence_collector", nodes)
        self.assertIn("claim_extractor", nodes)
        self.assertIn("verifier", nodes)
        self.assertIn("verdict", nodes)
        self.assertIn("report_writer", nodes)
        self.assertIn("citation_validator", nodes)
        self.assertIn("critic", nodes)


if __name__ == "__main__":
    unittest.main()
