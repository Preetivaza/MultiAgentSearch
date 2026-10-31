import unittest
from unittest.mock import MagicMock, patch

from schemas import (
    Claim,
    ClaimVerification,
    EvidenceCollection,
    ResearchState,
    SourceMetadata,
    VerdictType,
    VerificationReport,
)
from verification import (
    claim_extraction_node,
    evidence_verification_node,
    extract_claims_heuristic,
    normalize_url,
    verdict_node,
    verify_single_claim,
)


class TestPart3ClaimVerification(unittest.TestCase):
    """
    Tests for Part 3: Claim Verification
    - [x] Claims are extracted individually.
    - [x] Every verdict links to real source records.
    - [x] Contradictory evidence is identified.
    - [x] Invalid source references are rejected.
    - [x] Failure cases are covered by automated tests.
    """

    def setUp(self):
        self.verified_source_1 = SourceMetadata(
            url="https://arxiv.org/abs/2401.00001",
            title="Transformer Scaling and Test-Time Compute",
            snippet="Test-time computation allows language models to explore alternative reasoning branches, boosting accuracy on competition math."
        )
        self.verified_source_2 = SourceMetadata(
            url="https://nature.com/articles/quantum-error-2025",
            title="Topological Qubit Fault Tolerance Study",
            snippet="The experimental architecture failed to suppress physical bit-flip errors and did not achieve fault-tolerant threshold."
        )
        self.verified_sources_map = {
            normalize_url(self.verified_source_1.url): self.verified_source_1,
            normalize_url(self.verified_source_2.url): self.verified_source_2,
        }

    def test_claims_are_extracted_individually(self):
        """Criterion: Claims are extracted individually as discrete atomic statements."""
        sample_text = (
            "Test-time compute boosts model accuracy on competition math problems. "
            "Topological qubits achieved fault-tolerant quantum error correction in late 2024. "
            "DeepMind announced the Gemini 2.0 Flash architecture with native multimodal tools."
        )
        claims = extract_claims_heuristic(sample_text)

        self.assertGreaterEqual(len(claims), 3)
        for c in claims:
            self.assertIsInstance(c, Claim)
            self.assertIsInstance(c.id, int)
            self.assertTrue(len(c.claim_text) >= 10)

        # Check unique claim contents
        texts = [c.claim_text for c in claims]
        self.assertEqual(len(texts), len(set(texts)), "Extracted claims should be unique individual statements")

    def test_every_verdict_links_to_real_source_records(self):
        """Criterion: Every verdict links to real source records from Part 2."""
        claim = Claim(
            id=1,
            claim_text="Test-time computation allows language models to explore reasoning branches and boost accuracy.",
            referenced_urls=["https://arxiv.org/abs/2401.00001"]
        )

        verdict_res = verify_single_claim(claim, self.verified_sources_map)

        self.assertEqual(verdict_res.verdict, VerdictType.SUPPORTED)
        self.assertEqual(len(verdict_res.matched_sources), 1)
        # Verify it points to the EXACT real source metadata
        matched = verdict_res.matched_sources[0]
        self.assertEqual(matched.url, self.verified_source_1.url)
        self.assertEqual(matched.title, self.verified_source_1.title)
        self.assertEqual(matched.snippet, self.verified_source_1.snippet)
        self.assertEqual(len(verdict_res.rejected_urls), 0)

    def test_contradictory_evidence_is_identified(self):
        """Criterion: Contradictory evidence is identified when sources conflict with claims."""
        # Claim affirms success, but source explicitly states it failed to suppress errors
        claim = Claim(
            id=2,
            claim_text="The experimental architecture successfully suppressed physical bit-flip errors and achieved fault-tolerance.",
            referenced_urls=["https://nature.com/articles/quantum-error-2025"]
        )

        verdict_res = verify_single_claim(claim, self.verified_sources_map)

        self.assertEqual(verdict_res.verdict, VerdictType.CONTRADICTED)
        self.assertIn("Contradictory evidence identified", verdict_res.reasoning)
        self.assertEqual(len(verdict_res.matched_sources), 1)
        self.assertEqual(verdict_res.matched_sources[0].url, self.verified_source_2.url)

    def test_invalid_source_references_are_rejected(self):
        """Criterion: Invalid source references are rejected."""
        fake_url = "https://unverified-hallucinated-domain.com/bogus-paper"
        claim = Claim(
            id=3,
            claim_text="Quantum teleportation of solid-state macromolecules was demonstrated across 100 kilometers.",
            referenced_urls=[fake_url]
        )

        verdict_res = verify_single_claim(claim, self.verified_sources_map)

        # Must reject the invalid URL and assign Insufficient Evidence
        self.assertEqual(verdict_res.verdict, VerdictType.INSUFFICIENT_EVIDENCE)
        self.assertIn(fake_url, verdict_res.rejected_urls)
        self.assertEqual(len(verdict_res.matched_sources), 0)
        self.assertIn("Rejected 1 invalid source reference", verdict_res.reasoning)

    def test_partially_supported_verdict(self):
        """Criterion: Partially supported claims are identified when evidence covers only part of the statement."""
        claim = Claim(
            id=4,
            claim_text="Test-time computation requires massive cooling infrastructure and supercomputers to run.",
            referenced_urls=["https://arxiv.org/abs/2401.00001"]
        )

        verdict_res = verify_single_claim(claim, self.verified_sources_map)
        # Some words match (test-time computation), but infrastructure claims are uncorroborated
        self.assertIn(verdict_res.verdict, [VerdictType.PARTIALLY_SUPPORTED, VerdictType.INSUFFICIENT_EVIDENCE])

    def test_failure_cases_empty_context(self):
        """Criterion: Failure cases are covered (empty research findings)."""
        empty_state: ResearchState = {
            "query": "Empty test",
            "evidence": None,
            "research_results": [],
            "errors": []
        }
        res = claim_extraction_node(empty_state)
        self.assertEqual(res["claims"], [])
        self.assertEqual(len(res["errors"]), 1)
        self.assertIn("empty", res["errors"][0].lower())

    def test_failure_cases_empty_claims_to_verifier(self):
        """Criterion: Failure cases are covered (empty claims passed to verifier)."""
        state_no_claims: ResearchState = {
            "query": "No claims",
            "claims": [],
            "evidence": {"unique_sources": [self.verified_source_1.model_dump()]},
            "errors": []
        }
        res = evidence_verification_node(state_no_claims)
        report = res["verification_report"]
        self.assertEqual(report["total_claims"], 0)
        self.assertEqual(len(res["errors"]), 1)

    def test_evidence_verification_node_report_aggregation(self):
        """Test aggregate reporting across multiple claims."""
        claims = [
            Claim(
                id=1,
                claim_text="Test-time computation boosts model accuracy on competition math.",
                referenced_urls=["https://arxiv.org/abs/2401.00001"]
            ),
            Claim(
                id=2,
                claim_text="Topological qubits succeeded and suppressed all physical bit-flip errors.",
                referenced_urls=["https://nature.com/articles/quantum-error-2025"]
            ),
            Claim(
                id=3,
                claim_text="Cold fusion was stabilized in an unverified kitchen microwave.",
                referenced_urls=["https://unverified-hallucinated.org/fake"]
            )
        ]

        state: ResearchState = {
            "query": "Quantum and AI",
            "claims": claims,
            "evidence": {
                "unique_sources": [
                    self.verified_source_1.model_dump(),
                    self.verified_source_2.model_dump()
                ]
            },
            "errors": []
        }

        output = evidence_verification_node(state)
        report = output["verification_report"]

        self.assertEqual(report["total_claims"], 3)
        self.assertEqual(report["supported_count"], 1)
        self.assertEqual(report["contradicted_count"], 1)
        self.assertEqual(report["insufficient_evidence_count"], 1)
        self.assertEqual(report["rejected_references_count"], 1)
        self.assertIn("1 Supported", report["summary"])
        self.assertIn("1 Contradicted", report["summary"])


if __name__ == "__main__":
    unittest.main()
