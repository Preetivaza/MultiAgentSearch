import json
import time
import unittest
from unittest.mock import MagicMock, patch

from schemas import (
    EvidenceCollection,
    ResearchPlan,
    ResearchState,
    SourceMetadata,
    SubQuestion,
    SubQuestionResult,
)
from planner import (
    build_planner_chain,
    planner_node,
    validate_plan,
)
from orchestrator import (
    collect_evidence,
    execute_subquestion_research,
    orchestrate_research_tasks,
    parse_search_output,
    research_orchestrator_node,
    evidence_collector_node,
)
from pipeline import route_after_planning


class TestPart1ResearchPlanner(unittest.TestCase):
    """
    Tests for Part 1: Research Planner
    - Query analysis
    - LangGraph State handling (query, plan, errors)
    - Planner Node subquestion generation
    - Validated structured output (ResearchPlan)
    """

    def setUp(self):
        self.sample_plan_data = {
            "topic": "Quantum Computing Breakthroughs 2025",
            "objective": "Evaluate recent fault-tolerant quantum computing achievements and topological qubits.",
            "subquestions": [
                {
                    "id": 1,
                    "question": "What are the latest breakthroughs in topological qubits in 2025?",
                    "search_query": "topological qubits breakthroughs 2025 quantum computing",
                    "focus_area": "Architecture"
                },
                {
                    "id": 2,
                    "question": "Which quantum hardware companies achieved quantum advantage benchmarks?",
                    "search_query": "quantum advantage benchmarks 2025 IBM Google Quantinuum",
                    "focus_area": "Benchmarks"
                },
                {
                    "id": 3,
                    "question": "What remain the primary error-correction bottlenecks in scaling qubits?",
                    "search_query": "quantum error correction bottlenecks physical qubits scaling",
                    "focus_area": "Limitations"
                }
            ],
            "suggested_sources": ["arxiv.org", "nature.com", "ibm.com/quantum"]
        }

    def test_plan_validation_from_dict(self):
        """Test that validate_plan correctly creates a validated ResearchPlan Pydantic model."""
        plan = validate_plan(self.sample_plan_data)
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(len(plan.subquestions), 3)
        self.assertEqual(plan.subquestions[0].focus_area, "Architecture")
        self.assertTrue(plan.subquestions[0].search_query.startswith("topological"))

    def test_plan_validation_from_json_string(self):
        """Test parsing plan from markdown code-fenced JSON string."""
        raw_json_str = f"```json\n{json.dumps(self.sample_plan_data)}\n```"
        plan = validate_plan(raw_json_str)
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(plan.topic, "Quantum Computing Breakthroughs 2025")

    def test_planner_generates_validated_plan_with_subquestions(self):
        """
        Test that planner_node analyzes query and produces a validated ResearchPlan
        with structured subquestions ready for the next agent.
        """
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ResearchPlan.model_validate(self.sample_plan_data)

        with patch("planner.build_planner_chain", return_value=mock_chain):
            state: ResearchState = {
                "query": "Quantum Computing Breakthroughs 2025",
                "plan": None,
                "errors": []
            }
            output = planner_node(state, llm=mock_llm)

            self.assertIn("plan", output)
            plan = output["plan"]
            self.assertIsInstance(plan, ResearchPlan)
            self.assertEqual(len(plan.subquestions), 3)
            self.assertEqual(len(output["errors"]), 0)

            # Validate structured properties of subquestions
            for sq in plan.subquestions:
                self.assertIsInstance(sq.id, int)
                self.assertTrue(len(sq.question) > 5)
                self.assertTrue(len(sq.search_query) > 3)
                self.assertIsNotNone(sq.focus_area)

    def test_planner_empty_query_error_handling(self):
        """Test that planner gracefully catches empty queries and reports error in state."""
        state: ResearchState = {
            "query": "   ",
            "plan": None,
            "errors": []
        }
        output = planner_node(state)
        self.assertIsNone(output["plan"])
        self.assertEqual(len(output["errors"]), 1)
        self.assertIn("empty", output["errors"][0].lower())

    def test_planner_llm_failure_resilience(self):
        """Test that if LLM fails, planner reports error into state['errors'] without crashing."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("API Rate limit exceeded")

        with patch("planner.build_planner_chain", return_value=mock_chain):
            state: ResearchState = {
                "query": "Autonomous AI Agents",
                "plan": None,
                "errors": []
            }
            output = planner_node(state, llm=MagicMock())
            self.assertIsNone(output["plan"])
            self.assertEqual(len(output["errors"]), 1)
            self.assertIn("Rate limit", output["errors"][0])

    def test_routing_after_planning(self):
        """Test LangGraph conditional edge routing logic after planning node."""
        # Case 1: Valid plan -> route to orchestrator
        state_valid: ResearchState = {
            "query": "Test",
            "plan": ResearchPlan.model_validate(self.sample_plan_data),
            "errors": []
        }
        self.assertEqual(route_after_planning(state_valid), "orchestrator")

        # Case 2: Missing plan -> terminate early to end
        state_invalid: ResearchState = {
            "query": "Test",
            "plan": None,
            "errors": ["Failed planning"]
        }
        self.assertEqual(route_after_planning(state_invalid), "end")


class TestPart2ResearchExecution(unittest.TestCase):
    """
    Tests for Part 2: Research Execution (LangGraph Research Orchestrator)
    - Subquestions dispatched to independent research tasks
    - Concurrent execution verification
    - Preserving source URLs and relevant excerpts
    - Handling duplicate sources
    - Resilient error handling (failed search reported without discarding successful ones)
    - 100% Offline tests without real web requests
    """

    def setUp(self):
        self.subquestions = [
            SubQuestion(
                id=1,
                question="What is the architecture of Transformer-based reasoning models?",
                search_query="Transformer reasoning model architecture chain of thought",
                focus_area="Architecture"
            ),
            SubQuestion(
                id=2,
                question="What are the latest benchmark scores for OpenAI o1 and Gemini 1.5 Pro?",
                search_query="OpenAI o1 Gemini 1.5 Pro benchmark comparison 2024 2025",
                focus_area="Benchmarks"
            ),
            SubQuestion(
                id=3,
                question="What are the test-time compute limitations in modern LLMs?",
                search_query="test-time compute scaling laws limitations LLM inference",
                focus_area="Limitations"
            ),
        ]

    def test_each_subquestion_gets_own_research_result(self):
        """Criterion: Each planned subquestion gets its own research result."""
        def mock_search(query: str):
            return json.dumps({
                "success": True,
                "results": [
                    {
                        "title": f"Result for {query[:15]}",
                        "url": f"https://example.com/{abs(hash(query))}",
                        "snippet": f"Detailed snippet content answering: {query}"
                    }
                ]
            })

        results = orchestrate_research_tasks(
            subquestions=self.subquestions,
            search_func=mock_search
        )

        self.assertEqual(len(results), len(self.subquestions))
        for idx, sq in enumerate(self.subquestions):
            res = results[idx]
            self.assertEqual(res.subquestion_id, sq.id)
            self.assertEqual(res.question, sq.question)
            self.assertIsNone(res.error)
            self.assertTrue(len(res.sources) >= 1)

    def test_langchain_search_tools_are_called_with_structured_input(self):
        class MockTool:
            def __init__(self):
                self.inputs = []

            def __call__(self, query):
                raise AssertionError("LangChain tools should be called with invoke().")

            def invoke(self, tool_input):
                self.inputs.append(tool_input)
                return json.dumps({
                    "success": True,
                    "results": [{
                        "title": "Structured tool result",
                        "url": "https://example.com/structured-tool",
                        "snippet": "Evidence returned from a LangChain tool invocation.",
                    }],
                })

        tool = MockTool()
        result = execute_subquestion_research(self.subquestions[0], search_func=tool)

        self.assertIsNone(result.error)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(
            tool.inputs,
            [{"query": self.subquestions[0].search_query}],
        )

    def test_independent_research_tasks_can_execute_concurrently(self):
        """Criterion: Independent research tasks can execute concurrently."""
        def slow_search(query: str):
            time.sleep(0.1)  # Simulate network latency
            return json.dumps({
                "success": True,
                "results": [{"title": "Slow result", "url": "https://example.com/slow", "snippet": "Text"}]
            })

        start_time = time.time()
        results = orchestrate_research_tasks(
            subquestions=self.subquestions,
            search_func=slow_search,
            max_workers=3
        )
        total_duration = time.time() - start_time

        self.assertEqual(len(results), 3)
        # 3 tasks of 0.1s in parallel should finish in significantly less than sequential 0.3s (e.g. < 0.25s)
        self.assertLess(total_duration, 0.28)

    def test_results_preserve_source_urls_and_relevant_excerpts(self):
        """Criterion: Results preserve source URLs and relevant excerpts."""
        expected_url = "https://deepmind.google/research/gemini-scaling"
        expected_title = "Gemini Scaling Laws Report"
        expected_snippet = "We demonstrate novel inference-time computation techniques that boost mathematical reasoning."

        def mock_search(query: str):
            return json.dumps({
                "success": True,
                "results": [
                    {
                        "title": expected_title,
                        "url": expected_url,
                        "snippet": expected_snippet
                    }
                ]
            })

        single_sq = self.subquestions[0]
        res = execute_subquestion_research(single_sq, search_func=mock_search)

        self.assertIsNone(res.error)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0].url, expected_url)
        self.assertEqual(res.sources[0].title, expected_title)
        self.assertEqual(res.sources[0].snippet, expected_snippet)
        self.assertIn(expected_url, res.summary)
        self.assertIn("inference-time computation", res.summary)

    def test_duplicate_sources_are_handled(self):
        """Criterion: Duplicate sources are handled (deduplicated by URL while preserving metadata)."""
        shared_url = "https://arxiv.org/abs/2310.00000"
        unique_url_1 = "https://github.com/reasoning-model"
        unique_url_2 = "https://huggingface.co/blog/reasoning"

        task_results = [
            SubQuestionResult(
                subquestion_id=1,
                question="Question 1",
                search_query="Query 1",
                sources=[
                    SourceMetadata(url=shared_url, title="Shared ArXiv Paper", snippet="Excerpt from task 1"),
                    SourceMetadata(url=unique_url_1, title="GitHub Repo", snippet="Repo readme excerpt")
                ],
                summary="Task 1 summary"
            ),
            SubQuestionResult(
                subquestion_id=2,
                question="Question 2",
                search_query="Query 2",
                sources=[
                    SourceMetadata(url=shared_url, title="Shared ArXiv Paper", snippet="Excerpt from task 2"),
                    SourceMetadata(url=unique_url_2, title="Hugging Face Blog", snippet="Blog post excerpt")
                ],
                summary="Task 2 summary"
            ),
        ]

        evidence = collect_evidence(task_results)

        # 4 total source citations across 2 subquestions, but only 3 unique URLs
        self.assertEqual(evidence.total_tasks, 2)
        self.assertEqual(evidence.successful_tasks, 2)
        self.assertEqual(evidence.unique_sources_count, 3)

        urls = [s.url for s in evidence.unique_sources]
        self.assertEqual(len(urls), 3)
        self.assertEqual(urls.count(shared_url), 1)
        self.assertIn(unique_url_1, urls)
        self.assertIn(unique_url_2, urls)

    def test_failed_search_reported_without_discarding_successful_results(self):
        """
        Criterion: A failed search is reported without discarding successful results.
        If Subquestion 2 fails due to a network or search error, Subquestions 1 and 3
        remain completely intact and recorded in the Evidence Collection.
        """
        def flawed_search(query: str):
            if "benchmark" in query.lower():
                raise ConnectionError("Tavily search timeout on benchmark query")
            return json.dumps({
                "success": True,
                "results": [
                    {
                        "title": f"Valid source for: {query[:10]}",
                        "url": f"https://sources.org/{abs(hash(query))}",
                        "snippet": "Reliable excerpt for answering the question."
                    }
                ]
            })

        results = orchestrate_research_tasks(
            subquestions=self.subquestions,
            search_func=flawed_search
        )

        self.assertEqual(len(results), 3)

        # Subquestion 1: Success
        self.assertIsNone(results[0].error)
        self.assertEqual(len(results[0].sources), 1)

        # Subquestion 2: Flawed (Benchmark query) -> reported error without crashing
        self.assertIsNotNone(results[1].error)
        self.assertIn("timeout", results[1].error)
        self.assertEqual(len(results[1].sources), 0)

        # Subquestion 3: Success
        self.assertIsNone(results[2].error)
        self.assertEqual(len(results[2].sources), 1)

        # Now collect evidence
        evidence = collect_evidence(results)

        self.assertEqual(evidence.total_tasks, 3)
        self.assertEqual(evidence.successful_tasks, 2)
        self.assertEqual(evidence.failed_tasks, 1)
        self.assertEqual(evidence.unique_sources_count, 2)

        # Ensure failed subquestion status is transparently preserved in compiled context
        self.assertIn("Failed to retrieve sources", evidence.compiled_research_context)
        # Ensure successful subquestion findings are also preserved in compiled context
        self.assertIn("Reliable excerpt for answering the question", evidence.compiled_research_context)

    def test_evidence_collector_node_in_langgraph_state(self):
        """Test evidence_collector_node interaction with LangGraph state dictionary."""
        mock_result = SubQuestionResult(
            subquestion_id=1,
            question="What is LangGraph?",
            search_query="LangGraph documentation",
            sources=[
                SourceMetadata(url="https://langchain.com/langgraph", title="LangGraph Docs", snippet="Agent framework")
            ],
            summary="LangGraph enables multi-agent cyclic state graphs."
        )

        state: ResearchState = {
            "query": "LangGraph multi-agent architecture",
            "plan": None,
            "errors": [],
            "research_results": [mock_result]
        }

        output = evidence_collector_node(state)
        self.assertIn("evidence", output)
        ev = output["evidence"]
        self.assertEqual(ev["successful_tasks"], 1)
        self.assertEqual(ev["unique_sources_count"], 1)
        self.assertEqual(ev["unique_sources"][0]["url"], "https://langchain.com/langgraph")


if __name__ == "__main__":
    unittest.main()
