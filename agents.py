
import os
import logging

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
# pyrefly: ignore[missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url

logger = logging.getLogger(__name__)

# Load environment variables from the project-root .env file.
load_dotenv(override=True)

# --------------------------------------------------
# Gemini configuration
# --------------------------------------------------

api_key = os.getenv("GOOGLE_API_KEY", "").strip()

if not api_key:
    logger.warning(
        "GOOGLE_API_KEY is missing. Live research requires "
        "a valid Google Gemini API key."
    )

# Centralize the model name so all components use the same model.
best_model = "gemini-3.5-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=best_model,
    google_api_key=api_key,
    thinking_level="minimal",
)
logger.info("ResearchMind configured with model: %s", best_model)


# --------------------------------------------------
# Search Agent
# --------------------------------------------------

def build_search_agent():
    """Create an agent that searches the web."""
    return create_react_agent(
        model=llm,
        tools=[web_search],
    )


# --------------------------------------------------
# Reader Agent
# --------------------------------------------------

def build_reader_agent():
    """Create an agent that retrieves and reads source URLs."""
    return create_react_agent(
        model=llm,
        tools=[scrape_url],
    )


# --------------------------------------------------
# Writer Chain
# --------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. "
            "Write clear, structured, evidence-based reports. "
            "Do not invent facts or citations.",
        ),
        (
            "human",
            """Write a detailed research report on the topic below.

Topic:
{topic}

Research gathered:
{research}

Structure the report as:
1. Introduction
2. Key Findings (at least three well-explained points)
3. Limitations or Uncertainties
4. Conclusion
5. Sources (include only URLs provided in the research)

Use a factual, professional tone. Clearly distinguish evidence
from uncertainty. Do not fabricate source URLs.""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()


# --------------------------------------------------
# Critic Chain
# --------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a rigorous, constructive research critic. "
            "Evaluate factual support, source quality, completeness, "
            "reasoning, and citation accuracy.",
        ),
        (
            "human",
            """Review the research report below.

Report:
{report}

Respond in this format:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
...""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()


# --------------------------------------------------
# Project exports
# Keep these imports for compatibility with existing modules.
# --------------------------------------------------

from schemas import (
    ResearchState,
    ResearchPlan,
    SubQuestion,
    SubQuestionResult,
    SourceMetadata,
    EvidenceCollection,
    Claim,
    ClaimVerification,
    VerificationReport,
    VerdictType,
    CitationValidationResult,
    ReportMetadata,
    ReviewAssessment,
    QualityMetrics,
    ExecutionMetrics,
    EvaluationResult,
)

from planner import (
    planner_node,
    build_planner_chain,
    validate_plan,
)

from orchestrator import (
    research_orchestrator_node,
    evidence_collector_node,
    orchestrate_research_tasks,
    collect_evidence,
    execute_subquestion_research,
)

from verification import (
    claim_extraction_node,
    evidence_verification_node,
    verdict_node,
    verify_single_claim,
    extract_claims_heuristic,
)

from writer import (
    report_writer_node,
    citation_validation_node,
    prepare_sources_and_citations,
    build_fallback_markdown_report,
)

from reviewer import (
    reviewer_node,
    targeted_revision_node,
    route_after_review,
    evaluate_report_heuristics,
    build_reviewer_chain,
)

from evaluation import (
    evaluation_node,
    compute_quality_metrics,
    measure_execution_metrics,
    export_evaluation_markdown,
)
