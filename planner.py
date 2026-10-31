
import json
import logging
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import ResearchPlan, ResearchState

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """
You are an expert Research Director and AI Planning Agent.

Analyze the user's research query and create an actionable research plan.

Requirements:
1. Generate 2 to 5 distinct, relevant subquestions.
2. Cover complementary research dimensions where appropriate:
   - Current progress and state of the art
   - Core mechanisms, architecture, and key players
   - Real-world applications, benchmarks, and limitations
   - Challenges, trade-offs, and future directions
3. For every subquestion, provide:
   - id: A unique integer starting from 1
   - question: A specific analytical research question
   - search_query: An optimized web search query
   - focus_area: A concise research category
4. Include an overarching topic summary and research objective.
5. Do not invent sources or claim that web searches have already been performed.
6. Return output conforming to the ResearchPlan schema.
"""


def build_planner_chain(llm: Any):
    """
    Build a planner chain.

    Prefer native structured output when supported.
    Fall back to Pydantic JSON parsing if structured output
    is unavailable or fails during chain construction.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            (
                "human",
                "Research query: {query}\n"
                "Create a comprehensive research plan.",
            ),
        ]
    )

    try:
        structured_llm = llm.with_structured_output(ResearchPlan)
        return prompt | structured_llm

    except (AttributeError, NotImplementedError, TypeError, ValueError) as exc:
        logger.warning(
            "Structured output unavailable; using Pydantic parser: %s",
            exc,
        )

    parser = PydanticOutputParser(pydantic_object=ResearchPlan)

    fallback_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                PLANNER_SYSTEM_PROMPT
                + "\n\n{format_instructions}",
            ),
            (
                "human",
                "Research query: {query}\n"
                "Create a comprehensive research plan.",
            ),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    return fallback_prompt | llm | parser


def validate_plan(plan_data: Any) -> ResearchPlan:
    """
    Convert supported model outputs into a validated ResearchPlan.
    """
    if isinstance(plan_data, ResearchPlan):
        plan = plan_data

    elif isinstance(plan_data, dict):
        plan = ResearchPlan.model_validate(plan_data)

    elif isinstance(plan_data, str):
        clean_text = plan_data.strip()

        # Support JSON wrapped in Markdown code fences.
        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            clean_text,
            flags=re.IGNORECASE,
        )

        if match:
            clean_text = match.group(1).strip()

        data = json.loads(clean_text)
        plan = ResearchPlan.model_validate(data)

    else:
        raise ValueError(
            f"Unsupported planner output type: "
            f"{type(plan_data).__name__}"
        )

    if not plan.subquestions:
        raise ValueError(
            "Planner returned no subquestions."
        )

    if not 2 <= len(plan.subquestions) <= 5:
        raise ValueError(
            "Planner must return between 2 and 5 subquestions; "
            f"received {len(plan.subquestions)}."
        )

    ids = [question.id for question in plan.subquestions]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Planner returned duplicate subquestion IDs."
        )

    for question in plan.subquestions:
        if not question.question.strip():
            raise ValueError(
                f"Subquestion {question.id} has an empty question."
            )

        if not question.search_query.strip():
            raise ValueError(
                f"Subquestion {question.id} has an empty search query."
            )

        if not question.focus_area.strip():
            raise ValueError(
                f"Subquestion {question.id} has an empty focus area."
            )

    return plan


def planner_node(
    state: ResearchState,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    LangGraph planner node.

    Reads the query, generates a research plan, validates it,
    and returns the plan and errors in LangGraph state format.
    """
    query = (state.get("query") or "").strip()

    if not query:
        error_msg = (
            "Planner error: query is empty. "
            "Enter a research question and try again."
        )

        logger.error(error_msg)

        return {
            "plan": None,
            "errors": [error_msg],
        }

    try:
        if llm is None:
            from agents import llm as default_llm
            llm = default_llm

        chain = build_planner_chain(llm)

        logger.info("Generating research plan for query: %s", query)

        raw_output = chain.invoke({"query": query})
        plan = validate_plan(raw_output)

        logger.info(
            "Generated validated research plan with %d subquestions.",
            len(plan.subquestions),
        )

        return {
            "plan": plan,
            "errors": [],
        }

    except Exception as exc:
        error_msg = (
            "Planner failed while generating the research plan: "
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(error_msg)

        return {
            "plan": None,
            "errors": [error_msg],
        }
