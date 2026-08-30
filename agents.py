import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

# agents.py — Builds agents using langgraph.
# ── Load environment 
# this is dotenv file which is used to load environment variables  like API keys, secrets, etc. ─────────────────────────────────────────────────────
load_dotenv(override=True)

# Model setup using Google Gemini
api_key = os.getenv("GOOGLE_API_KEY", "dummy_key_to_prevent_startup_crash")
best_model = "gemini-1.5-pro" # Default Gemini model for advanced agent reasoning

llm = ChatGoogleGenerativeAI(
    model=best_model,
    temperature=0,
    google_api_key=api_key
)

# 1st agent: Search Agent
def build_search_agent():
    return create_react_agent(
        llm,
        tools=[web_search]
    )

# 2nd agent: Reader Agent
def build_reader_agent():
    return create_react_agent(
        llm,
        tools=[scrape_url]
    )

# Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()