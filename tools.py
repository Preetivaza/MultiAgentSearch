import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web and return structured search results."""

    try:
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            return json.dumps({
                "success": False,
                "error": "TAVILY_API_KEY is missing.",
                "results": []
            })

        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            max_results=5
        )

        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")
            })

        return json.dumps({
            "success": True,
            "results": results
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "results": []
        })


@tool
def scrape_url(url: str) -> str:
    """Scrape and extract clean text content from a given web page URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav, footer, header, and aside tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
            element.decompose()

        # Extract and clean text
        text = soup.get_text(separator="\n", strip=True)
        return text[:4000] if len(text) > 4000 else text
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"