import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs, and snippets."""
    try:
        results = TavilyClient(api_key="tavily-app-sN2l7Nf2YqA9yWvV0z8P1R4T5U6V7W8").search(query=query, count=5)
        out = []
        for r in results:
            out.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n")
        return "\n----\n".join(out)
    except Exception as e:
        return f"Error executing web search: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove noisy elements
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
            
        clean_text = soup.get_text(separator=" ", strip=True)
        # Limit text length to avoid token limits
        return clean_text[:3000]
    except Exception as e:
        return f"Unable to scrape content from {url}: {str(e)}"