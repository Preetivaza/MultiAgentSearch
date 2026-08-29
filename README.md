# ResearchMind - Multi-Agent AI System 🔬

ResearchMind is a powerful, multi-agent artificial intelligence system designed to automate comprehensive research tasks. Built with **Streamlit**, **LangChain**, **LangGraph**, and powered by **Google Gemini** models, this application orchestrates specialized AI agents to gather information, scrape deep content, draft structured reports, and provide critical feedback on any given topic.

## 🌟 Features

ResearchMind utilizes a four-step pipeline to deliver high-quality research reports:

1.  **🔍 Search Agent**: Utilizes web search tools (like Tavily) to scour the internet for recent, reliable, and detailed information about the user-provided topic.
2.  **📄 Reader Agent**: Analyzes search results to identify the most relevant URLs and scrapes them for in-depth content extraction.
3.  **✍️ Writer Chain**: Synthesizes the gathered research into a structured, detailed, and professional markdown report, including an introduction, key findings, conclusion, and a list of sources.
4.  **🧐 Critic Chain**: Acts as a rigorous reviewer, evaluating the drafted report to provide a score out of 10, highlighting strengths, suggesting areas for improvement, and giving a final verdict.

## 🛠️ Tech Stack

*   **Frontend UI**: [Streamlit](https://streamlit.io/) with custom CSS for a modern, sleek interface.
*   **LLM Framework**: [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
*   **AI Models**: Google Gemini (`gemini-1.5-pro` via `langchain-google-genai`)
*   **Web Search & Scraping**: `tavily-python`, `beautifulsoup4`, `requests`
*   **Environment Management**: `python-dotenv`

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Google Gemini API Key
*   Tavily API Key (for web search tool capabilities)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Multi-agent-research-system.git
    cd Multi-agent-research-system
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_gemini_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    ```

### Running the Application

Execute the following command to start the Streamlit server:

```bash
streamlit run app.py
```

The application will be accessible in your web browser, typically at `http://localhost:8501`.

## 💡 How to Use

1.  Open the ResearchMind web application.
2.  Enter your desired research topic in the input field (e.g., "Quantum computing breakthroughs in 2025").
3.  Click the **"⚡ Run Research Pipeline"** button.
4.  Watch the pipeline progress through the Search, Reader, Writer, and Critic stages.
5.  Once complete, view the final research report and the critic's feedback.
6.  You can download the generated report as a Markdown (`.md`) file.

## 📂 Project Structure

*   `app.py`: The main Streamlit application script containing the UI and pipeline execution logic.
*   `agents.py`: Defines the LangChain models, React agents (Search, Reader), and Prompt Chains (Writer, Critic).
*   `tools.py`: (Assumed) Contains the custom tools `web_search` and `scrape_url` used by the agents.
*   `requirements.txt`: Lists all Python dependencies required for the project.
*   `.env`: Configuration file for storing environment variables and API keys.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.
