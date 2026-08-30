# ResearchMind - Multi-Agent AI System 🔬

ResearchMind is a powerful, multi-agent artificial intelligence system designed to automate comprehensive research tasks. Built with **Streamlit**, **LangChain**, and powered by **Google Gemini** models, this application orchestrates specialized AI agents to gather information, scrape deep content, draft structured reports, and provide critical feedback on any given topic.

## ✨ What It Does

The app works in 4 simple steps to give you a great report:

1. **🔍 Search**: The AI searches the internet to find the newest and best information on your topic.
2. **📄 Read**: It picks the best website from the search and reads the full article to get the deep details.
3. **✍️ Write**: The AI takes all the information and writes a clear, organized report for you.
4. **🧐 Check (Critic)**: Another AI checks the report, gives it a score out of 10, and tells you what is good and what can be better.

## 🛠️ What We Used to Build It

* **Website**: Streamlit (makes the app look nice and easy to use).
* **AI Brain**: Google Gemini (`gemini-1.5-pro`).
* **AI Manager**: LangChain and LangGraph (connects the AI steps together).
* **Web Search Tool**: Tavily API and BeautifulSoup (to search and read websites).

## 🚀 How to Run It on Your Computer

### What You Need
* Python installed on your computer.
* A Google Gemini API Key (to use the AI).

### Setup Steps

1. **Download the project files** to your computer.
2. **Open your terminal** and go to the project folder.
3. **Install the required tools** by running:
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your API Key**: Create a file named `.env` in the main folder and add your Google API key like this:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   *(Note: The Tavily search key is already set up in `tools.py`)*

### Start the App

Run this command in your terminal to start the website:

```bash
streamlit run app.py
```

The app will open in your web browser!

## 💡 How to Use the App

1. Open the app in your browser.
2. Type a topic you want to learn about in the box (for example, "Latest electric cars 2025").
3. Click the **"⚡ Run Research Pipeline"** button.
4. Watch the AI do the work! It will show you the search results, what it read, and the final report.
5. You can click a button to download the finished report to your computer.

## 📂 Files in This Project

* `app.py`: The main website code and how it looks.
* `pipeline.py`: A script to run the research process in your terminal instead of the website.
* `agents.py`: The instructions for the AI agents (Search, Read, Write, Check).
* `tools.py`: The tools the AI uses to search and read the internet.
* `requirements.txt`: The list of Python tools you need to install.
