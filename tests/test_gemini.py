from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

if __name__ == "__main__":
    print("Testing Gemini API...")
    response = llm.invoke("Explain Python in two sentences.")
    print("\nGemini response:")
    print(response.content)
