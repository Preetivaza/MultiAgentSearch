from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
)

if __name__ == "__main__":
    print("Testing Gemini 3.8 Flash...")
    response = llm.invoke("Explain Python in two sentences.")
    print("\nSUCCESS!")
    print(response.content)
