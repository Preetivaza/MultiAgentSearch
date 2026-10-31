from dotenv import load_dotenv
import os
load_dotenv(override=True)

from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available models for generateContent:")
for m in client.models.list():
    if hasattr(m, "supported_actions") and "generateContent" in (m.supported_actions or []):
        print(" ", m.name)
    elif hasattr(m, "supported_generation_methods") and "generateContent" in (m.supported_generation_methods or []):
        print(" ", m.name)
    else:
        print(" ", m.name)
