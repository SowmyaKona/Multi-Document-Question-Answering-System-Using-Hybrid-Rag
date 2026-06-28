import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0
    )
    return llm

# llm = ChatAnthropic(
#         model="claude-sonnet-4-20250514",
#         api_key=os.getenv("ANTHROPIC_API_KEY")