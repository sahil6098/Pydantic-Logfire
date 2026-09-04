import os
from langchain_openai import ChatOpenAI

#Intializing Groq LLM

def groq_llm(temperature : float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        base_url = "https://api.groq.com/openai/v1",
        api_key = os.environ.get("GROQ_API_KEY"),
        model = "openai/gpt-oss-20b",
        temperature = temperature,
    )

# Initializing gemini llm

def gemini_llm(temperature : float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key = os.environ.get("GEMINI_API_KEY"),
        model = "gemini-2.5-flash-lite",
        temperature = temperature,
    )