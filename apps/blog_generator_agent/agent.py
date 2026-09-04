import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from prompt import RESEARCHER_PROMPT, WRITER_PROMPT, EDITOR_PROMPT

prod = os.getenv("NODE_ENV")


# GET LLM
def get_llm(model: str = "gemma4:31b-cloud", temperature: float = 0.5):
    if prod:
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        return llm
    llm = ChatOllama(model=model, temperature=temperature)
    return llm


def researcher_agent(
    llm: ChatOpenAI | ChatOllama, topic: str, audience: str, feedback: str = ""
) -> str:
    revision_hints = f"The human provided this feedback on previous research - please address it : {feedback}."
    if not feedback:
        revision_hints = "This is your first attempt."

    chain = RESEARCHER_PROMPT | llm

    result = chain.invoke(
        {"topic": topic, "audience": audience, "revision_hints": revision_hints}
    )

    return result.content


def writer_agent(
    llm: ChatOpenAI | ChatOllama,
    topic: str,
    audience: str,
    research: str = "",
    feedback: str = "",
) -> str:
    revision_hints = f"The human provided this feedback on previous draft and asked for these changes : {feedback}. Please apply these changes during writing the blog."
    if not feedback:
        revision_hints = "This is your first attempt."

    chain = WRITER_PROMPT | llm

    result = chain.invoke(
        {
            "topic": topic,
            "audience": audience,
            "research": research,
            "revision_hints": revision_hints,
        }
    )

    return result.content


def editor_agent(
    llm: ChatOpenAI | ChatOllama,
    topic: str,
    draft: str,
) -> str:
    chain = EDITOR_PROMPT | llm

    result = chain.invoke(
        {
            "topic": topic,
            "draft": draft,
        }
    )

    return result.content
