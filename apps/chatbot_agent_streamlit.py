from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

load_dotenv()

model = ChatOllama(model="minimax-m3:cloud", disable_streaming=False)

tool = TavilySearch(
    max_results=1,
    topic="general",
)

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=model,
    tools=[tool],
    system_prompt="You are a agent and can search any question.",
    checkpointer=st.session_state.memory,
)

# Building web interface with streamlit

st.subheader("AI Chatbot with Streamlit 🤖")

for msg in st.session_state.history:
    role = msg["role"]
    content = msg["content"]

    st.chat_message(role).markdown(content)

question = st.chat_input("Ask anything ?")

if question:
    st.chat_message("user").markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    res = agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": "chat_id"}},
        stream_mode="messages",
    )

    ai_container = st.chat_message("ai")

    with ai_container:
        space = st.empty()

        full_message = ""

        for chunk in res:
            full_message += chunk[0].content
            space.markdown(full_message)

        if not full_message:
            full_message = "I performed a search but didn't generate any new text."
            space.markdown(full_message)

    st.session_state.history.append({"role": "ai", "content": full_message})
