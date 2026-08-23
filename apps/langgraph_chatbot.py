from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

load_dotenv()


class ChatState(BaseModel):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="minimax-m3:cloud")


def ChatBot(state: ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state


graph = StateGraph(ChatState)
graph.add_node("chat_bot", ChatBot)

memory = InMemorySaver()

graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)

finalGraph = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "langgraph_chat_bot"}}

try:
    print("Press Ctrl+C to stop.")
    while True:
        question = input("User: ")

        res = finalGraph.invoke(
            {"messages": [{"role": "user", "content": question}]}, config
        )

        print(f"AI: {res["messages"][-1].content}")
        pass

except KeyboardInterrupt:
    print("\nSuccessfully Exited.")
