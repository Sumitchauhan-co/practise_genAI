from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOllama(model="minimax-m3:cloud")

tool = TavilySearch(
    max_results=1,
    topic="general",
)

agent = create_agent(
    model=model,
    tools=[tool],
    system_prompt="You are a agent and can search any question.",
    checkpointer=InMemorySaver(),
)

try:
    print("Press Ctrl+C to stop.")
    while True:
        question = input("Ask Agent: ")
        res = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": "1"}},
        )

        print(res["messages"][-1].content, end="\n")
except KeyboardInterrupt:
    print("\nSuccessfully Exited.")
