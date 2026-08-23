from langchain_ollama import ChatOllama

llm = ChatOllama(model="minimax-m3:cloud")

try:
    print("Press Ctrl+C to stop.")
    while True:
        question = input("User: ")
        res = llm.invoke(question)

        print(f"AI: {res.content}")
        pass
except KeyboardInterrupt:
    print("\nSuccessfully Exited.")
