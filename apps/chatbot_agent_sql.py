from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///tasks.db")

print("DB created successfully!")

db.run("""
  CREATE TABLE IF NOT EXISTS Tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('pending','in_progress','completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
""")

print("DB table 'Tasks' created successfully")

llm = ChatOllama(model="minimax-m3:cloud")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


memory = InMemorySaver()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""


@st.cache_resource
def create_todo_agent():
    agent = create_agent(
        model=llm, tools=tools, system_prompt=system_prompt, checkpointer=memory
    )

    return agent


agent = create_todo_agent()

st.title("TaskBot - Your Tasks manager.")

st.subheader("Manage your tasks with TaskBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask to manage your todos...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("ai"):
        with st.spinner("Processing..."):

            res = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": "sql_chat_id"}},
            )

            result = res["messages"][-1].content

            st.markdown(result)
    st.session_state.messages.append({"role": "ai", "content": result})
