from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import PromptTemplate
from typing import Literal

from state import BlogState, DecisionSchema
from agent import get_llm, researcher_agent, writer_agent, editor_agent
from prompt import REVIEW_RESEARCH_PROMPT, REVIEW_DRAFT_PROMPT

MAX_REVISION = 3

llm = get_llm()


# Define Node
def researcher_node(state: BlogState) -> BlogState:
    """
    Researcher Agent generates (or revise) research outline
    """

    research_data = researcher_agent(
        llm,
        topic=state.topic,
        audience=state.audience,
        feedback=state.research_feedback,
    )

    state.research = research_data
    state.research_feedback = ""

    return state


def human_review_research_node(state: BlogState):
    """
    Pause and ask the human to approve the research or send the feedback
    """

    decision = interrupt(
        {
            "stage": "researcher_review",
            "research": state.research,
            "instructions": (
                "Reply with 'approve' to continue to writing",
                "or decribe the change to send it back to the researcher",
            ),
        }
    )

    if isinstance(decision, dict) and "action" in decision:
        state.research_feedback = (
            "" if decision["action"] == "approve" else decision.get("feedback", "")
        )
        return state

    user_input = str(decision)

    structured_llm = llm.with_structured_output(DecisionSchema, method="json_mode")
    chain = REVIEW_RESEARCH_PROMPT | structured_llm
    result = chain.invoke({"user_input": user_input})

    state.research_feedback = "" if result.action == "approve" else result.feedback
    return state


def writer_node(state: BlogState) -> BlogState:
    """
    Writer Agent produce the full draft blog (or revise it)
    """

    writer_data = writer_agent(
        llm,
        topic=state.topic,
        audience=state.audience,
        research=state.research,
        feedback=state.draft_feedback,
    )

    state.draft = writer_data
    state.draft_feedback = ""

    return state


def human_review_draft_node(state: BlogState):
    """
    Pause and ask the human to approve the draft or send the feedback
    """

    decision = interrupt(
        {
            "stage": "draft_review",
            "draft": state.draft,
            "instructions": (
                "Reply with 'approve' to continue to editing",
                "or decribe the change to send it back to the editor",
            ),
        }
    )

    if isinstance(decision, dict) and "action" in decision:
        state.draft_feedback = (
            "" if decision["action"] == "approve" else decision.get("feedback", "")
        )
        return state

    user_input = str(decision)

    structured_llm = llm.with_structured_output(DecisionSchema, method="json_mode")
    chain = REVIEW_DRAFT_PROMPT | structured_llm
    result = chain.invoke({"user_input": user_input})

    state.draft_feedback = "" if result.action == "approve" else result.feedback
    return state


def editor_node(state: BlogState) -> BlogState:
    final_response = editor_agent(llm=llm, topic=state.topic, draft=state.draft)
    state.final_blog = final_response
    return state


# Conditional Edges


def route_after_research_review(
    state: BlogState,
) -> Literal["researcher_node", "writer_node"]:
    if state.research_feedback:
        return "researcher_node"
    return "writer_node"


def route_after_draft_review(state: BlogState) -> Literal["writer_node", "editor_node"]:
    if state.draft_feedback and state.revision_count < MAX_REVISION:
        return "writer_node"
    return "editor_node"


# Build and compile graph


def build_blog_graph():
    builder = StateGraph(BlogState)

    # add nodes
    builder.add_node("researcher_node", researcher_node)
    builder.add_node("researcher_review_node", human_review_research_node)
    builder.add_node("writer_node", writer_node)
    builder.add_node("draft_review_node", human_review_draft_node)
    builder.add_node("editor_node", editor_node)

    # add edges
    builder.add_edge(START, "researcher_node")
    builder.add_edge("researcher_node", "researcher_review_node")
    builder.add_edge("writer_node", "draft_review_node")
    builder.add_edge("editor_node", END)

    # add conditional edges
    builder.add_conditional_edges(
        "researcher_review_node",
        route_after_research_review,
        {"researcher_node": "researcher_node", "writer_node": "writer_node"},
    )
    builder.add_conditional_edges(
        "draft_review_node",
        route_after_draft_review,
        {"writer_node": "writer_node", "editor_node": "editor_node"},
    )

    # compile graph
    graph = builder.compile(checkpointer=InMemorySaver())

    return graph
