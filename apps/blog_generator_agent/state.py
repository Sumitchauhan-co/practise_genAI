from pydantic import BaseModel, Field


class BlogState(BaseModel):
    # User Input
    topic: str = ""
    audience: str = "general"

    # Researcher Output
    research: str = ""
    research_feedback: str = ""

    # Writer Output
    draft: str = ""
    draft_feedback: str = ""

    # Editor Output
    final_blog: str = ""

    # MetaData
    revision_count: int = 0


class DecisionSchema(BaseModel):
    action: str = Field(description="Must be either 'approve' or 'reject'")
    feedback: str = Field(
        description="The user's modifications requests. Empty if approved."
    )
