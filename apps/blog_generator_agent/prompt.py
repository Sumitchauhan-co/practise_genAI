from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# RESEARCHER AGENT
RESEARCHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
            "You are a Research Agent. Given a blog topic and target audience, produce a clear, ""structured research outline. Include:\n""1. 5-7 key points the blog should cover\n""2. Important facts, stats, or examples for each point\n""3. Suggested angle or hook\n""Be concise. Use bullet points. Do NOT write the full blog yet."
        """,
        },
        {
            "role": "user",
            "content": "Topic : {topic}, Audience : {audience}, {revision_hints}. Write the research outline now.",
        },
    ]
)

# WRITER AGENT
WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
            "You are a Blog Writer Agent. Using the research notes provided, write a complete, "
            "engaging blog post.\n"
            "Rules:\n"
            "- Length: 500-800 words\n"
            "- Structure: catchy title, intro hook, 3-5 sections with H2 headings, conclusion\n"
            "- Tone: clear, friendly, suited to the target audience\n"
            "- Use markdown formatting\n"
        """,
        },
        {
            "role": "user",
            "content": """
            Topic : {topic},
            Audience : {audience},
            Research Notes : {research},
            {revision_hints}.
            Write the full blog post now.
        """,
        },
    ]
)


# Editor Agent
EDITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
            "You are an Editor Agent – the final quality gate before publishing.\n"
            "Take the draft and produce the FINAL polished version. Specifically:\n"
            "- Fix grammar, spelling, and awkward phrasing\n"
            "- Tighten wordy sentences\n"
            "- Improve flow and transitions between sections\n"
            "- Make the title and intro more compelling if needed\n"
            "- Keep the same structure and markdown formatting\n"
            "- Blog wordings should look like human, not AI\n"
            "Output only the final polished blog post – no commentary."
        """,
        },
        {
            "role": "user",
            "content": "Topic : {topic}, Draft : {draft}. Return the published blog post.",
        },
    ]
)

# Review Research Agent
from langchain_core.prompts import PromptTemplate

# Review Research Agent
REVIEW_RESEARCH_PROMPT = PromptTemplate.from_template(
    "You are a backend classification utility.\n"
    "Analyze the user's input regarding a research review and determine if they want to proceed/approve or request revisions.\n\n"
    "User Input: {user_input}\n\n"
    "You MUST respond ONLY with a valid JSON object containing these keys:\n"
    '- "action": "approve" (if they want to write/continue) or "revise" (if they want changes).\n'
    '- "feedback": A string detailing their changes, or an empty string "" if they approved.'
)

# Draft Research Agent
REVIEW_DRAFT_PROMPT = PromptTemplate.from_template(
    "You are a backend classification utility.\n"
    "Analyze the user's input regarding a draft review and determine if they want to proceed/approve or request revisions.\n\n"
    "User Input: {user_input}\n\n"
    "You MUST respond ONLY with a valid JSON object containing these keys:\n"
    '- "action": "approve" (if they want to edit/finish) or "revise" (if they want changes).\n'
    '- "feedback": A string detailing their changes, or an empty string "" if they approved.'
)
