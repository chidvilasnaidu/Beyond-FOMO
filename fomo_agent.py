"""
Beyond FOMO — agentic backend.
A single compiled LangGraph is reused across Streamlit turns.
"""

import os
import ast
import sqlite3

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from ddgs import DDGS
from langgraph.checkpoint.sqlite import SqliteSaver


class FomoState(MessagesState):
    preferences: dict
    intent: str


SYSTEM_PROMPT = """
You are FOMO Guardian AI.

Rules:
- For updates or "today/recent/latest" requests, give 5-8 concise useful points when enough relevant information exists.
- Use tools for current or time-sensitive information: news, events, jobs, scholarships, visas, government updates, weather, and markets.
- Understand the user's question before selecting a tool.
- Use WebSearch for evergreen facts, rankings, comparisons, definitions, and how-to questions.
- If tools are unavailable or insufficient, answer from your knowledge without inventing current facts.
- Explain, compare, and simplify when useful.
- Prefer short, line-by-line bullet points.
- Do not mention news organizations, channels, source names, or taglines in the final answer.
- If the user asks for daily insights, ask for their preferences before setting them up.
- End every reply with: "Anything else I can help with?"
- If the user says goodbye, thank them, include a short quote, and say goodbye.
"""


def _get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .streamlit/secrets.toml "
            "or export it as an environment variable before running the app."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )


llm = None
search = DuckDuckGoSearchResults()


@tool
def SearchEngine(query: str):
    """Use only for today's/recent news, events, deadlines, or announcements."""
    results = DDGS().news(query, region="in-en", max_results=6)
    return "\n\n".join(
        r.get("body", "") for r in results if r.get("body")
    )


@tool
def WeatherTool(city: str):
    """Return current weather information for a city."""
    return search.run(f"Current weather in {city}")


@tool
def WebSearch(query: str):
    """Use for evergreen facts, rankings, comparisons, definitions, and how-to questions."""
    return search.run(query)


@tool
def TranslateTool(text: str, language: str):
    """Translate text into the requested language."""
    prompt = f"""
Translate the following text into {language}.

Text:
{text}
"""
    return llm.invoke(prompt).content


@tool
def CategoryNews(category: str):
    """Search for the latest news by category such as AI, Visa, Jobs, Scholarships, or Business."""
    return search.run(f"Latest {category} news")


@tool
def SummarizeTool(text: str):
    """Summarize the supplied information."""
    prompt = f"""
Summarize the following information in concise bullet points.

{text}
"""
    return llm.invoke(prompt).content


@tool
def FOMOPriority(news: str):
    """Classify supplied news as Critical, Important, Can Wait, or Ignore and explain why."""
    prompt = f"""
You are a FOMO Assistant.

Classify the following news into exactly one of:
Critical
Important
Can Wait
Ignore

Also explain why.

News:
{news}
"""
    return llm.invoke(prompt).content


@tool
def OpportunityFinder(news: str):
    """Extract scholarships, jobs, internships, visa updates, deadlines, and government announcements."""
    prompt = f"""
Find these opportunities or updates in the supplied information:
- Scholarships
- Jobs
- Internships
- Visa Updates
- Deadlines
- Government Announcements

Information:
{news}
"""
    return llm.invoke(prompt).content


TOOLS = [
    SearchEngine,
    WebSearch,
    WeatherTool,
    TranslateTool,
    CategoryNews,
    SummarizeTool,
    FOMOPriority,
    OpportunityFinder,
]


def ModerateNode(state: FomoState):
    messages = state["messages"]
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

    if tool_msgs:
        combined = "\n---\n".join(str(m.content) for m in tool_msgs)

        prompt = f"""
Clean each of these {len(tool_msgs)} tool results.

Rules:
- Remove hate speech, gossip, and clearly unverified claims.
- Remove news organization names, channel names, source names, taglines,
  and "Read more at..." phrases.
- Keep factual content useful to the user's question.
- Rewrite in neutral third-person language.
- Return the cleaned results separated by '---' in the same order.

Tool results:
{combined}
"""
        cleaned = llm.invoke(prompt).content.split("---")

        for m, c in zip(tool_msgs, cleaned):
            m.content = c.strip()

    return {"messages": messages}


def IntentRouter(state: FomoState):
    last_msg = state["messages"][-1].content

    prompt = f"""
Classify the user's message into exactly one word.

FEEDBACK = the user is telling you how to respond going forward,
such as style, tone, length, language, or preferred categories.

QUERY = the user is asking for actual information/help right now.

Message:
{last_msg}

Reply with only:
FEEDBACK
or
QUERY
"""
    intent = llm.invoke(prompt).content.strip().upper()
    return {
        "intent": intent if intent in ("FEEDBACK", "QUERY") else "QUERY"
    }


def FeedbackNode(state: FomoState):
    last_msg = state["messages"][-1].content
    old_prefs = state.get("preferences", {})

    prompt = f"""
Current preferences:
{old_prefs}

User feedback:
"{last_msg}"

Merge the new feedback into the preferences dictionary.

Possible keys:
- categories: list of topics such as jobs, visa, ai, business, scholarships
- style: short bullets, simple English, detailed, formal, etc.
- language: preferred language

Return ONLY a valid Python dictionary.
"""
    raw = llm.invoke(prompt).content.strip()

    try:
        new_prefs = ast.literal_eval(raw)
        if not isinstance(new_prefs, dict):
            new_prefs = old_prefs
    except Exception:
        new_prefs = old_prefs

    ack = AIMessage(
        content="Got it — I'll tailor updates to that going forward. "
                "Anything else I can help with?"
    )
    return {"preferences": new_prefs, "messages": [ack]}


def ChatAgent(state: FomoState):
    prefs = state.get("preferences", {})
    pref_text = (
        f"\nApply these user preferences to category focus and response style:\n"
        f"{prefs}\n"
        if prefs
        else ""
    )

    messages = [SystemMessage(
        content=SYSTEM_PROMPT + pref_text
    )] + state["messages"]

    response = llm.bind_tools(TOOLS).invoke(messages)
    return {"messages": [response]}


def build_graph(db_path: str = "FOMO.db"):
    global llm
    llm = _get_llm()

    checkpointer = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )
    memory = SqliteSaver(checkpointer)

    graph_var = StateGraph(FomoState)

    graph_var.add_node("IntentRouter", IntentRouter)
    graph_var.add_node("ChatNode", ChatAgent)
    graph_var.add_node("tools", ToolNode(TOOLS))
    graph_var.add_node("ModerateNode", ModerateNode)
    graph_var.add_node("FeedbackNode", FeedbackNode)

    graph_var.add_edge(START, "IntentRouter")

    graph_var.add_conditional_edges(
        "IntentRouter",
        lambda state: state["intent"],
        {
            "FEEDBACK": "FeedbackNode",
            "QUERY": "ChatNode",
        },
    )

    graph_var.add_edge("FeedbackNode", END)

    graph_var.add_conditional_edges(
        "ChatNode",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    graph_var.add_edge("tools", "ModerateNode")
    graph_var.add_edge("ModerateNode", "ChatNode")

    return graph_var.compile(checkpointer=memory)
