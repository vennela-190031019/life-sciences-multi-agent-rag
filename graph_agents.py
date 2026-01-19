# graph_agents.py
from typing import TypedDict, Literal, List, Dict, Any

from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


AgentName = Literal["clinical", "regulatory", "general"]

class GraphState(TypedDict):
    question: str
    agent: AgentName
    context: str
    citations: List[Dict[str, Any]]
    answer: str


class RouteDecision(BaseModel):
    agent: AgentName = Field(description="Which agent should handle the user question: clinical, regulatory, or general")
    reason: str = Field(description="Short reason for why this agent was chosen")

def _format_citations(docs) -> List[Dict[str, Any]]:
    out = []
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}
        out.append({
            "source": md.get("source", "knowledge.txt"),
            "chunk_id": md.get("chunk_id"),
        })
    return out

def make_graph(retriever, model: str = "gpt-4o-mini"):
    """
    Creates a LangGraph:
      router -> (clinical | regulatory | general) -> END
    Each agent uses RAG context + role prompt.
    """

    llm = ChatOpenAI(model=model, temperature=0.2)

# --- Router Node (simple deterministic routing: great for beginners) ---
def router_node(state: GraphState) -> GraphState:
    q = state["question"].strip()

    router_llm = ChatOpenAI(model=model, temperature=0.0)

    router = router_llm.with_structured_output(RouteDecision)

    decision = router.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a router for a multi-agent life sciences assistant.\n"
                    "Pick the best agent:\n"
                    "- regulatory: FDA/EMA/ICH/GxP, compliance, submissions (IND/NDA), labeling\n"
                    "- clinical: clinical trial phases, endpoints, enrollment, AE/SAE, study design\n"
                    "- general: anything else or unclear\n"
                    "Return a structured decision."
                ),
            },
            {"role": "user", "content": f"Question: {q}"},
        ]
    )

    state["agent"] = decision.agent
    return state

    # --- Shared retrieval step ---
    def retrieval_step(state: GraphState) -> GraphState:
        docs = retriever.invoke(state["question"])
        state["context"] = "\n\n".join(d.page_content for d in docs) if docs else ""
        state["citations"] = _format_citations(docs)
        return state

    # --- Clinical Agent ---
    def clinical_agent(state: GraphState) -> GraphState:
        retrieval_step(state)
        prompt = f"""
You are a Clinical Trials expert.
Rules:
- Answer using ONLY the Context.
- If the Context does NOT contain the answer, say: "I don't have enough information in my knowledge base."
- Do not add facts not present in the Context.

Context:
{state["context"] if state["context"] else "No relevant context found."}

Question:
{state["question"]}
""".strip()

        state["answer"] = llm.invoke(prompt).content.strip()
        return state

    # --- Regulatory Agent ---
    def regulatory_agent(state: GraphState) -> GraphState:
        retrieval_step(state)
        prompt = f"""
You are a Regulatory Affairs expert (FDA/EMA/ICH/GxP).
Rules:
- Answer using ONLY the Context.
- If the Context does NOT contain the answer, say: "I don't have enough information in my knowledge base."
- Do not add facts not present in the Context.


Context:
{state["context"] if state["context"] else "No relevant context found."}

Question:
{state["question"]}
""".strip()

        state["answer"] = llm.invoke(prompt).content.strip()
        return state

    # --- General Agent ---
    def general_agent(state: GraphState) -> GraphState:
        retrieval_step(state)
        prompt = f"""
You are a helpful life sciences assistant.
Rules:
- Answer using ONLY the Context.
- If the Context does NOT contain the answer, say: "I don't have enough information in my knowledge base."
- Do not add facts not present in the Context.


Context:
{state["context"] if state["context"] else "No relevant context found."}

Question:
{state["question"]}
""".strip()

        state["answer"] = llm.invoke(prompt).content.strip()
        return state

    # --- LangGraph wiring ---
    from langgraph.graph import StateGraph, END

    g = StateGraph(GraphState)

    g.add_node("router", router_node)
    g.add_node("clinical", clinical_agent)
    g.add_node("regulatory", regulatory_agent)
    g.add_node("general", general_agent)

    g.set_entry_point("router")

    # conditional routing
    def route_selector(state: GraphState) -> str:
        return state["agent"]

    g.add_conditional_edges(
        "router",
        route_selector,
        {
            "clinical": "clinical",
            "regulatory": "regulatory",
            "general": "general",
        },
    )

    # end edges
    g.add_edge("clinical", END)
    g.add_edge("regulatory", END)
    g.add_edge("general", END)

    return g.compile()
