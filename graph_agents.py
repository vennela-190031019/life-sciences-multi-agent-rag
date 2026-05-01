# graph_agents.py
from typing import TypedDict, Literal, List, Dict, Any

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

AgentName = Literal["clinical", "regulatory", "general"]


class GraphState(TypedDict):
    question: str
    agent: AgentName
    context: str
    citations: List[Dict[str, Any]]
    answer: str


class RouteDecision(BaseModel):
    agent: AgentName = Field(description="clinical, regulatory, or general")
    reason: str = Field(description="Short reason for choosing the agent")


def _format_citations(docs) -> List[Dict[str, Any]]:
    citations = []
    for d in docs or []:
        md = d.metadata or {}
        citations.append({
            "source": md.get("source", "knowledge.txt"),
            "chunk_id": md.get("chunk_id"),
        })
    return citations


def make_graph(retriever, model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0.2)

    def router_node(state: GraphState) -> GraphState:
        question = state["question"].strip()

        router_llm = ChatOpenAI(model=model, temperature=0.0)
        router = router_llm.with_structured_output(RouteDecision)

        decision = router.invoke([
            {
                "role": "system",
                "content": (
                    "You are a router for a life sciences multi-agent chatbot.\n"
                    "Choose only one agent:\n"
                    "clinical = clinical trials, phases, endpoints, patients, adverse events, safety reporting.\n"
                    "regulatory = FDA, EMA, ICH, GxP, GMP, GCP, compliance, submissions, approvals, labeling.\n"
                    "general = basic life sciences questions or unclear questions.\n"
                    "Return the best agent."
                ),
            },
            {"role": "user", "content": question},
        ])

        state["agent"] = decision.agent
        return state

    def retrieval_step(state: GraphState) -> GraphState:
        docs = retriever.invoke(state["question"])
        state["context"] = "\n\n".join(d.page_content for d in docs) if docs else ""
        state["citations"] = _format_citations(docs)
        return state

    def clinical_agent(state: GraphState) -> GraphState:
        retrieval_step(state)

        prompt = f"""
You are the Clinical Trials Agent.

Answer the question using ONLY the context.
If the context does not contain the answer, say:
"I don't have enough information in my knowledge base."

Context:
{state["context"] if state["context"] else "No relevant context found."}

Question:
{state["question"]}
""".strip()

        state["answer"] = llm.invoke(prompt).content.strip()
        return state

    def regulatory_agent(state: GraphState) -> GraphState:
        retrieval_step(state)

        prompt = f"""
You are the Regulatory Affairs Agent.

Answer the question using ONLY the context.
Focus on FDA, EMA, ICH, GxP, GMP, GCP, approvals, submissions, and compliance.
If the context does not contain the answer, say:
"I don't have enough information in my knowledge base."

Context:
{state["context"] if state["context"] else "No relevant context found."}

Question:
{state["question"]}
""".strip()

        state["answer"] = llm.invoke(prompt).content.strip()
        return state

    def general_agent(state: GraphState) -> GraphState:
        retrieval_step(state)

        prompt = f"""
You are the General Life Sciences Agent.

Answer the question using ONLY the context.
If the context does not contain the answer, say:
"I don't have enough information in my knowledge base."

Context:
{state["context"] if state["context"] else "No relevant context found."}

Question:
{state["question"]}
""".strip()

        state["answer"] = llm.invoke(prompt).content.strip()
        return state

    def safety_validator(state: GraphState) -> GraphState:
        guard_llm = ChatOpenAI(model=model, temperature=0.0)

        prompt = f"""
You are a safety validator for a life sciences chatbot.

Compare the answer with the context.

Rules:
- If the answer is supported by the context, return the answer unchanged.
- If the answer contains facts not supported by the context, return exactly:
"I don't have enough information in my knowledge base."
- Return only the final answer.

Context:
{state["context"] if state["context"] else "No context."}

Answer:
{state["answer"]}
""".strip()

        state["answer"] = guard_llm.invoke(prompt).content.strip()
        return state

    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("clinical", clinical_agent)
    graph.add_node("regulatory", regulatory_agent)
    graph.add_node("general", general_agent)
    graph.add_node("safety", safety_validator)

    graph.set_entry_point("router")

    def route_selector(state: GraphState) -> str:
        return state["agent"]

    graph.add_conditional_edges(
        "router",
        route_selector,
        {
            "clinical": "clinical",
            "regulatory": "regulatory",
            "general": "general",
        },
    )

    graph.add_edge("clinical", "safety")
    graph.add_edge("regulatory", "safety")
    graph.add_edge("general", "safety")
    graph.add_edge("safety", END)

    return graph.compile()