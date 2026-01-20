# life-sciences-multi-agent-rag

Life Sciences Multi-Agent RAG Chatbot

A simple GenAI chatbot for Life Sciences that answers questions using a knowledge base, routes queries to specialized agents, and prevents hallucinations.

Built as part of a GenAI Internship Project.

---

## What this project does

- Accepts user questions through a web UI  
- Routes the question to the right expert agent  
- Retrieves relevant content from a knowledge base (RAG)  
- Generates grounded answers  
- Blocks answers if information is not available  

---

## Architecture (Simple View)

```mermaid
flowchart TD
    U[User] --> UI[Web UI]
    UI --> API[FastAPI Backend]
    API --> R[Router Agent]
    R --> C[Clinical Agent]
    R --> G[Regulatory Agent]
    R --> N[General Agent]
    C --> S[Safety Validator]
    G --> S
    N --> S
    S --> A[Final Answer]