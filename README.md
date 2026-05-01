# Life Sciences Multi-Agent RAG Chatbot

A production-style GenAI chatbot for Life Sciences that answers questions using a knowledge base, routes queries to specialized agents, and prevents hallucinations using a safety validation layer.

Built as part of a GenAI Internship Project.

---

## What this project does

- Accepts user questions through a web UI  
- Routes queries to specialized agents (clinical, regulatory, general)  
- Retrieves relevant context using Retrieval-Augmented Generation (RAG)  
- Generates grounded and domain-specific answers  
- Prevents hallucinations using a safety validator  


---


## Live Demo

[Open Live App] https://life-sciences-multi-agent-rag.onrender.com

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