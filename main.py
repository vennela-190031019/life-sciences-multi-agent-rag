# main.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_basic import build_retriever
from graph_agents import make_graph

app = FastAPI(title="Life Sciences Multi-Agent RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = build_retriever(k=3)
graph = make_graph(retriever)


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Life Sciences Multi-Agent RAG</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 30px auto; background: #f5f5f5; }
    h2 { text-align: center; }
    #chat-container { background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
    #messages { border: 1px solid #ddd; border-radius: 8px; padding: 12px; height: 420px; overflow-y: auto; background: #fafafa; }
    .msg { margin: 10px 0; display: flex; }
    .user { justify-content: flex-end; }
    .bot { justify-content: flex-start; }
    .bubble { max-width: 80%; padding: 10px 12px; border-radius: 14px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.35; }
    .user .bubble { background: #007bff; color: white; border-bottom-right-radius: 6px; }
    .bot .bubble { background: #e7e7e7; color: #111; border-bottom-left-radius: 6px; }
    #input-area { margin-top: 12px; display: flex; gap: 8px; }
    #message { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #ccc; outline: none; }
    button { padding: 10px 16px; border: none; border-radius: 8px; background: #007bff; color: white; cursor: pointer; }
    button:disabled { background: #9a9a9a; cursor: not-allowed; }
    #small { margin-top: 8px; color: #666; font-size: 12px; text-align: center; }
  </style>
</head>

<body>
  <h2>Life Sciences Multi-Agent RAG Chatbot</h2>

  <div id="chat-container">
    <div id="messages"></div>

    <div id="input-area">
      <input id="message" type="text" placeholder="Ask about FDA, EMA, clinical trials..." />
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <div id="small">Answers are grounded in your knowledge base.</div>
  </div>

<script>
  const messagesDiv = document.getElementById("messages");
  const input = document.getElementById("message");
  const btn = document.getElementById("send-btn");

  function addMessage(role, text) {
    const row = document.createElement("div");
    row.className = "msg " + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    messagesDiv.appendChild(row);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    return bubble;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) {
      addMessage("bot", "Please type a message.");
      return;
    }

    addMessage("user", text);
    input.value = "";
    input.focus();

    const botBubble = addMessage("bot", "Thinking...");
    btn.disabled = true;
    input.disabled = true;

    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ message: text })
      });

      const data = await resp.json();
      botBubble.textContent = data.reply || "No reply.";
    } catch (e) {
      botBubble.textContent = "Request failed: " + e;
    } finally {
      btn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  });

  addMessage("bot", "Hi! Ask about FDA/EMA/clinical trials. I route queries to specialized agents.");
</script>
</body>
</html>
"""


@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        return {"reply": "Please type a question."}

    result = graph.invoke({
        "question": question,
        "agent": "general",
        "context": "",
        "citations": [],
        "answer": ""
    })

    agent = result.get("agent", "general")
    answer = result.get("answer", "").strip()
    citations = result.get("citations", []) or []

    sources = sorted({c.get("source", "knowledge.txt") for c in citations})
    if sources:
        answer += "\n\nSource: " + ", ".join(sources)

    final = f"[Agent: {agent}]\n\n{answer}"
    return {"reply": final}