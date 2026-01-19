# main.py
from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from graph_agents import make_graph


from rag_basic import build_retriever



# LangChain retriever (RAG)
retriever = build_retriever()

graph = make_graph(retriever)


app = FastAPI(title="LangChain RAG Chatbot")

# (Optional) allow JS frontends to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# 🏠 Home page: simple HTML chat UI
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8" />
        <title>LangChain + RAG Chatbot</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 30px auto;
            background: #f5f5f5;
          }
          h1 {
            text-align: center;
          }
          #chat-container {
            background: #ffffff;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
          }
          #messages {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            height: 350px;
            overflow-y: auto;
            background: #fafafa;
          }
          .msg-user {
            margin-bottom: 10px;
            text-align: right;
          }
          .msg-user span {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 6px 10px;
            border-radius: 12px;
            max-width: 80%;
          }
          .msg-bot {
            margin-bottom: 10px;
            text-align: left;
          }
          .msg-bot span {
            display: inline-block;
            background: #e0e0e0;
            padding: 6px 10px;
            border-radius: 12px;
            max-width: 80%;
            white-space: pre-wrap;
          }
          #input-area {
            margin-top: 12px;
            display: flex;
            gap: 8px;
          }
          #message {
            flex: 1;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ccc;
          }
          button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
          }
          button:disabled {
            background: #999999;
            cursor: not-allowed;
          }
        </style>
      </head>
      <body>
        <h1>LangChain + RAG Chatbot</h1>
        <div id="chat-container">
          <div id="messages"></div>

          <div id="input-area">
            <input id="message" type="text" placeholder="Ask about clinical trials, FDA, EMA, etc..." />
            <button id="send-btn" onclick="sendMessage()">Send</button>
          </div>
        </div>

        <script>
          async function sendMessage() {
            const input = document.getElementById("message");
            const btn = document.getElementById("send-btn");
            const messagesDiv = document.getElementById("messages");
            const text = input.value.trim();

            if (!text) {
              alert("Please type a question.");
              return;
            }

            // show user message
            const userMsg = document.createElement("div");
            userMsg.className = "msg-user";
            userMsg.innerHTML = "<span>" + text + "</span>";
            messagesDiv.appendChild(userMsg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            input.value = "";
            input.focus();

            // show "thinking"
            const botMsg = document.createElement("div");
            botMsg.className = "msg-bot";
            botMsg.innerHTML = "<span>Thinking...</span>";
            messagesDiv.appendChild(botMsg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            btn.disabled = true;

            try {
              const resp = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text }),
              });

              if (!resp.ok) {
                botMsg.innerHTML = "<span>Error: " + resp.status + "</span>";
              } else {
                const data = await resp.json();
                botMsg.innerHTML = "<span>" + data.reply + "</span>";
              }
            } catch (err) {
              botMsg.innerHTML = "<span>Request failed: " + err + "</span>";
            } finally {
              btn.disabled = false;
              messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
          }

          // allow pressing Enter to send
          document.getElementById("message").addEventListener("keydown", function (e) {
            if (e.key === "Enter") {
              e.preventDefault();
              sendMessage();
            }
          });
        </script>
      </body>
    </html>
    """


# 🤖 Chat endpoint using LangChain + little RAG
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_question = req.message.strip()
    if not user_question:
        return {"reply": "Please type a question."}

    result = graph.invoke({
        "question": user_question,
        "agent": "general",
        "context": "",
        "citations": [],
        "answer": ""
    })

    sources = sorted({c.get("source", "knowledge.txt") for c in (result.get("citations") or [])})
    reply = result.get("answer") or "No answer generated."
    agent = result.get("agent") or "unknown"

    if sources:
        reply += "\n\nSource: " + ", ".join(sources)

    reply = f"[Agent: {agent}]\n\n" + reply
    return {"reply": reply}

