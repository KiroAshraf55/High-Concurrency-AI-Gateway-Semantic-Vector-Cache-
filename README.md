<div align="center">

# 🌌 Distributed AI Gateway & Semantic Cache

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)](https://reactjs.org)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF.svg)](https://vitejs.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-FF5252.svg)](https://qdrant.tech/)

An enterprise-grade, containerized AI routing gateway that intercepts LLM requests, resolves conversational context, and serves responses from a localized Semantic Vector Cache to drastically reduce API latency and inference costs.

<br />

*(Replace this text with your UI screenshot! For example: `<img src="./docs/ui-screenshot.png" width="800" />`)*

</div>

---

## ✨ Core Architecture

This system is built on a clean monorepo architecture, physically separating the stateless presentation layer from the heavy backend processing engine.

### 1. 🧠 The Semantic Cache Layer (Qdrant + HuggingFace)
Instead of sending every user query to a paid LLM, the gateway intercepts the request and embeds it into a high-dimensional vector. It queries a local **Qdrant** vector database for semantic similarity. If a similar question was asked previously (e.g., *"What's the capital of France?"* vs *"What is the French capital?"*), it returns the cached response instantly, bypassing the LLM entirely.

### 2. 🔄 The AI Decontextualizer
Conversational AI requires memory, but caching raw conversational fragments ruins cache hit rates. This system employs an internal "Interpreter Model" to rewrite fragmented user follow-ups into standalone queries *before* they hit the cache.

### 3. ⚡ Stateless React Frontend
A hyper-fast, Vite-powered React UI utilizing standard JavaScript and custom CSS. By maintaining volatile state locally in the browser, the backend remains completely stateless, decoupled, and horizontally scalable.

---

## 💻 Code Showcase: The Decontextualizer

To ensure maximum cache hit rates, user inputs are intercepted and passed through a highly restricted LLM router. By hardcoding the `temperature` to `0` and using strict negative constraints, the interpreter reliably converts fragmented dialogue into standalone vector-searchable queries.

```python
async def decontextualize_prompt(messages_payload: list[dict], client: httpx.AsyncClient, headers: dict) -> str:
    """
    Passes conversation history to an ultra-fast model to rewrite the query.
    If it's single turn or doesn't need context, it safely returns it as-is.
    """
    latest_user_message = messages_payload[-1]["content"]

    if len(messages_payload) == 1:
        return latest_user_message
    
    system_instruction = {
        "role": "system",
        "content": (
            "You are a routing assistant. Look at the conversation history and the user's latest message. "
            "If the latest message relies on previous context (e.g., uses pronouns like 'it', 'there', 'he', 'that'), "
            "rewrite it into a single, standalone question in English. "
            "If the latest message is ALREADY self-contained and standalone, return it EXACTLY as the user wrote it. "
            "DO NOT answer the question. ONLY output the rewritten string or original string with absolutely no commentary."
        )
    }

    # Build the interpreter payload: System instructions + full history
    interpreter_payload = {
        "model": "llama-3.1-8b-instant",  # Light, fast, and fraction-of-a-penny cheap
        "messages": [system_instruction] + messages_payload,
        "max_tokens": 100,
        "temperature": 0  # Strict, deterministic outputs
    }

    try:
        response = await client.post(
            "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)",
            headers=headers,
            json=interpreter_payload,
            timeout=10.0
        )
        if response.status_code == 200:
            data = response.json()
            cleaned_query = data["choices"][0]["message"]["content"].strip()
            print(f"[Decontextualizer] Raw: '{latest_user_message}' -> Standalone: '{cleaned_query}'")
            return cleaned_query
    except Exception as e:
        print(f"[Decontextualizer Warning] Fallback triggered due to error: {e}")
    
    return latest_user_message
```
## 💻 Code Showcase: The Decontextualizer
ai-gateway-cache/
├── backend/                  # Containerized Python Engine
│   ├── gateway_v3.py         # Main FastAPI entry point
│   ├── semantic_cache.py     # Vector DB connection & Embedding logic
│   ├── requirements.txt      
│   ├── Dockerfile            # Lightweight Python 3.11-slim build
│   └── docker-compose.yml    # Orchestrates Gateway + Qdrant DB
│
└── frontend/                 # Stateless React UI
    ├── src/
    │   ├── components/       # UI Components (ChatInterface)
    │   ├── hooks/            # Local State Management (useChat)
    │   └── services/         # API Routing to Gateway
    ├── package.json
    └── vite.config.js

## 🛠️ Quick Start (Local Development)
### Prerequisites
* Docker Desktop
* Node.js (v18+)
* A valid Groq API Key

### 1. Spin up the Backend (Docker)
```bash
# Navigate to the backend directory
cd backend

# Create your environment variables file
echo "GROQ_API_KEY=your_api_key_here" > .env

# Build and start the FastAPI gateway and Qdrant database containers
docker compose up --build
```

### 2. Spin up the Frontend (Vite)
```bash
# In a new terminal window, navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
Navigate to http://localhost:5173 to interact with the gateway!