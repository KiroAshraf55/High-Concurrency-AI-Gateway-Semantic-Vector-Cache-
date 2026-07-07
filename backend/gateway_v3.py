import os
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from semantic_cache import SemanticCache

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

app = FastAPI(title="Distributed Multi-Turn AI Gateway & Semantic Cache")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, put your React app URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = SemanticCache(threshold=0.91, max_size=1000)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

async def decontextualize_prompt(messages_payload: list[dict], client: httpx.AsyncClient, headers: dict) -> str:
    """
    Passes conversation history to a ultra-fast model to rewrite the query.
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
            "https://api.groq.com/openai/v1/chat/completions",
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "The Distributed AI Gateway V3 is running."}

@app.post("/chat")
async def chat_proxy(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Multi-turn endpoint that resolves context, queries the cache, and routes to Groq.
    """
    # Convert Pydantic models to standard list of dicts for the Groq API
    messages_payload = [msg.model_dump() for msg in request.messages]
    
    if not messages_payload:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            # STEP 1: Turn the potentially messy follow-up into a clean standalone query string
            standalone_query = await decontextualize_prompt(messages_payload, client, headers)

            # STEP 2: Check your high-precision Qdrant cache with the clean standalone query
            cached_response = cache.query(standalone_query)
            if cached_response:
                return {
                    "gateway_status": "success",
                    "source": "cache",
                    "standalone_query_evaluated": standalone_query,
                    "ai_response": cached_response["response"]
                }
            
            # STEP 3: Cache Miss! Forward the FULL original multi-turn chat payload to the main brain LLM
            main_payload = {
                "model": "llama-3.1-8b-instant", 
                "messages": messages_payload, # Keeping the full context intact for the answer
                "max_tokens": 500
            }

            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=main_payload,
                timeout=30.0
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            ai_text = data["choices"][0]["message"]["content"]
            
            # STEP 4: Store the clean standalone query paired with the answer for future users
            background_tasks.add_task(cache.insert, standalone_query, ai_text)

            return {
                "gateway_status": "success",
                "source": "Groq",
                "standalone_query_evaluated": standalone_query,
                "ai_response": ai_text
            }
        
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
