import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from semantic_cache import SemanticCache

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

app = FastAPI(title="Distributed AI Gateway & Semantic Cache")

cache = SemanticCache(threshold=0.8, max_size=1000, storage_dir="cache_storage")

class chatRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health_check():
    """A simple endpoint to check if our server is alive."""
    return {"status": "healthy", "message": "The Distributed AI Gateway is running."}

@app.post("/chat")
async def chat_proxy(request: chatRequest):
    """
    The main proxy endpoint that intercepts prompts and applies semantic caching.
    """
    cached_response = cache.query(request.prompt)
    if cached_response:
        return {
            "gateway_status": "success",
            "source": "cache",
            "ai_response": cached_response["response"]
        }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": 500
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            ai_text = data["choices"][0]["message"]["content"]

            cache.insert(request.prompt, ai_text)

            return {
                "gateway_status": "success",
                "source": "Groq",
                "ai_response": ai_text
            }
        
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")