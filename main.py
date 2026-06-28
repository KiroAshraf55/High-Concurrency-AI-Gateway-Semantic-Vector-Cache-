import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
'''1. os (Operating System Interface)
What it is: This is a built-in Python tool. You don't even have to install it.

What it does in our code: It talks directly to your computer's operating system (in your case, Windows/Git Bash). We specifically use os.getenv("GROQ_API_KEY") to reach into your computer's temporary memory (RAM) and pull out your secret key so we can use it to securely unlock GROQ.

2. dotenv (load_dotenv)
What it is: The vault manager.

What it does in our code: By itself, Python's os tool doesn't know what a .env file is. When we call load_dotenv(), this library sneaks into your hidden .env file, reads the GROQ_API_KEY, and violently shoves it into your operating system's memory so the os tool can find it later.

3. fastapi (FastAPI, HTTPException)
What it is: The Kitchen Manager. This is the massive framework running the whole show.

What it does in our code: * FastAPI creates the actual server instance (app = FastAPI(...)). It is the thing listening to port 8000 waiting for users to connect.

HTTPException is your emergency brake. If someone tries to chat but your API key is missing, or if the internet drops, you use this to cleanly crash the request and send a professional error message back to the user (like a 500 Internal Server Error) instead of taking down the whole server.

4. pydantic (BaseModel)
What it is: The Bouncer / Quality Control.

What it does in our code: When a user sends data to your /chat endpoint, they might send garbage. They might send a number, an empty file, or a weird JSON structure. By creating the ChatRequest class using BaseModel, you are telling the bouncer: "If the user's request does not contain exactly a string named 'prompt', reject them instantly with a 422 Unprocessable Entity error before my code even runs." It protects your server from bad data.

5. httpx
What it is: The Asynchronous Delivery Driver.

What it does in our code: As we discussed, this is the tool that packages up your prompt, drives out of your computer, travels across the internet to GROQ's servers, waits for the model to generate a response, and brings the text back—all while allowing your FastAPI server to keep helping other users. '''

# load the secret keys from .env file into memory
load_dotenv()

# initialize FastAPI app
app = FastAPI(title="Distributed AI Gateway & Semantic Cache")

# Define the structure of the incoming json body
class ChatRequest(BaseModel):
    prompt: str

# endpoints
@app.get("/health")
async def health_check():
    """A simple endpoint to check if our server is alive."""
    return {"status": "healthy", "message": "The Distributed AI Gateway is running."}

@app.post("/chat")
async def chat_proxy(request: ChatRequest):
    """The main proxy endpoint that intercepts the user's prompt and calls OpenAI."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ API key is not set in the environment variables.")
    
    # The headers metadata (The shipping label)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # The json payload (The package contents)
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": 150
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0  # Set a timeout for the request
            )

            # If OpenAI returns an error (like a bad API key), pass it to the user
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            # extract the response from OpenAI
            data = response.json()
            ai_text = data["choices"][0]["message"]["content"]

            return {"gateway_status": "success", "ai_response": ai_text}
        
        except httpx.RequestError as e:
            # Handle network-related errors
            raise HTTPException(status_code=503, detail=f"Failed to reach OpenAI: {str(e)}")
            