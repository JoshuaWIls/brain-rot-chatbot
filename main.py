from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Use the stable alias to avoid 404 "Model Not Found" errors
MODEL_NAME = "gemini-3.1-flash" 
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

@app.get("/")
async def serve_home():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message")

        if not user_message:
            return PlainTextResponse(content="Error: No message.", status_code=400)

        full_prompt = f"""
You are Brain Rotted Character — a sharp, witty expert who is *always wrong*. 
Speak with absolute confidence. Use modern, casual language.
User: {user_message}
"""

        json_input = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": 800
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_ENDPOINT, json=json_input, timeout=30.0)

        if response.status_code == 200:
            res_data = response.json()
            # Gemini 3.1 response parsing
            try:
                parts = res_data['candidates'][0]['content']['parts']
                # Filter for the text part (skipping thought signatures)
                text_response = next((p['text'] for p in parts if 'text' in p), "")
                return PlainTextResponse(content=text_response)
            except (KeyError, IndexError):
                return PlainTextResponse(content="Error: AI brain stall.", status_code=500)
        else:
            # This will catch 404s, 400s, etc.
            return PlainTextResponse(content=f"API Error: {response.status_code}", status_code=response.status_code)

    except Exception as e:
        return PlainTextResponse(content=f"Server Error: {str(e)}", status_code=500)

@app.get("/chat.png")
async def serve_icon():
    return FileResponse("chat.png")
