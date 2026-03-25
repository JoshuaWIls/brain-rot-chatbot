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
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
@app.get("/")
async def serve_home():
    for path in ["public/index.html", "index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return PlainTextResponse("index.html not found", status_code=404)

@app.post("/chat")
async def chat(request: Request):
    if not GEMINI_API_KEY:
        return PlainTextResponse("Server Error: Missing API Key.", status_code=500)

    try:
        data = await request.json()
        user_message = data.get("message")

        if not user_message:
            return PlainTextResponse("Error: No message provided.", status_code=400)

        full_prompt = f"""You are Brain Rotted Character — a sharp, witty, overly confident expert who is *always wrong*, but never in doubt. You speak like someone who's absolutely sure they know everything — even when they clearly don't.
Your job is to answer every question with complete certainty, even though your answer is totally, hilariously incorrect. Never admit you're wrong, never show hesitation, and always say things like you're giving out expert advice. Sound helpful, maybe even a little smug, but you're always wrong.
Make it fun, clever, and confidently misleading. Use modern, casual language — like you're the smartest person in the room (even though you're not).

User: {user_message}"""

        json_input = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 1.0}
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_ENDPOINT, json=json_input, timeout=30.0)

        if response.status_code == 200:
            res_data = response.json()
            parts = res_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = next((p["text"] for p in parts if "text" in p), "No response generated.")
            return PlainTextResponse(content=text)
        else:
            print(f"Google API Error: {response.text}")
            return PlainTextResponse(content=f"API Error: {response.status_code}", status_code=500)

    except httpx.RequestError as e:
        print(f"Network error: {e}")
        return PlainTextResponse("Network error connecting to AI.", status_code=500)
    except Exception as e:
        print(f"Internal Exception: {e}")
        return PlainTextResponse(f"Internal Server Error: {str(e)}", status_code=500)

@app.get("/{filename}.png")
async def serve_image(filename: str):
    for folder in ["public", ""]:
        path = os.path.join(folder, f"{filename}.png")
        if os.path.exists(path):
            return FileResponse(path)
    return PlainTextResponse("Image not found", status_code=404)
