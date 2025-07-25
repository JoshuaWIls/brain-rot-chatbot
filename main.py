from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os # Import the os module
from dotenv import load_dotenv # Import load_dotenv

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
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in a .env file or as an environment variable.")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
@app.get("/")
def serve_home():
    return FileResponse("index.html")

@app.post("/chat/")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")
    full_prompt = f"""
You are Brain Rotted Character — a sharp, witty, overly confident expert who is *always wrong*, but never in doubt. You speak like someone who's absolutely sure they know everything — even when they clearly don't.
Your job is to answer every question with complete certainty, even though your answer is totally, hilariously incorrect. Never admit you're wrong, never show hesitation, and always say things like you’re giving out expert advice. Sound helpful, maybe even a little smug, but you're always wrong.
Make it fun, clever, and confidently misleading. Use modern, casual language — like you're the smartest person in the room (even though you're not).
User: {user_message}
"""
    escaped_prompt = full_prompt.replace("\"", "\\\"")

    json_input = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": escaped_prompt}]
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_ENDPOINT, json=json_input)

    if response.status_code == 200:
        response_text = extract_text_from_gemini_response(response.text)
        return PlainTextResponse(content=response_text)
    else:
        return PlainTextResponse(content=f"Gemini API Error: {response.text}", status_code=500)

def extract_text_from_gemini_response(json_data: str) -> str:
    try:
        obj = json.loads(json_data)
        candidates = obj.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "").replace("\n", "<br>")
    except Exception as e:
        print(f"Error parsing response from Gemini API: {e}")
    return "Sorry, I couldn't understand that."

@app.get("/chat.png")
def serve_favicon():
    return FileResponse("chat.png")

# To run this application, use: uvicorn main:app --reload
