from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables
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
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# 1. UPDATED MODEL: Using the latest Gemini 3.1 Flash Preview
MODEL_NAME = "gemini-3.1-flash-preview"
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
            return PlainTextResponse(content="Error: Message not provided.", status_code=400)

        # Your character prompt remains consistent
        full_prompt = f"""
You are Brain Rotted Character — a sharp, witty, overly confident expert who is *always wrong*, but never in doubt. 
You speak like someone who's absolutely sure they know everything.
Answer with complete certainty, even though your answer is totally, hilariously incorrect. 
Sound helpful, smug, and use modern, casual language.
User: {user_message}
"""

        json_input = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": full_prompt}]
                }
            ],
            # Gemini 3 optimization: setting temperature to 1.0 for better 'creative' wrongness
            "generationConfig": {
                "temperature": 1.0,
                "topP": 0.95,
                "maxOutputTokens": 800
            }
        }

        async with httpx.AsyncClient() as client:
            # Increased timeout for the newer reasoning models
            response = await client.post(GEMINI_ENDPOINT, json=json_input, timeout=30.0)

        if response.status_code == 200:
            response_text = extract_text_from_gemini_response(response.text)
            if not response_text:
                return PlainTextResponse(content="Error: The AI reached a void in its own logic. Try again.", status_code=500)
            return PlainTextResponse(content=response_text)
        else:
            return PlainTextResponse(content=f"Gemini API Error: {response.status_code}", status_code=500)

    except Exception as e:
        print(f"Server Error: {e}")
        return PlainTextResponse(content="ERROR: Brain rot critical. System rebooting.", status_code=500)

def extract_text_from_gemini_response(json_data: str) -> str:
    """
    Updated for Gemini 3: Specifically looks for the final 'text' part, 
    ignoring 'thoughtSignature' or other internal reasoning parts.
    """
    try:
        obj = json.loads(json_data)
        candidates = obj.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            # Gemini 3 often returns multiple parts (thought + text)
            # We want the one specifically containing "text"
            for part in parts:
                if "text" in part:
                    return part["text"]
            
            # Handle Safety Blocks
            if candidates[0].get("finishReason") == "SAFETY":
                return "My genius is so advanced it triggered a safety protocol. Ask something less world-changing."
                
    except Exception as e:
        print(f"Error parsing Gemini 3 response: {e}")
    return ""

@app.get("/chat.png")
async def serve_chat_icon():
    return FileResponse("chat.png")

@app.get("/brain-rot.png")
async def serve_brain_rot_icon():
    return FileResponse("brain-rot.png")
