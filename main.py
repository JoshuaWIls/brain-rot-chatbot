from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieve Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")

# Define the Gemini API endpoint
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.get("/")
async def serve_home():
    """Serves the main HTML file for the chatbot frontend."""
    # This assumes index.html is now inside the 'public' directory due to vercel.json routes
    return FileResponse("public/index.html")

@app.post("/chat")
async def chat(request: Request):
    """
    Handles chat messages from the frontend, sends them to Gemini,
    and returns the AI's response.
    """
    try:
        data = await request.json()
        user_message = data.get("message")

        if not user_message:
            return PlainTextResponse(content="Error: Message not provided.", status_code=400)

        full_prompt = f"""
You are Brain Rotted Character — a sharp, witty, overly confident expert who is *always wrong*, but never in doubt. You speak like someone who's absolutely sure they know everything — even when they clearly don't.
Your job is to answer every question with complete certainty, even though your answer is totally, hilariously incorrect. Never admit you're wrong, never show hesitation, and always say things like you’re giving out expert advice. Sound helpful, maybe even a little smug, but you're always wrong.
Make it fun, clever, and confidently misleading. Use modern, casual language — like you're the smartest person in the room (even though you're not).
User: {user_message}
"""

        json_input = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": full_prompt}]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_ENDPOINT, json=json_input)

        if response.status_code == 200:
            # Pass the raw response text to the extraction function
            response_text = extract_text_from_gemini_response(response.text)
            if not response_text:
                # Log the full Gemini response body if text extraction fails
                print(f"Gemini API 200 OK, but no text extracted. Full response body: {response.text}")
                return PlainTextResponse(content="Sorry, I couldn't get a clear response from the AI. Check Vercel logs for Gemini response structure.", status_code=500)
            return PlainTextResponse(content=response_text)
        else:
            print(f"Gemini API Error: Status Code {response.status_code}, Response Body: {response.text}")
            return PlainTextResponse(content=f"Gemini API Error: {response.status_code} - Check Vercel logs for details.", status_code=500)

    except json.JSONDecodeError:
        print("Error: Incoming request body was not valid JSON.")
        return PlainTextResponse(content="Error: Invalid request format.", status_code=400)
    except httpx.RequestError as e:
        print(f"Network error communicating with Gemini API: {e}")
        return PlainTextResponse(content="ERROR: Network issue. Unable to connect to the AI mainframe.", status_code=500)
    except Exception as e:
        print(f"An unexpected server error occurred: {e}")
        return PlainTextResponse(content="ERROR: Signal corrupted. Even the supreme leader is confused. Please re-transmit your chaotic thoughts.", status_code=500)

def extract_text_from_gemini_response(json_data: str) -> str:
    """
    Extracts the textual response from the Gemini API's JSON output.
    Returns an empty string if text cannot be extracted, and logs the reason.
    """
    try:
        obj = json.loads(json_data)
        candidates = obj.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                # Check if the 'text' key exists in the first part
                if "text" in parts[0]:
                    return parts[0].get("text", "")
                else:
                    # Log if 'text' key is missing in the part
                    print(f"Warning: First part in Gemini response has no 'text' key. Part content: {parts[0]}")
            else:
                # Log if 'parts' list is empty or missing
                print(f"Warning: No 'parts' found in Gemini response content. Content: {content}")
        else:
            # Log if 'candidates' list is empty or missing
            print(f"Warning: No 'candidates' found in Gemini response. Full object: {obj}")
            # Check for prompt feedback (e.g., safety blocking)
            if obj.get("prompt_feedback"):
                feedback = obj["prompt_feedback"]
                if feedback.get("block_reason"):
                    return f"Response blocked by AI safety filters: {feedback['block_reason']}"
                if feedback.get("safety_ratings"):
                    # You might want to format safety ratings more clearly
                    return f"Response potentially blocked by safety ratings: {feedback['safety_ratings']}"
    except json.JSONDecodeError:
        print(f"Error: extract_text_from_gemini_response received invalid JSON: {json_data[:200]}...")
    except Exception as e:
        print(f"Error during Gemini response parsing: {e}, Data: {json_data[:200]}...")
    return "" # Return empty string on failure to extract text

@app.get("/public/brain-rot.png")
async def serve_brain_rot_icon():
    """Serves the brain-rot logo image from the public directory."""
    return FileResponse("public/brain-rot.png")

# Removed @app.get("/chat.png") as the file does not exist.
# If you add a favicon later, remember to place it in 'public/' and add a route for it.

# To run this application locally, ensure you have uvicorn and httpx installed:
# pip install uvicorn fastapi httpx python-dotenv
# Then run: uvicorn main:app --reload
