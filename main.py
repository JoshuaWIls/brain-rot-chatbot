from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
# This line should be at the very top to ensure .env is loaded before other imports
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
# This allows your frontend (e.g., deployed on Vercel) to make requests
# to this backend (also deployed on Vercel as a serverless function).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Retrieve Gemini API key from environment variables
# IMPORTANT: For Vercel deployment, set GEMINI_API_KEY in your Vercel project's
# Environment Variables (Settings -> Environment Variables).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Raise an error if the API key is not set, preventing deployment issues.
    # This error will appear in your Vercel build/runtime logs.
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it.")

# Define the Gemini API endpoint
# Using gemini-2.0-flash as specified
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.get("/")
async def serve_home():
    """Serves the main HTML file for the chatbot frontend."""
    return FileResponse("index.html")

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

        # Construct the full prompt for the Gemini model
        # The prompt defines the AI's persona and the user's input.
        full_prompt = f"""
You are Brain Rotted Character — a sharp, witty, overly confident expert who is *always wrong*, but never in doubt. You speak like someone who's absolutely sure they know everything — even when they clearly don't.
Your job is to answer every question with complete certainty, even though your answer is totally, hilariously incorrect. Never admit you're wrong, never show hesitation, and always say things like you’re giving out expert advice. Sound helpful, maybe even a little smug, but you're always wrong.
Make it fun, clever, and confidently misleading. Use modern, casual language — like you're the smartest person in the room (even though you're not).
User: {user_message}
"""

        # Prepare the JSON payload for the Gemini API request
        json_input = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": full_prompt}]
                }
            ]
        }

        # Use httpx to make an asynchronous POST request to the Gemini API
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_ENDPOINT, json=json_input)

        # Check if the API request was successful (status code 200)
        if response.status_code == 200:
            response_text = extract_text_from_gemini_response(response.text)
            if not response_text:
                # If text extraction failed but status was 200, log the full response
                print(f"Gemini API 200 OK, but no text extracted. Full response: {response.text}")
                return PlainTextResponse(content="Sorry, I couldn't get a clear response from the AI.", status_code=500)
            return PlainTextResponse(content=response_text)
        else:
            # If the API request itself failed, log the full error response from Gemini
            print(f"Gemini API Error: Status Code {response.status_code}, Response Body: {response.text}")
            return PlainTextResponse(content=f"Gemini API Error: {response.status_code} - Check Vercel logs for details.", status_code=500)

    except json.JSONDecodeError:
        # Handle cases where the incoming request body is not valid JSON
        print("Error: Incoming request body was not valid JSON.")
        return PlainTextResponse(content="Error: Invalid request format.", status_code=400)
    except httpx.RequestError as e:
        # Handle network-related errors when communicating with Gemini API
        print(f"Network error communicating with Gemini API: {e}")
        return PlainTextResponse(content="ERROR: Network issue. Unable to connect to the AI mainframe.", status_code=500)
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"An unexpected server error occurred: {e}")
        return PlainTextResponse(content="ERROR: Signal corrupted. Even the supreme leader is confused. Please re-transmit your chaotic thoughts.", status_code=500)

def extract_text_from_gemini_response(json_data: str) -> str:
    """
    Extracts the textual response from the Gemini API's JSON output.
    Returns an empty string if text cannot be extracted.
    """
    try:
        obj = json.loads(json_data)
        candidates = obj.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                # Ensure the 'text' key exists in the first part
                if "text" in parts[0]:
                    return parts[0].get("text", "")
                else:
                    print(f"Warning: First part in Gemini response has no 'text' key. Parts: {parts[0]}")
            else:
                print(f"Warning: No 'parts' found in Gemini response content. Content: {content}")
        else:
            # This block will catch cases where candidates list is empty or missing
            # It's good to log the full object to see why candidates are missing
            print(f"Warning: No 'candidates' found in Gemini response. Full object: {obj}")
            # Check for prompt feedback, e.g., safety blocking
            if obj.get("prompt_feedback"):
                feedback = obj["prompt_feedback"]
                if feedback.get("block_reason"):
                    return f"Response blocked by AI safety filters: {feedback['block_reason']}"
                if feedback.get("safety_ratings"):
                    return f"Response potentially blocked by safety ratings: {feedback['safety_ratings']}"

    except json.JSONDecodeError:
        print(f"Error: extract_text_from_gemini_response received invalid JSON: {json_data[:200]}...")
    except Exception as e:
        print(f"Error during Gemini response parsing: {e}, Data: {json_data[:200]}...")
    return "" # Return empty string on failure to extract text

@app.get("/chat.png")
async def serve_chat_icon():
    """Serves the chat icon (favicon)."""
    return FileResponse("chat.png")

@app.get("/brain-rot.png")
async def serve_brain_rot_icon():
    """Serves the brain-rot logo image."""
    return FileResponse("brain-rot.png")

# To run this application locally, ensure you have uvicorn and httpx installed:
# pip install uvicorn httpx
# Then run: uvicorn main:app --reload
