import os
import json
import urllib.request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Sahayak AI Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SerialValidation(BaseModel):
    serial_number: string

def get_valid_serials():
    url = "https://raw.githubusercontent.com/ayaanrdx12/sahayak-ai/main/serials.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("serials", [])
    except Exception as e:
        print(f"Error fetching serials: {e}")
        return []

@app.post("/validate-and-chat")
async def validate_and_chat(payload: SerialValidation):
    valid_serials = get_valid_serials()
    if payload.serial_number not in valid_serials:
        raise HTTPException(status_code=400, detail="Invalid Serial Number")
    
    # Retrieve API key and setup Gemini
    api_key = os.getenv("SAHAYAK_AI_ODOC")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Example interaction logic
    response = model.generate_content("नमस्ते, मैं आपकी किस प्रकार सहायता कर सकता हूँ?")
    return {"status": "connected", "reply": response.text}
