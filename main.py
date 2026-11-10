import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Sahayak AI Service")

# CORS middleware add kiya hai taaki Odoo se requests allow ho sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Retrieve API keys securely from environment variables
ODOO_API_KEY = os.getenv("sahayak-ai-odoo")
GEMINI_API_KEY = os.getenv("sahayak-ai")

# Configure Gemini AI client safely
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

class PromptRequest(BaseModel):
    prompt: str

# In-memory device status state
device_state = {"isConnected": False}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "details": "Sahayak AI backend is running smoothly."
    }

@app.post("/generate")
def generate_text(request: PromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API key is not configured on the server."
        )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(request.prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/odoo-status")
def odoo_status():
    if not ODOO_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Odoo API key is not configured on the server."
        )
    return {
        "status": "connected",
        "message": "Odoo API key is active and recognized."
    }

# Device connectivity status check karne ke liye GET endpoint
@app.get("/api/device-status")
def get_device_status():
    return {"isConnected": device_state["isConnected"]}

# Device status toggle karne ke liye POST endpoint
@app.post("/api/device-status")
def toggle_device_status():
    device_state["isConnected"] = not device_state["isConnected"]
    return {"isConnected": device_state["isConnected"]}
