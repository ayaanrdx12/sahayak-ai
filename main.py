from fastapi import FastAPI 
app = FastAPI()
import os
odoo_api_key = os.getenv('sahayak-ai-odoo')
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Sahayak AI Service")

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
