import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

app = FastAPI(title="Sahayak AI Direct")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    api_key = os.getenv("SAHAYAK_AI_ODOC")
    if not api_key:
        return {"status": "error", "reply": "API Key not configured"}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    
    try:
        response = model.generate_content(message)
        return {"status": "success", "reply": response.text}
    except Exception as e:
        return {"status": "error", "reply": str(e)}
