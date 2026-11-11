import asyncio
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sahayak_AI")

app = FastAPI(title="Sahayak AI Voice Backend")

# Model Definition
PRIMARY_LIVE_MODEL = "gemini-3-flash-live"

# Initialize Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
async def root():
    return {"status": "ok", "message": "Sahayak AI Live Service is running"}

@app.websocket("/ws/live")
async def live_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"]
    )

    try:
        async with client.aio.live.connect(model=PRIMARY_LIVE_MODEL, config=config) as session:
            logger.info("Connected to Gemini Live API session")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                except WebSocketDisconnect:
                    logger.info("Client disconnected from WebSocket")
                except Exception as e:
                    logger.error(f"Error receiving from client: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    await websocket.send_bytes(part.inline_data.data)
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            await asyncio.gather(receive_from_client(), receive_from_gemini())

    except Exception as e:
        logger.error(f"Gemini Live session error: {e}")
        await websocket.close()
