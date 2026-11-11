import asyncio
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sahayak AI Voice Backend")

# Model Definition
PRIMARY_LIVE_MODEL = "gemini-3-flash-live"

# Initialize Client
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    logger.critical(f"Failed to initialize GenAI client: {e}")
    client = None

@app.websocket("/ws/live")
async def live_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not client:
        logger.critical("GenAI client not initialized.")
        await websocket.close(code=1011)
        return

    logger.info("New WebSocket connection accepted.")
    
    # Configure the live session
    config = types.LiveConnectConfig(
        response_modalities=[types.LiveModality.AUDIO]
    )

    try:
        async with client.aio.live.connect(model=PRIMARY_LIVE_MODEL, config=config) as session:
            logger.info("Connected to Gemini Live API.")
            
            async def send_to_gemini():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send(input=data, end_of_turn=False)
                except WebSocketDisconnect:
                    logger.info("Client disconnected from WebSocket.")
                except Exception as e:
                    logger.error(f"Error sending to Gemini: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        server_content = response.server_content
                        if server_content and server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                if part.inline_data:
                                    await websocket.send_bytes(part.inline_data.data)
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            # Run send and receive concurrently
            send_task = asyncio.create_task(send_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())
            
            await asyncio.gather(send_task, receive_task)

    except Exception as e:
        logger.error(f"Live session error: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass # WebSocket already closed
        logger.info("WebSocket connection closed.")
