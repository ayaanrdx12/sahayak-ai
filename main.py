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

# Pure Voice Model Definitions
PRIMARY_LIVE_MODEL = "gemini-3.8-live"
FALLBACK_LIVE_MODEL = "gemini-3-flash-live"

# Initialize Client
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize GenAI client: {e}")
    client = None

async def proxy_media_stream(session, websocket: WebSocket):
    """Forward audio frames from the Gemini session to the WebSocket client."""
    try:
        async for response in session.receive():
            server_content = response.server_content
            if server_content is not None:
                model_turn = server_content.model_turn
                if model_turn is not None:
                    for part in model_turn.parts:
                        # Stream audio back to client
                        if part.inline_data is not None:
                            await websocket.bytes_send(part.inline_data.data)
    except Exception as e:
        logger.error(f"Error in media stream proxy: {e}")

async def handle_client_uplink(websocket: WebSocket, session):
    """Forward audio frames from the WebSocket client to the Gemini session."""
    try:
        while True:
            data = await websocket.receive_bytes()
            await session.send(input={
                "data": data,
                "mime_type": "audio/pcm"
            })
    except WebSocketDisconnect:
        logger.info("Client disconnected from uplink.")
    except Exception as e:
        logger.error(f"Error handling client uplink: {e}")

@app.websocket("/ws/live")
async def live_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection accepted.")

    if not client:
        logger.critical("GenAI client not initialized.")
        await websocket.close(code=1011, reason="Server configuration error.")
        return

    config = types.LiveConnectConfig(
        response_modalities=[types.LiveModality.AUDIO]
    )

    active_model = PRIMARY_LIVE_MODEL
    session = None

    # Attempt connection with primary live model
    try:
        logger.info(f"Attempting connection with primary model: {active_model}")
        session = await client.aio.live.connect(model=active_model, config=config).__aenter__()
        logger.info(f"Connected to Gemini Live using {active_model}")
    except Exception as e:
        logger.error(f"Failed to connect with primary model {active_model}: {e}")
        # Failover to secondary live model
        active_model = FALLBACK_LIVE_MODEL
        logger.info(f"Attempting failover to fallback model: {active_model}")
        try:
             session = await client.aio.live.connect(model=active_model, config=config).__aenter__()
             logger.info(f"Connected to Gemini Live using fallback: {active_model}")
        except Exception as fallback_error:
             logger.critical(f"Fallback connection also failed: {fallback_error}")
             await websocket.close(code=1011, reason="Live session connection failed.")
             return

    try:
        # Run uplink and downlink tasks concurrently
        uplink_task = asyncio.create_task(handle_client_uplink(websocket, session))
        downlink_task = asyncio.create_task(proxy_media_stream(session, websocket))
        await asyncio.gather(uplink_task, downlink_task)
    except Exception as e:
         logger.error(f"Error during live session with {active_model}: {e}")
    finally:
        if session:
            await session.__aexit__(None, None, None)
            logger.info("Live session closed.")
    
    logger.info("WebSocket connection closed.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
