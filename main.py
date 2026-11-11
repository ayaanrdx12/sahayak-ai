import asyncio
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sahayak_AI")

app = FastAPI(title="Sahayak AI Voice Backend")

PRIMARY_LIVE_MODEL = os.getenv(
    "LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@app.get("/")
async def root():
    return {"status": "ok", "message": "Sahayak AI Live Service is running"}


@app.websocket("/ws/live")
async def live_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="आप सहायक AI हैं। हिंदी में जवाब दें, छोटे और सीधे, ज़्यादा से ज़्यादा 1-2 वाक्य।",
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                silence_duration_ms=300,
            )
        ),
    )

    try:
        async with client.aio.live.connect(
            model=PRIMARY_LIVE_MODEL, config=config
        ) as session:
            logger.info("Connected to Gemini Live API session")

            async def receive_from_client():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=data, mime_type="audio/pcm;rate=16000"
                            )
                        )
                except WebSocketDisconnect:
                    logger.info("Client disconnected from WebSocket")
                except Exception as e:
                    logger.error(f"Error receiving from client: {e}")

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            sc = response.server_content
                            if sc:
                                if getattr(sc, "interrupted", False):
                                    await websocket.send_text("interrupted")
                                if sc.model_turn:
                                    for part in sc.model_turn.parts:
                                        if part.inline_data:
                                            await websocket.send_bytes(
                                                part.inline_data.data
                                            )
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            tasks = [
                asyncio.create_task(receive_from_client()),
                asyncio.create_task(receive_from_gemini()),
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in tasks:
                t.cancel()

    except Exception as e:
        logger.error(f"Gemini Live session error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
