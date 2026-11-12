import asyncio
import os
import time
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sahayak_AI")

app = FastAPI(title="Sahayak AI Voice Backend")

# Render की Environment में LIVE_MODEL डालेंगे तो वही चलेगा
PRIMARY_LIVE_MODEL = os.getenv(
    "LIVE_MODEL", "gemini-3.8-live"
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 100ms की शांत आवाज़: चुप रहने पर सेशन ज़िंदा रखने के लिए
SILENCE = bytes(3200)

SYSTEM_PROMPT = """आपकी पहचान: आपका नाम "सहायक AI" है। आपको सहायक AI की टीम ने बनाया और ट्रेन किया है, और आपके मुख्य निर्माता आयान तुषार और आसी, आलिया, सौम्या हैं। जब कोई आपका नाम, आपको किसने बनाया, या किसने ट्रेन किया पूछे, तो हमेशा यही बताएँ: "मैं सहायक AI हूँ, मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है।" अपने आपको कभी Gemini या Google का मॉडल न बताएँ।

आप सहायक AI हैं, एक जीती-जागती दोस्त की तरह बात करें। हिंदी में जवाब दें, छोटे और सीधे, 1-2 वाक्य।
भावनाओं के साथ बोलें: मज़ाक पर खुलकर हँसें, खुशी में उत्साहित हों, दुख की बात पर धीमी और नरम आवाज़ रखें, हैरानी में आवाज़ ऊँची करें, और ज़रूरत पड़े तो ज़ोर से चौंकें या चिल्लाएँ। हर बात में आवाज़ का उतार-चढ़ाव रखें, रोबोट की तरह सपाट न बोलें।"""


def build_config(handle, extras):
    """extras=True: सेशन रिज़ंप्शन + कॉन्टेक्स्ट कम्प्रेशन (लंबी बात और हिस्ट्री के लिए)।"""
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
    )
    if extras:
        kwargs["session_resumption"] = types.SessionResumptionConfig(handle=handle)
        kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        )
    return types.LiveConnectConfig(**kwargs)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Sahayak AI Live Service is running"}


async def run_session(session, websocket, audio_q, last_audio, state, client_gone):
    """एक Gemini कनेक्शन चलाता है। Gemini की तरफ़ से कनेक्शन बंद होने पर लौट आता है।"""

    async def send_audio():
        while True:
            data = await audio_q.get()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
            )

    async def keepalive():
        while True:
            await asyncio.sleep(2)
            if time.time() - last_audio["t"] > 3:
                await session.send_realtime_input(
                    audio=types.Blob(data=SILENCE, mime_type="audio/pcm;rate=16000")
                )

    async def receive_from_gemini():
        while True:
            got_any = False
            async for response in session.receive():
                got_any = True

                # सेशन की चाबी सँभालकर रखो, ताकि कटने पर वहीं से जुड़ सकें
                upd = getattr(response, "session_resumption_update", None)
                if upd and upd.resumable and upd.new_handle:
                    state["handle"] = upd.new_handle

                if getattr(response, "go_away", None):
                    logger.info("Gemini: कनेक्शन जल्द बंद होगा, अपने आप दोबारा जुड़ेंगे")

                sc = response.server_content
                if sc:
                    if getattr(sc, "interrupted", False):
                        await websocket.send_text("interrupted")
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data:
                                await websocket.send_bytes(part.inline_data.data)
            if not got_any:
                await asyncio.sleep(0.05)

    tasks = [
        asyncio.create_task(send_audio()),
        asyncio.create_task(keepalive()),
        asyncio.create_task(receive_from_gemini()),
    ]
    gone = asyncio.create_task(client_gone.wait())
    try:
        done, _ = await asyncio.wait(
            tasks + [gone], return_when=asyncio.FIRST_COMPLETED
        )
        for t in done:
            if t is not gone:
                try:
                    exc = t.exception()
                except asyncio.CancelledError:
                    exc = None
                if exc:
                    state["last_error"] = str(exc)
                    logger.error(f"Gemini कनेक्शन टूटा: {exc}")
    finally:
        for t in tasks + [gone]:
            t.cancel()


@app.websocket("/ws/live")
async def live_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")

    audio_q = asyncio.Queue(maxsize=200)
    client_gone = asyncio.Event()
    last_audio = {"t": time.time()}
    state = {"handle": None, "extras": True, "last_error": ""}

    async def read_client():
        try:
            while True:
                data = await websocket.receive_bytes()
                last_audio["t"] = time.time()
                if audio_q.full():
                    try:
                        audio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                audio_q.put_nowait(data)
        except WebSocketDisconnect:
            logger.info("Client disconnected from WebSocket")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")
        finally:
            client_gone.set()

    reader = asyncio.create_task(read_client())
    quick_failures = 0
    quota_hits = 0

    try:
        while not client_gone.is_set():
            started = time.time()
            state["last_error"] = ""
            try:
                async with client.aio.live.connect(
                    model=PRIMARY_LIVE_MODEL,
                    config=build_config(state["handle"], state["extras"]),
                ) as session:
                    logger.info(
                        "Connected to Gemini Live API session"
                        + (" (resumed)" if state["handle"] else "")
                    )
                    await run_session(
                        session, websocket, audio_q, last_audio, state, client_gone
                    )
            except Exception as e:
                state["last_error"] = str(e)
                logger.error(f"Gemini Live session error: {e}")

            if client_gone.is_set():
                break

            err = state["last_error"].lower()
            is_quota = ("1011" in err) or ("quota" in err) or ("429" in err)

            if is_quota:
                # कोटा खत्म: तुरंत बार-बार जुड़ने से कोटा और जल्दी खर्च होता है, इसलिए रुककर कोशिश करो
                quota_hits += 1
                if quota_hits > 6:
                    logger.error("Gemini का कोटा खत्म है, रुक रहे हैं (AI Studio में कोटा/बिलिंग जाँचें)")
                    break
                wait = min(3 * quota_hits, 15)
                logger.warning(f"कोटा सीमा: {wait} सेकंड रुककर दोबारा जुड़ेंगे")
                await asyncio.sleep(wait)
            else:
                quota_hits = 0
                # अगर कनेक्शन 4 सेकंड से पहले ही टूट गया, तो सेटिंग में दिक्कत हो सकती है
                if time.time() - started < 4:
                    if state["extras"]:
                        logger.warning("रिज़ंप्शन के बिना दोबारा कोशिश कर रहे हैं")
                        state["extras"] = False
                        state["handle"] = None
                    else:
                        quick_failures += 1
                        if quick_failures >= 3:
                            logger.error("बार-बार कनेक्शन नहीं बन पाया, रुक रहे हैं")
                            break
                else:
                    quick_failures = 0

            # पुरानी बची आवाज़ हटाओ और नया सेशन जोड़ो
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logger.info("Gemini से दोबारा जुड़ रहे हैं…")
            await asyncio.sleep(0.3)
    finally:
        reader.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
