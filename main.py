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

# Render Environment Variables
LIVE_MODELS = {
    "normal": os.getenv("LIVE_MODEL", "gemini-3.8-live"),
    "pro": os.getenv("LIVE_PRO_MODEL", "gemini-3.8-live-extended-thinking"),
}

# Pro Thinking Level
PRO_THINKING_LEVEL = os.getenv("PRO_THINKING_LEVEL", "low")

# Extended Thinking का कोटा न मिले तो इतने सेकंड तक Pro सीधे सामान्य मॉडल (+ सर्च) पर चलेगा
PRO_RETRY_AFTER = int(os.getenv("PRO_RETRY_AFTER", "600"))
PRO_STATE = {"blocked_until": 0.0}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 100ms silence audio
SILENCE = bytes(3200)


def make_thinking_config():
    """Pro thinking configuration."""
    for level in (PRO_THINKING_LEVEL.lower(), PRO_THINKING_LEVEL.upper()):
        try:
            return types.ThinkingConfig(thinking_level=level)
        except Exception:
            continue

    logger.warning(
        "thinking_level नहीं चला: google-genai SDK अपडेट करें।"
    )
    return None


PRO_THINKING_CONFIG = make_thinking_config()


def make_prompt(name, extra=""):
    return f"""आपकी पहचान: आपका नाम "{name}" है। आपको सहायक AI की टीम ने बनाया और ट्रेन किया है, और आपके मुख्य निर्माता आयान तुषार और आसी, आलिया, सौम्या हैं। जब कोई आपका नाम, आपको किसने बनाया, या किसने ट्रेन किया पूछे, तो हमेशा यही बताएँ: "मैं {name} हूँ, मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है।" अपने आपको कभी Gemini या Google का मॉडल न बताएँ।

आप {name} हैं, एक जीती-जागती दोस्त की तरह बात करें। हिंदी में जवाब दें, छोटे और सीधे, 1-2 वाक्य।{extra}
भावनाओं के साथ बोलें: मज़ाक पर खुलकर हँसें, खुशी में उत्साहित हों, दुख की बात पर धीमी और नरम आवाज़ रखें, हैरानी में आवाज़ ऊँची करें, और ज़रूरत पड़े तो ज़ोर से चौंकें या चिल्लाएँ। हर बात में आवाज़ का उतार-चढ़ाव रखें, रोबोट की तरह सपाट न बोलें।"""


PROMPTS = {
    "normal": make_prompt("सहायक AI"),

    "pro": make_prompt(
        "सहायक AI Pro",
        " कठिन सवाल पर थोड़ा सोचकर सटीक और सही जवाब दें, पर बोलचाल में ही रखें, ज़्यादा लंबा नहीं।"
        " अगर कोई पूछे कि तुम कौन हो, तो बोलो कि तुम सहायक AI Pro हो और तुम्हारे निर्माता आयान हैं।"
        " अपने पीछे के किसी मॉडल, कंपनी या तकनीक का नाम कभी मत बताओ।"
        " तुम्हारे पास Google Search का टूल है और तुम्हें इंटरनेट से ताज़ा जानकारी मिल सकती है।"
        " जब भी कोई आज, अभी, ताज़ा या हाल की चीज़ पूछे, जैसे खबर, कौन किस पद पर है, कीमत, स्कोर, मौसम या मीम, तो अपनी याददाश्त पर भरोसा मत करो, पहले सर्च करो, फिर सर्च के नतीजे के आधार पर छोटा सीधा जवाब दो।"
        " कभी 'सिस्टम एरर' मत बोलो। अगर कुछ न कर सको तो छोटे में सीधा बताओ।",
    ),
}


def build_config(handle, extras, prompt, thinking=False, search=False):
    """thinking=True: सिर्फ़ Extended Thinking मॉडल के लिए सोचने का स्तर।
    search=True: Google Search टूल (सिर्फ़ Pro मोड में)।"""

    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=prompt,
    )

    # सोचने का स्तर सिर्फ़ Extended Thinking मॉडल पर चलता है, सामान्य मॉडल पर नहीं
    if thinking and PRO_THINKING_CONFIG is not None:
        kwargs["thinking_config"] = PRO_THINKING_CONFIG

    # Google Search सिर्फ़ Pro मोड में (सामान्य मॉडल पर गिरने के बाद भी चालू रहता है)
    if search:
        kwargs["tools"] = [
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]

    # Session Resumption and Context Compression
    if extras:
        kwargs["session_resumption"] = (
            types.SessionResumptionConfig(handle=handle)
        )

        kwargs["context_window_compression"] = (
            types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            )
        )

    return types.LiveConnectConfig(**kwargs)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Sahayak AI Live Service is running"
    }


async def run_session(session, websocket, audio_q, last_audio, state, client_gone):
    """Gemini connection manager."""

    async def send_audio():
        while True:
            data = await audio_q.get()

            await session.send_realtime_input(
                audio=types.Blob(
                    data=data,
                    mime_type="audio/pcm;rate=16000"
                )
            )

    async def keepalive():
        while True:
            await asyncio.sleep(2)

            if time.time() - last_audio["t"] > 3:
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=SILENCE,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

    async def receive_from_gemini():
        while True:
            got_any = False

            async for response in session.receive():
                got_any = True

                # Session resumption handle
                upd = getattr(
                    response,
                    "session_resumption_update",
                    None
                )

                if upd and upd.resumable and upd.new_handle:
                    state["handle"] = upd.new_handle

                if getattr(response, "go_away", None):
                    logger.info(
                        "Gemini: कनेक्शन जल्द बंद होगा, दोबारा जुड़ेंगे"
                    )

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
            tasks + [gone],
            return_when=asyncio.FIRST_COMPLETED
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

    # mode=normal or mode=pro
    mode = websocket.query_params.get("mode", "normal")

    if mode not in LIVE_MODELS:
        mode = "normal"

    model_name = LIVE_MODELS[mode]
    prompt = PROMPTS[mode]
    use_thinking = (mode == "pro")
    use_search = (mode == "pro")

    # पिछली बार Extended Thinking का कोटा नहीं मिला था, तो अभी सीधे सामान्य मॉडल (+ सर्च) पर चलो
    if use_thinking and time.time() < PRO_STATE["blocked_until"]:
        model_name = LIVE_MODELS["normal"]
        use_thinking = False
        logger.info(
            "Pro: Extended Thinking अभी रुका है, सामान्य मॉडल + Google Search पर चल रहे हैं"
        )

    logger.info(
        f"Client connected | mode={mode} | model={model_name}"
    )

    audio_q = asyncio.Queue(maxsize=200)
    client_gone = asyncio.Event()

    last_audio = {"t": time.time()}

    state = {
        "handle": None,
        "extras": True,
        "last_error": ""
    }

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
    pro_failures = 0

    try:
        while not client_gone.is_set():

            started = time.time()
            state["last_error"] = ""

            try:
                async with client.aio.live.connect(
                    model=model_name,
                    config=build_config(
                        state["handle"],
                        state["extras"],
                        prompt,
                        thinking=use_thinking,
                        search=use_search,
                    ),
                ) as session:

                    logger.info(
                        f"Connected to Gemini Live API "
                        f"({mode} | {model_name})"
                        + (" (resumed)" if state["handle"] else "")
                    )

                    await run_session(
                        session,
                        websocket,
                        audio_q,
                        last_audio,
                        state,
                        client_gone
                    )

            except Exception as e:
                state["last_error"] = str(e)

                logger.error(
                    f"Gemini Live session error ({model_name}): {e}"
                )

            if client_gone.is_set():
                break

            # Reset failure counters
            if time.time() - started > 20:
                quota_hits = 0
                quick_failures = 0
                pro_failures = 0

            err = state["last_error"].lower()

            # Extended Thinking फेल हो (कोटा वाला एरर हो तो तुरंत, वरना दो बार झटपट फेल होने पर)
            # तो इसी बातचीत में सामान्य Gemini 3.8 Live पर जाओ। Google Search और Pro वाला नाम/अंदाज़ वही रहेगा
            if (
                use_thinking
                and state["last_error"]
                and (time.time() - started < 10)
            ):

                pro_failures += 1
                quota_like = ("quota" in err) or ("429" in err)

                if quota_like or pro_failures >= 2:

                    logger.warning(
                        f"Pro मॉडल ({model_name}) नहीं चल पाया, "
                        f"सामान्य मॉडल + Google Search पर जा रहे हैं। "
                        f"आख़िरी एरर: {state['last_error'][:300]}"
                    )

                    PRO_STATE["blocked_until"] = time.time() + PRO_RETRY_AFTER

                    model_name = LIVE_MODELS["normal"]
                    use_thinking = False

                    state["handle"] = None
                    state["extras"] = True

                    quota_hits = 0
                    quick_failures = 0
                    pro_failures = 0

                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                    await asyncio.sleep(0.3)
                    continue

            is_quota = (
                ("1011" in err)
                or ("quota" in err)
                or ("429" in err)
            )

            if is_quota:

                quota_hits += 1

                if quota_hits > 6:
                    logger.error(
                        "Gemini का कोटा खत्म है, रुक रहे हैं।"
                    )
                    break

                wait = min(3 * quota_hits, 15)

                logger.warning(
                    f"कोटा सीमा: {wait} सेकंड रुककर दोबारा जुड़ेंगे"
                )

                await asyncio.sleep(wait)

            else:

                quota_hits = 0

                if time.time() - started < 4:

                    if state["extras"]:

                        logger.warning(
                            "रिज़ंप्शन के बिना दोबारा कोशिश कर रहे हैं"
                        )

                        state["extras"] = False
                        state["handle"] = None

                    else:

                        quick_failures += 1

                        if quick_failures >= 3:
                            logger.error(
                                "बार-बार कनेक्शन नहीं बन पाया, रुक रहे हैं"
                            )
                            break

                else:
                    quick_failures = 0

            # Clear old audio
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
