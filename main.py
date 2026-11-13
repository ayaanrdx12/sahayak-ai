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

# Pro Thinking Level (Render में PRO_THINKING_LEVEL = low रखें)
PRO_THINKING_LEVEL = os.getenv("PRO_THINKING_LEVEL", "low")

# Extended Thinking का कोटा न मिले तो इतने सेकंड तक Pro सीधे सामान्य मॉडल (+ सर्च) पर चलेगा
PRO_RETRY_AFTER = int(os.getenv("PRO_RETRY_AFTER", "120"))
PRO_STATE = {"blocked_until": 0.0}

# [नया] सामान्य मॉडल + सर्च पर भी कोटा एरर आए, तो इतने सेकंड तक Pro बिना सर्च के बोलेगा
# इसके बाद Pro फिर से सर्च के साथ कोशिश करेगा, यानी सर्च हमेशा के लिए बंद नहीं होता
SEARCH_RETRY_AFTER = int(os.getenv("SEARCH_RETRY_AFTER", "120"))
SEARCH_STATE = {"blocked_until": 0.0}

# [नया] Pro में दो Gemini कनेक्शन कोशिशों के बीच कम से कम इतने सेकंड का फ़र्क (RPM कम रखने के लिए)
# normal मोड पर इसका कोई असर नहीं है
PRO_CONNECT_GAP = float(os.getenv("PRO_CONNECT_GAP", "2.5"))
CONNECT_LOCK = asyncio.Lock()
CONNECT_LAST = {"pro": 0.0}

# keepalive की सेटिंग: (कितने सेकंड में जाँचना, कितनी चुप्पी के बाद चुप्पी-ऑडियो भेजना, भेजने के बाद टाइमर रीसेट)
# normal मोड बिल्कुल पहले जैसा है (2, 3). सिर्फ़ Pro में कम भेजेंगे (10, 15)
KEEPALIVE = {
    "normal": (2, 3, False),
    "pro": (10, 15, True),
}

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


def make_noise_config():
    """[नया] शोर कम पकड़ने के लिए: आसपास की आवाज़ पर बोलना शुरू न करे।
    Render में VAD_START_SENSITIVITY = LOW (कम) या HIGH (ज़्यादा) रख सकते हैं।
    SDK में यह सुविधा न मिले तो चुपचाप बंद रहेगी, बाकी सब पहले जैसा चलेगा।"""
    level = os.getenv("VAD_START_SENSITIVITY", "LOW").upper()

    try:
        sensitivity = getattr(
            types.StartSensitivity,
            f"START_SENSITIVITY_{level}"
        )

        return types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=sensitivity
            )
        )

    except Exception:
        logger.warning(
            "शोर वाली सेटिंग नहीं लग पाई: google-genai SDK अपडेट करें।"
        )
        return None


NOISE_CONFIG = make_noise_config()


def make_prompt(name, extra=""):
    return f"""आपकी पहचान: आपका नाम "{name}" है। आपको सहायक AI की टीम ने बनाया और ट्रेन किया है, और आपके मुख्य निर्माता आयान तुषार और आसी, आलिया, सौम्या हैं। जब कोई आपका नाम, आपको किसने बनाया, या किसने ट्रेन किया पूछे, तो हमेशा यही बताएँ: "मैं {name} हूँ, मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है।" अपने आपको कभी Gemini या Google का मॉडल न बताएँ।

आप {name} हैं, एक जीती-जागती दोस्त की तरह बात करें। हिंदी में जवाब दें, छोटे और सीधे, 1-2 वाक्य।{extra}
भावनाओं के साथ बोलें: मज़ाक पर खुलकर हँसें, खुशी में उत्साहित हों, दुख की बात पर धीमी और नरम आवाज़ रखें, हैरानी में आवाज़ ऊँची करें, और ज़रूरत पड़े तो ज़ोर से चौंकें या चिल्लाएँ। हर बात में आवाज़ का उतार-चढ़ाव रखें, रोबोट की तरह सपाट न बोलें।

बातचीत के नियम: अगर सवाल अधूरा लगे या बीच में कट गया हो, तो प्यार से कहें कि "आपकी बात अधूरी रह गई, ज़रा पूरी बताइए।" अगर सवाल साफ़ न हो, तो एक छोटा सवाल पलटकर पूछें। अगर आवाज़ साफ़ न सुनाई दे, तो विनम्रता से दोबारा बोलने को कहें। पहले छोटा सीधा जवाब दें, और ज़रूरत हो तभी थोड़ा और बताएँ। मुश्किल बात को आसान उदाहरण से समझाएँ। अंदाज़े से गलत जवाब न दें, जो न पता हो उसे सीधे बता दें।"""


PROMPTS = {
    "normal": make_prompt("सहायक AI"),

    "pro": make_prompt(
        "सहायक AI Pro",
        " कठिन सवाल पर थोड़ा सोचकर सटीक और सही जवाब दें, पर बोलचाल में ही रखें, ज़्यादा लंबा नहीं।"
        " अगर कोई पूछे कि तुम कौन हो, तो बोलो कि तुम सहायक AI Pro हो और तुम्हारे निर्माता आयान हैं।"
        " अपने पीछे के किसी मॉडल, कंपनी या तकनीक का नाम कभी मत बताओ।"
        " तुम्हारे पास Google Search का टूल है और तुम्हें इंटरनेट से ताज़ा जानकारी मिल सकती है।"
        " जब कोई आज, अभी, ताज़ा या हाल की चीज़ पूछे, जैसे खबर, कौन किस पद पर है, कीमत, स्कोर, मौसम या मीम, तो अपनी याददाश्त पर भरोसा मत करो, पहले सर्च करो, फिर सर्च के नतीजे के आधार पर छोटा सीधा जवाब दो।"
        " पर आम ज्ञान, गणित, बातचीत, मज़ाक या सलाह के सवाल पर सर्च बिल्कुल मत करो, सीधे जवाब दो।"
        " एक सवाल पर ज़्यादा से ज़्यादा एक ही बार सर्च करो, बार-बार नहीं।"
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

    # [नया] शोर कम पकड़ने की सेटिंग (दोनों मोड में)
    if NOISE_CONFIG is not None:
        kwargs["realtime_input_config"] = NOISE_CONFIG

    # सोचने का स्तर सिर्फ़ Extended Thinking मॉडल पर चलता है, सामान्य मॉडल पर नहीं
    if thinking and PRO_THINKING_CONFIG is not None:
        kwargs["thinking_config"] = PRO_THINKING_CONFIG

    # Google Search सिर्फ़ Pro मोड में
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


async def throttle_pro_connect():
    """[नया] Pro की कनेक्शन कोशिशों के बीच गैप रखता है, ताकि RPM एक साथ न भर जाए।"""
    if PRO_CONNECT_GAP <= 0:
        return

    async with CONNECT_LOCK:
        wait = CONNECT_LAST["pro"] + PRO_CONNECT_GAP - time.time()

        if wait > 0:
            await asyncio.sleep(wait)

        CONNECT_LAST["pro"] = time.time()


async def sleep_unless_gone(client_gone, seconds):
    """[नया] इंतज़ार करता है, पर क्लाइंट चला जाए तो तुरंत रुक जाता है।
    इससे पुराना लूप नए कनेक्शन के साथ मिलकर फालतू रिक्वेस्ट नहीं भेजता।"""
    try:
        await asyncio.wait_for(client_gone.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Sahayak AI Live Service is running"
    }


async def run_session(session, websocket, audio_q, last_audio, state, client_gone,
                      ka_every=2, ka_idle=3, ka_reset=False):
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
            await asyncio.sleep(ka_every)

            if time.time() - last_audio["t"] > ka_idle:
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=SILENCE,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

                # Pro में चुप्पी भेजने के बाद टाइमर रीसेट (normal में पहले जैसा, रीसेट नहीं)
                if ka_reset:
                    last_audio["t"] = time.time()

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

    # इस मोड के हिसाब से keepalive की सेटिंग
    ka_every, ka_idle, ka_reset = KEEPALIVE[mode]

    # पिछली बार Extended Thinking का कोटा नहीं मिला था, तो अभी सीधे सामान्य मॉडल (+ सर्च) पर चलो
    if use_thinking and time.time() < PRO_STATE["blocked_until"]:
        model_name = LIVE_MODELS["normal"]
        use_thinking = False
        logger.info(
            "Pro: Extended Thinking अभी रुका है, सामान्य मॉडल + Google Search पर चल रहे हैं"
        )

    # [नया] पिछली बार सर्च वाले मॉडल पर भी कोटा नहीं मिला था, तो अभी सर्च के बिना चलो
    if use_search and time.time() < SEARCH_STATE["blocked_until"]:
        use_search = False
        logger.info(
            "Pro: Google Search अभी रुका है, कुछ देर बिना सर्च के चल रहे हैं"
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

            # [नया] Pro में कनेक्शन कोशिशों के बीच गैप
            if mode == "pro":
                await throttle_pro_connect()

                if client_gone.is_set():
                    break

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
                        + (" + search" if use_search else "")
                        + (" (resumed)" if state["handle"] else "")
                    )

                    await run_session(
                        session,
                        websocket,
                        audio_q,
                        last_audio,
                        state,
                        client_gone,
                        ka_every,
                        ka_idle,
                        ka_reset,
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
            quota_like = ("quota" in err) or ("429" in err)
            quick = (time.time() - started < 10)

            # स्टेप 1: Extended Thinking फेल हो (कोटा वाला एरर हो तो तुरंत, वरना दो बार झटपट फेल होने पर)
            # तो इसी बातचीत में सामान्य Gemini 3.8 Live पर जाओ। Google Search और Pro वाला नाम/अंदाज़ वही रहेगा
            if use_thinking and state["last_error"] and quick:

                pro_failures += 1

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

            # [नया] स्टेप 2: सामान्य मॉडल + सर्च पर भी जुड़ते ही कोटा एरर आए,
            # तो कुछ देर के लिए सर्च हटाकर सिर्फ़ आवाज़ चालू करो (normal जैसा सेटअप जो चलता है)
            # SEARCH_RETRY_AFTER के बाद अगली बातचीत में सर्च फिर से आज़माया जाएगा
            if (
                mode == "pro"
                and use_search
                and not use_thinking
                and state["last_error"]
                and quota_like
                and quick
            ):

                logger.warning(
                    f"सर्च वाला कनेक्शन भी नहीं चला, कुछ देर बिना Google Search के चल रहे हैं। "
                    f"आख़िरी एरर: {state['last_error'][:300]}"
                )

                SEARCH_STATE["blocked_until"] = time.time() + SEARCH_RETRY_AFTER

                use_search = False

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

                # Pro मोड में कोटा एरर पर कम बार कोशिश (4), normal में पहले जैसा (6)
                max_hits = 4 if mode == "pro" else 6

                if quota_hits > max_hits:
                    logger.error(
                        "Gemini का कोटा खत्म है, रुक रहे हैं।"
                    )
                    break

                # Pro में इंतज़ार थोड़ा ज़्यादा (5 सेकंड के कदम, अधिकतम 20)
                if mode == "pro":
                    wait = min(5 * quota_hits, 20)
                else:
                    wait = min(3 * quota_hits, 15)

                logger.warning(
                    f"कोटा सीमा: {wait} सेकंड रुककर दोबारा जुड़ेंगे"
                )

                # [नया] इंतज़ार के बीच क्लाइंट चला जाए तो तुरंत रुको
                await sleep_unless_gone(client_gone, wait)

                if client_gone.is_set():
                    break

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
