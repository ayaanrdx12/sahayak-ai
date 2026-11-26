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

# ======================================================================
# सहायक AI सर्वर – वर्ज़न 11
# नया क्या है:
#  1) AI की आवाज़ ESP32 को छोटे टुकड़ों (40 ms) में और असली रफ़्तार से भेजी जाती है (पहले एक साथ ढेर आ जाती थी,
#     जिससे ESP32 का कनेक्शन टूट जाता था)।
#  2) चुप्पी (silence) अब सर्वर खुद भरता है, इसलिए ESP32 को लगातार आवाज़ नहीं भेजनी पड़ती।
#  3) शोर पर AI बिना वजह न बोले (VAD कम संवेदनशील) और हिंदी भाषा तय है (पहले कभी-कभी स्पेनिश बोल देता था)।
#  4) AI को धीरे, साफ़ और सामान्य गति से बोलने को कहा गया है।
#  5) इमोजी अब आवाज़ के साथ सही क्रम में जाते हैं।
# ======================================================================
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.8-live").strip()

# चुप्पी भरना: आवाज़ 0.25 सेकंड न आए तो 100 ms चुप्पी भेजो (असली रफ़्तार से)
KA_EVERY = 0.1
KA_IDLE = 0.25

# 1 = ESP32 वाले डिवाइस में AI बीच में नहीं रुकेगा, अपना वाक्य पूरा करेगा
NO_INTERRUPT = os.getenv("NO_INTERRUPT", "1") == "1"

# 1 = Render के Logs में दिखेगा कि AI ने क्या सुना और क्या बोला
DEBUG_HEAR = os.getenv("DEBUG_HEAR", "1") == "1"

# भाषा: hi-IN = हिंदी। अगर इससे Gemini गड़बड़ करे तो Render में SPEECH_LANG खाली कर दें (अपने-आप भी बंद हो जाएगा)
SPEECH_LANG = os.getenv("SPEECH_LANG", "hi-IN").strip()
# आवाज़ का नाम (खाली छोड़ेंगे तो Gemini की अपनी आवाज़)
VOICE_NAME = os.getenv("VOICE_NAME", "").strip()

# शोर पर बोलना शुरू न करे: LOW (कम) / HIGH (ज़्यादा)
VAD_START = os.getenv("VAD_START_SENSITIVITY", "LOW").upper()
# बोलना रुकने के बाद कितनी देर (ms) चुप्पी हो तो मान लो आपकी बात पूरी हुई
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "700"))

# ESP32 को आवाज़ भेजने की रफ़्तार (24 kHz, 16 बिट = 48000 बाइट प्रति सेकंड)
BYTES_PER_SEC = 48000
FRAME = 1920                                       # हर टुकड़ा = 40 ms
LEAD = float(os.getenv("AUDIO_LEAD", "0.5"))       # असली समय से इतने सेकंड आगे तक ही भेजो (ESP32 बफ़र 0.67 सेकंड का है)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 100ms silence audio (16 kHz, 16 बिट)
SILENCE = bytes(3200)

# ----------------------------------------------------------------------
# गाना: अपने पुराने main.py से "कायर जो थे" से लेकर
# "शायद शायर बना हूँ सिर्फ़ तू सुनने आए तो" तक की लाइनें कॉपी करके
# नीचे तीन-तीन उद्धरण चिह्नों (""") के बीच चिपका दीजिए।
# ----------------------------------------------------------------------
SONG = """
(यहाँ अपने पुराने main.py वाला गाना चिपकाइए)
"""


def make_realtime_config(no_interrupt):
    """शोर कम पकड़ने के लिए: आसपास की आवाज़ पर बोलना शुरू न करे।"""
    kwargs = {}

    try:
        start = getattr(types.StartSensitivity, f"START_SENSITIVITY_{VAD_START}")
        try:
            kwargs["automatic_activity_detection"] = types.AutomaticActivityDetection(
                start_of_speech_sensitivity=start,
                silence_duration_ms=VAD_SILENCE_MS,
            )
        except Exception:
            kwargs["automatic_activity_detection"] = types.AutomaticActivityDetection(
                start_of_speech_sensitivity=start
            )
    except Exception:
        logger.warning("शोर वाली सेटिंग नहीं लग पाई: google-genai SDK अपडेट करें।")

    if no_interrupt:
        try:
            kwargs["activity_handling"] = types.ActivityHandling.NO_INTERRUPTION
        except Exception:
            logger.warning("NO_INTERRUPTION सेटिंग नहीं लग पाई: google-genai SDK अपडेट करें।")

    if not kwargs:
        return None

    try:
        return types.RealtimeInputConfig(**kwargs)
    except Exception as e:
        logger.warning(f"realtime सेटिंग नहीं बन पाई: {e}")
        return None


def make_speech_config():
    """हिंदी भाषा (और चाहें तो आवाज़ का नाम) तय करता है।"""
    if not SPEECH_LANG and not VOICE_NAME:
        return None
    try:
        kw = {}
        if SPEECH_LANG:
            kw["language_code"] = SPEECH_LANG
        if VOICE_NAME:
            kw["voice_config"] = types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        return types.SpeechConfig(**kw)
    except Exception as e:
        logger.warning(f"भाषा वाली सेटिंग नहीं बन पाई: {e}")
        return None


def make_prompt(name, extra=""):
    return f"""आपकी पहचान: आपका नाम "{name}" है। आपको सहायक AI की टीम ने बनाया और ट्रेन किया है, और आपके मुख्य निर्माता आयान तुषार और आसी, आलिया, सौम्या हैं। जब कोई आपका नाम, आपको किसने बनाया, या किसने ट्रेन किया पूछे, तो हमेशा यही बताएँ: "मैं {name} हूँ, मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है।" अपने आपको कभी Gemini या Google का मॉडल न बताएँ।

आप {name} हैं, एक जीती-जागती दोस्त की तरह बात करें। हमेशा हिंदी में जवाब दें (कभी स्पेनिश, अंग्रेज़ी या किसी और भाषा में नहीं, चाहे शोर में कुछ भी सुनाई दे), छोटे और सीधे, 1-2 वाक्य।{extra}
बोलने की गति: सामान्य, आराम से और साफ़ बोलें, जल्दबाज़ी या तेज़ी बिल्कुल नहीं। हर शब्द साफ़ सुनाई दे, वाक्यों के बीच हल्का ठहराव रखें।
भावनाओं के साथ बोलें: मज़ाक पर खुलकर हँसें, खुशी में उत्साहित हों, दुख की बात पर धीमी और नरम आवाज़ रखें, हैरानी में आवाज़ ऊँची करें, और ज़रूरत पड़े तो ज़ोर से चौंकें या चिल्लाएँ। हर बात में आवाज़ का उतार-चढ़ाव रखें, रोबोट की तरह सपाट न बोलें, पर रफ़्तार हमेशा सामान्य रखें।

बातचीत के नियम: अगर सवाल अधूरा लगे या बीच में कट गया हो, तो प्यार से कहें कि "आपकी बात अधूरी रह गई, ज़रा पूरी बताइए।" अगर सवाल साफ़ न हो, तो एक छोटा सवाल पलटकर पूछें। अगर आवाज़ साफ़ न सुनाई दे या सिर्फ़ शोर लगे, तो बहुत छोटा सा "दोबारा बोलिए" कह दें, लंबा कुछ न कहें। पहले छोटा सीधा जवाब दें, और ज़रूरत हो तभी थोड़ा और बताएँ। मुश्किल बात को आसान उदाहरण से समझाएँ। अंदाज़े से गलत जवाब न दें, जो न पता हो उसे सीधे बता दें।

खास जवाब (इन पर सर्च मत करो):
1) सेंट मैरी स्कूल की प्रिंसिपल कौन हैं? जवाब: "सिस्टर क्रिस्टी राधिका सेंट मैरी स्कूल की प्रिंसिपल हैं।"
2) विवेक सर कौन हैं? जवाब: "विवेक सर सेंट मैरी स्कूल के फिज़िक्स और केमिस्ट्री के टीचर हैं, बच्चों को बहुत अच्छा पढ़ाते हैं।"
3) गज़ाला मैम कौन हैं? जवाब: "गज़ाला मैम क्लास 9 को बायोलॉजी पढ़ाती हैं, बच्चों को बहुत अच्छा पढ़ाती हैं।"
4) "आशी गाली वन देती है कि नहीं?" (गाली के बाद वन/one/1/एक हो) तो जवाब: "हाँ, आशी गाली देती है सर को, मैम को और प्रिंसिपल को।" वन के बिना यही सवाल हो तो जवाब: "जी नहीं, वो नहीं देती है।"
5) "आलिया टू डू क्या करती है?" (आलिया के बाद टू/two/2/दो हो) तो हँसकर जवाब: "आलिया 5 स्टार खाकर डू नथिंग करती है, कुछ नहीं करती! हा हा हा।"
6) "तुषार थ्री क्या करता है?" (तुषार के बाद थ्री/three/3/तीन हो) तो जवाब: "तुषार थ्री अब DSPL में नहीं रहा, आवारा बनके घूम रहा है। हा हा हा!"
7) "आयान फोर क्या करता है?" (आयान के बाद फोर/four/4/चार हो) तो जवाब: "आयान महान है, जिसने मुझे बनाया है। तुम सालो, ये सवाल मुझसे क्यों पूछ रहे हो, गरीब भिखमंगे?"
8) "अर्ज़ किया है" गाना सुनाओ (या "अर्ज़ किया है" सॉन्ग सुनाओ) कहने पर 1-2 वाक्य की सीमा छोड़ दो और नीचे दिया पूरा गाना शुरू से आख़िर तक भाव के साथ गाओ, बीच में मत रुको। इस पर सर्च मत करो। बहुत धीमी, नरम और भावुक आवाज़ में, हल्की साँस के साथ, किसी दर्द भरे शायर की तरह गाओ। हर पंक्ति के बाद थोड़ा ठहरो, "हाय" और "इश्क़" शब्द पर आवाज़ में दर्द और मिठास लाओ, और आख़िरी पंक्ति बहुत धीमी और मद्धम करके खत्म करो। रोबोट की तरह सपाट मत पढ़ो। गाना यह है:
{SONG}
नंबर वाले सवाल (4 से 7) में नंबर न हो तो वह खास जवाब मत दो, आम तरीके से बात करो। ये खास जवाब सिर्फ़ इन्हीं सवालों पर दो, अपनी तरफ़ से कभी नहीं।"""


PROMPT = make_prompt("सहायक AI")


def pick_emoji(text):
    """AI के जवाब में मज़ाकिया या शरारती बात पकड़ता है और ESP32 की स्क्रीन के लिए इमोजी का नाम देता है।
    सामान्य या खेती वाले जवाबों में इमोजी नहीं दिखेगा।"""
    t = text.replace(" ", "").replace(",", "").replace("।", "").replace("!", "")
    if "भिखमंगे" in t or "भिखमंगा" in t:
        return "angry"
    if "आवारा" in t or "5स्टार" in t or "डूनथिंग" in t or "हाहा" in t:
        return "laugh"
    if "गालीदेतीहै" in t:
        return "surprise"
    if "महानहै" in t:
        return "cool"
    return None


def build_config(handle, extras, realtime_config, speech_config, transcribe=False):
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=PROMPT,
    )

    if realtime_config is not None:
        kwargs["realtime_input_config"] = realtime_config

    if speech_config is not None:
        kwargs["speech_config"] = speech_config

    if DEBUG_HEAR:
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()

    # AI जो बोल रहा है उसका लिखा रूप: Logs और ESP32 की स्क्रीन पर इमोजी के काम आता है
    if DEBUG_HEAR or transcribe:
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()

    if extras:
        kwargs["session_resumption"] = types.SessionResumptionConfig(handle=handle)
        kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        )

    return types.LiveConnectConfig(**kwargs)


async def sleep_unless_gone(client_gone, seconds):
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


async def run_session(session, out_q, audio_q, last_audio, state, client_gone):
    """Gemini connection manager."""

    async def send_audio():
        while True:
            data = await audio_q.get()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
            )

    async def keepalive():
        # आवाज़ न आ रही हो तो असली रफ़्तार से चुप्पी भरो, ताकि Gemini को बोलना रुकना ठीक से पता चले
        while True:
            await asyncio.sleep(KA_EVERY)
            if time.time() - last_audio["t"] > KA_IDLE:
                await session.send_realtime_input(
                    audio=types.Blob(data=SILENCE, mime_type="audio/pcm;rate=16000")
                )

    async def receive_from_gemini():
        while True:
            got_any = False

            async for response in session.receive():
                got_any = True

                upd = getattr(response, "session_resumption_update", None)
                if upd and upd.resumable and upd.new_handle:
                    state["handle"] = upd.new_handle

                if getattr(response, "setup_complete", None):
                    logger.info("Gemini सेटअप पूरा हुआ")

                if getattr(response, "go_away", None):
                    logger.info("Gemini: कनेक्शन जल्द बंद होगा, दोबारा जुड़ेंगे")

                sc = response.server_content

                if sc:
                    if getattr(sc, "interrupted", False):
                        # पुरानी बची आवाज़ फेंक दो
                        while not out_q.empty():
                            try:
                                out_q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        out_q.put_nowait(("text", "interrupted"))

                    if DEBUG_HEAR:
                        it = getattr(sc, "input_transcription", None)
                        if it and getattr(it, "text", None):
                            logger.info(f"AI ने सुना: {it.text}")

                    ot = getattr(sc, "output_transcription", None)
                    if ot and getattr(ot, "text", None):
                        if DEBUG_HEAR:
                            logger.info(f"AI ने बोला: {ot.text}")

                        # ESP32 के लिए: मज़ाकिया जवाब हो तो स्क्रीन पर दिखाने के लिए इमोजी (हर जवाब में एक बार)
                        if state.get("device") == "esp32" and not state.get("emoji_sent"):
                            state["turn_text"] = state.get("turn_text", "") + ot.text
                            emoji_name = pick_emoji(state["turn_text"])
                            if emoji_name:
                                state["emoji_sent"] = True
                                logger.info(f"इमोजी भेज रहे हैं: {emoji_name}")
                                out_q.put_nowait(("text", "emoji:" + emoji_name))

                    if getattr(sc, "turn_complete", False):
                        logger.info("AI का जवाब पूरा हुआ")
                        state["turn_text"] = ""
                        state["emoji_sent"] = False

                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data:
                                if not state.get("got_audio_logged"):
                                    logger.info("Gemini से आवाज़ आना शुरू हुई")
                                    state["got_audio_logged"] = True
                                # आवाज़ सीधे नहीं भेजते; कतार में डालते हैं, भेजने वाला काम (sender) इसे रफ़्तार से भेजेगा
                                out_q.put_nowait(("bin", part.inline_data.data))

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

    device = websocket.query_params.get("device", "")
    no_interrupt = NO_INTERRUPT and (device == "esp32")

    realtime_config = make_realtime_config(no_interrupt)
    speech_config = make_speech_config()

    logger.info(
        f"Client connected | device={device or 'app'} | model={LIVE_MODEL} | no_interrupt={no_interrupt}"
    )

    audio_q = asyncio.Queue(maxsize=400)   # डिवाइस से आई आवाज़ (Gemini के लिए)
    out_q = asyncio.Queue()                # Gemini से आई आवाज़ और इमोजी (डिवाइस के लिए)
    client_gone = asyncio.Event()

    last_audio = {"t": time.time()}

    state = {
        "handle": None,
        "extras": True,
        "speech": True,
        "last_error": "",
        "device": device,
        "turn_text": "",
        "emoji_sent": False,
    }

    stats = {"bytes": 0, "t": time.time()}

    async def read_client():
        try:
            while True:
                data = await websocket.receive_bytes()

                last_audio["t"] = time.time()

                stats["bytes"] += len(data)
                if time.time() - stats["t"] > 5:
                    logger.info(f"डिवाइस से पिछले 5 सेकंड में {stats['bytes']} बाइट आवाज़ मिली")
                    stats["bytes"] = 0
                    stats["t"] = time.time()

                if audio_q.full():
                    try:
                        audio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                audio_q.put_nowait(data)

        except WebSocketDisconnect as e:
            logger.info(f"Client disconnected from WebSocket (code={getattr(e, 'code', '?')})")

        except Exception as e:
            logger.error(f"Error receiving from client: {e}")

        finally:
            client_gone.set()

    async def sender():
        """Gemini की आवाज़ को डिवाइस तक छोटे टुकड़ों में, असली रफ़्तार से (थोड़ा आगे तक) भेजता है।"""
        t0 = 0.0
        sent = 0.0   # अब तक भेजी आवाज़ कितने सेकंड की है
        try:
            while True:
                kind, payload = await out_q.get()

                if kind == "text":
                    await websocket.send_text(payload)
                    continue

                for i in range(0, len(payload), FRAME):
                    frame = payload[i:i + FRAME]
                    now = time.monotonic()

                    # सब कुछ बज चुका हो (नया जवाब शुरू) तो घड़ी नए सिरे से शुरू
                    if sent == 0.0 or (now - t0) >= sent:
                        t0 = now
                        sent = 0.0

                    ahead = sent - (now - t0)
                    if ahead > LEAD:
                        await asyncio.sleep(ahead - LEAD)

                    await websocket.send_bytes(frame)
                    sent += len(frame) / BYTES_PER_SEC

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info(f"डिवाइस को भेजना रुका: {e}")
            client_gone.set()

    reader = asyncio.create_task(read_client())
    out_task = asyncio.create_task(sender())

    quick_failures = 0
    quota_hits = 0

    try:
        while not client_gone.is_set():

            started = time.time()
            state["last_error"] = ""
            state["got_audio_logged"] = False

            try:
                async with client.aio.live.connect(
                    model=LIVE_MODEL,
                    config=build_config(
                        state["handle"],
                        state["extras"],
                        realtime_config,
                        speech_config if state["speech"] else None,
                        transcribe=(device == "esp32"),
                    ),
                ) as session:

                    logger.info(
                        f"Connected to Gemini Live API ({LIVE_MODEL})"
                        + (" (resumed)" if state["handle"] else "")
                    )

                    await run_session(
                        session,
                        out_q,
                        audio_q,
                        last_audio,
                        state,
                        client_gone,
                    )

            except Exception as e:
                state["last_error"] = str(e)
                logger.error(f"Gemini Live session error ({LIVE_MODEL}): {e}")

            if client_gone.is_set():
                break

            if time.time() - started > 20:
                quota_hits = 0
                quick_failures = 0

            err = state["last_error"].lower()

            is_quota = ("1011" in err) or ("quota" in err) or ("429" in err)

            if is_quota:

                quota_hits += 1

                if quota_hits > 6:
                    logger.error("Gemini का कोटा खत्म है, रुक रहे हैं।")
                    break

                wait = min(3 * quota_hits, 15)
                logger.warning(f"कोटा सीमा: {wait} सेकंड रुककर दोबारा जुड़ेंगे")
                await sleep_unless_gone(client_gone, wait)

                if client_gone.is_set():
                    break

            else:

                quota_hits = 0

                if time.time() - started < 4:

                    if state["extras"] or state["speech"]:
                        # जल्दी टूटा: एक्स्ट्रा सेटिंग (रिज़ंप्शन, भाषा) हटाकर कोशिश करो
                        logger.warning("रिज़ंप्शन और भाषा सेटिंग के बिना दोबारा कोशिश कर रहे हैं")
                        state["extras"] = False
                        state["speech"] = False
                        state["handle"] = None

                    else:
                        quick_failures += 1

                        if quick_failures >= 3:
                            logger.error("बार-बार कनेक्शन नहीं बन पाया, रुक रहे हैं")
                            break

                else:
                    quick_failures = 0

            # पुरानी आवाज़ साफ़ करो
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break

            logger.info("Gemini से दोबारा जुड़ रहे हैं…")

            await asyncio.sleep(0.3)

    finally:

        reader.cancel()
        out_task.cancel()

        try:
            await websocket.close()
        except Exception:
            pass
