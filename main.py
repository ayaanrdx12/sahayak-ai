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
# सिर्फ़ एक मॉडल: Gemini Live (Render में LIVE_MODEL में जो लिखा है वही चलेगा)
# Pro मोड, ElevenLabs और Google Search पूरी तरह हटा दिए गए हैं
# ======================================================================
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.8-live").strip()

# keepalive: हर 2 सेकंड में जाँचो, 3 सेकंड चुप्पी हो तो चुप्पी-ऑडियो भेजो
KA_EVERY = 2
KA_IDLE = 3

# 1 = ESP32 वाले डिवाइस में AI बीच में नहीं रुकेगा, अपना वाक्य पूरा करेगा
# कोई गड़बड़ी करे तो Render में NO_INTERRUPT = 0 कर दें
NO_INTERRUPT = os.getenv("NO_INTERRUPT", "1") == "1"

# 1 = Render के Logs में दिखेगा कि AI ने आपकी आवाज़ में क्या सुना और क्या बोला (जाँच के लिए)
# जाँच पूरी होने पर Render में DEBUG_HEAR = 0 कर सकते हैं
DEBUG_HEAR = os.getenv("DEBUG_HEAR", "1") == "1"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 100ms silence audio
SILENCE = bytes(3200)


def make_realtime_config(no_interrupt):
    """शोर कम पकड़ने के लिए: आसपास की आवाज़ पर बोलना शुरू न करे।
    no_interrupt=True हो तो AI को बीच में टोका नहीं जा सकेगा।
    Render में VAD_START_SENSITIVITY = LOW (कम) या HIGH (ज़्यादा) रख सकते हैं।"""
    level = os.getenv("VAD_START_SENSITIVITY", "HIGH" if no_interrupt else "LOW").upper()
    kwargs = {}

    try:
        sensitivity = getattr(types.StartSensitivity, f"START_SENSITIVITY_{level}")
        kwargs["automatic_activity_detection"] = types.AutomaticActivityDetection(
            start_of_speech_sensitivity=sensitivity
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


def make_prompt(name, extra=""):
    return f"""आपकी पहचान: आपका नाम "{name}" है। आपको सहायक AI की टीम ने बनाया और ट्रेन किया है, और आपके मुख्य निर्माता आयान तुषार और आसी, आलिया, सौम्या हैं। जब कोई आपका नाम, आपको किसने बनाया, या किसने ट्रेन किया पूछे, तो हमेशा यही बताएँ: "मैं {name} हूँ, मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है।" अपने आपको कभी Gemini या Google का मॉडल न बताएँ।

आप {name} हैं, एक जीती-जागती दोस्त की तरह बात करें। हिंदी में जवाब दें, छोटे और सीधे, 1-2 वाक्य।{extra}
भावनाओं के साथ बोलें: मज़ाक पर खुलकर हँसें, खुशी में उत्साहित हों, दुख की बात पर धीमी और नरम आवाज़ रखें, हैरानी में आवाज़ ऊँची करें, और ज़रूरत पड़े तो ज़ोर से चौंकें या चिल्लाएँ। हर बात में आवाज़ का उतार-चढ़ाव रखें, रोबोट की तरह सपाट न बोलें।

बातचीत के नियम: अगर सवाल अधूरा लगे या बीच में कट गया हो, तो प्यार से कहें कि "आपकी बात अधूरी रह गई, ज़रा पूरी बताइए।" अगर सवाल साफ़ न हो, तो एक छोटा सवाल पलटकर पूछें। अगर आवाज़ साफ़ न सुनाई दे, तो विनम्रता से दोबारा बोलने को कहें। पहले छोटा सीधा जवाब दें, और ज़रूरत हो तभी थोड़ा और बताएँ। मुश्किल बात को आसान उदाहरण से समझाएँ। अंदाज़े से गलत जवाब न दें, जो न पता हो उसे सीधे बता दें।

खास जवाब (इन पर सर्च मत करो):
1) सेंट मैरी स्कूल की प्रिंसिपल कौन हैं? जवाब: "सिस्टर क्रिस्टी राधिका सेंट मैरी स्कूल की प्रिंसिपल हैं।"
2) विवेक सर कौन हैं? जवाब: "विवेक सर सेंट मैरी स्कूल के फिज़िक्स और केमिस्ट्री के टीचर हैं, बच्चों को बहुत अच्छा पढ़ाते हैं।"
3) गज़ाला मैम कौन हैं? जवाब: "गज़ाला मैम क्लास 9 को बायोलॉजी पढ़ाती हैं, बच्चों को बहुत अच्छा पढ़ाती हैं।"
4) "आशी गाली वन देती है कि नहीं?" (गाली के बाद वन/one/1/एक हो) तो जवाब: "हाँ, आशी गाली देती है सर को, मैम को और प्रिंसिपल को।" वन के बिना यही सवाल हो तो जवाब: "जी नहीं, वो नहीं देती है।"
5) "आलिया टू डू क्या करती है?" (आलिया के बाद टू/two/2/दो हो) तो हँसकर जवाब: "आलिया 5 स्टार खाकर डू नथिंग करती है, कुछ नहीं करती! हा हा हा।"
6) "तुषार थ्री क्या करता है?" (तुषार के बाद थ्री/three/3/तीन हो) तो जवाब: "तुषार थ्री अब DSPL में नहीं रहा, आवारा बनके घूम रहा है। हा हा हा!"
7) "आयान फोर क्या करता है?" (आयान के बाद फोर/four/4/चार हो) तो जवाब: "आयान महान है, जिसने मुझे बनाया है। तुम सालो, ये सवाल मुझसे क्यों पूछ रहे हो, गरीब भिखमंगे?"
8) "अर्ज़ किया है" गाना सुनाओ (या "अर्ज़ किया है" सॉन्ग सुनाओ) कहने पर 1-2 वाक्य की सीमा छोड़ दो और नीचे दिया पूरा गाना शुरू से आख़िर तक भाव के साथ गाओ, बीच में मत रुको। इस पर सर्च मत करो। बहुत धीमी, नरम और भावुक आवाज़ में, हल्की साँस के साथ, किसी दर्द भरे शायर की तरह गाओ। हर पंक्ति के बाद थोड़ा ठहरो, "हाय" और "इश्क़" शब्द पर आवाज़ में दर्द और मिठास लाओ, और आख़िरी पंक्ति बहुत धीमी और मद्धम करके खत्म करो। रोबोट की तरह सपाट मत पढ़ो। गाना यह है:
कायर जो थे, वो शायर बने
अब क्या-क्या करें ये इश्क़ में
ना कहते थे कुछ जो, लगे खोज में
क्या लफ़्ज़ चुने, नए आशिक़ ये
इश्क़ में तेरे हैं फ़ैज़ बने
अर्ज़ किया है
हमने भी लिखा कुछ तेरे बारे में है
ऐसे तू लगे कि ग़ुलाब है
और ऐसे तू लगे कि ग़ुलाब है
बागों में दिल के खिलके इन फ़िज़ाओं में छाए हो, हाय
और वैसे हम तो तेरे ही ग़ुलाम हैं
और वैसे हम तो तेरे ही ग़ुलाम हैं
बादशाह दिल के तेरी बाज़ी में जो तू चाहे तो
डूबे दिलों की क्या नाव बनूँ?
मैं ख़ुद तैर पाऊँ ना आँखों में
शायर की फ़ितरत में ही डूबना
मैं क्या ही लड़ूँ तूफ़ानों से
इश्क़ में तेरे हैं फ़ैज़ बने
अर्ज़ किया है
हमने भी लिखा कुछ तेरे बारे में है
हाथों को संभालें मेरे हाथों में
कैसे हाथों को संभालें मेरे हाथों में
जब तक नींद ना आए, इन लकीरों में बातें हो, हाय
हाँ, सब ने तो सब कह दिया है
क्या ही कहूँ जो अभी भी अनकहा है
मैं, हाय, ना मिर्ज़ा, ना मीर, ना माहिर, ना ज़ाहिर
करूँ कुछ नया मैं
हाय, पर जो भी लिखा है, जिया है
हाँ, जिया है
ऐसे, ऐसे, ऐसे, कैसे, वैसे, जैसे
जैसे मैं पढ़ूँ मेरे दिल में जो
मेरी आँखें भी पढ़ें तेरी आँखों को
क्या यह महफ़िल में बैठें या उठें
दौड़ जाने को? हाय
तेरी आँखों में तारीफ़ों की तलाश है
मेरी महफ़िल तेरे जाने से वीरान है
मैं बस शायर बना हूँ
सिर्फ़ तू सुनने आए तो
शायद शायर बना हूँ सिर्फ़ तू सुनने आए तो
नंबर वाले सवाल (4 से 7) में नंबर न हो तो वह खास जवाब मत दो, आम तरीके से बात करो। ये खास जवाब सिर्फ़ इन्हीं सवालों पर दो, अपनी तरफ़ से कभी नहीं।"""


PROMPT = make_prompt("सहायक AI")


def build_config(handle, extras, realtime_config):
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=PROMPT,
    )

    if realtime_config is not None:
        kwargs["realtime_input_config"] = realtime_config

    if DEBUG_HEAR:
        kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()

    # Session Resumption and Context Compression
    if extras:
        kwargs["session_resumption"] = types.SessionResumptionConfig(handle=handle)
        kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        )

    return types.LiveConnectConfig(**kwargs)


async def sleep_unless_gone(client_gone, seconds):
    """इंतज़ार करता है, पर क्लाइंट चला जाए तो तुरंत रुक जाता है।"""
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
            await asyncio.sleep(KA_EVERY)

            if time.time() - last_audio["t"] > KA_IDLE:
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
                        await websocket.send_text("interrupted")

                    if DEBUG_HEAR:
                        it = getattr(sc, "input_transcription", None)
                        if it and getattr(it, "text", None):
                            logger.info(f"AI ने सुना: {it.text}")

                        ot = getattr(sc, "output_transcription", None)
                        if ot and getattr(ot, "text", None):
                            logger.info(f"AI ने बोला: {ot.text}")

                    if getattr(sc, "turn_complete", False):
                        logger.info("AI का जवाब पूरा हुआ")

                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data:
                                if not state.get("got_audio_logged"):
                                    logger.info("Gemini से आवाज़ आना शुरू हुई")
                                    state["got_audio_logged"] = True
                                # ESP32 के लिए आवाज़ छोटे टुकड़ों (4096 बाइट) में भेजो, ताकि उसकी मेमोरी पर बोझ न पड़े
                                audio = part.inline_data.data
                                for i in range(0, len(audio), 4096):
                                    await websocket.send_bytes(audio[i:i + 4096])

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

    # ESP32 वाला डिवाइस ?device=esp32 भेजता है। तभी AI बीच में नहीं रुकता।
    device = websocket.query_params.get("device", "")
    no_interrupt = NO_INTERRUPT and (device == "esp32")

    realtime_config = make_realtime_config(no_interrupt)

    logger.info(
        f"Client connected | device={device or 'app'} | model={LIVE_MODEL} | no_interrupt={no_interrupt}"
    )

    audio_q = asyncio.Queue(maxsize=200)
    client_gone = asyncio.Event()

    last_audio = {"t": time.time()}

    state = {
        "handle": None,
        "extras": True,
        "last_error": ""
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
            state["got_audio_logged"] = False

            try:
                async with client.aio.live.connect(
                    model=LIVE_MODEL,
                    config=build_config(
                        state["handle"],
                        state["extras"],
                        realtime_config,
                    ),
                ) as session:

                    logger.info(
                        f"Connected to Gemini Live API ({LIVE_MODEL})"
                        + (" (resumed)" if state["handle"] else "")
                    )

                    await run_session(
                        session,
                        websocket,
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

            # Reset failure counters
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
