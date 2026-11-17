import asyncio
import os
import time
import logging
import httpx
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

# सामान्य मॉडल + सर्च पर भी कोटा एरर आए, तो इतने सेकंड तक Pro बिना सर्च के बोलेगा
# इसके बाद Pro फिर से सर्च के साथ कोशिश करेगा, यानी सर्च हमेशा के लिए बंद नहीं होता
SEARCH_RETRY_AFTER = int(os.getenv("SEARCH_RETRY_AFTER", "120"))
SEARCH_STATE = {"blocked_until": 0.0}

# Pro में दो Gemini कनेक्शन कोशिशों के बीच कम से कम इतने सेकंड का फ़र्क (RPM कम रखने के लिए)
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

# ======================================================================
# [नया] ElevenLabs की आवाज़ (सिर्फ़ Pro मोड के लिए)
# Render में ये दो ज़रूरी हैं: ELEVEN_API_KEY और ELEVEN_VOICE_ID
# इनमें से कोई खाली हो तो Pro पहले जैसा Gemini की अपनी आवाज़ में चलेगा
# ======================================================================
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "").strip()

# बोलने वाला मॉडल (वैकल्पिक). यह हिंदी समेत बहुत सी भाषाएँ बोलता है
ELEVEN_MODEL = os.getenv("ELEVEN_MODEL", "eleven_multilingual_v2").strip()

# आवाज़ में भाव: stability जितनी कम, उतना ज़्यादा उतार-चढ़ाव. style जितनी ज़्यादा, उतना ज़्यादा अंदाज़
ELEVEN_STABILITY = float(os.getenv("ELEVEN_STABILITY", "0.35"))
ELEVEN_STYLE = float(os.getenv("ELEVEN_STYLE", "0.5"))

# ElevenLabs में एरर आए (जैसे क्रेडिट खत्म), तो इतने सेकंड तक Pro Gemini की आवाज़ में बोलेगा
ELEVEN_RETRY_AFTER = int(os.getenv("ELEVEN_RETRY_AFTER", "300"))
ELEVEN_STATE = {"blocked_until": 0.0}

# 1 = यूज़र बीच में बोले तो ElevenLabs की आवाज़ रुक जाए. गड़बड़ करे तो Render में 0 कर दें
ELEVEN_BARGE_IN = os.getenv("ELEVEN_BARGE_IN", "1") == "1"

# आवाज़ 24kHz, 16-bit PCM में आती है (Gemini की आवाज़ जैसी), इसलिए ऐप में कुछ बदलना नहीं पड़ता
ELEVEN_BYTES_PER_SEC = 48000

# बोलने के लिए वाक्य कम से कम इतने अक्षर का होने पर भेजेंगे
MIN_CHARS = 20
SENT_END = "।.?!\n"

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
    """शोर कम पकड़ने के लिए: आसपास की आवाज़ पर बोलना शुरू न करे।
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
        # [नया] आवाज़ पुरुष की है, इसलिए बोलने का तरीका भी पुल्लिंग में
        " तुम्हारी आवाज़ एक पुरुष की है, इसलिए हमेशा पुल्लिंग में बोलो, जैसे 'मैं बताता हूँ', 'मैं कर सकता हूँ'।"
        # [नया] भाव शब्दों में भी ज़ाहिर हों, ताकि ElevenLabs की आवाज़ में भी भावना आए
        " भावना शब्दों में भी ज़ाहिर करो, जैसे 'अरे वाह!', 'हा हा हा', 'ओह...', 'क्या बात है!', ताकि आवाज़ में असली भाव आए।"
        " तुम्हारे पास Google Search का टूल है और तुम्हें इंटरनेट से ताज़ा जानकारी मिल सकती है।"
        " जब कोई आज, अभी, ताज़ा या हाल की चीज़ पूछे, जैसे खबर, कौन किस पद पर है, कीमत, स्कोर, मौसम या मीम, तो अपनी याददाश्त पर भरोसा मत करो, पहले सर्च करो, फिर सर्च के नतीजे के आधार पर छोटा सीधा जवाब दो।"
        " पर आम ज्ञान, गणित, बातचीत, मज़ाक या सलाह के सवाल पर सर्च बिल्कुल मत करो, सीधे जवाब दो।"
        " एक सवाल पर ज़्यादा से ज़्यादा एक ही बार सर्च करो, बार-बार नहीं।"
        " कभी 'सिस्टम एरर' मत बोलो। अगर कुछ न कर सको तो छोटे में सीधा बताओ।",
    ),
}


def build_config(handle, extras, prompt, thinking=False, search=False, transcribe=False):
    """thinking=True: सिर्फ़ Extended Thinking मॉडल के लिए सोचने का स्तर।
    search=True: Google Search टूल (सिर्फ़ Pro मोड में)।
    transcribe=True: [नया] Gemini जो बोले उसका लिखा हुआ रूप भी मिले (ElevenLabs को बोलने के लिए)।"""

    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=prompt,
    )

    # शोर कम पकड़ने की सेटिंग (दोनों मोड में)
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

    # [नया] ElevenLabs वाले Pro के लिए: Gemini के जवाब का टेक्स्ट चाहिए
    if transcribe:
        kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()

        # यूज़र बीच में बोले तो पता चले (ताकि पुरानी आवाज़ रोकी जा सके)
        if ELEVEN_BARGE_IN:
            kwargs["input_audio_transcription"] = types.AudioTranscriptionConfig()

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


# ======================================================================
# [नया] ElevenLabs बोलने वाला हिस्सा
# ======================================================================
def eleven_ready():
    """ElevenLabs अभी इस्तेमाल हो सकता है या नहीं (चाबी, आवाज़ और कोई हालिया एरर)।"""
    return bool(
        ELEVEN_API_KEY
        and ELEVEN_VOICE_ID
        and time.time() >= ELEVEN_STATE["blocked_until"]
    )


def last_boundary(buf):
    """वाक्य खत्म होने की आख़िरी जगह। 3.5 जैसी दशमलव संख्या को बीच में नहीं काटता।"""
    for i in range(len(buf) - 1, -1, -1):
        ch = buf[i]

        if ch not in SENT_END:
            continue

        if ch == ".":
            prev_digit = i > 0 and buf[i - 1].isdigit()
            next_digit = i + 1 < len(buf) and buf[i + 1].isdigit()

            # अंक.अंक, या अंक के बाद आख़िर में "." (आगे अंक आ सकते हैं): यहाँ मत काटो
            if prev_digit and (next_digit or i == len(buf) - 1):
                continue

        return i

    return -1


def take_ready(buf, force=False):
    """बफ़र से बोलने लायक हिस्सा निकालता है। (बोलने वाला टेक्स्ट, बचा हुआ टेक्स्ट)"""
    if force:
        return buf.strip(), ""

    cut = last_boundary(buf)

    if cut >= 0 and len(buf[:cut + 1].strip()) >= MIN_CHARS:
        return buf[:cut + 1].strip(), buf[cut + 1:]

    # बहुत लंबा वाक्य बिना विराम के आए, तो किसी खाली जगह पर काट दो
    if len(buf) > 150:
        sp = buf.rfind(" ", 0, 150)

        if sp > 0:
            return buf[:sp].strip(), buf[sp + 1:]

    return "", buf


class ElevenSpeaker:
    """Gemini के जवाब का टेक्स्ट लेकर ElevenLabs की आवाज़ में ऐप तक भेजता है।
    ElevenLabs में एरर आए (जैसे क्रेडिट खत्म) तो उस जवाब में Gemini की अपनी आवाज़ चला देता है।"""

    def __init__(self, websocket, client_gone):
        self.ws = websocket
        self.client_gone = client_gone

        self.q = asyncio.Queue()

        # यह संख्या बढ़ाने पर चल रही और कतार वाली पुरानी आवाज़ रुक जाती है
        self.generation = 0

        self.turn_id = 0
        self.in_turn = False
        self.text_buf = ""
        self.prev_text = ""

        # इस जवाब में Gemini की अपनी आवाज़ चलानी है या नहीं
        self.turn_gemini_voice = False
        self.turn_audio = []
        self.sent_audio = False

        self.speaking = False
        self.play_until = 0.0
        self.last_model_event = 0.0

    # ---------- जवाब की शुरुआत ----------
    def _ensure_turn(self):
        self.last_model_event = time.time()

        if self.in_turn:
            return

        self.in_turn = True
        self.turn_id += 1
        self.text_buf = ""
        self.prev_text = ""
        self.turn_audio = []
        self.sent_audio = False
        self.turn_gemini_voice = not eleven_ready()

    # ---------- Gemini से आने वाली चीज़ें ----------
    async def on_gemini_audio(self, data):
        self._ensure_turn()

        if self.turn_gemini_voice:
            await self.ws.send_bytes(data)
        else:
            # ElevenLabs चल रहा है, तो Gemini की आवाज़ रोककर रखो (एरर आए तभी काम आएगी)
            self.turn_audio.append(data)

    def on_text(self, text):
        self._ensure_turn()

        if self.turn_gemini_voice:
            return

        self.text_buf += text
        self._push_ready(force=False)

    def end_turn(self):
        self.last_model_event = time.time()

        if self.in_turn and not self.turn_gemini_voice:
            self._push_ready(force=True)

        self.in_turn = False

    def _push_ready(self, force):
        ready, rest = take_ready(self.text_buf, force)
        self.text_buf = rest

        if ready:
            self.q.put_nowait((self.generation, self.turn_id, ready))

    # ---------- रोकना ----------
    def _drain(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def interrupt(self):
        """आवाज़ तुरंत रोको और ऐप को भी बताओ कि बजना बंद करे।"""
        self.generation += 1
        self._drain()

        self.play_until = 0.0
        self.text_buf = ""
        self.turn_audio = []
        self.in_turn = False

        try:
            await self.ws.send_text("interrupted")
        except Exception:
            pass

    def busy(self):
        return (
            self.speaking
            or (not self.q.empty())
            or time.time() < self.play_until
        )

    async def on_user_speech(self):
        """यूज़र पुरानी आवाज़ के बीच में बोलने लगा।"""
        if not ELEVEN_BARGE_IN:
            return

        # Gemini जवाब दे रहा है: वहाँ Gemini का अपना interrupted संकेत काम करता है
        if self.in_turn:
            return

        # अभी-अभी जवाब खत्म हुआ है: यह उसी बात का देर से आया टुकड़ा हो सकता है
        if time.time() - self.last_model_event < 1.5:
            return

        if self.busy():
            logger.info("यूज़र बीच में बोला: ElevenLabs की आवाज़ रोक रहे हैं")
            await self.interrupt()

    # ---------- ElevenLabs से बुलवाना ----------
    async def fail(self, gen, turn_id, err):
        logger.error(f"ElevenLabs एरर: {err}")

        # कुछ देर ElevenLabs बंद, Pro Gemini की आवाज़ में चलेगा
        ELEVEN_STATE["blocked_until"] = time.time() + ELEVEN_RETRY_AFTER

        if self.client_gone.is_set():
            return

        if gen != self.generation or turn_id != self.turn_id:
            return

        # इस जवाब में ElevenLabs से कुछ नहीं बजा था, तो Gemini की आवाज़ यहीं से चला दो
        if self.sent_audio or self.turn_gemini_voice:
            return

        logger.warning("इस जवाब में Gemini की अपनी आवाज़ चला रहे हैं")

        self.turn_gemini_voice = True
        self.generation += 1
        self._drain()

        saved = self.turn_audio
        self.turn_audio = []

        for chunk in saved:
            await self.ws.send_bytes(chunk)

    async def speak(self, http, gen, turn_id, text):
        body = {
            "text": text,
            "model_id": ELEVEN_MODEL,
            "voice_settings": {
                "stability": ELEVEN_STABILITY,
                "similarity_boost": 0.8,
                "style": ELEVEN_STYLE,
                "use_speaker_boost": True,
            },
        }

        # पिछला वाक्य बताने से आवाज़ का लहजा जुड़ा हुआ लगता है
        if self.prev_text and "v3" not in ELEVEN_MODEL:
            body["previous_text"] = self.prev_text[-300:]

        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{ELEVEN_VOICE_ID}/stream"
        )

        self.speaking = True

        try:
            async with http.stream(
                "POST",
                url,
                params={"output_format": "pcm_24000"},
                headers={
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json",
                },
                json=body,
            ) as r:

                if r.status_code != 200:
                    detail = (await r.aread()).decode("utf-8", "ignore")[:300]
                    raise RuntimeError(f"HTTP {r.status_code}: {detail}")

                carry = b""

                async for chunk in r.aiter_bytes():
                    if gen != self.generation:
                        return

                    carry += chunk

                    # 16-bit आवाज़ है, इसलिए हमेशा जोड़ी (2 बाइट) में ही भेजो
                    n = len(carry) - (len(carry) % 2)

                    if n:
                        out, carry = carry[:n], carry[n:]

                        await self.ws.send_bytes(out)

                        if turn_id == self.turn_id:
                            self.sent_audio = True

                        now = time.time()
                        self.play_until = (
                            max(self.play_until, now)
                            + len(out) / ELEVEN_BYTES_PER_SEC
                        )

            if gen == self.generation:
                self.prev_text = text

        finally:
            self.speaking = False

    async def worker(self):
        timeout = httpx.Timeout(20.0, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as http:
            while True:
                gen, turn_id, text = await self.q.get()

                if gen != self.generation:
                    continue

                try:
                    await self.speak(http, gen, turn_id, text)

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    if self.client_gone.is_set():
                        return

                    await self.fail(gen, turn_id, str(e))


async def throttle_pro_connect():
    """Pro की कनेक्शन कोशिशों के बीच गैप रखता है, ताकि RPM एक साथ न भर जाए।"""
    if PRO_CONNECT_GAP <= 0:
        return

    async with CONNECT_LOCK:
        wait = CONNECT_LAST["pro"] + PRO_CONNECT_GAP - time.time()

        if wait > 0:
            await asyncio.sleep(wait)

        CONNECT_LAST["pro"] = time.time()


async def sleep_unless_gone(client_gone, seconds):
    """इंतज़ार करता है, पर क्लाइंट चला जाए तो तुरंत रुक जाता है।
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
                      ka_every=2, ka_idle=3, ka_reset=False, speaker=None):
    """Gemini connection manager.
    speaker: [नया] सिर्फ़ ElevenLabs वाले Pro में होता है। normal मोड में None रहता है,
    तब सब कुछ बिल्कुल पहले जैसा चलता है।"""

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
                        if speaker is not None:
                            await speaker.interrupt()
                        else:
                            await websocket.send_text("interrupted")

                    # [नया] यूज़र बोलना शुरू करे तो पुरानी ElevenLabs आवाज़ रोको
                    if speaker is not None:
                        it = getattr(sc, "input_transcription", None)

                        if it and getattr(it, "text", None):
                            await speaker.on_user_speech()

                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data:
                                if speaker is None:
                                    await websocket.send_bytes(
                                        part.inline_data.data
                                    )
                                else:
                                    await speaker.on_gemini_audio(
                                        part.inline_data.data
                                    )

                    # [नया] Gemini जो बोल रहा है उसका टेक्स्ट ElevenLabs को दो
                    if speaker is not None:
                        ot = getattr(sc, "output_transcription", None)

                        if ot and getattr(ot, "text", None):
                            speaker.on_text(ot.text)

                        if getattr(sc, "turn_complete", False):
                            speaker.end_turn()

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

    # पिछली बार सर्च वाले मॉडल पर भी कोटा नहीं मिला था, तो अभी सर्च के बिना चलो
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

    # [नया] Pro मोड में ElevenLabs की आवाज़ (normal मोड में कुछ नहीं बदलता)
    speaker = None
    speaker_task = None

    if mode == "pro":
        if ELEVEN_API_KEY and ELEVEN_VOICE_ID:
            speaker = ElevenSpeaker(websocket, client_gone)
            speaker_task = asyncio.create_task(speaker.worker())
            logger.info("Pro: ElevenLabs की आवाज़ चालू है")
        else:
            logger.warning(
                "Pro: ELEVEN_API_KEY या ELEVEN_VOICE_ID नहीं मिला, Gemini की आवाज़ में चलेगा"
            )

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

            # Pro में कनेक्शन कोशिशों के बीच गैप
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
                        transcribe=(speaker is not None),
                    ),
                ) as session:

                    logger.info(
                        f"Connected to Gemini Live API "
                        f"({mode} | {model_name})"
                        + (" + search" if use_search else "")
                        + (" + ElevenLabs" if speaker is not None else "")
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
                        speaker,
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

            # स्टेप 2: सामान्य मॉडल + सर्च पर भी जुड़ते ही कोटा एरर आए,
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

                # इंतज़ार के बीच क्लाइंट चला जाए तो तुरंत रुको
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

        # [नया] ElevenLabs वाला काम भी बंद करो
        if speaker_task is not None:
            speaker_task.cancel()

        try:
            await websocket.close()
        except Exception:
            pass
