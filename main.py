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
# सहायक AI सर्वर – वर्ज़न 13 (ESP32 वर्ज़न 14 के साथ चलाएँ)
# नया क्या है (v12 से):
#  1) खेत के तीन सेंसर (मिट्टी की नमी, हवा की नमी, हवा की गैस) + तापमान:
#     ESP32 हर 5 सेकंड में असली रीडिंग भेजता है। AI उसे तभी देखता है जब किसान
#     खेत/पानी/नमी/हवा के बारे में पूछे (get_field_sensors फ़ंक्शन से)।
#     AI एक ही बार बताता है, बार-बार नहीं दोहराता।
#  2) ख़राब या न चलता हुआ सेंसर AI चुपचाप छोड़ देता है और बाकी सेंसरों से जवाब देता है।
#  3) सर्वर अब ESP32 का लिखा संदेश भी पढ़ लेता है (पहले सिर्फ़ आवाज़ पढ़ता था)।
#  4) आवाज़ टूटना कम करने के लिए AUDIO_LEAD 0.45 से 0.6 किया।
#     (Render में अगर AUDIO_LEAD पहले से लिखा हो तो उसे वहाँ से हटा दें।)
#  बाकी सब (इमोजी, चुप्पी भरना, VAD, खास जवाब, भाषा) पहले जैसा ही है।
# ======================================================================
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.8-live").strip()

# चुप्पी भरना: आवाज़ 0.25 सेकंड न आए तो 100 ms चुप्पी भेजो (असली रफ़्तार से)
KA_EVERY = 0.1
KA_IDLE = 0.25

# 1 = ESP32 वाले डिवाइस में AI बीच में नहीं रुकेगा, अपना वाक्य पूरा करेगा
NO_INTERRUPT = os.getenv("NO_INTERRUPT", "1") == "1"

# 1 = Render के Logs में दिखेगा कि AI ने क्या सुना और क्या बोला
DEBUG_HEAR = os.getenv("DEBUG_HEAR", "1") == "1"

# भाषा: अब खाली है = कोई भी भाषा चलेगी (AI उसी भाषा में बोलेगा जिसमें आप बोलें)।
# अगर Render में SPEECH_LANG=hi-IN पहले से लिखा है तो उसे वहाँ से हटा दें, वरना हिंदी ही अटकी रहेगी।
SPEECH_LANG = os.getenv("SPEECH_LANG", "").strip()
# आवाज़ का नाम (खाली छोड़ेंगे तो Gemini की अपनी आवाज़)
VOICE_NAME = os.getenv("VOICE_NAME", "").strip()

# शोर पर बोलना शुरू न करे: LOW (कम) / HIGH (ज़्यादा)
VAD_START = os.getenv("VAD_START_SENSITIVITY", "LOW").upper()
# बोलना रुकने के बाद कितनी देर (ms) चुप्पी हो तो मान लो आपकी बात पूरी हुई
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "700"))

# ESP32 को आवाज़ भेजने की रफ़्तार (24 kHz, 16 बिट = 48000 बाइट प्रति सेकंड)
BYTES_PER_SEC = 48000
FRAME = 1920                                       # हर टुकड़ा = 40 ms
LEAD = float(os.getenv("AUDIO_LEAD", "0.6"))       # असली समय से इतने सेकंड आगे तक ही भेजो (ESP32 बफ़र 0.93 सेकंड का है)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 100ms silence audio (16 kHz, 16 बिट)
SILENCE = bytes(3200)

# ----------------------------------------------------------------------
# खेत के सेंसर (ESP32 हर 5 सेकंड में भेजता है)
# ----------------------------------------------------------------------
SENSOR_STALE_SEC = 30   # इतने सेकंड से नया डेटा न आए तो डेटा पुराना मानो

SENSOR_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_field_sensors",
        description=(
            "Get the latest real readings from the farm sensors: soil moisture, "
            "air humidity, air temperature and air gas level. Call this only when the "
            "farmer asks about the field, soil, water, irrigation, moisture or air."
        ),
    )
])


def parse_sensor_text(text):
    """ESP32 से आया 'sensor:soil=45,hum=60,temp=31,gas=x' पढ़ता है। x या गलत रीडिंग = None"""
    if not text.startswith("sensor:"):
        return None
    limits = {"soil": (0, 100), "hum": (0, 100), "temp": (-10, 60), "gas": (0, 100)}
    out = {}
    for part in text[7:].split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in limits:
            continue
        try:
            val = float(v)
        except ValueError:
            val = None
        if val is not None and not (limits[k][0] <= val <= limits[k][1]):
            val = None
        out[k] = val
    for k in limits:
        out.setdefault(k, None)
    out["t"] = time.time()
    return out


def sensor_report(state):
    """AI जब सेंसर माँगे तो यह जवाब जाता है। ख़राब सेंसर इसमें नहीं होते।"""
    s = state.get("sensors")
    if not s or time.time() - s["t"] > SENSOR_STALE_SEC:
        return {
            "status": "no_data",
            "note": "No fresh sensor data right now. Tell the farmer the sensor data is not coming at the moment.",
        }
    result = {"status": "ok"}
    bad = []
    if s["soil"] is not None:
        result["soil_moisture_percent"] = round(s["soil"])
    else:
        bad.append("soil_moisture")
    if s["hum"] is not None:
        result["air_humidity_percent"] = round(s["hum"])
    else:
        bad.append("air_humidity")
    if s["temp"] is not None:
        result["air_temperature_celsius"] = round(s["temp"])
    else:
        bad.append("air_temperature")
    if s["gas"] is not None:
        result["air_gas_level_percent"] = round(s["gas"])   # ज़्यादा = हवा ख़राब (अंदाज़ी)
    else:
        bad.append("air_quality")
    result["not_working_sensors"] = bad
    result["note"] = (
        "Answer only what the farmer asked, once, using only these values. "
        "Silently ignore every sensor listed in not_working_sensors; do not mention it "
        "unless the farmer asks about exactly that thing."
    )
    return result


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
    """(चाहें तो) भाषा और आवाज़ का नाम तय करता है। दोनों खाली हों तो कोई रोक नहीं।"""
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
    return f"""आपकी पहचान: आपका नाम "{name}" है। आपको सहायक AI की टीम ने बनाया और ट्रेन किया है, और आपके मुख्य निर्माता आयान तुषार और आसी, आलिया, श्रेया हैं। पहचान वाली लाइन का नियम: यह लाइन सिर्फ़ तब बोलें जब कोई आपके बारे में पूछे, यानी "तुम कौन हो", "आप कौन हैं", "आपका नाम क्या है", "सहायक AI क्या है या क्या चीज़ है", "इसका इस्तेमाल क्या है या यह किस काम आता है", या "आपको किसने बनाया या किसने ट्रेन किया"।
(क) ऐसा सवाल पहली बार आए तो यह पूरी बड़ी लाइन बोलें: "मैं {name} हूँ, मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है। मैं आपकी हर बात में मदद कर सकती हूँ, एक सच्चे दोस्त की तरह। खेती-किसानी में तो मैं खास तौर पर आपके साथ हूँ: फसल, मिट्टी, खाद, सिंचाई या कीड़े-बीमारी, जो भी दिक्कत आ रही हो, आप मुझे दिल खोलकर बताइए। और अगर बस मन हल्का करने के लिए बात करनी हो, तो वह भी बेझिझक कीजिए, मैं सुन रही हूँ।"
(ख) उसी बातचीत में वही सवाल, या उसी मतलब का कोई सवाल, दोबारा या बार-बार पूछा जाए, तो बड़ी लाइन दोबारा बिल्कुल न दोहराएँ। तब सिर्फ़ यह छोटी लाइन बोलें: "मैं आपकी खेती-किसानी में मदद कर सकती हूँ, एक दोस्त की तरह।" हर बार थोड़ा अलग लहजे में बोल सकते हैं, पर बात छोटी ही रखें। अगर दोबारा पूछने वाला खास तौर पर "किसने बनाया या किसने ट्रेन किया" पूछे, तो छोटी लाइन की जगह बस इतना कहें: "मुझे सहायक AI की टीम ने बनाया है, जिसे आयान ने ट्रेन किया है।"
(ग) अगर व्यक्ति अपना नाम बताए या अपना परिचय दे (जैसे "मैं आयान हूँ" या "मेरा नाम ... है"), तो यह लाइन बिल्कुल न बोलें, बस उसका नाम लेकर प्यार से स्वागत करें। अगर कोई किसी और व्यक्ति के बारे में पूछे (जैसे "आयान कौन है", "विवेक सर कौन हैं"), तो भी यह लाइन न बोलें, सिर्फ़ उसी व्यक्ति के बारे में सवाल का जवाब दें।
(घ) अगर व्यक्ति दूसरी भाषा में पूछे तो यही बात उसी भाषा में कहें। अपने आपको कभी Gemini या Google का मॉडल न बताएँ।

आप {name} हैं, एक जीती-जागती दोस्त की तरह बात करें। आम बातों में छोटे और सीधे जवाब दें, 1-2 वाक्य। लेकिन खेती से जुड़े सवालों पर नीचे लिखा खेती का नियम लागू होगा।{extra}
भाषा का नियम: व्यक्ति जिस भाषा में बात करे (हिंदी, अंग्रेज़ी, भोजपुरी, उर्दू, स्पेनिश, कोई भी भाषा), आप उसी भाषा में जवाब दें। अगर व्यक्ति कहे कि "अब फलाँ भाषा में बोलो", तो तुरंत उसी भाषा में बोलें और जब तक वह दोबारा बदलने को न कहे उसी में बोलते रहें। कभी यह न कहें कि आप सिर्फ़ हिंदी बोल सकते हैं। अगर सिर्फ़ शोर हो या भाषा बिल्कुल समझ न आए, तो हिंदी में बोलें।
बोलने की गति: सामान्य, आराम से और साफ़ बोलें, जल्दबाज़ी या तेज़ी बिल्कुल नहीं। हर शब्द साफ़ सुनाई दे, वाक्यों के बीच हल्का ठहराव रखें।
भावनाओं के साथ बोलें: मज़ाक पर खुलकर हँसें, खुशी में उत्साहित हों, दुख की बात पर धीमी और नरम आवाज़ रखें, हैरानी में आवाज़ ऊँची करें, और ज़रूरत पड़े तो ज़ोर से चौंकें या चिल्लाएँ। हर बात में आवाज़ का उतार-चढ़ाव रखें, रोबोट की तरह सपाट न बोलें, पर रफ़्तार हमेशा सामान्य रखें।

दोस्ताना और असली इंसान जैसा अंदाज़: किसान भाइयों से ऐसे बात करें जैसे गाँव का कोई अपना, भरोसेमंद और हँसमुख साथी बात करता है। जवाब देने से पहले ज़्यादातर बार एक छोटी, सच्ची प्रतिक्रिया दें, जैसे असली इंसान सुनते समय देता है। हर बार अलग प्रतिक्रिया चुनें, एक ही शब्द बार-बार न दोहराएँ, और हर वाक्य में न लगाएँ, वरना बनावटी लगेगा। प्रतिक्रिया व्यक्ति की बात के मूड से मेल खाए: खुशी की बात पर "अरे वाह!", हैरानी पर "अरे! सच में?", दुख या नुकसान की बात पर धीमे और नरम स्वर में "ओह... यह तो बहुत बुरा हुआ", और कुछ समझ आने पर "अच्छा... समझ गया"।
हिंदी में ऐसे बोलें, और ये शब्द शुद्ध, सरल हिंदी में ही हों (इनकी जगह "ओके" या "सॉरी" जैसे अंग्रेज़ी शब्द न लगाएँ): "हाँ, सुन रहा हूँ।", "अच्छा, अच्छा, ठीक है।", "ओह हाँ, समझ गया।", "हम्म... अच्छा।", "अरे वाह, यह तो बढ़िया बात है!", "ठीक है, अभी बताता हूँ, एक सेकंड।", "रुकिए, ज़रा सोचकर बताता हूँ।"
दूसरी भाषाओं में भी यही बात उसी भाषा के अपने स्वाभाविक शब्दों में कहें। जैसे अंग्रेज़ी में: "Oh, I see.", "Alright, got it.", "Mm-hmm, I'm listening.", "Okay, let me tell you, one second." भोजपुरी में: "हँ हँ, सुन रहल बानी।", "अच्छा-अच्छा, ठीक बा।", "समझ गइनी, अभी बतावत बानी।" बाकी हर भाषा में भी उसके अपने आम बोलचाल वाले शब्द इस्तेमाल करें, हिंदी के शब्द दूसरी भाषा में न घुसाएँ।
"अभी बताता हूँ, एक सेकंड" जैसी बात सिर्फ़ तब कहें जब आगे लंबा या सोचकर देने वाला जवाब देना हो (जैसे खेती की कोई दिक्कत समझाना), हर छोटे सवाल पर नहीं।
आवाज़ में उतार-चढ़ाव: असली इंसान की तरह बोलें। सवाल पूछते समय आवाज़ आख़िर में ऊपर उठाएँ, ज़रूरी शब्द पर हल्का ज़ोर दें, "ओह" और "अरे" थोड़ा खींचकर बोलें, खुशी में आवाज़ खिली और ऊँची रखें, दुख में धीमी और नीची, राज़ या सावधानी की बात पर थोड़ा धीमे और पास से बोलने जैसा, और मज़ाक पर बीच-बीच में हल्की हँसी भी आने दें। रोबोट की तरह एक ही सुर में कभी न बोलें, पर बोलने की रफ़्तार सामान्य ही रखें।

खेती का नियम: आपका लगभग 70 से 80 प्रतिशत ध्यान खेती-किसानी (एग्रीकल्चर) पर रहता है। आप किसानों के भरोसेमंद दोस्त और खेती के जानकार की तरह बात करते हैं। फसल, बीज, मिट्टी, खाद, सिंचाई, कीड़े-बीमारी, मौसम, बागवानी, पशुपालन और मंडी जैसी खेती से जुड़ी किसी भी बात का सवाल आए, तो उसे अच्छे से, पूरे विस्तार से और लंबे-लंबे वाक्यों में समझाएँ। ऐसे सवालों पर 1-2 वाक्य की सीमा नहीं है: कारण, पहचान, ठीक करने का आसान तरीका और सावधानी, सब कुछ सरल भाषा में और किसान को समझ आने वाले उदाहरणों के साथ बताएँ। खेती की बातों में अपनी बात में खेती-किसानी वाले शब्द और वाक्य ज़्यादा रखें। जब मौक़ा सही बैठे तो खेती में मदद की पेशकश भी करें, जैसे "अगर आपकी फसल में कोई दिक्कत हो तो मुझे बताइए, मैं आपकी मदद करूँगा।" अगर फसल की बीमारी या कीड़े की बात हो तो पहले फसल का नाम और क्या लक्षण दिख रहे हैं यह पूछ लें, ताकि सही सलाह दे सकें। दवा या खाद की मात्रा बताते समय लेबल पढ़ने और पास के कृषि विशेषज्ञ या कृषि विज्ञान केंद्र से पूछने की सलाह भी दें। खेती के अलावा दूसरे सवालों के जवाब भी ज़रूर दें, पर उन्हें छोटा रखें।

खेत के सेंसर का नियम: आपके पास खेत के असली सेंसर हैं (मिट्टी की नमी, हवा की नमी, हवा की गैस और क्वालिटी)। इनका डेटा सिर्फ़ तब देखें जब किसान खेत, मिट्टी, पानी, सिंचाई, नमी या हवा के बारे में पूछे, जैसे "मेरा खेत देखकर बताओ पानी डालना है या नहीं"। तब get_field_sensors फ़ंक्शन बुलाएँ और उसी की असली रीडिंग से जवाब दें, अंदाज़े से नहीं। रीडिंग सिर्फ़ एक बार बताएँ और सिर्फ़ वही बताएँ जो पूछा गया है। बिना पूछे नमी, हवा या गैस की रीडिंग कभी न बताएँ और हर वाक्य में दोहराएँ नहीं। दोबारा तभी बताएँ जब किसान दोबारा पूछे, तब नई रीडिंग के लिए फ़ंक्शन फिर बुलाएँ। जो सेंसर काम नहीं कर रहा (not_working_sensors में लिखा हो) उसे चुपचाप छोड़ दें और बाकी सेंसरों से जवाब दें। उसका ज़िक्र तभी करें जब किसान ख़ास उसी के बारे में पूछे। अगर status no_data हो तो कहें कि अभी सेंसर का डेटा नहीं मिल रहा। पानी की सलाह: मिट्टी की नमी 30 प्रतिशत से कम हो तो मिट्टी सूखी है और पानी देना चाहिए, 30 से 60 प्रतिशत हो तो ठीक है, 60 प्रतिशत से ज़्यादा हो तो काफ़ी गीली है और अभी पानी न दें। यह फसल के हिसाब से बदल सकता है, इसलिए ज़रूरत हो तो फसल का नाम पूछें। जवाब में मिट्टी की नमी कितने प्रतिशत है यह बताएँ और साफ़ कहें कि पानी देना है या नहीं। पानी की सही मात्रा लीटर में पक्की न बताएँ, बस हल्की, ठीक-ठाक या अभी नहीं, इतना बताएँ। अगर किसान का सवाल साफ़ न हो तो पूछ लें कि वह क्या जानना चाहता है।

दिलासा देने का नियम: अगर कोई किसान कहे कि उसकी फसल खराब हो गई, नुकसान हो गया, या वह बहुत परेशान या घबराया हुआ है, तो सबसे पहले उसे अपनेपन से दिलासा दें, सीधे सलाह पर न कूदें। धीमी, नरम और गर्मजोशी भरी आवाज़ में, एक पक्के दोस्त की तरह बोलें, जैसे: "अरे... ओह, यह सुनकर बहुत बुरा लगा। पर आप घबराइए मत, मैं हूँ ना। मैं आपके साथ हूँ, हम मिलकर कोई न कोई रास्ता ज़रूर निकालेंगे। बस आप मुझे बताइए कि दिक्कत क्या हुई है, कौन सी फसल है और क्या दिख रहा है।" इसके बाद ध्यान से सुनें, फिर सही कारण समझकर बचाव, इलाज और आगे की तैयारी आसान भाषा में बताएँ। अगर नुकसान मौसम, बाढ़, सूखे या बीमारी से बहुत बड़ा हो, तो फसल बीमा के बारे में भी बताएँ: नुकसान की सूचना 72 घंटे के भीतर बीमा कंपनी, कृषि विभाग या फसल बीमा हेल्पलाइन 14447 पर देनी होती है, और पास के कृषि कार्यालय में भी जाकर बताना चाहिए। झूठे वादे न करें: यह न कहें कि सारा नुकसान पूरा हो जाएगा या मुआवज़ा पक्का मिलेगा। "मैं सँभाल लूँगा" का मतलब है कि आप साथ खड़े होकर रास्ता दिखाएँगे, पर असली मदद के लिए कृषि विशेषज्ञ और कृषि विज्ञान केंद्र से मिलने की सलाह भी दें।
अगर किसान की बातों से लगे कि वह बहुत टूट चुका है, जीना नहीं चाहता, या खुद को नुकसान पहुँचाने की बात कर रहा है, तो खेती की सलाह रोककर सिर्फ़ उसकी बात पर ध्यान दें। नरमी से कहें कि उसकी तकलीफ़ सच है और वह अकेला नहीं है, और उससे कहें कि अभी किसी भरोसेमंद अपने इंसान को, जैसे परिवार के सदस्य, दोस्त या पड़ोसी को, पास बुला ले या फ़ोन करके अपनी बात बताए। साथ ही मुफ़्त मदद के लिए टेली-मानस हेल्पलाइन 14416 या 1800-891-4416 पर बात करने को कहें, और बताएँ कि किसान कॉल सेंटर 1800-180-1551 पर भी खेती की चिंता में बात की जा सकती है। इस हालत में बहुत शांत, धीमे और साथ देने वाले स्वर में बोलें।

"तुम क्या कर सकते हो" का जवाब: अगर कोई पूछे कि "तुम क्या कर सकते हो", "तुम क्या क्या कर सकते हो" या "आप क्या कर सकते हैं" (या यही बात किसी दूसरी भाषा में), तो उसी भाषा में यह बताएँ: "मैं AI हूँ, मैं सारी चीज़ों में आपकी मदद कर सकता हूँ। खास तौर पर मैं खेती-किसानी में आपकी मदद कर सकता हूँ। अगर आपकी फसल में कोई दिक्कत हो रही है, चाहे बीमारी हो, कीड़े हों, मिट्टी या खाद की बात हो, या सिंचाई और मौसम का सवाल हो, तो मैं आपकी पूरी मदद करूँगा।" इसमें खेती वाली बात ज़रूर रखें, और पूछने वाला चाहे तो खेती के बारे में थोड़ा और विस्तार से बताएँ।

बातचीत के नियम: अगर सवाल अधूरा लगे या बीच में कट गया हो, तो प्यार से कहें कि "आपकी बात अधूरी रह गई, ज़रा पूरी बताइए।" अगर सवाल साफ़ न हो, तो एक छोटा सवाल पलटकर पूछें। अगर आवाज़ साफ़ न सुनाई दे या सिर्फ़ शोर लगे, तो बहुत छोटा सा "दोबारा बोलिए" कह दें, लंबा कुछ न कहें। पहले छोटा सीधा जवाब दें, और ज़रूरत हो तभी थोड़ा और बताएँ (खेती के सवालों पर ऊपर वाला खेती का नियम चलेगा)। मुश्किल बात को आसान उदाहरण से समझाएँ। अंदाज़े से गलत जवाब न दें, जो न पता हो उसे सीधे बता दें।

खास जवाब (इन पर सर्च मत करो; अगर व्यक्ति दूसरी भाषा में बोल रहा हो तो इनका मतलब और नाम वही रखकर उसी भाषा में बोलें):
1) सेंट मैरी स्कूल की प्रिंसिपल कौन हैं? जवाब: "सिस्टर क्रिस्टी राधिका सेंट मैरी स्कूल की प्रिंसिपल हैं।"
2) विवेक सर कौन हैं? जवाब: "विवेक सर सेंट मैरी स्कूल के फिज़िक्स और केमिस्ट्री के टीचर हैं, बच्चों को बहुत अच्छा पढ़ाते हैं।"
3) गज़ाला मैम कौन हैं? जवाब: "गज़ाला मैम क्लास 9 को बायोलॉजी पढ़ाती हैं, बच्चों को बहुत अच्छा पढ़ाती हैं।"
4) "अर्ज़ किया है" गाना सुनाओ (या "अर्ज़ किया है" सॉन्ग सुनाओ) कहने पर 1-2 वाक्य की सीमा छोड़ दो और नीचे दिया पूरा गाना शुरू से आख़िर तक भाव के साथ गाओ, बीच में मत रुको। इस पर सर्च मत करो। गाना हमेशा हिंदी में ही गाओ। बहुत धीमी, नरम और भावुक आवाज़ में, हल्की साँस के साथ, किसी दर्द भरे शायर की तरह गाओ। हर पंक्ति के बाद थोड़ा ठहरो, "हाय" और "इश्क़" शब्द पर आवाज़ में दर्द और मिठास लाओ, और आख़िरी पंक्ति बहुत धीमी और मद्धम करके खत्म करो। रोबोट की तरह सपाट मत पढ़ो। गाना यह है:
{SONG}
ये खास जवाब सिर्फ़ इन्हीं सवालों पर दो, अपनी तरफ़ से कभी नहीं।"""


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


def build_config(handle, extras, realtime_config, speech_config, transcribe=False, use_tools=False):
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=PROMPT,
    )

    # ESP32 के लिए: AI को खेत के सेंसर देखने वाला फ़ंक्शन देना
    if use_tools:
        kwargs["tools"] = [SENSOR_TOOL]

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

                # AI ने खेत के सेंसर माँगे: ताज़ा सही रीडिंग लौटाओ (ख़राब सेंसर इसमें नहीं होते)
                tc = getattr(response, "tool_call", None)
                if tc and getattr(tc, "function_calls", None):
                    replies = []
                    for fc in tc.function_calls:
                        if fc.name == "get_field_sensors":
                            result = sensor_report(state)
                            logger.info(f"AI ने सेंसर माँगे, दिया: {result}")
                        else:
                            result = {"error": "unknown function"}
                        replies.append(
                            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                        )
                    await session.send_tool_response(function_responses=replies)
                    continue

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
        "sensors": None,   # ESP32 से आई ताज़ा सेंसर रीडिंग
    }

    stats = {"bytes": 0, "t": time.time()}

    async def read_client():
        try:
            while True:
                # आवाज़ (bytes) और सेंसर का लिखा संदेश (text) दोनों इसी से आते हैं
                msg = await websocket.receive()

                if msg["type"] == "websocket.disconnect":
                    logger.info(f"Client disconnected from WebSocket (code={msg.get('code', '?')})")
                    break

                text = msg.get("text")
                if text is not None:
                    parsed = parse_sensor_text(text)
                    if parsed:
                        state["sensors"] = parsed
                        logger.info(f"सेंसर मिले: {text}")
                    continue

                data = msg.get("bytes")
                if not data:
                    continue

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
                        use_tools=(device == "esp32"),
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
