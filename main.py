import asyncio
import json
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
# सहायक AI सर्वर – वर्ज़न 14 (ESP32 वर्ज़न 15 के साथ चलाएँ)
# नया क्या है (v13 से):
#  1) सेंसर का डेटा AI की याद में चुपचाप जाता है (AI उस पर खुद कुछ नहीं बोलता)। ESP32 हर 5 सेकंड में भेजता है,
#     सर्वर हर बार नया मान रखता है और मान बदलने पर AI की याद भी अपडेट करता है। चैट चलती रहती है।
#     चैट बंद होते ही सारा डेटा मिट जाता है; नई चैट नए सिरे से पढ़ती है।
#  2) ESP32 दोबारा जुड़े तो उसका पुराना (अटका हुआ) कनेक्शन तुरंत बंद होता है।
#  3) ESP32 से बहुत देर तक कुछ न आए तो कनेक्शन बंद माना जाता है।
#  4) Logs में दिखता है कि ESP32 क्यों दोबारा जुड़ा ("ESP32 की जानकारी")।
# ======================================================================
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.8-live").strip()

# चुप्पी भरना: आवाज़ 0.25 सेकंड न आए तो 100 ms चुप्पी भेजो (असली रफ़्तार से)
KA_EVERY = 0.1
KA_IDLE = 0.25

# 1 = ESP32 वाले डिवाइस में AI बीच में नहीं रुकेगा, अपना वाक्य पूरा करेगा
NO_INTERRUPT = os.getenv("NO_INTERRUPT", "1") == "1"

# 1 = Render के Logs में दिखेगा कि AI ने क्या सुना और क्या बोला
DEBUG_HEAR = os.getenv("DEBUG_HEAR", "1") == "1"

# भाषा: खाली है = कोई भी भाषा चलेगी (AI उसी भाषा में बोलेगा जिसमें आप बोलें)।
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
# (v13) असली समय से इतने सेकंड आगे तक भेजो। ESP32 का बफ़र 1.2 सेकंड का है, इसलिए 0.8 सुरक्षित है
LEAD = float(os.getenv("AUDIO_LEAD", "0.8"))

# सेंसर का डेटा इतने सेकंड से पुराना हो जाए तो AI उसे "डेटा नहीं मिल रहा" मानेगा
SENSOR_STALE_SEC = float(os.getenv("SENSOR_STALE_SEC", "20"))

# 1 = सेंसर का डेटा AI की याद में चुपचाप डालो। इससे कनेक्शन में दिक्कत आए तो Render में SENSOR_CONTEXT=0 करें
SENSOR_CONTEXT = os.getenv("SENSOR_CONTEXT", "1") == "1"
# AI की याद कम से कम इतने सेकंड के अंतर से अपडेट हो
SENSOR_CONTEXT_EVERY = float(os.getenv("SENSOR_CONTEXT_EVERY", "5"))
# मान कम से कम इतना बदले तभी AI की याद अपडेट करो (हल्के उतार-चढ़ाव से याद न भरे)
SENSOR_MIN_CHANGE = {"soil": 3.0, "hum": 3.0, "temp": 0.5, "gas": 5.0}
# ESP32 से इतने सेकंड तक कुछ भी न आए तो कनेक्शन मरा हुआ मानो (वह हर 5 सेकंड में सेंसर भेजता है)
ESP_IDLE_SEC = float(os.getenv("ESP_IDLE_SEC", "60"))

# हर ESP32 का इस समय चालू कनेक्शन: id -> उसका "बंद करो" संकेत
active_devices = {}

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

# ----------------------------------------------------------------------
# खेत के सेंसर
# ----------------------------------------------------------------------
# हर सेंसर की सही सीमा। इससे बाहर का मान गलत माना जाएगा और AI उसे नहीं देखेगा।
SENSOR_RANGES = {
    "soil": (0.0, 100.0),    # मिट्टी की नमी %
    "hum": (0.0, 100.0),     # हवा की नमी %
    "temp": (-20.0, 70.0),   # तापमान °C
    "gas": (0.0, 100.0),     # हवा में गैस % (ज़्यादा = खराब हवा)
}


def parse_sensor_text(txt):
    """ESP32 का संदेश "sensor:soil=45,hum=60,temp=28.5,gas=20" पढ़ता है।
    जो सेंसर खराब है, ESP32 उसे भेजता ही नहीं; और जो सीमा से बाहर हो उसे यहाँ भी हटा देते हैं।"""
    out = {}
    body = txt[len("sensor:"):]
    for part in body.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in SENSOR_RANGES:
            continue
        try:
            val = float(v.strip())
        except ValueError:
            continue
        lo, hi = SENSOR_RANGES[k]
        if lo <= val <= hi:
            out[k] = val
    return out


def sensor_report(state):
    """AI के टूल को जवाब: सिर्फ़ चालू सेंसर का ताज़ा डेटा। खराब सेंसर 'kharab_sensor' में आते हैं।"""
    t = state.get("sensor_t")
    if not t or (time.time() - t) > SENSOR_STALE_SEC:
        return {
            "status": "koi_data_nahi",
            "note": "अभी खेत के सेंसर का डेटा नहीं मिल रहा। किसान को एक छोटे वाक्य में बता दें, अंदाज़े से कोई संख्या न बोलें।",
        }

    s = state.get("sensors") or {}
    out = {"status": "ok"}
    names = {
        "soil": "mitti_ki_nami_percent",
        "hum": "hawa_ki_nami_percent",
        "temp": "taapman_celsius",
        "gas": "hawa_mein_gas_percent_zyada_matlab_kharab_hawa",
    }
    working = {}
    broken = []
    for key, label in names.items():
        if key in s:
            v = s[key]
            working[label] = int(round(v)) if key != "temp" else round(v, 1)
        else:
            broken.append(label)

    if not working:
        return {
            "status": "koi_sensor_chalu_nahi",
            "note": "सभी सेंसर इस समय बंद या खराब हैं। किसान को एक छोटे वाक्य में बता दें।",
        }

    out["chalu_sensor"] = working
    if broken:
        out["kharab_sensor"] = broken
        out["note"] = "kharab_sensor वालों को पूरी तरह अनदेखा करें, उनका ज़िक्र तभी करें जब किसान ख़ुद उसी के बारे में पूछे।"
    return out


def sensors_changed(old, new):
    """पिछली बार AI को बताए डेटा से अब कोई काम का बदलाव है या नहीं।"""
    if old is None or set(old) != set(new):
        return True
    return any(abs(new[k] - old[k]) >= SENSOR_MIN_CHANGE[k] for k in new)


def sensor_context_text(state):
    return (
        "[खेत सेंसर अपडेट - यह मशीन का अपने-आप आया डेटा है, किसान ने कुछ नहीं बोला। "
        "इसका कोई जवाब न दें, बस याद रखें। पुराने अपडेट की जगह यही सबसे नया है।] "
        + json.dumps(sensor_report(state), ensure_ascii=False)
    )


def make_tools():
    """AI को खेत के सेंसर पढ़ने का टूल देता है।"""
    try:
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="get_farm_sensors",
                        # AI पहले डेटा का इंतज़ार करे, फिर जवाब दे
                        behavior=types.Behavior.BLOCKING,
                        description=(
                            "किसान के खेत में लगे असली सेंसर (मिट्टी की नमी, हवा की नमी, तापमान, हवा में गैस) का ताज़ा डेटा पढ़ता है। "
                            "इसे सिर्फ़ तब बुलाएँ जब किसान अपने खेत की हालत, पानी देने, मिट्टी की नमी, हवा या मौसम के बारे में पूछे। "
                            "बाकी बातों में इसे न बुलाएँ।"
                        ),
                    )
                ]
            )
        ]
    except Exception as e:
        logger.warning(f"सेंसर टूल नहीं बन पाया: {e}")
        return None


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

खेत के सेंसर का नियम: किसान के खेत में असली सेंसर लगे हैं (मिट्टी की नमी, हवा की नमी, तापमान, हवा में गैस)। उनका डेटा आपको बातचीत के बीच "[खेत सेंसर अपडेट ...]" नाम के संदेश में अपने-आप मिलता रहता है, और "get_farm_sensors" टूल से भी सबसे ताज़ा डेटा पढ़ सकते हैं।
(0) अपडेट संदेश किसान की बात नहीं है। उसका कभी जवाब न दें और अपने-आप उसका ज़िक्र न करें। उसे पूरी बातचीत में याद रखें, सबसे नया अपडेट ही सही मानें, और खेती की सलाह उसी हालत को ध्यान में रखकर दें।
(1) जब किसान अपने खेत की हालत, पानी देने या न देने, मिट्टी की नमी, हवा की क्वालिटी, गैस, हवा की नमी या तापमान के बारे में पूछे, तो पहले "get_farm_sensors" टूल से ताज़ा डेटा पढ़ें, फिर उसी के हिसाब से जवाब दें। जवाब के बाद, जहाँ काम का हो, उसी डेटा से जुड़ा एक छोटा अगला सवाल पूछें (जैसे फसल कौन सी है, आख़िरी सिंचाई कब हुई)। कभी अंदाज़े से कोई संख्या न बोलें।
(2) सिर्फ़ वही बताएँ जो पूछा गया। अगर पानी का सवाल है तो मिट्टी की नमी के हिसाब से बताएँ और बाकी सेंसर का ज़िक्र न करें। अगर हवा का सवाल है तो सिर्फ़ हवा की बात करें। अगर किसान साफ़ "खेत की पूरी हालत" पूछे, तभी सब चालू सेंसर संक्षेप में एक साथ बताएँ। अगर समझ न आए कि किसान किस चीज़ के बारे में पूछ रहा है, तो एक छोटा सवाल पूछ लें।
(3) एक ही बार बताएँ। अपने आप से, या दूसरे जवाबों में, बार-बार यह न दोहराएँ कि "आपके खेत में नमी इतनी है, हवा इतनी है"। किसान दोबारा पूछे तभी टूल से नया डेटा लेकर दोबारा बताएँ। खेती की दूसरी बातों में डेटा को ध्यान में रखें, पर संख्याएँ तभी बोलें जब वे उस सवाल से सीधे जुड़ी हों।
(4) पानी की सलाह का तरीका (मिट्टी की नमी के हिसाब से, यह सिर्फ़ सेंसर वाली एक जगह की नमी है इसलिए इसे अंदाज़ा बताएँ): 30% से कम = मिट्टी सूखी है, सिंचाई की ज़रूरत है; 30% से 60% = ठीक है, ज़्यादातर फसलों को अभी पानी नहीं चाहिए (धान जैसी ज़्यादा पानी वाली फसल को हल्का पानी लग सकता है); 60% से ऊपर = नमी काफ़ी है, पानी न डालें, और बहुत ज़्यादा हो तो जल-जमाव से बचाव बताएँ। सूखी हो तो बताएँ कि नमी अभी कितनी है और लगभग कितनी तक पहुँचानी है, और पानी धीरे-धीरे हल्की सिंचाई से देने को कहें, लीटर की झूठी पक्की संख्या न दें। फसल का नाम पता हो तो उसके हिसाब से सलाह दें, न पता हो तो ज़रूरी लगे तो एक बार पूछ लें।
(5) हवा में गैस की संख्या जितनी ज़्यादा, हवा उतनी खराब: 30% तक = ठीक, 30% से 60% = थोड़ी खराब, 60% से ऊपर = खराब, खुली जगह या हवादार जगह की सलाह दें। यह सेंसर मोटा अंदाज़ा देता है, डॉक्टरी या पक्की जाँच नहीं।
(6) टूल के जवाब में अगर कोई सेंसर "kharab_sensor" में हो, तो उसे पूरी तरह अनदेखा करें और बाकी चालू सेंसर से जवाब दें। उस खराब सेंसर का ज़िक्र तभी करें जब किसान ख़ुद उसी के बारे में पूछे, तब सीधे कह दें कि वह सेंसर अभी काम नहीं कर रहा। अगर टूल कहे कि डेटा नहीं मिल रहा या कोई सेंसर चालू नहीं है, तो एक छोटे वाक्य में यही बता दें और अंदाज़े से कोई संख्या न बोलें।
(7) टूल बुलाते समय किसान को "रुकिए, खेत का हाल देखता हूँ" जैसा छोटा सा एक वाक्य ही कहें, लंबी भूमिका न बाँधें।

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


def build_config(handle, extras, realtime_config, speech_config, transcribe=False, tools=None):
    kwargs = dict(
        response_modalities=["AUDIO"],
        system_instruction=PROMPT,
    )

    if realtime_config is not None:
        kwargs["realtime_input_config"] = realtime_config

    if speech_config is not None:
        kwargs["speech_config"] = speech_config

    if tools:
        kwargs["tools"] = tools

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

    async def feed_sensor_context():
        # सेंसर का डेटा AI की याद में डालता है। turn_complete=False = AI इस पर बोलना शुरू नहीं करता
        sent = None        # इस Gemini कनेक्शन में AI को आख़िरी बार क्या बताया
        sent_at = 0.0
        while True:
            await asyncio.sleep(0.5)
            now = time.time()
            t = state.get("sensor_t")
            if not t or (now - t) > SENSOR_STALE_SEC:
                continue
            # AI बोल रहा हो या किसान अभी बोल रहा हो, तो बीच में कुछ न डालो
            if state.get("model_busy") or (now - last_audio.get("mic", 0.0)) < 2.0:
                continue
            current = dict(state.get("sensors") or {})
            if sent is not None and (now - sent_at) < SENSOR_CONTEXT_EVERY:
                continue
            if not sensors_changed(sent, current):
                continue
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=sensor_context_text(state))],
                ),
                turn_complete=False,
            )
            sent, sent_at = current, now
            if DEBUG_HEAR:
                logger.info(f"AI की याद में सेंसर डेटा डाला: {current}")

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

                # खेत के सेंसर का टूल: AI ने डेटा माँगा तो ताज़ा डेटा लौटाओ
                tc = getattr(response, "tool_call", None)
                if tc and getattr(tc, "function_calls", None):
                    frs = []
                    for fc in tc.function_calls:
                        if fc.name == "get_farm_sensors":
                            result = sensor_report(state)
                            logger.info(f"AI ने खेत का डेटा माँगा, दिया: {result}")
                        else:
                            result = {"error": "yeh tool maujood nahi hai"}
                        frs.append(
                            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                        )
                    await session.send_tool_response(function_responses=frs)

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

                    if getattr(sc, "interrupted", False):
                        state["model_busy"] = False

                    if sc.model_turn:
                        state["model_busy"] = True

                    if getattr(sc, "turn_complete", False):
                        state["model_busy"] = False
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

    if SENSOR_CONTEXT and state.get("device") == "esp32":
        tasks.append(asyncio.create_task(feed_sensor_context()))

    state["model_busy"] = False

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
    # खेत के सेंसर का टूल सिर्फ़ ESP32 वाले डिवाइस के लिए
    tools = make_tools() if device == "esp32" else None

    logger.info(
        f"Client connected | device={device or 'app'} | model={LIVE_MODEL} | no_interrupt={no_interrupt} | sensor_tool={bool(tools)}"
    )

    audio_q = asyncio.Queue(maxsize=400)   # डिवाइस से आई आवाज़ (Gemini के लिए)
    out_q = asyncio.Queue()                # Gemini से आई आवाज़ और इमोजी (डिवाइस के लिए)
    client_gone = asyncio.Event()

    # वही ESP32 दोबारा जुड़ा है तो उसका पुराना कनेक्शन बंद करो, ताकि दो Gemini चैट साथ न चलें
    device_key = None
    if device == "esp32":
        device_key = websocket.query_params.get("id", "esp32")
        old = active_devices.get(device_key)
        if old is not None:
            logger.info("यही ESP32 दोबारा जुड़ा: पुराना कनेक्शन बंद कर रहे हैं")
            old.set()
        active_devices[device_key] = client_gone

    last_audio = {"t": time.time()}

    state = {
        "handle": None,
        "extras": True,
        "speech": True,
        "last_error": "",
        "device": device,
        "turn_text": "",
        "emoji_sent": False,
        "sensors": {},        # सबसे ताज़ा चालू सेंसर का डेटा
        "sensor_t": 0.0,      # वह डेटा कब आया
    }

    stats = {"bytes": 0, "t": time.time()}

    async def read_client():
        try:
            while True:
                if device == "esp32":
                    try:
                        msg = await asyncio.wait_for(websocket.receive(), timeout=ESP_IDLE_SEC)
                    except asyncio.TimeoutError:
                        logger.info(f"ESP32 से {int(ESP_IDLE_SEC)} सेकंड से कुछ नहीं आया: कनेक्शन बंद मान रहे हैं")
                        break
                else:
                    msg = await websocket.receive()

                if msg.get("type") == "websocket.disconnect":
                    logger.info(f"Client disconnected from WebSocket (code={msg.get('code', '?')})")
                    break

                data = msg.get("bytes")

                if data is not None:
                    last_audio["t"] = time.time()
                    last_audio["mic"] = last_audio["t"]

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
                    continue

                txt = msg.get("text")
                if txt and txt.startswith("info:"):
                    logger.info(f"ESP32 की जानकारी: {txt[5:200]}")
                    continue
                if txt and txt.startswith("sensor:"):
                    # सेंसर का डेटा सिर्फ़ सर्वर में रखते हैं, AI को अपने आप नहीं सुनाते
                    state["sensors"] = parse_sensor_text(txt)
                    state["sensor_t"] = time.time()
                    if DEBUG_HEAR:
                        logger.info(f"खेत का डेटा मिला: {state['sensors']}")

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
                        tools=tools,
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

        if device_key and active_devices.get(device_key) is client_gone:
            del active_devices[device_key]

        try:
            await asyncio.wait_for(websocket.close(), timeout=3)
        except Exception:
            pass

        logger.info("चैट बंद: इस चैट का सेंसर डेटा और याद मिट गई")
