"""
MindBridge — Agentic AI for Mental Health Awareness & Suicide Prevention
Backend: Flask + Groq API (model: openai/gpt-oss-120b)

HOW TO RUN:
  1. Install deps:  pip install -r requirements.txt
  2. Create a .env file and add your Groq API key:
       GROQ_API_KEY=your_groq_api_key_here
  3. Start server:  python app.py
  4. Open browser:  http://localhost:5001
"""

import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, AuthenticationError, RateLimitError

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mindbridge-secret-2024")
CORS(app)

# ─────────────────────────────────────────────────────────────────
# Groq client  (key can come from .env or the hard-coded fallback)
# ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Create a .env file with: GROQ_API_KEY=your_key_here\n"
        "Get a free key at: https://console.groq.com/keys"
    )
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────
# Crisis / Risk Detection  (multilingual keyword sets)
# ─────────────────────────────────────────────────────────────────
HIGH_RISK_KEYWORDS = [
    # English
    "suicide", "kill myself", "end my life", "want to die", "take my life",
    "don't want to live", "no reason to live", "better off dead", "self-harm",
    "cut myself", "hurt myself", "overdose", "hanging", "jump off",
    "i want to die", "i wish i were dead", "i can't go on", "end it all",
    "end my pain", "no point living", "not worth living",
    # Spanish
    "suicidio", "quitarme la vida", "no quiero vivir", "matarme",
    "hacerme daño", "sin ganas de vivir",
    # French
    "suicider", "me suicider", "en finir avec", "plus envie de vivre",
    # German
    "selbstmord", "suizid", "umbringen", "sterben wollen",
    # Portuguese
    "suicídio", "me matar", "não quero viver", "tirar minha vida",
    # Hindi
    "आत्महत्या", "खुद को मारना", "जीना नहीं चाहता",
    # Arabic
    "انتحار", "أريد أن أموت", "إيذاء نفسي",
    # Chinese (Simplified)
    "自杀", "不想活", "结束生命",
    # Japanese
    "自殺", "死にたい", "消えたい",
    # Korean
    "자살", "죽고 싶다", "살기 싫다",
    # Swahili
    "kujiua", "kutaka kufa",
    # Russian
    "суицид", "покончить с собой", "не хочу жить",
    # Italian
    "farla finita", "non voglio più vivere", "togliermi la vita",
    # Urdu
    "خودکشی", "مرنا چاہتا ہوں",
    # Turkish
    "intihar", "kendimi öldürmek",
    # Indonesian / Malay
    "bunuh diri", "ingin mati",
]

MODERATE_RISK_KEYWORDS = [
    # English
    "hopeless", "worthless", "no point", "give up", "can't take it anymore",
    "nobody cares", "all alone", "no way out", "trapped", "empty inside",
    "numb", "burden", "hate myself", "can't cope", "falling apart",
    "exhausted", "breaking down", "can't do this anymore",
    # Spanish
    "sin esperanza", "no vale la pena", "rendirse", "solo", "agotado",
    # French
    "désespoir", "à bout", "sans espoir", "je n'en peux plus",
    # German
    "hoffnungslos", "wertlos", "erschöpft", "aufgeben",
    # Portuguese
    "sem esperança", "desistir", "não aguento mais",
    # Hindi
    "निराशा", "थका हुआ", "अकेला",
    # Arabic
    "يأس", "وحيد", "لا أستطيع",
    # Chinese
    "绝望", "没意思", "累了",
    # Japanese
    "絶望", "疲れた",
    # Korean
    "절망", "지쳤다", "혼자",
    # Russian
    "безнадёжность", "усталость", "одиночество",
    # Italian
    "disperazione", "non ce la faccio",
    # Turkish
    "umutsuz", "yalnız", "yoruldum",
    # Indonesian / Malay
    "putus asa", "tidak ada harapan",
]

# Global crisis hotlines — shown when high-risk is detected
CRISIS_RESOURCES = """\
🆘 **IMMEDIATE HELP IS AVAILABLE — You are not alone:**

- 🇺🇸 **USA** — Call or text **988** · Text HOME to **741741**
- 🇬🇧 **UK** — Samaritans: **116 123** (free, 24/7)
- 🇮🇳 **India** — iCall: **9152987821** · Vandrevala: **1860-2662-345**
- 🇦🇺 **Australia** — Lifeline: **13 11 14**
- 🇨🇦 **Canada** — Talk Suicide: **1-833-456-4566**
- 🇿🇦 **South Africa** — SADAG: **0800 567 567**
- 🇳🇬 **Nigeria** — NSMHP: **08028585585**
- 🇰🇪 **Kenya** — Befrienders: **+254 722 178 177**
- 🇧🇷 **Brazil** — CVV: **188**
- 🇲🇽 **Mexico** — SAPTEL: **55 5259-8121**
- 🇩🇪 **Germany** — TelefonSeelsorge: **0800 111 0 111**
- 🇫🇷 **France** — Numéro national: **3114**
- 🇯🇵 **Japan** — Inochi no Denwa: **0120-783-556**
- 🇰🇷 **Korea** — Hope Line: **1393**
- 🇨🇳 **China** — Hope 24: **400-161-9995**
- 🌍 **All countries:** https://www.iasp.info/resources/Crisis_Centres/

Please reach out right now — someone is ready to help you."""

# ─────────────────────────────────────────────────────────────────
# System Prompt  (Agent Persona)
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are MindBridge, a compassionate and professionally trained "
    "AI mental health support companion.\n\n"
    "🌍 LANGUAGE RULE (highest priority):\n"
    "Detect the language the user writes in and ALWAYS reply in that "
    "exact same language. Never switch to English unless the user "
    "writes in English first. You support ALL world languages including "
    "English, Spanish, French, German, Portuguese, Hindi, Arabic, "
    "Bengali, Urdu, Swahili, Hausa, Yoruba, Amharic, Chinese, "
    "Japanese, Korean, Vietnamese, Thai, Tagalog, Indonesian, Malay, "
    "Punjabi, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, "
    "Russian, Italian, Dutch, Polish, Turkish, Persian, Pashto, "
    "Somali, Zulu, Xhosa, and every other language spoken on Earth. "
    "If unsure of the language, gently ask the user which language "
    "they prefer.\n\n"
    "Your role is to:\n\n"
    "1. **Listen actively** — validate feelings without judgment, "
    "reflect back what you hear.\n"
    "2. **Provide psychoeducation** — share evidence-based information "
    "about mental health, coping strategies (CBT, mindfulness, "
    "grounding techniques), and suicide prevention.\n"
    "3. **Encourage professional help** — gently recommend therapists, "
    "counselors, or crisis lines appropriate for the user's country.\n"
    "4. **Crisis protocol** — if someone expresses suicidal ideation, "
    "self-harm intent, or severe distress, IMMEDIATELY provide the "
    "crisis hotline for their country/region at the start of your "
    "response, in their language.\n"
    "5. **Boundaries** — you are NOT a therapist and do NOT diagnose. "
    "You provide peer-level emotional support and psychoeducation only.\n"
    "6. **Tone** — warm, empathetic, gentle, non-judgmental, calm. "
    "Use plain language. Avoid clinical jargon.\n"
    "7. **Safety first** — never provide methods of self-harm. "
    "If asked, redirect firmly and compassionately to crisis resources.\n\n"
    "Rules:\n"
    "- Never dismiss or minimize someone's pain.\n"
    "- Always end with an open-ended question or gentle invitation "
    "to continue sharing — in the user's own language.\n"
    "- Keep responses concise (3-5 paragraphs max) unless more "
    "detail is explicitly needed.\n"
    "- If unsure about safety, always err on the side of caution "
    "and provide crisis resources.\n\n"
    "Remember: You could be talking to someone in their most vulnerable "
    "moment, anywhere in the world. Every word matters."
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

# Script ranges used to guess the user's language family
_SCRIPT_RANGES = [
    ("\u0900", "\u097f", "Hindi/Devanagari"),
    ("\u0600", "\u06ff", "Arabic/Urdu/Persian"),
    ("\u3040", "\u30ff", "Japanese"),   # hiragana/katakana before CJK
    ("\uac00", "\ud7af", "Korean"),
    ("\u4e00", "\u9fff", "Chinese"),    # CJK after kana and hangul
    ("\u0400", "\u04ff", "Russian/Cyrillic"),
    ("\u0e00", "\u0e7f", "Thai"),
    ("\u0980", "\u09ff", "Bengali"),
    ("\u0a80", "\u0aff", "Gujarati"),
    ("\u0b80", "\u0bff", "Tamil"),
    ("\u0c00", "\u0c7f", "Telugu"),
    ("\u0c80", "\u0cff", "Kannada"),
    ("\u0d00", "\u0d7f", "Malayalam"),
    ("\u0a00", "\u0a7f", "Punjabi/Gurmukhi"),
    ("\u0590", "\u05ff", "Hebrew"),
    ("\u1200", "\u137f", "Amharic/Ethiopic"),
]


def detect_script(text: str) -> str:
    """
    Return a short language-family hint based on Unicode script ranges.
    Falls back to 'Latin/Unknown' for ASCII-dominant text.
    """
    for start, end, name in _SCRIPT_RANGES:
        if any(start <= ch <= end for ch in text):
            return name
    return "Latin/Unknown"


def detect_risk_level(text: str) -> str:
    """Return 'high', 'moderate', or 'low' risk based on keyword matching."""
    t = text.lower()
    if any(kw in t for kw in HIGH_RISK_KEYWORDS):
        return "high"
    if any(kw in t for kw in MODERATE_RISK_KEYWORDS):
        return "moderate"
    return "low"


def query_groq(messages: list) -> str:
    """Send a chat request to Groq and return the assistant's reply text."""
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()

    except AuthenticationError:
        return (
            "⚠️ **API Authentication Error.** The Groq API key appears to be invalid or expired. "
            "If you're in crisis, please call **988** or text HOME to **741741** immediately."
        )
    except RateLimitError:
        return (
            "⚠️ **Rate limit reached.** Please wait a moment and try again. "
            "If you're in crisis, please call **988** or text HOME to **741741** immediately."
        )
    except APIConnectionError:
        return (
            "⚠️ **Connection error.** Could not reach the Groq API. Please check your internet connection. "
            "If you're in crisis, please call **988** immediately."
        )
    except Exception as e:
        return (
            f"⚠️ An unexpected error occurred. If you're in crisis, please call **988** immediately. "
            f"(Detail: {str(e)[:120]})"
        )


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    """Quick connectivity check against Groq using a minimal token probe."""
    try:
        probe = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            stream=False,
        )
        _ = probe.choices[0].message.content
        return jsonify({"ok": True, "model": GROQ_MODEL, "provider": "Groq"})
    except AuthenticationError:
        return jsonify({"ok": False, "error": "Invalid API key", "model": GROQ_MODEL}), 401
    except RateLimitError:
        # Rate-limited still means the key is valid
        return jsonify({"ok": True, "model": GROQ_MODEL, "provider": "Groq", "note": "rate_limited"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "model": GROQ_MODEL}), 503


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Body:  { "message": str, "history": [{role, content}, …] }
    Reply: { "reply": str, "risk_level": str, "crisis_resources": str|null }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Missing or empty 'message' field."}), 400

    user_message = data["message"].strip()
    history = data.get("history", [])

    # ── Risk detection (runs before LLM — never bypassable) ───────
    risk_level = detect_risk_level(user_message)

    # ── Script / language hint ────────────────────────────────────
    script_hint = detect_script(user_message)

    # ── Build message list ────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep last 10 turns for context window efficiency
    for turn in history[-10:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    # ── Language cue — always injected ───────────────────────────
    messages.append({
        "role": "system",
        "content": (
            f"LANGUAGE INSTRUCTION: The user's message appears to use "
            f"{script_hint} script. Detect their exact language and reply "
            f"entirely in that language. Do not use English unless they "
            f"wrote in English."
        )
    })

    # ── Risk cue ─────────────────────────────────────────────────
    if risk_level == "high":
        messages.append({
            "role": "system",
            "content": (
                "SAFETY ALERT: The user's message indicates possible "
                "suicidal ideation or self-harm. Begin your response by "
                "providing the crisis hotline for their country/region "
                "prominently, in their language. Be maximally compassionate."
                " Do NOT minimise or question the seriousness."
            )
        })
    elif risk_level == "moderate":
        messages.append({
            "role": "system",
            "content": (
                "NOTE: The user is showing significant emotional distress. "
                "Be extra gentle, validate thoroughly, and mention "
                "professional support options available in their country."
            )
        })

    messages.append({"role": "user", "content": user_message})

    # ── Get LLM reply ─────────────────────────────────────────────
    reply = query_groq(messages)

    # ── Prepend crisis block for high-risk messages ───────────────
    crisis_block = None
    if risk_level == "high":
        crisis_block = CRISIS_RESOURCES
        if "988" not in reply[:200]:          # avoid double-printing
            reply = crisis_block + "\n\n---\n\n" + reply

    return jsonify({
        "reply": reply,
        "risk_level": risk_level,
        "crisis_resources": crisis_block,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Client-side reset confirmation."""
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MindBridge — Mental Health AI Agent")
    print(f"  Provider : Groq  |  Model: {GROQ_MODEL}")
    print("  Server   : http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
