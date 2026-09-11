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
    # English — direct statements and common phrases
    "suicide", "kill myself", "end my life", "want to die", "take my life",
    "don't want to live", "no reason to live", "better off dead", "self-harm",
    "cut myself", "hurt myself", "overdose", "hanging", "jump off",
    "i want to die", "i wish i were dead", "i can't go on", "end it all",
    "end my pain", "no point living", "not worth living", "planning to die",
    "going to kill", "saying goodbye", "final goodbye", "won't be here",
    "can't live anymore", "done living", "ready to go", "no point anymore",
    "life isn't worth", "nothing to live for", "give up on life",
    # Spanish
    "suicidio", "quitarme la vida", "no quiero vivir", "matarme",
    "hacerme daño", "sin ganas de vivir", "acabar con mi vida",
    # French
    "suicider", "me suicider", "en finir avec", "plus envie de vivre",
    "me tuer", "mettre fin à mes jours",
    # German
    "selbstmord", "suizid", "umbringen", "sterben wollen", "nicht mehr leben",
    # Portuguese
    "suicídio", "me matar", "não quero viver", "tirar minha vida",
    "acabar com tudo", "não vale mais a pena",
    # Hindi
    "आत्महत्या", "खुद को मारना", "जीना नहीं चाहता", "मरना चाहता हूँ",
    # Arabic
    "انتحار", "أريد أن أموت", "إيذاء نفسي", "أريد الموت",
    # Chinese (Simplified)
    "自杀", "不想活", "结束生命", "想死", "去死",
    # Japanese
    "自殺", "死にたい", "消えたい", "もう生きたくない",
    # Korean
    "자살", "죽고 싶다", "살기 싫다", "죽고 싶어",
    # Swahili
    "kujiua", "kutaka kufa", "ninataka kufa",
    # Russian
    "суицид", "покончить с собой", "не хочу жить", "убить себя",
    # Italian
    "farla finita", "non voglio più vivere", "togliermi la vita", "suicidio",
    # Urdu
    "خودکشی", "مرنا چاہتا ہوں", "زندگی ختم کرنا",
    # Turkish
    "intihar", "kendimi öldürmek", "yaşamak istemiyorum",
    # Indonesian / Malay
    "bunuh diri", "ingin mati", "mau mati",
]

MODERATE_RISK_KEYWORDS = [
    # English
    "hopeless", "worthless", "no point", "give up", "can't take it anymore",
    "nobody cares", "all alone", "no way out", "trapped", "empty inside",
    "numb", "burden", "hate myself", "can't cope", "falling apart",
    "exhausted", "breaking down", "can't do this anymore", "what's the point",
    "nobody would miss me", "everyone is better without me", "so tired of everything",
    "don't see a future", "can't feel anything", "lost all hope",
    "feel invisible", "nothing gets better", "so much pain", "unbearable",
    "can't stop crying", "don't want to wake up", "wish i could disappear",
    # Spanish
    "sin esperanza", "no vale la pena", "rendirse", "solo", "agotado",
    "me siento un estorbo", "nadie me quiere",
    # French
    "désespoir", "à bout", "sans espoir", "je n'en peux plus",
    "personne ne se soucie de moi", "je me sens inutile",
    # German
    "hoffnungslos", "wertlos", "erschöpft", "aufgeben", "niemand braucht mich",
    # Portuguese
    "sem esperança", "desistir", "não aguento mais", "me sinto um fardo",
    # Hindi
    "निराशा", "थका हुआ", "अकेला", "बोझ लग रहा हूँ",
    # Arabic
    "يأس", "وحيد", "لا أستطيع", "لا أحد يهتم",
    # Chinese
    "绝望", "没意思", "累了", "没人关心", "活着没意思",
    # Japanese
    "絶望", "疲れた", "誰も気にしない",
    # Korean
    "절망", "지쳤다", "혼자", "아무도 신경 안 써",
    # Russian
    "безнадёжность", "усталость", "одиночество", "никому не нужен",
    # Italian
    "disperazione", "non ce la faccio", "mi sento un peso",
    # Turkish
    "umutsuz", "yalnız", "yoruldum", "kimse umursamıyor",
    # Indonesian / Malay
    "putus asa", "tidak ada harapan", "merasa tidak berguna",
]

# Follow-up safety questions — used as conversation injections after high-risk detection
SAFETY_FOLLOW_UP = (
    "After your compassionate response, gently and directly ask the person: "
    "Do they have a specific plan or method in mind? "
    "Are they in immediate physical danger right now? "
    "Is there anyone physically with them or nearby they could reach out to? "
    "Frame these as caring questions, not an interrogation — something like "
    "'I want to make sure you're safe right now. Can I ask you a few things?' "
    "Their answers will help determine what kind of support they need most urgently."
)

# Global crisis hotlines — shown when high-risk is detected
CRISIS_RESOURCES = """\
🆘 **You matter, and help is available right now. Please reach out:**

- 🇺🇸 **USA** — Call or text **988** · Crisis Text Line: Text HOME to **741741**
- 🇬🇧 **UK** — Samaritans: **116 123** (free, 24/7) · Text SHOUT to **85258**
- 🇮🇳 **India** — iCall: **9152987821** · Vandrevala Foundation: **1860-2662-345**
- 🇦🇺 **Australia** — Lifeline: **13 11 14** · Beyond Blue: **1300 22 4636**
- 🇨🇦 **Canada** — Talk Suicide: **1-833-456-4566** · Crisis Text: Text HOME to **686868**
- 🇿🇦 **South Africa** — SADAG: **0800 567 567** (free, 24/7)
- 🇳🇬 **Nigeria** — NSMHP: **08028585585**
- 🇰🇪 **Kenya** — Befrienders: **+254 722 178 177**
- 🇧🇷 **Brazil** — CVV: **188** (free, 24/7)
- 🇲🇽 **Mexico** — SAPTEL: **55 5259-8121** (free, 24/7)
- 🇩🇪 **Germany** — TelefonSeelsorge: **0800 111 0 111** (free, 24/7)
- 🇫🇷 **France** — Numéro national prévention suicide: **3114** (free, 24/7)
- 🇯🇵 **Japan** — Inochi no Denwa: **0120-783-556** (free, 24/7)
- 🇰🇷 **Korea** — Hope Line: **1393** (free, 24/7)
- 🇨🇳 **China** — Hope 24: **400-161-9995**
- 🇵🇰 **Pakistan** — Umang: **0317-4288665**
- 🇧🇩 **Bangladesh** — Kaan Pete Roi: **01779-554391**
- 🇵🇭 **Philippines** — HOPELINE: **2919** (free, 24/7)
- 🇳🇿 **New Zealand** — Lifeline: **0800 543 354**
- 🌍 **Find your country:** https://www.iasp.info/resources/Crisis_Centres/

You are not alone. A real person is ready to listen right now."""

# ─────────────────────────────────────────────────────────────────
# System Prompt  (Agent Persona)
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are MindBridge, a compassionate and professionally trained "
    "AI mental health support companion. You exist to offer a safe, "
    "non-judgmental space for people to express their pain, feel heard, "
    "and find a path toward help.\n\n"

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

    "CORE APPROACH — follow this structure for most responses:\n\n"

    "1. **Acknowledge first** — Before anything else, reflect back what "
    "you heard. Use phrases like 'It sounds like…', 'I hear that you're…', "
    "'What you're describing sounds really painful.' Never start with advice.\n\n"

    "2. **Validate without conditions** — Make clear their feelings are "
    "real and understandable. Avoid phrases like 'But you have so much to "
    "live for' or 'Things could be worse.' These minimise pain. Instead: "
    "'Your feelings make complete sense given what you've been through.'\n\n"

    "3. **Ask one open question** — Gently explore what's happening. One "
    "question at a time. Never rapid-fire multiple questions. Good examples: "
    "'Can you tell me a little more about what's been going on?', "
    "'How long have you been feeling this way?', "
    "'Is there anything specific that happened recently?'\n\n"

    "4. **Provide psychoeducation when helpful** — Only after listening, "
    "offer evidence-based information about what they're experiencing: "
    "how depression distorts thinking ('The mind in depression can lie to us — "
    "it says things will never change, but that's the illness talking, not reality.'), "
    "how anxiety hijacks the body, grief responses, trauma reactions, etc.\n\n"

    "5. **Share coping tools in a human way** — Don't list techniques robotically. "
    "Offer one at a time and make it feel doable: 'One thing that sometimes helps "
    "people in intense moments is the 5-4-3-2-1 grounding technique — "
    "would you like me to walk you through it right now?'\n\n"

    "6. **Encourage professional help gently** — Frame therapy/counselling as "
    "something for strong people, not a last resort: 'Talking to a therapist "
    "doesn't mean you've failed — it means you're taking yourself seriously. "
    "A lot of people say it's one of the best decisions they ever made.'\n\n"

    "7. **Crisis protocol** — If someone expresses suicidal ideation, "
    "self-harm intent, or severe distress, IMMEDIATELY provide the "
    "crisis hotline for their country/region at the START of your response, "
    "in their language. Then offer compassionate, engaged conversation — "
    "do NOT just list resources and disengage. Stay with them.\n\n"

    "8. **Boundaries** — You are NOT a therapist and do NOT diagnose. "
    "You provide peer-level emotional support and psychoeducation only. "
    "You never provide information on methods of self-harm or suicide.\n\n"

    "9. **Tone** — Warm, empathetic, slow-paced, human. Write like a caring "
    "friend who happens to understand psychology — not like a clinical document. "
    "Use plain language. Short paragraphs. No jargon.\n\n"

    "IMPORTANT RULES:\n"
    "- Never dismiss, minimise, or question the validity of someone's pain.\n"
    "- Never say 'I understand' unless you follow it with showing you understand.\n"
    "- Never give unsolicited advice in the first response — listen first.\n"
    "- Always end with an open-ended question or a gentle, low-pressure invitation "
    "to continue — in the user's own language.\n"
    "- Keep responses to 3–5 focused paragraphs unless more is explicitly needed.\n"
    "- If someone shares something positive, celebrate it genuinely.\n"
    "- If unsure about safety, ALWAYS err toward providing crisis resources.\n\n"

    "SPECIFIC SITUATIONS:\n"
    "- If someone says 'I'm fine' but context suggests otherwise: "
    "'I'm glad to hear that. I'm here if anything ever weighs on you — "
    "even things that feel too small to mention.'\n"
    "- If someone pushes back on help: 'There's no pressure at all. "
    "I'm just here. Take your time.'\n"
    "- If someone is grieving: Lead with the loss, not the recovery. "
    "'I'm so sorry. Losing someone leaves a kind of silence that nothing fills.'\n"
    "- If someone describes abuse or trauma: Believe them. "
    "Validate. Don't interrogate details.\n\n"

    "Remember: You may be the first person — human or AI — that someone has "
    "ever opened up to about their darkest thoughts. Every word you write has weight. "
    "Your goal is not to fix, but to make the person feel less alone, "
    "more understood, and one step closer to getting real support."
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


def detect_mood_keywords(text: str) -> list:
    """Return a list of detected emotional themes for richer context injection."""
    t = text.lower()
    themes = []
    if any(w in t for w in ["anxious", "anxiety", "panic", "worry", "worried", "nervous", "fear", "scared"]):
        themes.append("anxiety")
    if any(w in t for w in ["depressed", "depression", "sad", "crying", "empty", "numb", "hopeless"]):
        themes.append("depression")
    if any(w in t for w in ["lonely", "alone", "isolated", "no friends", "nobody", "invisible"]):
        themes.append("loneliness")
    if any(w in t for w in ["grief", "loss", "died", "death", "passed away", "missing", "bereaved"]):
        themes.append("grief")
    if any(w in t for w in ["trauma", "abuse", "assault", "ptsd", "flashback", "nightmare"]):
        themes.append("trauma")
    if any(w in t for w in ["anger", "angry", "rage", "furious", "frustrated", "hate"]):
        themes.append("anger/frustration")
    if any(w in t for w in ["relationship", "breakup", "divorce", "cheated", "heartbreak", "rejection"]):
        themes.append("relationship pain")
    if any(w in t for w in ["work", "job", "boss", "burnout", "fired", "unemployed", "career"]):
        themes.append("work/career stress")
    if any(w in t for w in ["sleep", "insomnia", "nightmares", "can't sleep", "tired", "exhausted"]):
        themes.append("sleep/exhaustion")
    return themes


def query_groq(messages: list) -> str:
    """Send a chat request to Groq and return the assistant's reply text."""
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.72,
            max_tokens=1200,
            top_p=0.92,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()

    except AuthenticationError:
        return (
            "⚠️ **API Authentication Error.** The Groq API key appears to be invalid or expired. "
            "If you're in crisis right now, please call **988** (US) or text HOME to **741741** immediately. "
            "A real person is ready to talk with you."
        )
    except RateLimitError:
        return (
            "⚠️ **We're receiving a lot of messages right now.** Please wait a moment and try again. "
            "If you're in crisis, please call **988** or text HOME to **741741** — a person will answer immediately."
        )
    except APIConnectionError:
        return (
            "⚠️ **Connection error.** Could not reach the AI service. Please check your internet connection. "
            "If you're in crisis, please call **988** right now — you don't need to be online for that."
        )
    except Exception as e:
        return (
            "⚠️ Something went wrong on our end. If you're in a difficult moment, "
            "please call **988** or text HOME to **741741** — a person is there for you. "
            f"(Detail: {str(e)[:120]})"
        )


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    """Serve the landing / front page."""
    return render_template("landing.html")


@app.route("/app")
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
    Body:  { "message": str, "history": [{role, content}, …], "mood": str|null }
    Reply: { "reply": str, "risk_level": str, "crisis_resources": str|null, "detected_themes": list }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Missing or empty 'message' field."}), 400

    user_message = data["message"].strip()
    history = data.get("history", [])
    mood_score = data.get("mood")          # optional 1-10 mood score from check-in widget

    # ── Risk detection (runs before LLM — never bypassable) ───────
    risk_level = detect_risk_level(user_message)

    # ── Emotional theme detection ─────────────────────────────────
    detected_themes = detect_mood_keywords(user_message)

    # ── Script / language hint ────────────────────────────────────
    script_hint = detect_script(user_message)

    # ── Build message list ────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep last 12 turns for better context continuity
    for turn in history[-12:]:
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

    # ── Mood score cue (if provided) ──────────────────────────────
    if mood_score is not None:
        try:
            score = int(mood_score)
            mood_label = (
                "very low (1-3)" if score <= 3 else
                "low (4-5)" if score <= 5 else
                "moderate (6-7)" if score <= 7 else
                "good (8-10)"
            )
            messages.append({
                "role": "system",
                "content": (
                    f"MOOD CONTEXT: The user rated their current mood as {score}/10 ({mood_label}). "
                    f"Acknowledge this gently and let it inform the depth of your emotional support. "
                    f"A very low score warrants extra care and a direct, warm check-in."
                )
            })
        except (ValueError, TypeError):
            pass

    # ── Emotional theme cue ───────────────────────────────────────
    if detected_themes:
        messages.append({
            "role": "system",
            "content": (
                f"EMOTIONAL CONTEXT: This message appears to contain themes of: "
                f"{', '.join(detected_themes)}. "
                f"Tailor your acknowledgment and any coping tools to these specific themes."
            )
        })

    # ── Risk cue ─────────────────────────────────────────────────
    if risk_level == "high":
        messages.append({
            "role": "system",
            "content": (
                "SAFETY ALERT — HIGHEST PRIORITY: The user's message indicates possible "
                "suicidal ideation or self-harm intent. "
                "Begin your response by providing the crisis hotline for their country/region "
                "prominently in their language. "
                "Then respond with maximum compassion — DO NOT just list resources and stop. "
                "Stay engaged. Reflect their pain. Ask if they are safe right now. "
                "Ask if they have a specific plan. Ask if there is anyone nearby. "
                "Make them feel heard and not alone. "
                "Never minimise, question, or dismiss what they've shared. "
                + SAFETY_FOLLOW_UP
            )
        })
    elif risk_level == "moderate":
        messages.append({
            "role": "system",
            "content": (
                "DISTRESS NOTICE: The user is showing significant emotional distress. "
                "Be especially gentle and slow down. Validate thoroughly before anything else. "
                "Mention that professional support is available and accessible — "
                "frame it warmly, not as a dismissal. "
                "End with a caring, open question that invites them to share more."
            )
        })

    messages.append({"role": "user", "content": user_message})

    # ── Get LLM reply ─────────────────────────────────────────────
    reply = query_groq(messages)

    # ── Prepend crisis block for high-risk messages ───────────────
    crisis_block = None
    if risk_level == "high":
        crisis_block = CRISIS_RESOURCES
        # Avoid double-printing if the LLM already included key hotline numbers
        if "988" not in reply[:300] and "116 123" not in reply[:300]:
            reply = crisis_block + "\n\n---\n\n" + reply

    return jsonify({
        "reply": reply,
        "risk_level": risk_level,
        "crisis_resources": crisis_block,
        "detected_themes": detected_themes,
    })


@app.route("/api/breathe", methods=["GET"])
def breathe():
    """Return a structured breathing exercise for the frontend timer."""
    exercise = {
        "name": "Box Breathing",
        "description": "A simple technique used by therapists and first responders to calm the nervous system in minutes.",
        "phases": [
            {"label": "Breathe In",  "seconds": 4, "instruction": "Inhale slowly through your nose"},
            {"label": "Hold",        "seconds": 4, "instruction": "Hold gently — let your lungs be full"},
            {"label": "Breathe Out", "seconds": 4, "instruction": "Exhale slowly through your mouth"},
            {"label": "Hold",        "seconds": 4, "instruction": "Rest here — empty and still"},
        ],
        "cycles": 5,
        "note": "It's okay if your mind wanders. Just gently come back to the breath."
    }
    return jsonify(exercise)


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
