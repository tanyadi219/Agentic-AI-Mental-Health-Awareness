# 🧠 MindBridge — Agentic AI for Mental Health & Suicide Prevention

> A compassionate, context-aware AI companion powered by **openai/gpt-oss-120b** via the **Groq API** — built with Flask, HTML, CSS, and Vanilla JavaScript.

---

## 🆘 Crisis Resources (Always Available)

| Service | Contact |
|---|---|
| 988 Suicide & Crisis Lifeline (US) | Call or text **988** (free, 24/7) |
| Crisis Text Line | Text **HOME** to **741741** |
| International Crisis Centres | https://www.iasp.info/resources/Crisis_Centres/ |
| Emergency Services | **911** (US) or your local number |

---

## 📌 Overview

**MindBridge** is an Agentic AI application designed to:

- Provide **empathetic, non-judgmental** emotional support
- Raise **mental health awareness** through psychoeducation
- Deliver **suicide prevention** resources instantly when risk is detected
- Guide users toward **professional help** and evidence-based coping strategies

The agent uses a multi-layer safety architecture — a keyword-based crisis scanner runs **before** every LLM call, ensuring emergency resources are never delayed by AI reasoning.

---

## 🖼️ Features

| Feature | Details |
|---|---|
| 💬 **AI Chat** | Context-aware multi-turn conversation (last 10 turns) |
| 🚨 **3-Tier Risk Detection** | High / Moderate / Low — keyword scan on every message |
| 🆘 **Auto Crisis Escalation** | High-risk messages instantly surface 988, Crisis Text Line, 911 |
| 🧠 **GPT-OSS 120B** | Open-source 120B model via Groq — fast & high quality |
| 📚 **Resources Panel** | Grounding techniques, breathing exercises, hotlines, communities |
| ℹ️ **About Panel** | Transparent AI model info & disclaimer |
| 📱 **Responsive Design** | Works on desktop and mobile |
| 🔒 **Privacy First** | No conversation history stored server-side |

---

## 🗂️ Project Structure

```
agentic-ai-mental-health/
├── app.py                  ← Flask backend + Groq AI agent logic
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Frontend chat UI (3 panels)
├── static/
│   └── style.css           ← Full responsive stylesheet
└── README.md               ← This file
```

---

## 🤖 Tech Stack

| Layer | Technology |
|---|---|
| **AI Model** | `openai/gpt-oss-120b` (open-source, 120B parameters) |
| **AI Provider** | [Groq API](https://console.groq.com) — ultra-fast inference |
| **Backend** | Python 3 + Flask |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Safety Layer** | Keyword crisis detection + LLM risk prompting |
| **Auth** | Groq API key via `.env` or environment variable |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mindbridge-mental-health-ai.git
cd mindbridge-mental-health-ai
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key

Create a `.env` file in the project root:

```bash
touch .env
```

Add your key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at → https://console.groq.com/keys

### 5. Run the app

```bash
python app.py
```

### 6. Open in browser

```
http://localhost:5001
```

> **Note for macOS users:** If port 5001 is also taken, go to  
> **System Settings → General → AirDrop & Handoff → AirPlay Receiver → Off**  
> or change the port in `app.py` at the bottom: `app.run(port=XXXX)`

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model to use |
| `SECRET_KEY` | `mindbridge-secret-2024` | Flask session secret |

---

## 🛡️ Safety Architecture

MindBridge uses a **defence-in-depth** approach — 5 independent layers:

```
User Message
     │
     ▼
┌─────────────────────────────────┐
│  Layer 1: Keyword Scanner       │  ← Runs BEFORE the LLM, always
│  (HIGH / MODERATE / LOW risk)   │
└────────────────┬────────────────┘
                 │ HIGH risk?
                 ▼
┌─────────────────────────────────┐
│  Layer 2: Silent Risk Cue       │  ← Injected as system message to LLM
│  (SAFETY ALERT prompt)          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Layer 3: System Prompt         │  ← Agent persona with crisis protocol
│  (MindBridge persona)           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Layer 4: Crisis Block Prepend  │  ← 988 / 741741 / 911 prepended to reply
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Layer 5: UI Crisis Banner      │  ← Red bar + flashing sticky banner
└─────────────────────────────────┘
```

---

## 📸 Screenshots

> *(Add screenshots of the Chat, Resources, and About panels here)*

| Chat Panel | Resources Panel | About Panel |
|---|---|---|
| ![chat](screenshots/chat.png) | ![resources](screenshots/resources.png) | ![about](screenshots/about.png) |

---

## 📋 API Reference

### `GET /api/status`
Check if the Groq API is reachable.

**Response:**
```json
{ "ok": true, "model": "openai/gpt-oss-120b", "provider": "Groq" }
```

---

### `POST /api/chat`
Send a message to the AI agent.

**Request body:**
```json
{
  "message": "I've been feeling really anxious lately",
  "history": [
    { "role": "user",      "content": "Hello" },
    { "role": "assistant", "content": "Hi! How are you feeling today?" }
  ]
}
```

**Response:**
```json
{
  "reply": "I hear you, and I'm really glad you're sharing this with me...",
  "risk_level": "moderate",
  "crisis_resources": null
}
```

**`risk_level` values:**

| Value | Meaning |
|---|---|
| `"low"` | No crisis indicators detected |
| `"moderate"` | Distress keywords detected — gentle escalation |
| `"high"` | Crisis keywords detected — emergency resources prepended |

---

### `POST /api/reset`
Reset the conversation (client-side confirmation).

**Response:**
```json
{ "status": "ok" }
```

---

## ⚠️ Disclaimer

**MindBridge is NOT a substitute for professional mental health care.**

- It is not a licensed therapist, psychologist, or medical professional.
- It cannot diagnose conditions or prescribe treatment.
- It is intended for **emotional support and psychoeducation only**.
- Always consult a qualified mental health professional for diagnosis or treatment.

**If you or someone you know is in crisis, call 988 or go to the nearest emergency room immediately.**

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — for ultra-fast open-source LLM inference
- [988 Suicide & Crisis Lifeline](https://988lifeline.org) — for life-saving resources
- [NAMI](https://www.nami.org) — National Alliance on Mental Illness
- [IASP](https://www.iasp.info) — International Association for Suicide Prevention

---

<div align="center">
  <sub>Built with ❤️ to make mental health support more accessible.</sub><br/>
  <sub>If this project helped you or someone you know, please ⭐ star the repo.</sub>
</div>
