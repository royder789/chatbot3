# 🍳 CULINA — Intelligent Multi-Agent AI Chef & Kitchen Assistant

> *Your personal AI chef, nutritionist, and kitchen guide — powered by LangGraph, Groq, and FastAPI.*

---

## ✨ Features

| Category | Features |
|---|---|
| **AI Agents** | LangGraph multi-agent: Intent Router → Recipe Generator / Nutrition Analyst / Meal Planner / Technique Expert |
| **Vision** | Upload fridge/dish photos — CULINA identifies ingredients and suggests recipes |
| **Safety** | Big-9 allergen detection BEFORE the LLM, prompt injection blocking |
| **Memory** | Persistent PostgreSQL conversation history per kitchen session |
| **RAG Ready** | Architecture supports recipe knowledge base + nutrition DB injection |
| **Rate Limiting** | 20 req/min via SlowAPI per IP |
| **Context Mgmt** | Auto-summarization when context exceeds 40 messages |
| **Profiles** | Dietary restrictions (Vegetarian, Vegan, Gluten-Free, etc.) + skill level |
| **Recipe Cards** | Save recipes, favorite them, open in step-by-step Cooking Mode |
| **Cooking Mode** | Interactive step navigator for saved recipes |
| **Multi-Session** | "My Kitchens" sidebar for multiple concurrent sessions |

---

## 🗂 Folder Structure

```
culina/
├── app.py                  # FastAPI backend — all routes, agents, safety
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
└── static/
    └── index.html          # Full frontend (single file, no build step)
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Groq API key → [console.groq.com](https://console.groq.com)

### 2. Create PostgreSQL Database

```sql
CREATE DATABASE culina_db;
```

### 3. Clone & Install

```bash
git clone <repo-url>
cd culina
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your values
```

Required variables:
```env
GROQ_API_KEY=gsk_...
SHARED_SECRET=your-secret-here
POSTGRES_PASSWORD=your-db-password
```

### 5. Update Frontend Secret

In `static/index.html`, find this line and set to match `SHARED_SECRET`:
```js
const API_SECRET = "your-secret-here";
```

### 6. Run

```bash
python app.py
# → http://localhost:8000
```

---

## 🔒 Security Architecture

```
Request
  │
  ├─ X-Api-Secret header check           (shared secret auth)
  ├─ Pydantic input validation           (type safety + sanitization)
  ├─ Rate limiting (20 req/min)          (SlowAPI)
  ├─ Prompt injection detection          (regex patterns)
  ├─ Allergen guardrail (BEFORE LLM)     (Big-9 allergen DB)
  │
  └─ LangGraph Agent Pipeline
       ├─ Intent Router
       ├─ Response Generator (Chat or Vision LLM)
       └─ Pydantic output validation
```

---

## 🤖 Agent Architecture

```
User Message
     │
     ▼
[Intent Router] ──────────────────────────────────┐
     │                                             │
     ├─ recipe_generation → 🍳 Recipe Agent        │
     ├─ nutrition_analysis → 📊 Nutrition Agent    │
     ├─ meal_planning → 📅 Planner Agent           │
     ├─ ingredient_sub → 🔄 Substitution Agent     │
     ├─ cooking_technique → 🎓 Technique Agent     │
     ├─ vision_analysis → 👁 Vision Agent (LLaVA) │
     └─ general → 💬 General Chef Agent            │
                                                   │
     ◄─────────────────────────────────────────────┘
     │
[Pydantic Output Validation]
     │
     ▼
[Save to PostgreSQL] → Response to User
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/chat` | POST | Main chat endpoint (text + optional image) |
| `/recipes/{session_id}` | GET | Get saved recipes |
| `/recipes/save` | POST | Save a recipe |
| `/history/{session_id}` | GET | Get conversation history |
| `/history/{session_id}` | DELETE | Clear conversation history |

All endpoints (except `/health`) require `X-Api-Secret` header.

### Chat Request Body
```json
{
  "message": "Give me a pasta recipe",
  "session_id": "KITCHEN-ABC123",
  "image_data": "data:image/jpeg;base64,...",   // optional
  "dietary_restrictions": ["vegan", "gluten-free"],
  "skill_level": "intermediate"
}
```

---

## 🎨 UI Screenshots Description

- **Dark kitchen-themed UI** with warm ember/amber/sage color palette
- **Sidebar** with session list ("My Kitchens"), dietary restriction chips, skill selector
- **Chat interface** with typing indicator, markdown rendering, recipe save buttons
- **Cooking Mode** — step-by-step recipe navigator modal
- **Saved Recipes** tab — card grid with recipe previews
- **Image upload** with live preview for fridge/dish analysis

---

## 🔧 Production Checklist

- [ ] Change `SHARED_SECRET` to a strong random string
- [ ] Update `API_SECRET` in `static/index.html`
- [ ] Set `debug=False` in uvicorn (already done)
- [ ] Put behind nginx + SSL
- [ ] Set up PostgreSQL backups
- [ ] Add `ALLOWED_ORIGINS` for CORS restriction
- [ ] Monitor logs — all errors logged, none leaked to frontend

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| AI Agents | LangGraph + Langchain |
| LLM | Groq (llama-3.3-70b + llama-3.2-90b vision) |
| Database | PostgreSQL + SQLAlchemy |
| Rate Limiting | SlowAPI |
| Validation | Pydantic v2 |
| Frontend | Vanilla JS + CSS (no build step) |
| Fonts | Playfair Display + DM Sans |
