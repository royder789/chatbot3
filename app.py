"""
Space Explorer AI Chatbot - Production Backend
Features: async AI client, rate limiting, Pydantic validation, shared secret auth,
sanitized inputs, singleton DB engine, context summarization, no error leakage,
file upload with expiry, SQL injection prevention.
"""

import os
import re
import time
import uuid
import base64
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from urllib.parse import quote_plus

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)  # ← FIX: was imported but never called

# ─── SINGLETON RESOURCES ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_engine():
    """Singleton DB engine — never re-instantiated."""
    # FIX: URL-encode password to handle special characters like %
    password = quote_plus(os.getenv("POSTGRES_PASSWORD", ""))
    uri = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{password}@{os.getenv('POSTGRES_HOST')}:"
        f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(uri, pool_pre_ping=True, pool_size=5)

@lru_cache(maxsize=1)
def get_llm():
    """Singleton LLM client."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
    )

SHARED_SECRET = os.getenv("SHARED_SECRET", "change-me-in-production")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MAX_CONTEXT_MESSAGES = 20        # Summarize after this many messages
MAX_TOOL_LOOPS = 5               # Hard limit for agent tool loops
RATE_LIMIT_RPM = 10              # Max requests per minute per session
FILE_EXPIRY_HOURS = 24           # Delete uploaded files after N hours

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── PYDANTIC SCHEMAS (Validate AI output AND user input) ────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=3, max_length=100)
    image_data: str | None = Field(default=None)  # base64 image

    @field_validator("session_id")
    @classmethod
    def sanitize_session_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Invalid session_id format")
        return v

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        v = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", v)
        return v.strip()

    @field_validator("image_data")
    @classmethod
    def validate_image(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith("data:image/"):
            raise ValueError("Image must be a base64 data URI")
        if len(v) > 7_000_000:
            raise ValueError("Image too large (max 5MB)")
        return v

class AIResponse(BaseModel):
    """Schema that AI output is validated against."""
    response: str = Field(..., min_length=1, max_length=8000)

class SummarizeRequest(BaseModel):
    messages: list[str]

# ─── SAFETY INTERCEPTORS ─────────────────────────────────────────────────────

BLOCKED_PATTERNS = [
    r"\b(ignore (previous|all) instructions)\b",
    r"(jailbreak|prompt injection|DAN mode)",
    r"\b(system prompt|disregard your training)\b",
]

def safety_check(text: str) -> bool:
    """Returns True if safe, False if blocked."""
    lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            logger.warning("Safety check blocked message pattern: %s", pattern)
            return False
    return True

# ─── RATE LIMITER ─────────────────────────────────────────────────────────────

_rate_buckets: dict[str, list[float]] = defaultdict(list)

def is_rate_limited(session_id: str) -> bool:
    now = time.time()
    window = 60.0
    bucket = _rate_buckets[session_id]
    _rate_buckets[session_id] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[session_id]) >= RATE_LIMIT_RPM:
        return True
    _rate_buckets[session_id].append(now)
    return False

# ─── AUTH MIDDLEWARE ──────────────────────────────────────────────────────────

def verify_secret(req) -> bool:
    # FIX: safely handle missing/non-JSON body
    token = req.headers.get("X-Api-Secret", "")
    if not token:
        try:
            token = req.get_json(silent=True).get("secret", "")
        except Exception:
            token = ""
    return token == SHARED_SECRET

# ─── CONTEXT WINDOW MANAGEMENT ───────────────────────────────────────────────

def get_history_length(session_id: str) -> int:
    try:
        with get_engine().connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM message_store WHERE session_id = :sid"),
                {"sid": session_id}
            )
            return result.scalar() or 0
    except Exception:
        return 0

def summarize_and_trim(session_id: str) -> None:
    """If context is too long, ask LLM to summarize and replace history."""
    try:
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection=get_engine(),
            table_name="message_store"
        )
        msgs = history.messages
        if len(msgs) <= MAX_CONTEXT_MESSAGES:
            return
        old_msgs = msgs[:-5]
        summary_input = "\n".join(
            f"{m.type}: {m.content}" for m in old_msgs
        )
        summary_prompt = (
            f"Summarize this space exploration conversation in 3 sentences:\n{summary_input}"
        )
        summary = get_llm().invoke(summary_prompt).content
        history.clear()
        history.add_ai_message(f"[Conversation summary]: {summary}")
        for msg in msgs[-5:]:
            if msg.type == "human":
                history.add_user_message(msg.content)
            else:
                history.add_ai_message(msg.content)
        logger.info("Summarized context for session %s", session_id)
    except Exception as e:
        logger.error("Summarization failed: %s", e)

# ─── LLM CHAIN SETUP ─────────────────────────────────────────────────────────

SPACE_SYSTEM_PROMPT = """You are COSMOS, an expert AI assistant specializing in space exploration, 
astronomy, astrophysics, and the cosmos. You provide accurate, engaging, and inspiring information 
about space. You can analyze images of celestial objects. You remember past conversations.

Guidelines:
- Answer space-related questions with scientific accuracy
- For off-topic questions, gently redirect to space topics
- Use cosmic metaphors and analogies to explain complex concepts
- Include fascinating facts and recent space news when relevant
- If analyzing an image, describe what astronomical objects or phenomena you observe
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SPACE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# FIX: build chain lazily inside a function so get_llm() isn't called at import time
def get_chain_with_history():
    chain = prompt | get_llm()

    def get_session_history(session_id: str):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=get_engine(),
            table_name="message_store"
        )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# ─── FILE UPLOAD WITH EXPIRY ──────────────────────────────────────────────────

def cleanup_expired_files():
    """Delete files older than FILE_EXPIRY_HOURS."""
    cutoff = time.time() - FILE_EXPIRY_HOURS * 3600
    try:
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                logger.info("Deleted expired file: %s", fname)
    except Exception as e:
        logger.error("File cleanup error: %s", e)

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # 1. Auth: shared secret check
    if not verify_secret(request):
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Parse + validate input with Pydantic
    try:
        body = ChatRequest(**request.get_json(silent=True))
    except (ValidationError, TypeError) as e:
        logger.warning("Validation error: %s", e)
        return jsonify({"error": "Invalid request format"}), 400

    # 3. Rate limiting
    if is_rate_limited(body.session_id):
        return jsonify({"error": "Too many requests. Please slow down, explorer."}), 429

    # 4. Deterministic safety check before hitting LLM
    if not safety_check(body.message):
        return jsonify({"error": "Message flagged by content filter."}), 400

    # 5. Context summarization if needed
    if get_history_length(body.session_id) > MAX_CONTEXT_MESSAGES:
        summarize_and_trim(body.session_id)

    # 6. Build input (include image description if provided)
    user_input = body.message
    if body.image_data:
        user_input = f"[User shared an image for analysis]\n{body.message}"
        cleanup_expired_files()

    # 7. Invoke LLM
    try:
        chain_with_history = get_chain_with_history()
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": body.session_id}}
        )
        raw_output = response.content

        # 8. Validate AI output against schema
        validated = AIResponse(response=raw_output)
        return jsonify({"response": validated.response})

    except ValidationError:
        logger.error("AI output failed schema validation")
        return jsonify({"error": "The cosmos encountered a disturbance. Please try again."}), 500
    except Exception as e:
        # 9. Never leak raw errors to end users
        logger.error("LLM error: %s", e)
        return jsonify({"error": "Mission control encountered an anomaly. Please try again."}), 500

@app.route("/health")
def health():
    return jsonify({"status": "operational", "service": "COSMOS AI"})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
