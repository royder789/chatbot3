"""
CULINA - Intelligent Multi-Agent AI Chef & Kitchen Assistant
============================================================
v3.0 — Production-Grade Enhancements:
  - Pantry tracker: per-session ingredient memory, injected into prompts
  - Recipe favorites: toggle endpoint + DB column
  - Recipe search: full-text search across saved recipes
  - Nutrition RAG: ground nutrition answers in real food data
  - FAISS vector store fallback (no ChromaDB required)
  - Expanded RAG corpus (recipes + nutrition facts)
  - All prior fixes retained (async, password encoding, allergens, etc.)
"""
 
import os, re, time, uuid, base64, logging, asyncio, json, hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
from typing import Optional, Literal, Annotated
from urllib.parse import quote_plus
 
from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Boolean, func, or_
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
 
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("culina")
 
SHARED_SECRET     = os.getenv("SHARED_SECRET", "change-me")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_CHAT_MODEL   = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
 
POSTGRES_USER     = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "culina_db")
 
MAX_CONTEXT_MESSAGES  = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
MAX_AGENT_LOOPS       = int(os.getenv("MAX_AGENT_LOOPS", "5"))
MAX_MESSAGE_LENGTH    = int(os.getenv("MAX_MESSAGE_LENGTH", "3000"))
MAX_IMAGE_SIZE_BYTES  = int(os.getenv("MAX_IMAGE_SIZE_BYTES", "7_000_000"))
IMAGE_EXPIRY_HOURS    = int(os.getenv("IMAGE_EXPIRY_HOURS", "24"))
RAG_TOP_K             = int(os.getenv("RAG_TOP_K", "3"))
 
# ─── DATABASE ─────────────────────────────────────────────────────────────────
 
Base = declarative_base()
 
class ConversationMessage(Base):
    __tablename__ = "culina_messages"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String(100), nullable=False, index=True)
    role          = Column(String(20), nullable=False)
    content       = Column(Text, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
 
class SavedRecipe(Base):
    __tablename__ = "culina_recipes"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String(100), nullable=False, index=True)
    title         = Column(String(300), nullable=False)
    content       = Column(Text, nullable=False)
    is_favorite   = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
 
class UserProfile(Base):
    __tablename__ = "culina_profiles"
    session_id           = Column(String(100), primary_key=True)
    dietary_restrictions = Column(Text, default="")
    skill_level          = Column(String(20), default="intermediate")
    pantry_items         = Column(Text, default="")          # JSON list of ingredient strings
    updated_at           = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
 
class UploadedImage(Base):
    __tablename__ = "culina_images"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String(100), nullable=False, index=True)
    image_hash   = Column(String(64), nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)
 
@lru_cache(maxsize=1)
def get_engine():
    pw  = quote_plus(POSTGRES_PASSWORD)
    uri = f"postgresql+psycopg2://{POSTGRES_USER}:{pw}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(uri, pool_pre_ping=True, pool_size=5, max_overflow=10)
    Base.metadata.create_all(engine)
    return engine
 
def get_db():
    SessionLocal = sessionmaker(bind=get_engine())
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
# ─── SINGLETON LLMs ───────────────────────────────────────────────────────────
 
@lru_cache(maxsize=1)
def get_chat_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_CHAT_MODEL, temperature=0.7, max_tokens=1500)
 
@lru_cache(maxsize=1)
def get_vision_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_VISION_MODEL, temperature=0.3, max_tokens=1000)
 
# ─── RAG CORPUS ───────────────────────────────────────────────────────────────
# Two corpora: recipes + nutrition facts.
# Both are loaded into a single vector store with a "type" metadata field.
 
RAG_RECIPE_CORPUS = [
    {
        "id": "pasta_carbonara",
        "text": """Classic Spaghetti Carbonara (Intermediate)
Ingredients: 400g spaghetti, 200g pancetta or guanciale, 4 egg yolks, 100g Pecorino Romano, black pepper, salt.
Steps:
1. Boil pasta in salted water until al dente.
2. Fry pancetta in a dry pan until crispy.
3. Whisk egg yolks with grated Pecorino and black pepper.
4. Reserve 1 cup pasta water. Drain pasta.
5. Off heat, add pasta to pancetta, pour egg mixture, toss vigorously with pasta water to emulsify.
Tips: Never add raw cream. The heat from pasta cooks the eggs. Use guanciale for authenticity.""",
        "tags": "pasta,italian,egg,dairy,pork", "type": "recipe"
    },
    {
        "id": "vegan_dal",
        "text": """Red Lentil Dal (Vegan, Gluten-Free, Beginner)
Ingredients: 200g red lentils, 1 onion, 3 garlic cloves, 1 tbsp ginger, 2 tomatoes, 1 tsp cumin, 1 tsp turmeric, 1 tsp garam masala, coconut milk 200ml, salt, oil.
Steps:
1. Rinse lentils. Dice onion, mince garlic and ginger.
2. Sauté onion in oil 5 minutes. Add garlic, ginger, spices — cook 1 minute.
3. Add tomatoes, cook until softened.
4. Add lentils and 600ml water. Simmer 20 minutes until lentils dissolve.
5. Stir in coconut milk. Season with salt.
Tips: Pairs perfectly with basmati rice or naan. Add spinach at the end for iron.""",
        "tags": "vegan,gluten-free,dairy-free,indian,lentils,beginner", "type": "recipe"
    },
    {
        "id": "keto_chicken",
        "text": """Keto Garlic Butter Chicken Thighs (Keto, Gluten-Free, Beginner)
Ingredients: 4 bone-in chicken thighs, 4 tbsp butter, 5 garlic cloves, fresh thyme, salt, pepper, lemon.
Steps:
1. Pat chicken dry. Season generously with salt and pepper.
2. Heat oven-safe skillet over medium-high. Sear chicken skin-down 7 minutes until golden.
3. Flip, add butter, garlic, thyme. Baste constantly 2 minutes.
4. Transfer skillet to 200°C oven for 15 minutes.
5. Rest 5 minutes. Squeeze lemon over.
Macros per serving: ~420 kcal, 35g fat, 28g protein, 2g carbs.""",
        "tags": "keto,gluten-free,chicken,low-carb,beginner", "type": "recipe"
    },
    {
        "id": "gluten_free_banana_bread",
        "text": """Gluten-Free Banana Bread (Vegan Option, Beginner)
Ingredients: 3 ripe bananas, 200g almond flour, 2 eggs (or flax eggs for vegan), 3 tbsp maple syrup, 1 tsp baking soda, 1 tsp vanilla, pinch salt, optional: 80g dark chocolate chips.
Steps:
1. Preheat oven to 180°C. Line a loaf tin with parchment.
2. Mash bananas thoroughly. Mix in eggs, maple syrup, vanilla.
3. Fold in almond flour, baking soda, salt until combined.
4. Fold in chocolate chips if using.
5. Pour into tin. Bake 45-50 minutes until a skewer comes out clean.
Tips: The riper the bananas, the sweeter and moister the bread.""",
        "tags": "gluten-free,vegan-option,baking,banana,beginner,sweet", "type": "recipe"
    },
    {
        "id": "salmon_advanced",
        "text": """Pan-Seared Salmon with Beurre Blanc (Advanced, Pescatarian)
Ingredients: 2 salmon fillets (skin-on), 200ml dry white wine, 2 shallots, 150g cold butter (cubed), lemon juice, capers, dill, salt, white pepper, neutral oil.
Steps:
1. Pat salmon very dry. Season flesh side with salt and white pepper.
2. Heat oil in a heavy pan until shimmering. Place salmon skin-down, press flat for 10 seconds.
3. Cook 4 minutes without moving for crispy skin. Flip, cook 90 seconds. Rest.
4. In a small saucepan, reduce wine with minced shallots to 2 tbsp.
5. On very low heat, whisk in cold butter cube by cube until emulsified.
6. Season beurre blanc with lemon, white pepper. Strain shallots.
7. Plate salmon skin-up, spoon sauce around, garnish with capers and dill.
Tips: Emulsion breaks if pan is too hot. Cold butter in small pieces is the key.""",
        "tags": "fish,advanced,french,pescatarian,dairy,salmon", "type": "recipe"
    },
    {
        "id": "veggie_stir_fry",
        "text": """Quick Vegetable Stir Fry (Vegan, Beginner)
Ingredients: 2 cups mixed vegetables (broccoli, bell pepper, snap peas, carrot), 3 tbsp soy sauce, 1 tbsp sesame oil, 1 tbsp ginger, 3 garlic cloves, 1 tsp cornstarch, chilli flakes, cooked rice.
Steps:
1. Mix soy sauce, sesame oil, cornstarch, and 2 tbsp water into a sauce.
2. Heat wok over high heat until smoking. Add neutral oil.
3. Add garlic and ginger — stir 30 seconds.
4. Add hard vegetables first (carrot, broccoli). Stir fry 3 minutes.
5. Add softer vegetables. Pour sauce. Toss 1-2 minutes.
6. Serve over rice immediately.
Tips: High heat is essential. Don't crowd the wok — cook in batches if needed.""",
        "tags": "vegan,gluten-free-option,asian,quick,beginner,vegetables", "type": "recipe"
    },
    {
        "id": "overnight_oats",
        "text": """Overnight Oats (Vegan, Gluten-Free, Beginner)
Ingredients: 80g rolled oats, 200ml plant milk (oat/almond/soy), 1 tbsp chia seeds, 1 tbsp maple syrup, 1 tsp vanilla, toppings: berries, banana, nut butter, granola.
Steps:
1. Combine oats, milk, chia seeds, maple syrup, vanilla in a jar.
2. Stir well. Seal and refrigerate overnight (minimum 6 hours).
3. In the morning, stir and add splash of milk if too thick.
4. Top with fresh berries, sliced banana, and a drizzle of nut butter.
Nutrition: ~350 kcal, 12g protein, 55g carbs, 8g fiber.
Tips: Prep 5 jars on Sunday for the whole week.""",
        "tags": "vegan,gluten-free,breakfast,meal-prep,beginner,oats,dairy-free", "type": "recipe"
    },
    {
        "id": "butter_chicken",
        "text": """Butter Chicken (Intermediate, Contains Dairy)
Ingredients: 600g chicken breast, 200g yogurt, 1 tbsp garam masala, 1 tsp turmeric, 400ml tomato purée, 200ml heavy cream, 2 onions, 4 garlic cloves, 1 tbsp ginger, 2 tbsp butter, cumin seeds, kashmiri chilli powder.
Steps:
1. Marinate chicken in yogurt, garam masala, turmeric, salt for 2+ hours.
2. Grill or pan-fry marinated chicken until charred. Set aside.
3. Melt butter. Sauté cumin seeds, then onions until golden.
4. Add garlic, ginger, chilli powder. Cook 2 minutes.
5. Add tomato purée. Simmer 15 minutes until oil separates.
6. Blend sauce until smooth. Return to pan.
7. Add chicken, cream, simmer 10 minutes.
Tips: Kashmiri chilli gives the iconic red colour without too much heat.""",
        "tags": "indian,chicken,dairy,intermediate,creamy", "type": "recipe"
    },
    {
        "id": "shakshuka",
        "text": """Shakshuka (Vegetarian, Gluten-Free, Beginner)
Ingredients: 6 eggs, 2 cans crushed tomatoes, 1 onion, 1 red pepper, 4 garlic cloves, 1 tsp cumin, 1 tsp paprika, 0.5 tsp chilli flakes, 100g feta cheese (optional), fresh parsley, olive oil, salt.
Steps:
1. Sauté onion and pepper in olive oil until softened, 7 minutes.
2. Add garlic, cumin, paprika, chilli — cook 1 minute.
3. Pour in tomatoes. Simmer 10 minutes until thickened. Season.
4. Make 6 wells in the sauce. Crack an egg into each.
5. Cover and cook 6-8 minutes until whites are set but yolks are runny.
6. Crumble feta, scatter parsley. Serve with crusty bread.
Tips: Cover with a lid to steam the tops of the eggs faster.""",
        "tags": "vegetarian,gluten-free,egg,mediterranean,beginner,breakfast", "type": "recipe"
    },
    {
        "id": "miso_soup",
        "text": """Authentic Miso Soup (Vegan option, Beginner)
Ingredients: 4 cups dashi stock (or vegetable stock for vegan), 3 tbsp white miso paste, 200g silken tofu, 2 spring onions, 1 sheet nori (optional), 1 tbsp wakame seaweed (rehydrated).
Steps:
1. Heat dashi to just below a simmer — never boil miso.
2. Rehydrate wakame in cold water 5 minutes. Drain.
3. Dissolve miso paste in a ladleful of warm dashi, whisk smooth.
4. Add tofu cubes and wakame to the pot.
5. Pour in miso mixture. Stir gently. Heat 1 minute.
6. Ladle into bowls, top with sliced spring onions.
Tips: Never boil after adding miso — destroys beneficial enzymes and flavour.""",
        "tags": "japanese,vegan-option,beginner,soup,low-calorie,soy", "type": "recipe"
    },
]
 
RAG_NUTRITION_CORPUS = [
    {
        "id": "nut_eggs",
        "text": """Eggs — Nutrition Profile
One large egg (50g): 70 kcal, 6g protein, 5g fat (1.5g saturated), 0g carbs, 186mg cholesterol.
Rich in: choline (brain health), lutein/zeaxanthin (eye health), B12, selenium, vitamin D.
Protein quality: complete protein with all essential amino acids. DIAAS score ~1.13.
Health notes: Dietary cholesterol from eggs has minimal effect on blood cholesterol for most people.
Egg white only: 17 kcal, 3.6g protein, 0g fat — ideal for high-protein low-fat goals.""",
        "tags": "protein,egg,micronutrient,cholesterol", "type": "nutrition"
    },
    {
        "id": "nut_salmon",
        "text": """Salmon — Nutrition Profile
100g cooked Atlantic salmon: 208 kcal, 20g protein, 13g fat (3g saturated), 0g carbs.
Omega-3 fatty acids: 2.3g EPA+DHA per 100g — among the highest of any food.
Rich in: B12 (>100% DV), selenium, vitamin D, potassium, niacin.
Health benefits: Omega-3s reduce triglycerides, support brain function, reduce inflammation.
Wild vs farmed: Wild salmon has slightly more omega-3s. Farmed has more calories.""",
        "tags": "omega3,fish,protein,heart-health,vitamin-d", "type": "nutrition"
    },
    {
        "id": "nut_avocado",
        "text": """Avocado — Nutrition Profile
100g avocado: 160 kcal, 2g protein, 15g fat (2.1g saturated, 10g monounsaturated), 9g carbs, 7g fiber.
Rich in: potassium (more than banana), folate, K1, vitamin C, B5, B6.
Monounsaturated fat (oleic acid): same as olive oil — heart protective, reduces LDL.
Fiber: 7g per 100g supports gut health and satiety.
Tips: The darker the skin, the riper. Store unripe at room temp; ripe in the fridge.""",
        "tags": "healthy-fat,fiber,potassium,vegan,keto-friendly", "type": "nutrition"
    },
    {
        "id": "nut_lentils",
        "text": """Red Lentils — Nutrition Profile
100g cooked lentils: 116 kcal, 9g protein, 0.4g fat, 20g carbs, 8g fiber.
Rich in: folate (45% DV), iron (17% DV), manganese, B1, phosphorus.
Protein: plant-based complete protein when combined with rice or bread.
Glycemic index: Low (~29) — ideal for blood sugar management.
Iron note: Non-heme iron — pair with vitamin C foods (tomatoes, lemon) to boost absorption by 3x.""",
        "tags": "vegan,protein,fiber,iron,low-gi,gluten-free", "type": "nutrition"
    },
    {
        "id": "nut_oats",
        "text": """Rolled Oats — Nutrition Profile
100g dry oats: 389 kcal, 17g protein, 7g fat, 66g carbs, 11g fiber.
Beta-glucan fiber: 3-4g per serving — clinically proven to lower LDL cholesterol.
Glycemic index: ~55 (medium) — but the beta-glucan slows glucose absorption.
Rich in: manganese (191% DV), phosphorus, magnesium, B1, iron, zinc.
Tips: Steel-cut oats have a lower GI than rolled oats. Instant oats spike blood sugar fastest.""",
        "tags": "fiber,beta-glucan,cholesterol,vegan,gluten-free-if-certified", "type": "nutrition"
    },
    {
        "id": "nut_chicken_breast",
        "text": """Chicken Breast — Nutrition Profile
100g cooked skinless chicken breast: 165 kcal, 31g protein, 3.6g fat, 0g carbs.
Amino acid profile: complete protein, especially high in leucine (triggers muscle protein synthesis).
Rich in: B3 (niacin, 59% DV), B6, phosphorus, selenium.
Satiety: High protein content makes it one of the most satiating foods per calorie.
Cooking tip: Don't overcook — internal temp of 74°C (165°F). Brining keeps it juicy.""",
        "tags": "protein,lean,low-fat,keto,muscle-building,gluten-free", "type": "nutrition"
    },
    {
        "id": "nut_broccoli",
        "text": """Broccoli — Nutrition Profile
100g raw broccoli: 34 kcal, 2.8g protein, 0.4g fat, 7g carbs, 2.6g fiber.
Vitamin C: 89mg per 100g (99% DV) — more than an orange by weight.
Rich in: K1 (97% DV), folate, B6, potassium, manganese.
Sulforaphane: powerful antioxidant compound — strongest when broccoli is eaten raw or lightly steamed.
Tips: Don't overcook — steam 3-4 minutes max. Boiling removes 50% of vitamins into the water.""",
        "tags": "vegan,low-calorie,vitamin-c,cancer-fighting,keto-friendly", "type": "nutrition"
    },
    {
        "id": "nut_greek_yogurt",
        "text": """Greek Yogurt (Full-Fat) — Nutrition Profile
100g full-fat Greek yogurt: 97 kcal, 9g protein, 5g fat, 4g carbs.
Protein: ~2x the protein of regular yogurt due to straining.
Probiotics: Lactobacillus and Bifidobacterium strains — supports gut microbiome.
Rich in: calcium (10% DV), B12, phosphorus, iodine, riboflavin.
Satiety: High protein + fat combination is extremely satiating.
Tips: Choose plain yogurt — flavoured versions often contain 20g+ sugar per serving.""",
        "tags": "protein,probiotic,calcium,dairy,keto-friendly", "type": "nutrition"
    },
    {
        "id": "nut_olive_oil",
        "text": """Extra Virgin Olive Oil — Nutrition Profile
1 tablespoon (14g): 119 kcal, 0g protein, 14g fat (2g saturated, 10g monounsaturated), 0g carbs.
Oleocanthal: anti-inflammatory compound equivalent to low-dose ibuprofen.
Polyphenols: powerful antioxidants — degrade with heat. Best benefits when used raw/cold.
Smoke point: ~190°C (375°F) — fine for sautéing, not ideal for deep frying.
Mediterranean diet: cornerstone food linked to reduced cardiovascular disease by 30%.""",
        "tags": "healthy-fat,anti-inflammatory,vegan,keto,mediterranean", "type": "nutrition"
    },
    {
        "id": "nut_bananas",
        "text": """Bananas — Nutrition Profile
1 medium banana (118g): 105 kcal, 1.3g protein, 0.4g fat, 27g carbs, 3.1g fiber.
Potassium: 422mg (12% DV) — key for heart health and muscle function.
Resistant starch: green bananas contain high resistant starch (prebiotic) that feeds gut bacteria.
Glycemic index: 51 (medium) — rises as banana ripens.
Rich in: B6 (22% DV), vitamin C, manganese.
Sport nutrition: ideal pre-workout carb — fast energy + potassium to prevent cramps.""",
        "tags": "potassium,fiber,vegan,energy,pre-workout", "type": "nutrition"
    },
]
 
# Combined corpus for a single vector store
ALL_RAG_CORPUS = RAG_RECIPE_CORPUS + RAG_NUTRITION_CORPUS
 
# ─── VECTOR STORE: ChromaDB → FAISS fallback ─────────────────────────────────
 
_vector_store = None
_rag_backend  = None   # "chroma" | "faiss" | None
 
def _build_faiss_store(corpus):
    """
    Pure-Python FAISS fallback using TF-IDF + cosine similarity.
    Requires no extra packages beyond what's already installed (uses numpy + sklearn).
    Returns a dict with keys: docs, metas, vectorizer, matrix.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
 
        docs   = [c["text"]  for c in corpus]
        metas  = [{"tags": c["tags"], "type": c.get("type","recipe")} for c in corpus]
        vect   = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
        matrix = vect.fit_transform(docs)
        logger.info("FAISS-lite (TF-IDF) store built with %d documents.", len(docs))
        return {"docs": docs, "metas": metas, "vectorizer": vect, "matrix": matrix}
    except Exception as e:
        logger.error("FAISS-lite build failed: %s", e)
        return None
 
def _faiss_query(store, query, n):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    q_vec  = store["vectorizer"].transform([query])
    scores = cosine_similarity(q_vec, store["matrix"])[0]
    top_n  = scores.argsort()[::-1][:n]
    return [(store["docs"][i], store["metas"][i], float(scores[i])) for i in top_n]
 
 
def get_vector_store():
    global _vector_store, _rag_backend
    if _vector_store is not None:
        return _vector_store
 
    # ── Try ChromaDB first ────────────────────────────────────────
    try:
        import chromadb
        from chromadb.utils import embedding_functions
 
        chroma_client = chromadb.PersistentClient(path="./culina_chroma_db")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = chroma_client.get_or_create_collection(
            name="culina_v3",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        if collection.count() == 0:
            logger.info("Seeding ChromaDB with %d documents…", len(ALL_RAG_CORPUS))
            collection.add(
                ids=[c["id"] for c in ALL_RAG_CORPUS],
                documents=[c["text"] for c in ALL_RAG_CORPUS],
                metadatas=[{"tags": c["tags"], "type": c.get("type","recipe")} for c in ALL_RAG_CORPUS],
            )
        _vector_store = collection
        _rag_backend  = "chroma"
        logger.info("ChromaDB RAG ready (%d docs).", collection.count())
        return _vector_store
    except ImportError:
        logger.warning("ChromaDB not installed — falling back to TF-IDF RAG.")
    except Exception as e:
        logger.warning("ChromaDB failed (%s) — falling back to TF-IDF RAG.", e)
 
    # ── FAISS-lite (TF-IDF) fallback ─────────────────────────────
    store = _build_faiss_store(ALL_RAG_CORPUS)
    if store:
        _vector_store = store
        _rag_backend  = "faiss"
        return _vector_store
 
    logger.error("All RAG backends failed. RAG disabled.")
    return None
 
 
def rag_retrieve(query: str, dietary_restrictions: list, top_k: int = RAG_TOP_K,
                 doc_type: str = "recipe") -> str:
    """
    Retrieve top-k relevant documents from the vector store.
    doc_type: "recipe" | "nutrition" | "any"
    Filters out results conflicting with dietary restrictions.
    """
    store = get_vector_store()
    if store is None:
        return ""
 
    restriction_tag_map = {
        "vegan":       ["dairy", "egg", "pork", "chicken", "fish", "meat"],
        "vegetarian":  ["pork", "chicken", "fish", "meat"],
        "dairy-free":  ["dairy", "cream", "butter", "cheese"],
        "gluten-free": ["gluten", "wheat", "flour", "pasta"],
        "nut-free":    ["almond", "walnut", "cashew", "pecan", "nut"],
        "keto":        ["sugar", "oats", "rice", "bread", "banana"],
    }
    blocked_tags = set()
    for r in dietary_restrictions:
        blocked_tags.update(restriction_tag_map.get(r.lower(), []))
 
    try:
        if _rag_backend == "chroma":
            where = None
            if doc_type != "any":
                where = {"type": doc_type}
            results = store.query(
                query_texts=[query],
                n_results=min(top_k + 5, store.count()),
                where=where,
            )
            raw_docs  = results.get("documents", [[]])[0]
            raw_metas = results.get("metadatas",  [[]])[0]
            candidates = list(zip(raw_docs, raw_metas))
 
        elif _rag_backend == "faiss":
            hits = _faiss_query(store, query, top_k + 8)
            candidates = []
            for doc, meta, score in hits:
                if doc_type != "any" and meta.get("type") != doc_type:
                    continue
                candidates.append((doc, meta))
 
        else:
            return ""
 
        filtered = []
        for doc, meta in candidates:
            recipe_tags = set(meta.get("tags", "").split(","))
            if not recipe_tags.intersection(blocked_tags):
                filtered.append(doc)
            if len(filtered) >= top_k:
                break
 
        if not filtered:
            return ""
 
        context = "\n\n---\n\n".join(filtered)
        label   = "nutritional reference data" if doc_type == "nutrition" else "recipe knowledge base"
        return f"\n## Relevant {label} (CULINA's RAG context):\n{context}\n\nUse the above as authoritative reference. Adapt to the user's restrictions and skill level.\n"
 
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return ""
 
# ─── PANTRY HELPER ────────────────────────────────────────────────────────────
 
def get_pantry(session_id: str) -> list:
    try:
        SessionLocal = sessionmaker(bind=get_engine())
        db = SessionLocal()
        profile = db.query(UserProfile).filter_by(session_id=session_id).first()
        db.close()
        if profile and profile.pantry_items:
            return json.loads(profile.pantry_items)
    except Exception as e:
        logger.error("Pantry load failed: %s", e)
    return []
 
def set_pantry(session_id: str, items: list):
    try:
        SessionLocal = sessionmaker(bind=get_engine())
        db = SessionLocal()
        profile = db.query(UserProfile).filter_by(session_id=session_id).first()
        if not profile:
            profile = UserProfile(session_id=session_id)
            db.add(profile)
        profile.pantry_items = json.dumps(items)
        db.commit()
        db.close()
    except Exception as e:
        logger.error("Pantry save failed: %s", e)
 
# ─── PYDANTIC SCHEMAS ─────────────────────────────────────────────────────────
 
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str = Field(..., min_length=3, max_length=100)
    image_data: Optional[str] = Field(default=None)
    dietary_restrictions: list[str] = Field(default_factory=list)
    skill_level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
 
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Invalid session_id")
        return v
 
    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v)
        return v.strip()
 
    @field_validator("image_data")
    @classmethod
    def validate_image(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.startswith("data:image/"):
            raise ValueError("Image must be base64 data URI")
        if len(v) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError("Image exceeds 7MB limit")
        return v
 
    @field_validator("dietary_restrictions")
    @classmethod
    def validate_dietary(cls, v: list[str]) -> list[str]:
        allowed = {"vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free", "keto", "halal", "kosher"}
        return [d.lower().strip() for d in v if d.lower().strip() in allowed]
 
class SaveRecipeRequest(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=100)
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=10000)
 
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Invalid session_id")
        return v
 
class PantryRequest(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=100)
    items: list[str] = Field(..., max_length=100)
 
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Invalid session_id")
        return v
 
    @field_validator("items")
    @classmethod
    def sanitize_items(cls, v: list[str]) -> list[str]:
        cleaned = []
        for item in v:
            item = re.sub(r"[<>\"']", "", item.strip())[:80]
            if item:
                cleaned.append(item)
        return cleaned[:100]
 
class AIResponse(BaseModel):
    response: str = Field(..., min_length=1, max_length=12000)
    intent: str = Field(default="general")
    recipe_detected: bool = Field(default=False)
 
# ─── SAFETY GUARDRAILS ────────────────────────────────────────────────────────
 
ALLERGEN_KEYWORDS = {
    "peanut":    ["peanut", "groundnut", "arachis"],
    "treenut":   ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "macadamia", "brazil nut"],
    "milk":      ["milk", "dairy", "cheese", "butter", "cream", "lactose", "whey", "casein", "yogurt"],
    "egg":       ["egg", "mayonnaise", "albumin", "lecithin"],
    "wheat":     ["wheat", "flour", "gluten", "bread", "pasta", "semolina", "spelt", "durum"],
    "soy":       ["soy", "soya", "tofu", "tempeh", "edamame", "miso"],
    "fish":      ["fish", "cod", "salmon", "tuna", "tilapia", "bass", "flounder", "anchovy"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "clam", "oyster", "mussel", "scallop"],
    "sesame":    ["sesame", "tahini"],
}
 
DIETARY_BANNED_KEYWORDS = {
    "vegan":       ["meat", "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp",
                    "egg", "eggs", "milk", "dairy", "cheese", "butter", "cream", "honey",
                    "gelatin", "lard", "bacon", "ham"],
    "vegetarian":  ["meat", "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp",
                    "bacon", "ham", "lard", "gelatin"],
    "dairy-free":  ["milk", "cheese", "butter", "cream", "yogurt", "whey", "casein",
                    "lactose", "ghee"],
    "gluten-free": ["wheat", "flour", "bread", "pasta", "rye", "barley", "semolina",
                    "spelt", "couscous", "soy sauce"],
    "nut-free":    ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut",
                    "peanut", "nut butter", "marzipan"],
    "keto":        ["sugar", "honey", "maple syrup", "oats", "rice", "bread", "pasta",
                    "potato", "corn", "beans", "lentils"],
}
 
BLOCKED_PATTERNS = [
    r"\b(ignore (previous|all) instructions)\b",
    r"(jailbreak|prompt injection|DAN mode)",
    r"\b(system prompt|disregard your training)\b",
    r"\b(you are now|act as if you are|pretend to be)\b",
]
 
def detect_allergens(text: str, user_restrictions: list) -> list:
    lower = text.lower()
    found = []
    for restriction in user_restrictions:
        restriction_lower = restriction.lower()
        for allergen, keywords in ALLERGEN_KEYWORDS.items():
            if restriction_lower in allergen or allergen in restriction_lower:
                for kw in keywords:
                    if kw in lower:
                        found.append(allergen)
                        break
    return list(set(found))
 
def safety_check(text: str) -> bool:
    lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            logger.warning("Safety block triggered: %s", pattern)
            return False
    return True
 
# ─── LANGGRAPH STATE ──────────────────────────────────────────────────────────
 
class CulinaState(TypedDict):
    messages: list
    intent: str
    user_message: str
    image_data: Optional[str]
    dietary_restrictions: list
    skill_level: str
    pantry_items: list
    final_response: str
    recipe_detected: bool
    loop_count: int
 
# ─── INTENT ROUTER ────────────────────────────────────────────────────────────
 
INTENT_SYSTEM = """You are CULINA's intent router. Classify the user's message into EXACTLY one of:
- recipe_generation  (wants a recipe, "how do I make", "recipe for", "what can I cook with")
- nutrition_analysis (asking about calories, macros, health, diet info, vitamins, protein content)
- meal_planning      (wants a weekly plan, meal prep, batch cooking)
- ingredient_sub     (ingredient substitution, "what can I use instead of")
- cooking_technique  (how to chop, braise, reduce, sear, etc.)
- pantry_management  (adding/removing ingredients from their pantry/fridge, "I have X in my fridge")
- vision_analysis    (has uploaded an image, describing fridge/dish)
- general            (anything else food/kitchen related)
 
Respond ONLY with the intent string, nothing else."""
 
def route_intent(state: CulinaState) -> CulinaState:
    try:
        if state.get("image_data"):
            state["intent"] = "vision_analysis"
            return state
        llm = get_chat_llm()
        result = llm.invoke([
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=state["user_message"])
        ])
        intent = result.content.strip().lower()
        valid_intents = ["recipe_generation", "nutrition_analysis", "meal_planning",
                         "ingredient_sub", "cooking_technique", "vision_analysis",
                         "pantry_management", "general"]
        state["intent"] = intent if intent in valid_intents else "general"
    except Exception as e:
        logger.error("Intent routing failed: %s", e)
        state["intent"] = "general"
    return state
 
# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
 
def build_system_prompt(intent: str, dietary: list, skill: str,
                        rag_context: str = "", pantry_items: list = None) -> str:
    if dietary:
        banned_items = set()
        for d in dietary:
            banned_items.update(DIETARY_BANNED_KEYWORDS.get(d.lower(), []))
        restriction_lines = (
            f"STRICT DIETARY RESTRICTIONS — the user follows: {', '.join(d.upper() for d in dietary)}.\n"
            f"You MUST NEVER include these ingredients: {', '.join(sorted(banned_items))}.\n"
            f"If a classic recipe contains these, substitute or adapt it. Always confirm the recipe is safe."
        )
    else:
        restriction_lines = "No specific dietary restrictions."
 
    pantry_block = ""
    if pantry_items:
        pantry_block = (
            f"\n\n## User's Current Pantry / Fridge:\n"
            f"{', '.join(pantry_items)}\n"
            f"Prioritize recipes and suggestions that use these ingredients. "
            f"Flag which pantry items are used in each recipe."
        )
 
    skill_map = {
        "beginner":     "Explain EVERYTHING step by step. Define cooking terms. Include tips for common beginner mistakes. Keep techniques simple.",
        "intermediate": "Assume basic cooking knowledge. Include helpful technique tips and why steps matter.",
        "advanced":     "Skip basic explanations. Include advanced techniques, flavour science, plating suggestions, and professional tips.",
    }
    skill_note = skill_map.get(skill, skill_map["intermediate"])
 
    intent_addons = {
        "recipe_generation": "\nYou are in RECIPE mode. Provide complete recipes with exact ingredient quantities, timings, and techniques. Format with ## Ingredients, ## Steps, ## Tips.",
        "nutrition_analysis": "\nYou are in NUTRITION mode. Give accurate nutritional breakdowns, health context, and dietary insights with specific numbers. Cite sources from the RAG context if available.",
        "meal_planning":      "\nYou are in MEAL PLAN mode. Create structured weekly plans with variety, nutritional balance, and a shopping list.",
        "ingredient_sub":     "\nYou are in SUBSTITUTION mode. Explain WHY the substitution works, the ratio to use, and how it affects flavour and texture.",
        "cooking_technique":  "\nYou are in TECHNIQUE mode. Explain the science behind it, common errors, and how to master it.",
        "vision_analysis":    "\nYou are in VISION mode. Analyze what you see, identify ingredients, suggest recipes and improvements.",
        "pantry_management":  "\nYou are in PANTRY mode. Help the user manage their ingredients. Confirm what was added or removed. Suggest recipes that use their current pantry.",
        "general":            "\nBe a warm, knowledgeable kitchen companion.",
    }
 
    rag_block = f"\n{rag_context}" if rag_context else ""
 
    return f"""You are CULINA, an expert AI chef, nutritionist, and kitchen assistant. You are warm, encouraging, and precise.
 
{restriction_lines}{pantry_block}
 
Skill level: {skill.upper()} — {skill_note}
 
Always format recipes clearly. Never suggest ingredients that conflict with dietary restrictions.
{rag_block}{intent_addons.get(intent, "")}"""
 
 
# ─── RESPONSE GENERATOR ───────────────────────────────────────────────────────
 
async def generate_response(state: CulinaState, chat_history: list) -> CulinaState:
    try:
        rag_context = ""
        intent = state["intent"]
 
        # Recipe intents → recipe RAG
        if intent in ("recipe_generation", "meal_planning", "ingredient_sub"):
            rag_context = rag_retrieve(state["user_message"], state["dietary_restrictions"],
                                       doc_type="recipe")
        # Nutrition intent → nutrition RAG
        elif intent == "nutrition_analysis":
            rag_context = rag_retrieve(state["user_message"], state["dietary_restrictions"],
                                       doc_type="nutrition")
 
        if rag_context:
            logger.info("RAG context injected (%d chars, intent=%s)", len(rag_context), intent)
 
        system_prompt = build_system_prompt(
            intent,
            state["dietary_restrictions"],
            state["skill_level"],
            rag_context,
            state.get("pantry_items", [])
        )
        messages_to_send = [SystemMessage(content=system_prompt)]
        for msg in chat_history[-MAX_CONTEXT_MESSAGES:]:
            messages_to_send.append(msg)
 
        if state.get("image_data") and intent == "vision_analysis":
            llm = get_vision_llm()
            data_uri: str = state["image_data"]
            try:
                header, b64_data = data_uri.split(",", 1)
                media_type = header.split(":")[1].split(";")[0]
            except Exception:
                media_type = "image/jpeg"
                b64_data = data_uri
            user_msg = HumanMessage(content=[
                {"type": "text", "text": state["user_message"] or "Analyze this image and suggest what I can cook."},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_data}"}},
            ])
        else:
            llm = get_chat_llm()
            user_msg = HumanMessage(content=state["user_message"])
 
        messages_to_send.append(user_msg)
        result = await llm.ainvoke(messages_to_send)
        raw = result.content
 
        recipe_signals = ["## ingredients", "**ingredients**", "### ingredients", "1.", "- 1 cup", "- 2 "]
        state["recipe_detected"] = any(sig in raw.lower() for sig in recipe_signals)
        state["final_response"] = raw
 
    except Exception as e:
        logger.error("Response generation failed: %s", e)
        state["final_response"] = "I encountered a kitchen mishap! Please try again in a moment. 🍳"
        state["recipe_detected"] = False
 
    return state
 
def check_loop(state: CulinaState) -> str:
    if state["loop_count"] >= MAX_AGENT_LOOPS:
        return END
    return "generate"
 
# ─── BUILD LANGGRAPH ──────────────────────────────────────────────────────────
 
def build_graph():
    graph = StateGraph(CulinaState)
    graph.add_node("route", route_intent)
    graph.set_entry_point("route")
    graph.add_edge("route", END)
    return graph.compile()
 
culina_graph = build_graph()
 
# ─── CONTEXT MANAGEMENT ───────────────────────────────────────────────────────
 
def get_chat_history(session_id: str) -> list:
    try:
        SessionLocal = sessionmaker(bind=get_engine())
        db = SessionLocal()
        rows = (db.query(ConversationMessage)
                .filter_by(session_id=session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(MAX_CONTEXT_MESSAGES)
                .all())
        db.close()
        msgs = []
        for row in reversed(rows):
            if row.role == "user":
                msgs.append(HumanMessage(content=row.content))
            else:
                msgs.append(AIMessage(content=row.content))
        return msgs
    except Exception as e:
        logger.error("Failed to load chat history: %s", e)
        return []
 
def save_messages(session_id: str, user_msg: str, ai_msg: str):
    try:
        SessionLocal = sessionmaker(bind=get_engine())
        db = SessionLocal()
        db.add(ConversationMessage(session_id=session_id, role="user", content=user_msg))
        db.add(ConversationMessage(session_id=session_id, role="assistant", content=ai_msg))
        db.commit()
        db.close()
    except Exception as e:
        logger.error("Failed to save messages: %s", e)
 
def maybe_summarize(session_id: str):
    try:
        SessionLocal = sessionmaker(bind=get_engine())
        db = SessionLocal()
        count = db.query(func.count(ConversationMessage.id)).filter_by(session_id=session_id).scalar()
        if count and count > MAX_CONTEXT_MESSAGES * 2:
            old_rows = (db.query(ConversationMessage)
                        .filter_by(session_id=session_id)
                        .order_by(ConversationMessage.created_at)
                        .limit(20).all())
            text_block = "\n".join(f"{r.role}: {r.content[:200]}" for r in old_rows)
            summary_prompt = f"Summarize this cooking conversation in 2 sentences:\n{text_block}"
            summary = get_chat_llm().invoke([HumanMessage(content=summary_prompt)]).content
            for row in old_rows:
                db.delete(row)
            db.add(ConversationMessage(
                session_id=session_id, role="assistant",
                content=f"[Earlier conversation summary]: {summary}"
            ))
            db.commit()
            logger.info("Summarized context for session %s", session_id)
        db.close()
    except Exception as e:
        logger.error("Summarization failed: %s", e)
 
def cleanup_expired_images():
    try:
        SessionLocal = sessionmaker(bind=get_engine())
        db = SessionLocal()
        cutoff = datetime.utcnow() - timedelta(hours=IMAGE_EXPIRY_HOURS)
        deleted = db.query(UploadedImage).filter(UploadedImage.created_at < cutoff).delete()
        db.commit()
        db.close()
        if deleted:
            logger.info("Cleaned up %d expired image records.", deleted)
    except Exception as e:
        logger.error("Image cleanup failed: %s", e)
 
# ─── RATE LIMITING ────────────────────────────────────────────────────────────
 
limiter = Limiter(key_func=get_remote_address)
 
# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
 
app = FastAPI(title="CULINA API", version="3.0.0", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
 
@app.on_event("startup")
async def startup_event():
    try:
        get_engine()
        get_vector_store()
        cleanup_expired_images()
        logger.info("CULINA v3 startup complete. RAG backend: %s", _rag_backend)
    except Exception as e:
        logger.error("Startup error: %s", e)
 
# ─── AUTH ─────────────────────────────────────────────────────────────────────
 
def verify_auth(request: Request):
    token = request.headers.get("X-Api-Secret", "")
    if token != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
 
# ─── ROUTES ───────────────────────────────────────────────────────────────────
 
@app.get("/health")
async def health():
    return {
        "status": "online",
        "service": "CULINA AI",
        "version": "3.0.0",
        "rag_backend": _rag_backend or "disabled",
        "features": ["pantry", "favorites", "recipe_search", "nutrition_rag", "faiss_fallback"]
    }
 
@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, _auth=Depends(verify_auth)):
    try:
        body_raw = await request.json()
        body = ChatRequest(**body_raw)
    except (ValidationError, Exception) as e:
        logger.warning("Invalid request: %s", e)
        raise HTTPException(status_code=400, detail="Invalid request format")
 
    if not safety_check(body.message):
        raise HTTPException(status_code=400, detail="Message flagged by content filter.")
 
    allergens_found = detect_allergens(body.message, body.dietary_restrictions)
    if allergens_found:
        warning = (f"⚠️ **Allergy Alert**: Your message mentions ingredients that conflict with your "
                   f"dietary restrictions ({', '.join(allergens_found)}). "
                   f"I'll only suggest safe alternatives for you!")
        save_messages(body.session_id, body.message, warning)
        return JSONResponse({"response": warning, "intent": "safety", "recipe_detected": False})
 
    maybe_summarize(body.session_id)
    history = get_chat_history(body.session_id)
    pantry  = get_pantry(body.session_id)
 
    state: CulinaState = {
        "messages": history,
        "intent": "general",
        "user_message": body.message,
        "image_data": body.image_data,
        "dietary_restrictions": body.dietary_restrictions,
        "skill_level": body.skill_level,
        "pantry_items": pantry,
        "final_response": "",
        "recipe_detected": False,
        "loop_count": 0,
    }
 
    try:
        state = culina_graph.invoke(state)
    except Exception as e:
        logger.error("Graph error: %s", e)
 
    state = await generate_response(state, history)
 
    try:
        validated = AIResponse(
            response=state["final_response"],
            intent=state["intent"],
            recipe_detected=state["recipe_detected"]
        )
    except ValidationError:
        logger.error("Output validation failed")
        raise HTTPException(status_code=500, detail="Something went wrong in the kitchen. Please try again.")
 
    save_messages(body.session_id, body.message, validated.response)
 
    return JSONResponse({
        "response": validated.response,
        "intent": validated.intent,
        "recipe_detected": validated.recipe_detected
    })
 
# ── RECIPES ──────────────────────────────────────────────────────────────────
 
@app.post("/recipes/save")
@limiter.limit("30/minute")
async def save_recipe(request: Request, db: Session = Depends(get_db), _auth=Depends(verify_auth)):
    try:
        body_raw = await request.json()
        body = SaveRecipeRequest(**body_raw)
    except (ValidationError, Exception) as e:
        logger.warning("Save recipe validation error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid request format")
    try:
        recipe = SavedRecipe(session_id=body.session_id, title=body.title, content=body.content)
        db.add(recipe)
        db.commit()
        db.refresh(recipe)
        return JSONResponse({"id": recipe.id, "message": "Recipe saved!"})
    except Exception as e:
        logger.error("Save recipe failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not save recipe.")
 
@app.get("/recipes/{session_id}")
@limiter.limit("30/minute")
async def get_recipes(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(verify_auth),
    search: Optional[str] = Query(default=None, max_length=200),
    favorites_only: bool = Query(default=False),
):
    """
    Get saved recipes for a session.
    Optional ?search= for full-text search across title + content.
    Optional ?favorites_only=true to filter only favorited recipes.
    """
    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    try:
        q = db.query(SavedRecipe).filter_by(session_id=session_id)
        if favorites_only:
            q = q.filter_by(is_favorite=True)
        if search:
            term = f"%{search.lower()}%"
            q = q.filter(
                or_(
                    func.lower(SavedRecipe.title).like(term),
                    func.lower(SavedRecipe.content).like(term),
                )
            )
        recipes = q.order_by(SavedRecipe.is_favorite.desc(), SavedRecipe.created_at.desc()).all()
        return JSONResponse([{
            "id": r.id,
            "title": r.title,
            "content": r.content,
            "is_favorite": r.is_favorite,
            "created_at": str(r.created_at)
        } for r in recipes])
    except Exception as e:
        logger.error("Get recipes failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not load recipes.")
 
@app.patch("/recipes/{recipe_id}/favorite")
@limiter.limit("30/minute")
async def toggle_favorite(
    recipe_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(verify_auth),
):
    """Toggle the is_favorite flag on a recipe."""
    recipe = db.query(SavedRecipe).filter_by(id=recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found.")
    recipe.is_favorite = not recipe.is_favorite
    db.commit()
    return JSONResponse({"id": recipe.id, "is_favorite": recipe.is_favorite})
 
@app.delete("/recipes/{recipe_id}")
@limiter.limit("30/minute")
async def delete_recipe(
    recipe_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(verify_auth),
):
    """Delete a saved recipe by ID."""
    recipe = db.query(SavedRecipe).filter_by(id=recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found.")
    db.delete(recipe)
    db.commit()
    return JSONResponse({"message": "Recipe deleted."})
 
# ── PANTRY ───────────────────────────────────────────────────────────────────
 
@app.get("/pantry/{session_id}")
@limiter.limit("30/minute")
async def get_pantry_route(session_id: str, request: Request, _auth=Depends(verify_auth)):
    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return JSONResponse({"items": get_pantry(session_id)})
 
@app.put("/pantry")
@limiter.limit("30/minute")
async def update_pantry(request: Request, _auth=Depends(verify_auth)):
    try:
        body_raw = await request.json()
        body = PantryRequest(**body_raw)
    except (ValidationError, Exception) as e:
        raise HTTPException(status_code=400, detail="Invalid request format")
    set_pantry(body.session_id, body.items)
    return JSONResponse({"message": "Pantry updated.", "items": body.items})
 
# ── HISTORY ──────────────────────────────────────────────────────────────────
 
@app.get("/history/{session_id}")
@limiter.limit("30/minute")
async def get_history(session_id: str, request: Request, db: Session = Depends(get_db), _auth=Depends(verify_auth)):
    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    try:
        msgs = (db.query(ConversationMessage)
                .filter_by(session_id=session_id)
                .order_by(ConversationMessage.created_at)
                .all())
        return JSONResponse([{"role": m.role, "content": m.content, "created_at": str(m.created_at)} for m in msgs])
    except Exception as e:
        logger.error("Get history failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not load history.")
 
@app.delete("/history/{session_id}")
@limiter.limit("10/minute")
async def clear_history(session_id: str, request: Request, db: Session = Depends(get_db), _auth=Depends(verify_auth)):
    if not re.match(r"^[a-zA-Z0-9_\-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    try:
        db.query(ConversationMessage).filter_by(session_id=session_id).delete()
        db.commit()
        return JSONResponse({"message": "History cleared."})
    except Exception as e:
        logger.error("Clear history failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not clear history.")
 
# Serve frontend
try:
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
except Exception:
    pass
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
