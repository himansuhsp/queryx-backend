import os
import re
import time
import uuid
import io
from typing import Optional, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai

# -------------------------
# ENV
# -------------------------
load_dotenv()

APP_NAME = "QueryX PCMB Solver"

API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing in environment")

genai.configure(api_key=API_KEY)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title=APP_NAME)

# -------------------------
# 🔥 CORS (FINAL, SIMPLE, WORKING)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all (safe for now)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Gemini Models
# -------------------------
MODEL_FALLBACK = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

# -------------------------
# Image support
# -------------------------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -------------------------
# Request schema
# -------------------------
class AskRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"
    style: Optional[str] = "detailed"
    language: Optional[str] = "hinglish"

# -------------------------
# Helpers
# -------------------------
MATH_RE = re.compile(r"^[0-9\.\+\-\*\/\%\(\)\^\s]+$")

def is_simple_math(q: str) -> bool:
    return bool(MATH_RE.match(q.strip()))

def safe_eval(expr: str) -> str:
    expr = expr.replace("^", "**")
    return str(eval(expr, {"__builtins__": {}}))

def build_prompt(q: str, level: str, style: str, lang: str) -> str:
    level = (level or "basic").lower()
    style = (style or "detailed").lower()
    lang = (lang or "hinglish").lower()

    return f"""
You are QueryX, an exam-focused JEE/NEET PCMB solver.

LANGUAGE:
- {lang} (If Hinglish: Hindi in Latin + English mix)

ANSWER FORMAT (MANDATORY):
- Use BULLET POINTS (•) and NUMBERED STEPS (1,2,3).
- Each step max 2 short lines.
- No paragraphs.

LATEX RULES:
- Use LaTeX ONLY inside $...$ or $$...$$
- ❌ Do NOT use \\[ \\] or \\( \\)

DECISION LOGIC:
1) If question is CONCEPTUAL / THEORY based
   (keywords: explain, define, what is, law, principle):
   → Give:
     • Definition (1 bullet)
     • Key points (3–5 bullets)
     • Formula (if any)
     • One small example / application
   → DO NOT ask follow-up questions.

2) If question is NUMERICAL:
   → Follow exact order:
     Step 1: Given
     Step 2: Formula
     Step 3: Substitution
     Step 4: Final Answer (boxed)

3) Say **"Given data missing: ___"** ONLY if numerical values are compulsory
   and not provided. Otherwise NEVER say this.

STYLE RULES:
- No introduction
- No conclusion
- No generic textbook theory
- Exam-oriented, crisp, direct

QUESTION:
{q}
""".strip()



def extract_text(resp) -> str:
    if getattr(resp, "text", None):
        return resp.text.strip()

    try:
        parts = []
        for c in resp.candidates:
            for p in c.content.parts:
                if p.text:
                    parts.append(p.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""

def gemini_call(prompt: str, image: Optional[Any], req_id: str) -> str:
    for model_name in MODEL_FALLBACK:
        try:
            model = genai.GenerativeModel(model_name)
            content = [prompt, image] if image else prompt
            resp = model.generate_content(
                content,
                generation_config=GENERATION_CONFIG
            )
            text = extract_text(resp)
            if text:
                return text
        except Exception as e:
            print(f"[{req_id}] {model_name} failed: {e}")
            time.sleep(0.7)

    return f"⚠️ Gemini failed | id={req_id}"

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME}

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "api_key_loaded": bool(API_KEY),
        "models": MODEL_FALLBACK,
    }

@app.post("/ask-text")
async def ask_text(payload: AskRequest):
    req_id = str(uuid.uuid4())[:8]
    q = payload.question.strip()

    if not q:
        return {"answer_text": "⚠️ Question empty hai."}

    if is_simple_math(q):
        try:
            return {"answer_text": safe_eval(q)}
        except Exception:
            pass

    prompt = build_prompt(q, payload.level, payload.style, payload.language)
    answer = await run_in_threadpool(gemini_call, prompt, None, req_id)
    return {"answer_text": answer}

@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    req_id = str(uuid.uuid4())[:8]

    if not PIL_AVAILABLE:
        return {"answer_text": "❌ Pillow missing"}

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return {"answer_text": f"❌ Image error: {e}"}

    prompt = build_prompt("Solve the question from image.", level, style, language)
    answer = await run_in_threadpool(gemini_call, prompt, img, req_id)
    return {"answer_text": answer}
