import os
import re
import time
import uuid
import json
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

import google.generativeai as genai

# Optional image support (PIL)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# -------------------------
# Load env (local dev only)
# -------------------------
load_dotenv()

# -------------------------
# Config
# -------------------------
APP_NAME = "QueryX Backend"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# You should set ONE of these in Railway:
# GOOGLE_API_KEY or GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    # Fail fast so you don't silently get "Gemini error occurred" everywhere
    raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment variables.")

genai.configure(api_key=API_KEY)

# Safety: do not log key
# print("Gemini configured.")  # keep minimal

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title=APP_NAME)

# CORS - allow all for beta; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later set to your Vercel domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request / Response Models
# -------------------------
class AskTextRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"     # basic / advanced
    style: Optional[str] = "detailed"  # detailed / short
    language: Optional[str] = "hinglish"  # hinglish / english


# -------------------------
# Helpers
# -------------------------
MATH_EXPR_RE = re.compile(r"^[\s0-9\.\+\-\*\/\%\(\)\^\s]+$")

def _is_pure_arithmetic(q: str) -> bool:
    q = (q or "").strip()
    if len(q) == 0:
        return False
    # allow ^ as power; no letters allowed
    return bool(MATH_EXPR_RE.match(q))

def _safe_eval_arithmetic(expr: str) -> float:
    """
    Very small safe arithmetic evaluator:
    - digits, + - * / % ( ) . and power ^
    - converts ^ to ** for Python
    - blocks names/attributes/calls
    """
    expr = expr.strip().replace("^", "**")

    # Hard block suspicious tokens
    blocked = ["__", "import", "os.", "sys.", "eval", "exec", "open(", "read(", "write(", "subprocess", "socket"]
    low = expr.lower()
    for b in blocked:
        if b in low:
            raise ValueError("Unsafe expression")

    # AST-based safe eval
    import ast

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.FloorDiv,
        ast.Tuple,  # not used but harmless
    )

    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed token: {type(node).__name__}")

    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, {})


def _build_prompt(question: str, level: str, style: str, language: str) -> str:
    # Keep prompt stable and deterministic-ish
    # (You can expand later with your formula-lock / verification system)
    level = (level or "basic").lower()
    style = (style or "detailed").lower()
    language = (language or "hinglish").lower()

    # Output constraint: plain text (your frontend renders it)
    return f"""
You are QueryX, a PCMB tutor for JEE/NEET.
Answer the user question with correct physics/maths reasoning.

User preferences:
- Level: {level} (basic => simpler, advanced => deeper)
- Style: {style} (short => concise steps, detailed => more explanation)
- Language: {language} (hinglish => Hindi+English mix, english => only English)

Rules:
1) Be accurate. If assumptions needed, state them.
2) Prefer step-by-step.
3) Use standard symbols. Keep it readable.
4) If user question is purely arithmetic, return just the computed result.

Question:
{question}
""".strip()


def _gemini_generate_text(prompt: str, request_id: str) -> str:
    """
    Gemini call with retries (handles intermittent failures & rate limits).
    """
    model_name = DEFAULT_MODEL
    model = genai.GenerativeModel(model_name)

    # Conservative generation settings (stable)
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1200,
    }

    # Retry strategy
    max_attempts = 4
    base_sleep = 1.0

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            # Some SDK responses put text in .text
            text = getattr(resp, "text", None)
            if text and text.strip():
                return text.strip()

            # Fallback: try candidates
            if hasattr(resp, "candidates") and resp.candidates:
                parts = []
                for c in resp.candidates:
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        for p in c.content.parts:
                            t = getattr(p, "text", None)
                            if t:
                                parts.append(t)
                joined = "\n".join(parts).strip()
                if joined:
                    return joined

            raise RuntimeError("Empty response from Gemini")

        except Exception as e:
            last_err = e

            # Log minimal (no secrets)
            print(f"[{request_id}] Gemini attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {str(e)[:200]}")

            if attempt < max_attempts:
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue

    # If all retries fail:
    raise RuntimeError(f"Gemini failed after retries: {type(last_err).__name__}: {str(last_err)}")


def _gemini_generate_with_image(prompt: str, image: "Image.Image", request_id: str) -> str:
    model_name = DEFAULT_MODEL
    model = genai.GenerativeModel(model_name)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1200,
    }

    max_attempts = 4
    base_sleep = 1.0
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = model.generate_content(
                [prompt, image],
                generation_config=generation_config,
            )
            text = getattr(resp, "text", None)
            if text and text.strip():
                return text.strip()
            raise RuntimeError("Empty response from Gemini (image)")
        except Exception as e:
            last_err = e
            print(f"[{request_id}] Gemini(image) attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {str(e)[:200]}")
            if attempt < max_attempts:
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue

    raise RuntimeError(f"Gemini(image) failed after retries: {type(last_err).__name__}: {str(last_err)}")


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "model": DEFAULT_MODEL}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask-text")
def ask_text(payload: AskTextRequest):
    request_id = str(uuid.uuid4())[:8]
    q = (payload.question or "").strip()

    if not q:
        return {"answer_text": "⚠️ Question empty hai."}

    # 1) If pure arithmetic, solve locally (no Gemini needed)
    if _is_pure_arithmetic(q):
        try:
            val = _safe_eval_arithmetic(q)
            # neat formatting
            if abs(val - int(val)) < 1e-12:
                return {"answer_text": str(int(val))}
            return {"answer_text": str(val)}
        except Exception:
            # If arithmetic parsing fails, fallback to Gemini
            pass

    # 2) Gemini normal solve
    prompt = _build_prompt(q, payload.level, payload.style, payload.language)
    try:
        answer = _gemini_generate_text(prompt, request_id)
        return {"answer_text": answer}
    except Exception as e:
        # Show user-friendly message, keep logs for debugging
        print(f"[{request_id}] ERROR ask-text: {type(e).__name__}: {str(e)}")
        # print(traceback.format_exc())
        return {"answer_text": "⚠️ Gemini error occurred. Please try again."}


@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    request_id = str(uuid.uuid4())[:8]

    if not PIL_AVAILABLE:
        return {"answer_text": "⚠️ Image support disabled (PIL not installed). requirements.txt me pillow add karo."}

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")  # type: ignore
    except Exception as e:
        print(f"[{request_id}] ERROR reading image: {type(e).__name__}: {str(e)}")
        return {"answer_text": "⚠️ Image read nahi ho pa raha. Clear photo + supported format (jpg/png) try karo."}

    prompt = _build_prompt("Solve the question from the image.", level, style, language)

    try:
        answer = _gemini_generate_with_image(prompt, img, request_id)
        return {"answer_text": answer}
    except Exception as e:
        print(f"[{request_id}] ERROR ask-image: {type(e).__name__}: {str(e)}")
        return {"answer_text": "⚠️ Gemini error occurred. Please try again."}


# Needed for ask-image (io import)
import io  # keep at bottom to avoid clutter
