import os
import re
import time
import uuid
import io
import traceback
from typing import Optional, List, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai

# Optional safety types (some versions may differ)
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    SAFETY_TYPES_AVAILABLE = True
except Exception:
    SAFETY_TYPES_AVAILABLE = False

# Image support (PIL)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# -------------------------
# Load environment (local dev only)
# -------------------------
load_dotenv()

APP_NAME = "QueryX PCMB Solver"

# -------------------------
# API KEY (STRIP VERY IMPORTANT)
# -------------------------
API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

# If empty -> fail fast (otherwise Gemini will keep throwing confusing errors)
if not API_KEY:
    # Don't crash import-time in production if you want health endpoint to show the issue
    # but better to fail early so Railway shows error clearly.
    raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY missing or empty in environment variables.")

genai.configure(api_key=API_KEY)

# -------------------------
# Model fallback
# -------------------------
ENV_MODEL = (os.getenv("GEMINI_MODEL") or "").strip()

MODEL_FALLBACK: List[str] = []
if ENV_MODEL:
    MODEL_FALLBACK.append(ENV_MODEL)

# safe/stable order
MODEL_FALLBACK += [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# -------------------------
# Generation config
# -------------------------
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

# -------------------------
# Safety settings (optional)
# -------------------------
SAFETY_SETTINGS = None
if SAFETY_TYPES_AVAILABLE:
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


# -------------------------
# FastAPI App + CORS
# -------------------------
app = FastAPI(title=APP_NAME)

# ✅ CORS best practice:
# - allow_credentials=True can't be used with allow_origins=["*"]
# - so we keep explicit origins
ALLOWED_ORIGINS = [
    os.getenv("FRONTEND_ORIGIN", "").strip(),  # e.g. https://queryx-frontend.vercel.app
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = [o for o in ALLOWED_ORIGINS if o]  # remove empty

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,  # keep false for easier deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Models
# -------------------------
class AskRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"      # basic / advanced
    style: Optional[str] = "detailed"   # short / detailed
    language: Optional[str] = "hinglish"


# -------------------------
# Helpers
# -------------------------
MATH_EXPR_RE = re.compile(r"^[\s0-9\.\+\-\*\/\%\(\)\^\s]+$")

def _is_math_only(q: str) -> bool:
    q = (q or "").strip()
    return bool(q) and bool(MATH_EXPR_RE.match(q))

def _safe_eval_arithmetic(expr: str) -> float:
    expr = (expr or "").strip().replace("^", "**")

    blocked = ["__", "import", "os.", "sys.", "eval", "exec", "open(", "subprocess", "socket"]
    low = expr.lower()
    for b in blocked:
        if b in low:
            raise ValueError("Unsafe expression")

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
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed token: {type(node).__name__}")

    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, {})

def _clean_latex(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", text, flags=re.IGNORECASE)
    return text

def _build_prompt(q: str, level: str, style: str, lang: str) -> str:
    level = (level or "basic").lower()
    style = (style or "detailed").lower()
    lang = (lang or "hinglish").lower()

    return f"""
You are QueryX, an expert JEE/NEET PCMB (Physics, Chemistry, Maths, Biology) solver.

User Preferences:
- Level: {level}
- Style: {style}
- Language: {lang}

STRICT INSTRUCTIONS:
1) Start directly with solution. No "Sure" / "I can help".
2) Use Markdown.
3) Use LaTeX ONLY inside $...$ or $$...$$ for ALL formulas.
4) Exam-oriented and accurate. Step-by-step.
5) If language is 'hinglish', use Hindi (Latin script) + English mix.
6) If input is pure arithmetic, return ONLY final number.

Question:
{q}
""".strip()

def _extract_text(resp: Any) -> str:
    # Preferred
    txt = getattr(resp, "text", None)
    if txt and str(txt).strip():
        return str(txt).strip()

    # Fallback
    try:
        if resp.candidates:
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
    except Exception:
        pass

    return ""


# -------------------------
# Gemini Call (sync) -> we run it in threadpool from async routes
# -------------------------
def _gemini_call(prompt: str, image: Optional[Any], request_id: str) -> str:
    last_err = None

    for model_name in MODEL_FALLBACK:
        model = genai.GenerativeModel(model_name)
        max_attempts = 2
        base_sleep = 0.8

        for attempt in range(1, max_attempts + 1):
            try:
                content = [prompt, image] if image is not None else prompt
                kwargs = {"generation_config": GENERATION_CONFIG}
                if SAFETY_SETTINGS is not None:
                    kwargs["safety_settings"] = SAFETY_SETTINGS

                resp = model.generate_content(content, **kwargs)
                text = _extract_text(resp)
                if text:
                    return _clean_latex(text)

                raise RuntimeError("Empty/blocked response")

            except Exception as e:
                last_err = e
                print(
                    f"[{request_id}] model={model_name} attempt={attempt}/{max_attempts} failed: "
                    f"{type(e).__name__}: {str(e)[:220]}"
                )
                if attempt < max_attempts:
                    time.sleep(base_sleep)

        print(f"[{request_id}] switching model fallback after failures: {model_name}")

    # Return real error message (truncated) so you can debug from frontend
    if last_err:
        msg = str(last_err)
        return f"⚠️ Gemini error: {msg[:180]} | id: {request_id}"

    return f"⚠️ Gemini error: unknown | id: {request_id}"


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "models": MODEL_FALLBACK}

@app.get("/health")
def health():
    # masked key prefix for debugging (safe)
    prefix = API_KEY[:6] + "..." if API_KEY else ""
    return {
        "ok": True,
        "service": APP_NAME,
        "models": MODEL_FALLBACK,
        "pil": PIL_AVAILABLE,
        "api_key_loaded": bool(API_KEY),
        "api_key_prefix": prefix,
        "origins": ALLOWED_ORIGINS,
    }

@app.post("/ask-text")
async def ask_text(payload: AskRequest):
    request_id = str(uuid.uuid4())[:8]
    q = (payload.question or "").strip()
    if not q:
        return {"answer_text": "⚠️ Sawal empty hai."}

    # safe arithmetic fast path
    if _is_math_only(q):
        try:
            val = _safe_eval_arithmetic(q)
            if abs(val - int(val)) < 1e-12:
                return {"answer_text": str(int(val))}
            return {"answer_text": str(val)}
        except Exception:
            pass  # fallback to gemini

    prompt = _build_prompt(q, payload.level, payload.style, payload.language)

    try:
        answer = await run_in_threadpool(_gemini_call, prompt, None, request_id)
        return {"answer_text": answer}
    except Exception as e:
        print(f"[{request_id}] ask-text fatal: {type(e).__name__}: {str(e)}")
        return {"answer_text": f"⚠️ Server error: {str(e)[:120]} | id: {request_id}"}

@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    request_id = str(uuid.uuid4())[:8]

    if not PIL_AVAILABLE:
        return {"answer_text": "⚠️ Pillow missing. requirements.txt me `pillow` add karo."}

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")  # type: ignore
    except Exception as e:
        return {"answer_text": f"⚠️ Image read error: {str(e)[:160]} | id: {request_id}"}

    prompt = _build_prompt("Solve the question from the image.", level, style, language)

    try:
        answer = await run_in_threadpool(_gemini_call, prompt, img, request_id)
        return {"answer_text": answer}
    except Exception as e:
        print(f"[{request_id}] ask-image fatal: {type(e).__name__}: {str(e)}")
        return {"answer_text": f"⚠️ Server error: {str(e)[:120]} | id: {request_id}"}
