import os
import re
import ast
import math
import time
import traceback
from typing import Literal, Optional, Any

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai

# ---------------------------------------------------------
# LOAD ENV
# ---------------------------------------------------------
load_dotenv()

# Prefer GOOGLE_API_KEY (official). Fallback to GEMINI_API_KEY (old name) if present.
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY", "") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or "").strip()

API_KEY = GOOGLE_API_KEY or GEMINI_API_KEY
GEMINI_MODEL = (os.getenv("GEMINI_MODEL", "gemini-2.0-flash") or "").strip()

# Make sure SDK sees GOOGLE_API_KEY
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

# Configure Gemini SDK (safe even if key missing; we'll handle in request)
try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
except Exception:
    # We'll surface error during request
    pass

# Lazy model init (so import doesn't crash)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = genai.GenerativeModel(GEMINI_MODEL)
    return _model

app = FastAPI(title="QueryX Backend", version="1.0.0")

# ---------------------------------------------------------
# MIDDLEWARE: normalize double slashes in path
# ---------------------------------------------------------
@app.middleware("http")
async def normalize_path_middleware(request: Request, call_next):
    scope = request.scope
    path = scope.get("path", "")
    if "//" in path:
        scope["path"] = re.sub(r"/{2,}", "/", path)
    return await call_next(request)

# ---------------------------------------------------------
# CORS (beta open)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# TYPES
# ---------------------------------------------------------
Level = Literal["basic", "advanced"]
Style = Literal["detailed", "short"]
Language = Literal["english", "hinglish"]

class AskTextRequest(BaseModel):
    question: str
    level: Level = "basic"
    style: Style = "detailed"
    language: Language = "hinglish"

class AnswerResponse(BaseModel):
    answer_text: str

# ---------------------------------------------------------
# SAFE MATH EVAL (ONLY MATH EXPRESSIONS)
# ---------------------------------------------------------
ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Mod,
)

def safe_eval(expr: str) -> Optional[float]:
    if not expr:
        return None

    expr = expr.strip()

    if any(x in expr for x in ["__", "import", "exec", "eval", "open", "os.", "sys.", ";", "{", "}", "[", "]"]):
        return None

    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return None

            if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
                return None

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    return None
                if node.func.id not in ALLOWED_NAMES:
                    return None

        val = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ALLOWED_NAMES)
        if isinstance(val, (int, float)):
            return float(val)
        return None
    except Exception:
        return None

def looks_like_plain_math_question(q: str) -> bool:
    q = q.strip()
    return bool(re.fullmatch(r"[0-9\.\s\+\-\*\/\^\(\)eEpi]+", q))

# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------
def build_system_prompt(level: str, style: str, language: str) -> str:
    if level == "advanced":
        level_line = "Explain with JEE/NEET depth, correct sign conventions, and avoid algebra mistakes."
    else:
        level_line = "Explain at class 11–12 level using only required formulas and clear steps."

    if style == "detailed":
        style_line = "Give step-by-step solution. Show formula first, then substitute, then final result."
    else:
        style_line = "Give short exam-ready solution: key formula + substitution + final answer only."

    if language == "hinglish":
        lang_line = "Use Hinglish (Roman Hindi + English). Equations strictly in LaTeX."
    else:
        lang_line = "Use clear English. Equations strictly in LaTeX."

    return (
        "You are QueryX, a JEE/NEET PCMB solver.\n"
        "Output must be clean markdown only.\n"
        "Use LaTeX for math. Be careful with signs.\n\n"
        f"{level_line}\n{style_line}\n{lang_line}\n"
    )

def make_prompt(system_prompt: str, question: str) -> str:
    return (
        system_prompt
        + "\n\nQuestion:\n"
        + question
        + "\n\nInstructions:\n"
        + "- Derive formula clearly.\n"
        + "- Do arithmetic carefully.\n"
        + "- Final answer clearly.\n"
    )

def extract_candidate_expression(text: str) -> Optional[str]:
    if not text:
        return None

    eq_matches = re.findall(r"=\s*([0-9\.\s\+\-\*\/\^\(\)eEpi]+)", text)
    if eq_matches:
        expr = eq_matches[-1].strip().replace("^", "**")
        return expr

    tokens = re.findall(r"([0-9\.\s\+\-\*\/\^\(\)eEpi]{3,})", text)
    if not tokens:
        return None
    expr = tokens[-1].strip().replace("^", "**")
    return expr

def reconsider_prompt(original_answer: str, python_value: float) -> str:
    return (
        "Your formula/derivation is fixed. Do NOT change formula.\n"
        "Only re-check arithmetic/sign substitution.\n\n"
        f"Python computed value: {python_value}\n\n"
        "Now rewrite only the calculation lines and final answer (markdown + LaTeX).\n\n"
        "Original answer:\n"
        + original_answer
    )

# ---------------------------------------------------------
# GEMINI CALL (with retries)
# ---------------------------------------------------------
def gemini_generate_text(prompt: Any) -> str:
    # If no key, throw a clean error
    if not (os.getenv("GOOGLE_API_KEY") or ""):
        raise RuntimeError("GOOGLE_API_KEY missing at runtime")

    last_err = None
    for _ in range(3):
        try:
            res = get_model().generate_content(prompt)
            return (getattr(res, "text", "") or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.6)

    raise RuntimeError(f"Gemini failed after retries. Last error: {repr(last_err)}")

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "has_google_api_key": bool((os.getenv("GOOGLE_API_KEY") or "").strip()),
    }

@app.get("/debug")
async def debug():
    k = (os.getenv("GOOGLE_API_KEY") or "").strip()
    return {
        "has_google_api_key": bool(k),
        "google_api_key_last4": k[-4:] if k else "",
        "model": GEMINI_MODEL,
    }

@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    q = (payload.question or "").strip()
    if not q:
        return AnswerResponse(answer_text="Please provide a question.")

    # Pure math → bypass Gemini
    if looks_like_plain_math_question(q):
        expr = q.replace("^", "**")
        val = safe_eval(expr)
        if val is None:
            return AnswerResponse(answer_text="Invalid expression.")
        return AnswerResponse(answer_text=f"Final Answer: $ {val:g} $")

    system_prompt = build_system_prompt(payload.level, payload.style, payload.language)
    prompt = make_prompt(system_prompt, q)

    try:
        first_text = gemini_generate_text(prompt)

        # optional python check
        expr = extract_candidate_expression(first_text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            second_prompt = reconsider_prompt(first_text, py_val)
            second_text = gemini_generate_text(second_prompt)
            if second_text:
                return AnswerResponse(answer_text=second_text)

        return AnswerResponse(answer_text=first_text or "Empty response from Gemini.")

    except Exception as e:
        print("❌ Gemini error in /ask-text:", repr(e))
        traceback.print_exc()
        # surface exact key-missing style error to help debugging
        msg = str(e)
        if "GOOGLE_API_KEY" in msg or "API Key not found" in msg:
            return AnswerResponse(answer_text="⚠️ Gemini API key issue on server. Check GOOGLE_API_KEY in Render/Railway env.")
        return AnswerResponse(answer_text="⚠️ Gemini error occurred. Please try again.")

@app.post("/ask-image", response_model=AnswerResponse)
async def ask_image(
    file: UploadFile = File(...),
    level: Level = "basic",
    style: Style = "detailed",
    language: Language = "hinglish",
):
    try:
        system_prompt = build_system_prompt(level, style, language)

        img_bytes = await file.read()
        if not img_bytes:
            return AnswerResponse(answer_text="Empty image uploaded.")

        contents = [
            system_prompt
            + "\nTask:\n"
            + "1) Rewrite the question clearly from the image.\n"
            + "2) Solve it carefully with correct sign and arithmetic.\n",
            {"mime_type": file.content_type or "image/jpeg", "data": img_bytes},
        ]

        first_text = gemini_generate_text(contents)

        expr = extract_candidate_expression(first_text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            second_prompt = reconsider_prompt(first_text, py_val)
            second_text = gemini_generate_text(second_prompt)
            if second_text:
                return AnswerResponse(answer_text=second_text)

        return AnswerResponse(answer_text=first_text or "Empty response from Gemini.")

    except Exception as e:
        print("❌ Gemini error in /ask-image:", repr(e))
        traceback.print_exc()
        msg = str(e)
        if "GOOGLE_API_KEY" in msg or "API Key not found" in msg:
            return AnswerResponse(answer_text="⚠️ Gemini API key issue on server. Check GOOGLE_API_KEY in Render/Railway env.")
        return AnswerResponse(answer_text="⚠️ Gemini error occurred. Please try again.")
