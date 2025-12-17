import os
import re
import ast
import math
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

DEBUG_ERRORS = (os.getenv("DEBUG_ERRORS", "") or "").strip() in {"1", "true", "True", "YES", "yes"}

def _get_api_key() -> str:
    # Prefer GOOGLE_API_KEY (Railway), fallback GEMINI_API_KEY (local)
    k = (os.getenv("GOOGLE_API_KEY", "") or "").strip()
    if k:
        return k
    k = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    if k:
        return k
    return ""

def _mask_key(k: str) -> str:
    k = (k or "").strip()
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}***{k[-4:]}"

API_KEY = _get_api_key()
GEMINI_MODEL = (os.getenv("GEMINI_MODEL", "gemini-2.0-flash") or "").strip()

if API_KEY:
    genai.configure(api_key=API_KEY)

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
# CORS
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
# SAFE MATH EVAL
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
# GEMINI HELPERS
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
        return eq_matches[-1].strip().replace("^", "**")
    tokens = re.findall(r"([0-9\.\s\+\-\*\/\^\(\)eEpi]{3,})", text)
    if not tokens:
        return None
    return tokens[-1].strip().replace("^", "**")


def reconsider_prompt(original_answer: str, python_value: float) -> str:
    return (
        "Your formula/derivation is fixed. Do NOT change formula.\n"
        "Only re-check arithmetic/sign substitution.\n\n"
        f"Python computed value: {python_value}\n\n"
        "Now rewrite only the calculation lines and final answer (markdown + LaTeX).\n\n"
        "Original answer:\n"
        + original_answer
    )


def gemini_generate_text(prompt: Any) -> str:
    if not API_KEY:
        raise RuntimeError("NO_API_KEY: Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    mdl = get_model()
    res = mdl.generate_content(prompt)
    return (getattr(res, "text", "") or "").strip()


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}

@app.get("/debug")
async def debug():
    google_key = (os.getenv("GOOGLE_API_KEY", "") or "").strip()
    gemini_key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    using = "GOOGLE_API_KEY" if google_key else ("GEMINI_API_KEY" if gemini_key else "NONE")
    return {
        "has_google_api_key": bool(google_key),
        "has_gemini_api_key": bool(gemini_key),
        "using_key": using,
        "key_masked": _mask_key(google_key or gemini_key),
        "model": GEMINI_MODEL,
        "debug_errors": DEBUG_ERRORS,
    }

@app.get("/debug-gemini")
async def debug_gemini():
    """
    This actually calls Gemini with a tiny prompt and returns success/error.
    """
    try:
        txt = gemini_generate_text("Reply with exactly: OK")
        return {"ok": True, "reply": txt[:200]}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    q = (payload.question or "").strip()
    if not q:
        return AnswerResponse(answer_text="Please provide a question.")

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
        if DEBUG_ERRORS:
            return AnswerResponse(answer_text=f"⚠️ DEBUG: {repr(e)}")
        if "NO_API_KEY" in repr(e):
            return AnswerResponse(answer_text="⚠️ Gemini API key missing on server. Set GOOGLE_API_KEY / GEMINI_API_KEY.")
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
        if DEBUG_ERRORS:
            return AnswerResponse(answer_text=f"⚠️ DEBUG: {repr(e)}")
        if "NO_API_KEY" in repr(e):
            return AnswerResponse(answer_text="⚠️ Gemini API key missing on server. Set GOOGLE_API_KEY / GEMINI_API_KEY.")
        return AnswerResponse(answer_text="⚠️ Gemini error occurred. Please try again.")
