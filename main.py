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
# LOAD ENV + CONFIG
# ---------------------------------------------------------
load_dotenv()

def pick_api_key() -> str:
    """
    Render/railway/vercel env me keys kabhi alag naam se hoti hain.
    We accept multiple names safely.
    """
    candidates = [
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),  # (same but kept for clarity)
        os.getenv("GENAI_API_KEY", ""),
    ]
    for k in candidates:
        k = (k or "").strip()
        if k:
            return k
    return ""

API_KEY = pick_api_key()
if not API_KEY:
    raise RuntimeError("No API key found. Set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY in environment.")

GEMINI_MODEL = (os.getenv("GEMINI_MODEL", "gemini-2.0-flash") or "").strip() or "gemini-2.0-flash"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

app = FastAPI(title="QueryX Backend", version="1.2.0")

# ---------------------------------------------------------
# MIDDLEWARE: normalize // paths
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
# SAFE MATH EVAL (ONLY math expressions)
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

    # quick reject
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

def is_theory_question(q: str) -> bool:
    """
    Theory / explanation type questions me python verification bilkul nahi chahiye.
    """
    ql = q.strip().lower()
    theory_keywords = [
        "explain", "define", "what is", "derive", "state", "law", "principle",
        "gauss", "nlm", "newton", "concept", "difference", "why", "how", "write note",
    ]
    # if no digits at all, it's almost surely theory
    has_digit = any(ch.isdigit() for ch in ql)
    if not has_digit:
        return True
    return any(k in ql for k in theory_keywords)

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
        "Use LaTeX for math.\n"
        "If question is conceptual, answer conceptually.\n"
        "If numerical, show steps and final answer.\n\n"
        f"{level_line}\n{style_line}\n{lang_line}\n"
    )

def make_prompt(system_prompt: str, question: str) -> str:
    return (
        system_prompt
        + "\n\nQuestion:\n"
        + question
        + "\n\nInstructions:\n"
        + "- Be accurate.\n"
        + "- If numerical: show steps and final.\n"
        + "- If conceptual: explain clearly with examples if needed.\n"
    )

def extract_final_answer_expression(text: str) -> Optional[str]:
    """
    IMPORTANT: Only try extracting expression if model explicitly gives a final answer line.
    This prevents theory answers from triggering python-check.
    """
    if not text:
        return None

    # Prefer "Final Answer" style
    m = re.search(r"(Final\s*Answer|Answer)\s*[:\-]\s*\$?\s*([0-9\.\s\+\-\*\/\^\(\)eEpi]+)\s*\$?", text, re.IGNORECASE)
    if m:
        expr = (m.group(2) or "").strip().replace("^", "**")
        return expr

    # Or last "= number/expression" ONLY if it looks like final line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = "\n".join(lines[-3:])
        eq_matches = re.findall(r"=\s*([0-9\.\s\+\-\*\/\^\(\)eEpi]+)", tail)
        if eq_matches:
            expr = eq_matches[-1].strip().replace("^", "**")
            return expr

    return None

def reconsider_prompt(original_answer: str, python_value: float) -> str:
    return (
        "Keep the same method/formula.\n"
        "ONLY re-check arithmetic/substitution.\n\n"
        f"Python computed value: {python_value}\n\n"
        "Rewrite only calculation + final answer (markdown + LaTeX).\n\n"
        "Original answer:\n"
        + original_answer
    )

def gemini_generate_text(prompt: Any, retries: int = 2) -> str:
    """
    Gemini call with small retries (helps with transient failures / cold starts).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            res = model.generate_content(prompt)
            return (getattr(res, "text", "") or "").strip()
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Gemini failed after retries. Last error: {repr(last_err)}")

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}

@app.get("/debug")
async def debug():
    # never expose full key
    key = API_KEY or ""
    return {
        "has_key": bool(key),
        "key_last4": key[-4:] if len(key) >= 4 else "",
        "model": GEMINI_MODEL,
        "python": os.getenv("PYTHON_VERSION", ""),  # may be empty on some platforms
    }

@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    q = (payload.question or "").strip()
    if not q:
        return AnswerResponse(answer_text="Please provide a question.")

    # Pure math => bypass gemini
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

        # ✅ Only do python verification for NON-theory questions
        if not is_theory_question(q):
            expr = extract_final_answer_expression(first_text)
            py_val = safe_eval(expr) if expr else None

            # only reconsider if we got a meaningful python value
            if py_val is not None:
                second_prompt = reconsider_prompt(first_text, py_val)
                second_text = gemini_generate_text(second_prompt, retries=1)
                if second_text:
                    return AnswerResponse(answer_text=second_text)

        return AnswerResponse(answer_text=first_text or "Empty response from Gemini.")

    except Exception as e:
        print("❌ Gemini error in /ask-text:", repr(e))
        traceback.print_exc()
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
            + "2) Solve it carefully.\n"
            + "If conceptual, explain concept.\n",
            {"mime_type": file.content_type or "image/jpeg", "data": img_bytes},
        ]

        first_text = gemini_generate_text(contents)

        # For image: python-check ONLY if output contains explicit "Final Answer"
        expr = extract_final_answer_expression(first_text)
        py_val = safe_eval(expr) if expr else None
        if py_val is not None:
            second_prompt = reconsider_prompt(first_text, py_val)
            second_text = gemini_generate_text(second_prompt, retries=1)
            if second_text:
                return AnswerResponse(answer_text=second_text)

        return AnswerResponse(answer_text=first_text or "Empty response from Gemini.")

    except Exception as e:
        print("❌ Gemini error in /ask-image:", repr(e))
        traceback.print_exc()
        return AnswerResponse(answer_text="⚠️ Gemini error occurred. Please try again.")
