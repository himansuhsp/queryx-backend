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

# =========================================================
# ENV + GEMINI CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# =========================================================
# APP
# =========================================================

app = FastAPI(title="QueryX Backend", version="1.0.2")

# ---------------------------------------------------------
# FIX: normalize double slashes in path
# (frontend BACKEND_URL endswith '/' => //ask-text, //ask-image)
# ---------------------------------------------------------
@app.middleware("http")
async def normalize_path_middleware(request: Request, call_next):
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/{2,}", "/", path)
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

# =========================================================
# TYPES
# =========================================================

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


# =========================================================
# SAFE MATH EVAL (NO EXEC / IMPORT)
# =========================================================

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
    """
    Evaluate ONLY math expressions safely.
    Returns float or None.
    """
    if not expr:
        return None

    expr = expr.strip()

    # Hard reject suspicious content
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
    """
    STRICT: Only bypass Gemini if question is literally a math expression like:
    2+3, (2*3)+4, 1/2 + 3/4, 2^3, sqrt(16)
    """
    q = (q or "").strip()
    if not q:
        return False

    # too long => not pure math
    if len(q) > 80:
        return False

    # must have at least one digit
    if not re.search(r"\d", q):
        return False

    # must have an operator or function call
    if not (re.search(r"[\+\-\*\/\^]", q) or re.search(r"\b(sqrt|sin|cos|tan|log|ln|exp|abs)\s*\(", q)):
        return False

    # Only allow safe characters (includes letters for allowed funcs/pi/e)
    # Reject any other letters
    tmp = q.lower()
    tmp = re.sub(r"\b(sqrt|sin|cos|tan|log|ln|exp|abs)\b", "", tmp)
    tmp = tmp.replace("pi", "").replace("e", "")
    if re.search(r"[a-df-z]", tmp):  # any letter except e (handled), pi removed already
        return False

    # Final allowlist check
    return bool(re.fullmatch(r"[0-9\.\s\+\-\*\/\^\(\),a-zA-Z]+", q))


# =========================================================
# GEMINI HELPERS
# =========================================================

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
        "Use LaTeX for all math.\n"
        "Be very careful with arithmetic signs (+/-) and substitutions.\n\n"
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
        + "- Do NOT output JSON.\n"
        + "- Do NOT use triple backticks.\n"
    )


def extract_candidate_expression(text: str) -> Optional[str]:
    """
    Extract a *candidate* numeric expression from the answer.
    Goal: only catch simple final arithmetic like "= 2*3" or "= (2*3)+4".
    """
    if not text:
        return None

    cleaned = text.replace("×", "*").replace("−", "-")
    # Prefer last "= <expr>" with a strict allowlist
    eq_matches = re.findall(r"=\s*([0-9\.\s\+\-\*\/\^\(\),pi eE]+)", cleaned)
    if eq_matches:
        expr = eq_matches[-1].strip().replace("^", "**")
        expr = expr.replace(",", "")  # remove thousands separators if any
        return expr

    # fallback: last token-like math segment (strict)
    tokens = re.findall(r"([0-9\.\s\+\-\*\/\^\(\),pi eE]{5,})", cleaned)
    if not tokens:
        return None
    expr = tokens[-1].strip().replace("^", "**").replace(",", "")
    return expr


def reconsider_prompt(original_answer: str, python_value: float) -> str:
    return (
        "Your formula/derivation is FIXED. Do NOT change the formula.\n"
        "Only re-check arithmetic/sign/substitution.\n\n"
        f"Python computed value: {python_value}\n\n"
        "Now rewrite only the calculation lines and final answer (markdown + LaTeX).\n\n"
        "Original answer:\n"
        + original_answer
    )


def gemini_generate(prompt: Any) -> str:
    res = model.generate_content(prompt)
    return (getattr(res, "text", "") or "").strip()


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}


@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    q = (payload.question or "").strip()
    if not q:
        return AnswerResponse(answer_text="Please provide a question.")

    # 1) PURE MATH BYPASS (ONLY if truly a math expression)
    if looks_like_plain_math_question(q):
        expr = q.replace("^", "**")
        val = safe_eval(expr)
        if val is None:
            # If parsing failed, fall back to Gemini instead of blocking user
            pass
        else:
            return AnswerResponse(answer_text=f"Final Answer: $ {val:g} $")

    system_prompt = build_system_prompt(payload.level, payload.style, payload.language)
    prompt = make_prompt(system_prompt, q)

    try:
        first_text = gemini_generate(prompt)

        # 2) PYTHON CHECK (best-effort)
        expr = extract_candidate_expression(first_text)
        py_val = safe_eval(expr) if expr else None

        # IMPORTANT: only reconsider if we actually extracted something meaningful
        if py_val is not None:
            second_text = gemini_generate(reconsider_prompt(first_text, py_val))
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
            + "2) Solve it carefully with correct sign and arithmetic.\n",
            {"mime_type": file.content_type or "image/jpeg", "data": img_bytes},
        ]

        first_text = gemini_generate(contents)

        # python check attempt
        expr = extract_candidate_expression(first_text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            second_text = gemini_generate(reconsider_prompt(first_text, py_val))
            if second_text:
                return AnswerResponse(answer_text=second_text)

        return AnswerResponse(answer_text=first_text or "Empty response from Gemini.")

    except Exception as e:
        print("❌ Gemini error in /ask-image:", repr(e))
        traceback.print_exc()
        return AnswerResponse(answer_text="⚠️ Gemini error occurred. Please try again.")
