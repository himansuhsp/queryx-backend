import os
import re
import ast
import math
import asyncio
from typing import Literal, Optional, Union

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------
# LOAD ENV + CONFIG
# ---------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    # NOTE: If this is missing on Render, /health will still run only if app starts.
    # But we crash here to make it obvious.
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Gemini model
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
model = genai.GenerativeModel(MODEL_NAME)

# timeouts (seconds)
GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "25"))
RECONSIDER_TIMEOUT_SEC = int(os.getenv("RECONSIDER_TIMEOUT_SEC", "20"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # beta
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
# SAFE PYTHON CALC HELPERS
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
    """
    Safely evaluate a mathematical expression.
    Only arithmetic + allowed math functions.
    """
    try:
        expr = expr.strip()
        if not expr:
            return None

        tree = ast.parse(expr, mode="eval")

        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                return None
            if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
                return None
            if isinstance(node, ast.Call):
                # allow only function calls like sqrt(x), sin(x)
                if not isinstance(node.func, ast.Name):
                    return None
                if node.func.id not in ALLOWED_NAMES:
                    return None

        val = eval(
            compile(tree, "<expr>", "eval"),
            {"__builtins__": {}},
            ALLOWED_NAMES,
        )
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
        return None
    except Exception:
        return None


def extract_candidate_expression(text: str) -> Optional[str]:
    """
    Try to extract a numeric expression to verify using python.
    We take LAST reasonable expression-like chunk.
    """
    if not text:
        return None

    # Remove LaTeX blocks to avoid capturing random stuff inside $$ $$
    cleaned = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"\$.*?\$", " ", cleaned, flags=re.DOTALL)

    # Candidates: numbers and arithmetic operators
    candidates = re.findall(r"[-+*/().0-9eE]+", cleaned)
    if not candidates:
        return None

    # Pick last candidate with at least a digit
    for cand in reversed(candidates):
        if re.search(r"\d", cand):
            return cand
    return None


# ---------------------------------------------------------
# PROMPT HELPERS
# ---------------------------------------------------------

def build_system_prompt(level: str, style: str, language: str) -> str:
    if level == "advanced":
        level_line = "Explain with JEE/NEET depth. Derivation and sign conventions correct."
    else:
        level_line = "Explain class 11–12 level. Only required formulas."

    if style == "detailed":
        style_line = "Give step-by-step solution. Clearly show formula then substitution."
    else:
        style_line = "Give short exam-ready solution. Final formula + final answer only."

    if language == "hinglish":
        lang_line = "Use Hinglish (Roman Hindi + English). Equations strictly in LaTeX."
    else:
        lang_line = "Use clear English. Equations strictly in LaTeX."

    return (
        "You are QueryX, a JEE/NEET problem solver.\n"
        "OUTPUT RULES:\n"
        "- Return clean markdown ONLY (no JSON, no code fences).\n"
        "- Use LaTeX for all equations.\n"
        "- Be careful with arithmetic sign (+/-) and units.\n\n"
        f"{level_line}\n{style_line}\n{lang_line}\n\n"
        "IMPORTANT:\n"
        "- Choose correct formula and keep it consistent.\n"
        "- Final numeric answer must be arithmetically correct.\n"
    )


def make_prompt(system_prompt: str, question: str) -> str:
    return (
        system_prompt
        + "\n\nQuestion:\n"
        + question
        + "\n\nSolve now:\n"
    )


def reconsider_prompt(original_answer: str, python_value: float) -> str:
    return (
        "Your derivation/formula selection is OK.\n"
        "But final arithmetic/sign seems off.\n\n"
        f"Python recalculated numeric value: {python_value}\n\n"
        "Task:\n"
        "- DO NOT change the chosen formula/approach.\n"
        "- ONLY fix arithmetic/sign and final numeric answer.\n"
        "- Provide corrected final steps + final answer.\n\n"
        "Original answer:\n"
        + original_answer
    )


# ---------------------------------------------------------
# GEMINI CALL WITH TIMEOUT (IMPORTANT FIX)
# ---------------------------------------------------------

async def gemini_generate(prompt_or_contents: Union[str, list], timeout_sec: int) -> str:
    """
    google-generativeai is sync and can hang sometimes.
    So we run it in a thread + enforce timeout.
    """
    def _run():
        res = model.generate_content(prompt_or_contents)
        return (res.text or "").strip()

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return "⚠️ Gemini timeout. Please try again (server was slow)."
    except Exception as e:
        print("Gemini error:", repr(e))
        return "⚠️ Gemini error occurred. Please try again."


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "QueryX backend is running. Visit /docs for Swagger UI."}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    q = (payload.question or "").strip()
    if not q:
        return AnswerResponse(answer_text="⚠️ Question empty. Please type a question.")

    system_prompt = build_system_prompt(payload.level, payload.style, payload.language)
    prompt = make_prompt(system_prompt, q)

    # 1) first answer
    text = await gemini_generate(prompt, GEMINI_TIMEOUT_SEC)

    # If Gemini timed out / error message, return it directly
    if text.startswith("⚠️"):
        return AnswerResponse(answer_text=text)

    # 2) python verify if possible
    expr = extract_candidate_expression(text)
    py_val = safe_eval(expr) if expr else None

    # If we got a valid python value, ask Gemini to re-check calc/sign once
    if py_val is not None:
        reconsider = reconsider_prompt(text, py_val)
        text2 = await gemini_generate(reconsider, RECONSIDER_TIMEOUT_SEC)
        if text2 and not text2.startswith("⚠️"):
            text = text2

    return AnswerResponse(answer_text=text)


@app.post("/ask-image", response_model=AnswerResponse)
async def ask_image(
    file: UploadFile = File(...),
    level: Level = "basic",
    style: Style = "detailed",
    language: Language = "hinglish",
):
    system_prompt = build_system_prompt(level, style, language)

    try:
        img_bytes = await file.read()
        if not img_bytes:
            return AnswerResponse(answer_text="⚠️ Empty image file. Please upload again.")

        contents = [
            system_prompt
            + "\nFirst rewrite the question clearly from the image. Then solve carefully.\n",
            {"mime_type": file.content_type or "image/jpeg", "data": img_bytes},
        ]

        text = await gemini_generate(contents, GEMINI_TIMEOUT_SEC)

        if text.startswith("⚠️"):
            return AnswerResponse(answer_text=text)

        expr = extract_candidate_expression(text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            reconsider = reconsider_prompt(text, py_val)
            text2 = await gemini_generate(reconsider, RECONSIDER_TIMEOUT_SEC)
            if text2 and not text2.startswith("⚠️"):
                text = text2

        return AnswerResponse(answer_text=text)

    except Exception as e:
        print("Error /ask-image:", repr(e))
        return AnswerResponse(answer_text="⚠️ Image processing error. Please try again.")
