import os
import re
import ast
import math
from typing import Literal, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# =========================================================
# ENV
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# =========================================================
# APP
# =========================================================

app = FastAPI()

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
# SAFE PYTHON SANDBOX
# =========================================================

ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
}

def safe_eval(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode="eval")

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
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
                    ast.Call,
                ),
            ):
                return None

            if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
                return None

        return float(
            eval(
                compile(tree, "<expr>", "eval"),
                {"__builtins__": {}},
                ALLOWED_NAMES,
            )
        )
    except Exception:
        return None

def extract_last_expression(text: str) -> Optional[str]:
    if not text:
        return None

    matches = re.findall(r"[-+*/().0-9eE]+", text)
    if not matches:
        return None

    candidate = matches[-1]
    if len(candidate) > 40:
        return None

    return candidate

# =========================================================
# PROMPTS
# =========================================================

def build_system_prompt(level: str, style: str, language: str) -> str:
    level_line = (
        "Use JEE/NEET depth with correct sign conventions."
        if level == "advanced"
        else "Use class 11–12 level concepts only."
    )

    style_line = (
        "Give step-by-step derivation."
        if style == "detailed"
        else "Give short exam-ready solution."
    )

    lang_line = (
        "Write in Hinglish (Roman Hindi + English)."
        if language == "hinglish"
        else "Write in clear English."
    )

    return f"""
You are QueryX, a JEE/NEET problem solver.

RULES:
- Clean markdown only
- All equations in LaTeX
- Show formula before calculation
- Be careful with signs and units

{level_line}
{style_line}
{lang_line}

IMPORTANT:
- Do NOT change formula once chosen
- Final answer must be numerically correct
"""

def make_prompt(system_prompt: str, question: str) -> str:
    return f"{system_prompt}\nQuestion:\n{question}\n\nSolve carefully:\n"

def reconsider_prompt(answer: str, correct_value: float) -> str:
    return f"""
Your derivation and formula are correct.
However, arithmetic/sign error detected.

Correct numeric value (verified by Python): {correct_value}

Re-check ONLY calculation and sign.
Do NOT change formula.

Corrected solution:
{answer}
"""

# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    try:
        prompt = make_prompt(
            build_system_prompt(payload.level, payload.style, payload.language),
            payload.question.strip(),
        )

        first = model.generate_content(prompt)
        text = (first.text or "").strip()

        expr = extract_last_expression(text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            second = model.generate_content(
                reconsider_prompt(text, py_val)
            )
            text = (second.text or text).strip()

        if not text:
            text = "Unable to generate a valid solution. Please try again."

        return AnswerResponse(answer_text=text)

    except Exception as e:
        print("ask-text error:", repr(e))
        return AnswerResponse(
            answer_text="Sorry, backend error occurred. Please try again."
        )

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

        first = model.generate_content(
            [
                system_prompt
                + "\nRewrite the question from the image and solve carefully.\n",
                {
                    "mime_type": file.content_type or "image/jpeg",
                    "data": img_bytes,
                },
            ]
        )

        text = (first.text or "").strip()

        expr = extract_last_expression(text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            second = model.generate_content(
                reconsider_prompt(text, py_val)
            )
            text = (second.text or text).strip()

        if not text:
            text = "Unable to solve the image question. Please try another image."

        return AnswerResponse(answer_text=text)

    except Exception as e:
        print("ask-image error:", repr(e))
        return AnswerResponse(
            answer_text="Sorry, image processing error occurred."
        )
