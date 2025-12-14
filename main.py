import os
import re
import ast
import math
from typing import Literal

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------
# LOAD ENV + CONFIG
# ---------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

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
# PYTHON CALCULATION HELPERS (SAFE)
# ---------------------------------------------------------

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


def safe_eval(expr: str) -> float | None:
    """
    Safely evaluate a mathematical expression.
    Only arithmetic + math functions allowed.
    """
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

        return eval(
            compile(tree, "<expr>", "eval"),
            {"__builtins__": {}},
            ALLOWED_NAMES,
        )
    except Exception:
        return None


def extract_last_expression(text: str) -> str | None:
    """
    Try to extract the final numeric expression from Gemini output.
    Example:
      Final Answer: 16 m
      => returns "16"
    """
    matches = re.findall(r"([-+*/().0-9eE]+)", text)
    if not matches:
        return None
    return matches[-1]


# ---------------------------------------------------------
# PROMPT HELPERS
# ---------------------------------------------------------

def build_system_prompt(level: str, style: str, language: str) -> str:
    if level == "advanced":
        level_line = (
            "Explain with JEE/NEET exam depth, derivations, and correct sign conventions."
        )
    else:
        level_line = (
            "Explain conceptually for class 11–12 level using only required formulas."
        )

    if style == "detailed":
        style_line = (
            "Give step-by-step solution with clear derivation and calculation steps."
        )
    else:
        style_line = (
            "Give short, exam-ready solution with final formula and answer only."
        )

    if language == "hinglish":
        lang_line = (
            "Use Hinglish (Roman Hindi + English). Equations strictly in LaTeX."
        )
    else:
        lang_line = "Use clear English. Equations strictly in LaTeX."

    return (
        "You are QueryX, a JEE/NEET problem solver.\n\n"
        "RULES:\n"
        "- Give clean markdown output only.\n"
        "- Use LaTeX for all equations.\n"
        "- Show formula clearly before substituting values.\n"
        "- Be careful with arithmetic signs (+ / -).\n\n"
        f"{level_line}\n"
        f"{style_line}\n"
        f"{lang_line}\n\n"
        "IMPORTANT:\n"
        "- Once a formula is chosen, DO NOT change it unless told.\n"
        "- Final numeric answer must match correct calculation.\n"
    )


def make_prompt(system_prompt: str, question: str) -> str:
    return (
        system_prompt
        + "\nQuestion:\n"
        + question
        + "\n\nNow solve:\n"
    )


def reconsider_prompt(original_answer: str, python_value: float) -> str:
    return (
        "Your formula and derivation are correct.\n"
        "However, there is an arithmetic / sign mismatch.\n\n"
        f"Python calculation gives: {python_value}\n\n"
        "Re-check ONLY the calculation and sign.\n"
        "DO NOT change the formula.\n\n"
        "Give corrected final steps and answer:\n"
        + original_answer
    )


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask-text", response_model=AnswerResponse)
async def ask_text(payload: AskTextRequest):
    system_prompt = build_system_prompt(
        payload.level, payload.style, payload.language
    )
    prompt = make_prompt(system_prompt, payload.question.strip())

    try:
        first = model.generate_content(prompt)
        text = (first.text or "").strip()

        expr = extract_last_expression(text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            # Ask Gemini to reconsider calculation once
            reconsider = reconsider_prompt(text, py_val)
            second = model.generate_content(reconsider)
            text = (second.text or text).strip()

    except Exception as e:
        print("Error /ask-text:", repr(e))
        text = "Sorry, calculation error occurred. Please try again."

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

        contents = [
            system_prompt
            + "\nRewrite the question from the image and solve carefully.\n",
            {"mime_type": file.content_type or "image/jpeg", "data": img_bytes},
        ]

        first = model.generate_content(contents)
        text = (first.text or "").strip()

        expr = extract_last_expression(text)
        py_val = safe_eval(expr) if expr else None

        if py_val is not None:
            reconsider = reconsider_prompt(text, py_val)
            second = model.generate_content(reconsider)
            text = (second.text or text).strip()

    except Exception as e:
        print("Error /ask-image:", repr(e))
        text = "Sorry, image processing error occurred."

    return AnswerResponse(answer_text=text)
