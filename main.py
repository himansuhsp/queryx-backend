import os
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

# Gemini 2.0 Flash
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

# 🔴 YAHAN PE GALTI THI – ab sahi:
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
# PROMPT HELPERS (LATEX + DETAILED/SHORT CLEAR DIFFERENCE)
# ---------------------------------------------------------


def build_system_prompt(level: str, style: str, language: str) -> str:
    # Level
    if level == "advanced":
        level_line = (
            "Explain with JEE/NEET exam depth, derivations, multiple cases "
            "and mention important limiting cases / approximations where useful."
        )
    else:
        level_line = (
            "Explain conceptually for class 11–12 level without unnecessary complications. "
            "Use simple language and only the formulas actually needed."
        )

    # Style – yahan difference bahut clear bana rahe hain
    if style == "detailed":
        style_line = (
            "Give a DETAILED, step-by-step solution:\n"
            "- Use clear headings like 'Concept', 'Given', 'Solution', 'Result'.\n"
            "- Use numbered steps for derivation and reasoning.\n"
            "- Length around 300–600 words (ya 6–12 bullet/steps).\n"
            "- Include 1–2 short examples or comments if helpful."
        )
    else:  # short
        style_line = (
            "Give a SHORT, exam-ready answer ONLY:\n"
            "- Maximum 4–6 lines OR up to 5 bullet points.\n"
            "- NO extra examples, stories, or side comments.\n"
            "- ONLY the key concept, main formula, and final result.\n"
            "- Target length ≈ 80–150 words total.\n"
            "- Do NOT add headings; sirf chhota crisp explanation do."
        )

    # Language
    if language == "hinglish":
        lang_line = (
            "Write using mixed Hindi + English (Hinglish) in Roman script. "
            "Sentences Hinglish me ho, lekin equations LaTeX me likho."
        )
    else:
        lang_line = "Write explanation in clear English. Equations in LaTeX."

    return (
        "You are QueryX, a PCMB question solver for JEE/NEET.\n\n"
        "GOAL:\n"
        "- Provide only the final answer in clean markdown.\n"
        "- Do NOT output JSON, and do NOT output code fences.\n"
        "- Use LaTeX for all mathematical expressions.\n"
        "  Inline examples: $F = ma$, $Q_{enc}$, $\\vec{E} \\cdot d\\vec{A}$.\n"
        "  Block examples:\n\n"
        "    $$ \\oint_S \\vec{E} \\cdot dA = \\frac{Q_{enc}}{\\varepsilon_0} $$\n"
        "    $$ V(r) = \\frac{1}{4\\pi\\varepsilon_0} \\frac{q}{r} $$\n\n"
        "STYLE RULES (VERY IMPORTANT):\n"
        f"- {level_line}\n"
        f"- {style_line}\n"
        f"- {lang_line}\n"
        "- Do not mention internal reasoning.\n"
        "- Do not mention that you are an AI model.\n"
        "- Final answer must be clean and formatted.\n\n"
        "STRICT LATEX RULES:\n"
        "- Always use $ ... $ for inline LaTeX.\n"
        "- Always use $$ ... $$ for block LaTeX on separate lines.\n"
        "- No HTML tags inside equations.\n"
        "- No unicode subscripts (Q₀). Use LaTeX subscripts (Q_0).\n"
        "- Never wrap LaTeX inside backticks.\n"
    )


def make_markdown_prompt(system_prompt: str, question: str) -> str:
    return (
        system_prompt
        + "\n\nQuestion:\n"
        + question
        + "\n\nAnswer format instructions (follow strictly):\n"
        + "- Use markdown paragraphs + bullet / numbered lists.\n"
        + "- All maths strictly in LaTeX.\n"
        + "- Inline examples: $F = ma$, $T = 2\\pi\\sqrt{L/g}$.\n"
        + "- Block examples:\n"
        + "    $$ W = \\int_{x_1}^{x_2} F(x) \\, dx $$\n"
        + "    $$ a(t) = \\frac{dv}{dt}, \\quad v(t) = \\frac{dx}{dt} $$\n"
        + "- Do NOT output triple backticks.\n"
        "- Do NOT output JSON.\n\n"
        "Now give the final answer:\n"
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
    full_prompt = make_markdown_prompt(system_prompt, payload.question.strip())

    try:
        result = model.generate_content(full_prompt)
        text = (result.text or "").strip()
    except Exception as e:
        print("Gemini error in /ask-text:", repr(e))
        text = "Sorry, backend me kuch error aa gaya. Please try again."

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
            + "\nRewrite the question clearly from the image. Then solve step-by-step.\n",
            {"mime_type": file.content_type or "image/jpeg", "data": img_bytes},
        ]

        result = model.generate_content(contents)
        text = (result.text or "").strip()

    except Exception as e:
        print("Gemini error in /ask-image:", repr(e))
        text = "Sorry, image se question read karte waqt error aa gaya."

    return AnswerResponse(answer_text=text)
