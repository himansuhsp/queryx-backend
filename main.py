import os
import re
import time
import uuid
import io
from typing import Optional, List

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


load_dotenv()

APP_NAME = "QueryX Backend"

ENV_MODEL = os.getenv("GEMINI_MODEL", "").strip()

MODEL_FALLBACK: List[str] = []
if ENV_MODEL:
    MODEL_FALLBACK.append(ENV_MODEL)

# safer fallback order (1.5-flash is most stable)
MODEL_FALLBACK += [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY in Railway variables.")

genai.configure(api_key=API_KEY)


app = FastAPI(title=APP_NAME)

ALLOWED_ORIGINS = [
    "https://queryx-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskTextRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"
    style: Optional[str] = "detailed"
    language: Optional[str] = "hinglish"


MATH_EXPR_RE = re.compile(r"^[\s0-9\.\+\-\*\/\%\(\)\^\s]+$")

def _is_pure_arithmetic(q: str) -> bool:
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
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Load, ast.FloorDiv,
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed token: {type(node).__name__}")

    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, {})


SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE | re.DOTALL)
SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)

def _html_to_latex(text: str) -> str:
    if not text:
        return text
    text = SUB_RE.sub(r"_{\1}", text)
    text = SUP_RE.sub(r"^{\1}", text)
    return text


def _build_prompt(question: str, level: str, style: str, language: str) -> str:
    level = (level or "basic").lower()
    style = (style or "detailed").lower()
    language = (language or "hinglish").lower()

    return f"""
You are QueryX, a PCMB tutor for JEE/NEET.

User preferences:
- Level: {level}
- Style: {style}
- Language: {language}

Output rules (VERY IMPORTANT):
1) Output MUST be Markdown.
2) For math, use LaTeX only inside $...$ or $$...$$.
3) DO NOT use HTML tags like <sub>, <sup>, <br>, etc.
4) Avoid ChatGPT tone. Be like a confident JEE teacher.
5) If input is purely arithmetic, return ONLY the final number.

Format:
FINAL ANSWER: ...
STEPS:
1) ...
2) ...

Question:
{question}
""".strip()


def _extract_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text and text.strip():
        return text.strip()

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

    return ""


def _err_brief(e: Exception) -> str:
    s = str(e) or repr(e)
    # keep logs readable
    s = s.replace("\n", " ")[:400]
    return f"{type(e).__name__}: {s}"


def _gemini_generate(prompt: str, request_id: str) -> str:
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1200,
    }

    max_attempts = 3
    base_sleep = 1.0
    last_err: Optional[Exception] = None

    for model_name in MODEL_FALLBACK:
        model = genai.GenerativeModel(model_name)

        for attempt in range(1, max_attempts + 1):
            try:
                resp = model.generate_content(prompt, generation_config=generation_config)
                text = _extract_text(resp)
                if text:
                    return _html_to_latex(text)
                raise RuntimeError("Empty response from Gemini")

            except Exception as e:
                last_err = e
                print(f"[{request_id}] model={model_name} attempt {attempt}/{max_attempts} failed -> {_err_brief(e)}")
                if attempt < max_attempts:
                    time.sleep(base_sleep * (2 ** (attempt - 1)))

        print(f"[{request_id}] switching model fallback after failures: {model_name}")

    raise RuntimeError(f"All models failed. Last error -> {_err_brief(last_err)}")  # type: ignore


def _gemini_generate_image(prompt: str, image: "Image.Image", request_id: str) -> str:
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1200,
    }

    max_attempts = 3
    base_sleep = 1.0
    last_err: Optional[Exception] = None

    for model_name in MODEL_FALLBACK:
        model = genai.GenerativeModel(model_name)

        for attempt in range(1, max_attempts + 1):
            try:
                resp = model.generate_content([prompt, image], generation_config=generation_config)
                text = _extract_text(resp)
                if text:
                    return _html_to_latex(text)
                raise RuntimeError("Empty response from Gemini (image)")

            except Exception as e:
                last_err = e
                print(f"[{request_id}] model={model_name} image attempt {attempt}/{max_attempts} failed -> {_err_brief(e)}")
                if attempt < max_attempts:
                    time.sleep(base_sleep * (2 ** (attempt - 1)))

        print(f"[{request_id}] switching model fallback after failures (image): {model_name}")

    raise RuntimeError(f"All models failed (image). Last error -> {_err_brief(last_err)}")  # type: ignore


@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "models": MODEL_FALLBACK}

@app.get("/health")
def health():
    return {"ok": True, "models": MODEL_FALLBACK, "pil": PIL_AVAILABLE}

@app.post("/ask-text")
def ask_text(payload: AskTextRequest):
    request_id = str(uuid.uuid4())[:8]
    q = (payload.question or "").strip()

    if not q:
        return {"answer_text": "⚠️ Question empty hai.", "request_id": request_id}

    if _is_pure_arithmetic(q):
        try:
            val = _safe_eval_arithmetic(q)
            if abs(val - int(val)) < 1e-12:
                return {"answer_text": str(int(val)), "request_id": request_id}
            return {"answer_text": str(val), "request_id": request_id}
        except Exception:
            pass

    prompt = _build_prompt(q, payload.level, payload.style, payload.language)

    try:
        answer = _gemini_generate(prompt, request_id)
        return {"answer_text": answer, "request_id": request_id}
    except Exception as e:
        print(f"[{request_id}] ERROR ask-text -> {_err_brief(e)}")
        return {"answer_text": f"⚠️ Gemini error occurred. Request id: {request_id}", "request_id": request_id}


@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    request_id = str(uuid.uuid4())[:8]

    if not PIL_AVAILABLE:
        return {"answer_text": "⚠️ Image support disabled. requirements.txt me pillow add karo.", "request_id": request_id}

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")  # type: ignore
    except Exception as e:
        print(f"[{request_id}] ERROR reading image -> {_err_brief(e)}")
        return {"answer_text": "⚠️ Image read nahi ho pa raha. Clear photo (jpg/png) try karo.", "request_id": request_id}

    prompt = _build_prompt("Solve the question from the image.", level, style, language)

    try:
        answer = _gemini_generate_image(prompt, img, request_id)
        return {"answer_text": answer, "request_id": request_id}
    except Exception as e:
        print(f"[{request_id}] ERROR ask-image -> {_err_brief(e)}")
        return {"answer_text": f"⚠️ Gemini error occurred. Request id: {request_id}", "request_id": request_id}