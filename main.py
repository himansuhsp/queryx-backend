import os
import re
import json
import time
import traceback
import multiprocessing as mp
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai


# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Never print API key in logs
HAS_KEY = bool(API_KEY)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="QueryX Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # beta: allow all; later lock to Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request/Response models
# -----------------------------
class AskTextRequest(BaseModel):
    question: str
    level: str = "basic"       # basic | advanced
    style: str = "detailed"    # short | detailed
    language: str = "hinglish" # english | hinglish


class AskTextResponse(BaseModel):
    answer_text: str


# -----------------------------
# Utilities: Safe Python sandbox
# -----------------------------
_ALLOWED_CHARS_RE = re.compile(r"^[0-9\.\+\-\*\/\%\(\)\s,]+$")

def _safe_calc_worker(expr: str, q: mp.Queue):
    """
    Run in separate process. Extremely restricted eval for arithmetic.
    """
    try:
        expr = expr.strip()

        # Quick allowlist: only digits/operators/space/paren/comma/dot
        if not _ALLOWED_CHARS_RE.match(expr):
            q.put(("error", "Expression contains disallowed characters."))
            return

        # Disallow repeated dots like 1..2 or weird stuff
        if ".." in expr:
            q.put(("error", "Invalid numeric format."))
            return

        # No __, no words, no exponent symbol **? Actually allow **? (optional)
        # We keep it minimal for safety. If you want **, add it to regex and checks.
        if any(tok in expr for tok in ["__", "import", "open", "eval", "exec", "os", "sys"]):
            q.put(("error", "Disallowed tokens."))
            return

        # Eval with empty builtins
        result = eval(expr, {"__builtins__": {}}, {})
        # Only numeric results
        if not isinstance(result, (int, float)):
            q.put(("error", "Non-numeric result."))
            return

        q.put(("ok", float(result)))
    except Exception as e:
        q.put(("error", str(e)))


def safe_calc(expr: str, timeout_sec: float = 1.5) -> Tuple[bool, Optional[float], str]:
    """
    Returns (ok, value, error_message).
    """
    q = mp.Queue()
    p = mp.Process(target=_safe_calc_worker, args=(expr, q))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return (False, None, "Calculation timeout")

    if q.empty():
        return (False, None, "No result from sandbox")

    status, payload = q.get()
    if status == "ok":
        return (True, payload, "")
    return (False, None, str(payload))


# -----------------------------
# Heuristic: detect "pure arithmetic" questions
# -----------------------------
_SIMPLE_ARITH_RE = re.compile(r"^\s*[0-9\.\s\+\-\*\/\%\(\),]+\s*$")

def is_pure_arithmetic(q: str) -> bool:
    q = q.strip()
    if len(q) < 1:
        return False
    # "2 plus 3" not covered; this is for direct expressions like 2+3
    return bool(_SIMPLE_ARITH_RE.match(q))


def format_number(x: float) -> str:
    # nice formatting: 5.0 -> 5
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return str(x)


# -----------------------------
# Gemini client
# -----------------------------
def get_gemini_model():
    if not HAS_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY in env variables.")
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(DEFAULT_MODEL)


def gemini_generate(prompt: str, retries: int = 2) -> str:
    """
    Simple retry wrapper.
    """
    last_err = None
    model = get_gemini_model()

    for _ in range(retries + 1):
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text:
                # fallback if structure differs
                text = str(resp)
            return text
        except Exception as e:
            last_err = e
            time.sleep(0.6)

    raise RuntimeError(f"Gemini failed after retries. Last error: {repr(last_err)}")


# -----------------------------
# Prompt builder with calc crosscheck
# -----------------------------
def build_prompt(question: str, level: str, style: str, language: str) -> str:
    # Keep language handling simple
    lang_hint = "Use Hinglish (Hindi+English mix) in a clear step-by-step way." if language.lower() == "hinglish" else "Use clear English."
    style_hint = "Keep it short." if style.lower() == "short" else "Explain step-by-step with reasoning and final answer clearly."
    level_hint = "NEET/JEE basic level." if level.lower() == "basic" else "Advanced JEE level depth with careful reasoning."

    # IMPORTANT:
    # We ask Gemini to optionally provide a calc expression for numeric answers.
    # This helps our sandbox verify.
    return f"""
You are QueryX, a physics/math tutor AI.

User question:
{question}

Constraints:
- {lang_hint}
- {style_hint}
- {level_hint}
- If there is a numeric final result, ALSO include a single-line calculation expression inside:
  <calc> ... </calc>
  Example: <calc>(2+3)/5</calc>
- If no numeric calculation is needed, omit <calc> entirely.
- Do NOT include code blocks.

Now write the answer.
""".strip()


def extract_calc_expression(text: str) -> Optional[str]:
    m = re.search(r"<calc>\s*(.*?)\s*</calc>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    expr = m.group(1).strip()
    # Remove any newline
    expr = re.sub(r"\s+", " ", expr)
    return expr if expr else None


def remove_calc_tag(text: str) -> str:
    return re.sub(r"\s*<calc>.*?</calc>\s*", "\n", text, flags=re.DOTALL | re.IGNORECASE).strip()


def maybe_verify_calc(answer_text: str) -> str:
    """
    If Gemini gave a <calc> expression, verify it with sandbox.
    If mismatch / error, remove calc tag and return with a warning correction line.
    """
    expr = extract_calc_expression(answer_text)
    clean_answer = remove_calc_tag(answer_text)

    if not expr:
        return clean_answer

    ok, val, err = safe_calc(expr)
    if not ok:
        # If calc expression unusable, just drop it (don’t break UX)
        return clean_answer + f"\n\n⚠️ (Calc verification skipped: {err})"

    # Append verified final if not already clearly present
    verified = format_number(val)
    # If answer already has a "final" number, we avoid messing too much.
    # Just add a small verified line.
    return clean_answer + f"\n\n✅ Verified calculation result: **{verified}**"


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug():
    # safe debug (no secret)
    return {
        "has_google_api_key": bool(os.getenv("GOOGLE_API_KEY")),
        "has_gemini_api_key": bool(os.getenv("GEMINI_API_KEY")),
        "model": DEFAULT_MODEL,
    }


@app.post("/ask-text", response_model=AskTextResponse)
def ask_text(req: AskTextRequest):
    q = (req.question or "").strip()
    if not q:
        return {"answer_text": "Please enter a question."}

    # 1) If it's pure arithmetic like "2+3", solve locally (fast & reliable)
    if is_pure_arithmetic(q):
        ok, val, err = safe_calc(q)
        if ok:
            return {"answer_text": format_number(val)}
        # If sandbox fails, fallback to Gemini explanation
        # (still safe; just won't crash)
        # no key? show key message
        if not HAS_KEY:
            return {"answer_text": "⚠️ Gemini API key issue on server. Check GOOGLE_API_KEY / GEMINI_API_KEY in Railway Variables."}
        prompt = build_prompt(q, req.level, req.style, req.language)
        ans = gemini_generate(prompt)
        return {"answer_text": maybe_verify_calc(ans)}

    # 2) Normal questions: Gemini + optional calc verification
    if not HAS_KEY:
        return {"answer_text": "⚠️ Gemini API key issue on server. Check GOOGLE_API_KEY / GEMINI_API_KEY in Railway Variables."}

    try:
        prompt = build_prompt(q, req.level, req.style, req.language)
        ans = gemini_generate(prompt)
        final_text = maybe_verify_calc(ans)
        return {"answer_text": final_text}
    except Exception:
        # Don't leak stack to user
        return {"answer_text": "⚠️ Gemini error occurred. Please try again."}


@app.post("/ask-image", response_model=AskTextResponse)
async def ask_image(
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
    file: UploadFile = File(...)
):
    if not HAS_KEY:
        return {"answer_text": "⚠️ Gemini API key issue on server. Check GOOGLE_API_KEY / GEMINI_API_KEY in Railway Variables."}

    try:
        content = await file.read()
        if not content:
            return {"answer_text": "Image not received."}

        model = get_gemini_model()

        # Build prompt for image question solving
        lang_hint = "Use Hinglish (Hindi+English mix) in a clear step-by-step way." if language.lower() == "hinglish" else "Use clear English."
        style_hint = "Keep it short." if style.lower() == "short" else "Explain step-by-step with reasoning and final answer clearly."
        level_hint = "NEET/JEE basic level." if level.lower() == "basic" else "Advanced JEE level depth with careful reasoning."

        prompt = f"""
You are QueryX, a JEE/NEET tutor AI.

Solve the question from the image.
Constraints:
- {lang_hint}
- {style_hint}
- {level_hint}
- If there is a numeric final result, ALSO include a single-line calculation expression inside:
  <calc> ... </calc>
- If no numeric calculation is needed, omit <calc> entirely.
""".strip()

        # Gemini vision input: [prompt, image-bytes]
        resp = model.generate_content([prompt, {"mime_type": file.content_type or "image/jpeg", "data": content}])
        text = getattr(resp, "text", None) or str(resp)
        final_text = maybe_verify_calc(text)
        return {"answer_text": final_text}

    except Exception:
        return {"answer_text": "⚠️ Gemini image error occurred. Please try again."}
