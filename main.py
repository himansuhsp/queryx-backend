import os
import re
import io
import json
import time
import uuid
from typing import Optional, List, Dict, Any

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


# -------------------------
# Load env (local dev only)
# -------------------------
load_dotenv()

# -------------------------
# Config
# -------------------------
APP_NAME = "QueryX Backend"

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in Railway.")

genai.configure(api_key=API_KEY)

ENV_MODEL = (os.getenv("GEMINI_MODEL") or "").strip()

MODEL_FALLBACK: List[str] = []
if ENV_MODEL:
    MODEL_FALLBACK.append(ENV_MODEL)

# Fallback chain (safe)
MODEL_FALLBACK += [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# CORS
DEFAULT_ALLOWED_ORIGINS = [
    "https://queryx-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
# Optional override via env (comma separated)
origins_env = (os.getenv("ALLOWED_ORIGINS") or "").strip()
if origins_env:
    ALLOWED_ORIGINS = [x.strip() for x in origins_env.split(",") if x.strip()]
else:
    ALLOWED_ORIGINS = DEFAULT_ALLOWED_ORIGINS

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Models
# -------------------------
class AskTextRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"        # basic / advanced
    style: Optional[str] = "detailed"     # detailed / short
    language: Optional[str] = "hinglish"  # hinglish / english


# -------------------------
# Small helpers
# -------------------------
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
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.FloorDiv,
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed token: {type(node).__name__}")

    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, {})


# Convert accidental HTML sub/sup to LaTeX style
SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE | re.DOTALL)
SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)

def _html_to_latex(text: str) -> str:
    if not text:
        return text
    text = SUB_RE.sub(r"_{\1}", text)
    text = SUP_RE.sub(r"^{\1}", text)
    return text


def _extract_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text and str(text).strip():
        return str(text).strip()

    # fallback candidates
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


def _gemini_call(prompt: Any, request_id: str) -> str:
    """
    prompt can be str OR [str, PIL.Image]
    """
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1400,
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
                text = _html_to_latex(text)

                if text:
                    return text

                raise RuntimeError("Empty response from Gemini")

            except Exception as e:
                last_err = e
                print(f"[{request_id}] model={model_name} attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {str(e)[:220]}")
                if attempt < max_attempts:
                    time.sleep(base_sleep * (2 ** (attempt - 1)))

        print(f"[{request_id}] Switching model fallback after failures: {model_name}")

    raise RuntimeError(f"All models failed. Last error: {type(last_err).__name__}: {str(last_err)}")


def _safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()

    # if model wrapped in ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _detect_options_in_text(q: str) -> List[str]:
    """
    Detect common option patterns:
    A) ... B) ...
    (A) ... (B) ...
    A. ... B. ...
    """
    if not q:
        return []
    # rough split capture
    patterns = [
        r"\(A\)(.*?)(?=\(B\)|$)",
        r"\(B\)(.*?)(?=\(C\)|$)",
        r"\(C\)(.*?)(?=\(D\)|$)",
        r"\(D\)(.*?)(?=\(E\)|$)",
    ]
    opts = []
    for i, p in enumerate(patterns):
        m = re.search(p, q, flags=re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            label = ["A", "B", "C", "D"][i]
            opts.append(f"({label}) {content}")
    if opts:
        return opts

    # Another format A) B) ...
    for label in ["A", "B", "C", "D"]:
        m = re.search(rf"{label}\)\s*(.*?)(?=(A\)|B\)|C\)|D\)|$)", q, flags=re.DOTALL)
        if m:
            opts.append(f"({label}) {m.group(1).strip()}")

    return [o for o in opts if len(o) > 4]


# -------------------------
# 1) Classify + Extract (JSON)
# -------------------------
def _prompt_extract(question: str, level: str, style: str, language: str) -> str:
    # keep it deterministic
    return f"""
You are a strict parser for JEE/NEET PCMB questions.

Return ONLY valid JSON (no markdown, no extra text).
JSON schema:
{{
  "subject": "physics|chem|math|bio|general",
  "question_clean": "...",
  "is_objective": true/false,
  "options": ["(A) ...", "(B) ..."] ,
  "asked": "...",
  "givens": [{{"name":"", "value":"", "unit":""}}],
  "expected_answer": "option|numeric|text"
}}

Rules:
- If options exist, set is_objective=true and fill options.
- question_clean should be the cleaned problem statement (without junk).
- expected_answer:
  - "option" if MCQ,
  - "numeric" if asks value,
  - else "text".
- Keep strings short & clean.

User preferences context:
level={level}, style={style}, language={language}

Question:
{question}
""".strip()


def _extract_structured(question: str, level: str, style: str, language: str, request_id: str) -> Dict[str, Any]:
    # pre-detect options as a hint (helps extraction)
    detected_opts = _detect_options_in_text(question)
    hint = ""
    if detected_opts:
        hint = "\n\nOptions detected (hint):\n" + "\n".join(detected_opts)

    prompt = _prompt_extract(question + hint, level, style, language)
    raw = _gemini_call(prompt, request_id)
    data = _safe_json_parse(raw)

    # fallback if JSON fails
    if not data:
        # minimal extraction
        return {
            "subject": "general",
            "question_clean": question.strip(),
            "is_objective": bool(detected_opts),
            "options": detected_opts,
            "asked": "",
            "givens": [],
            "expected_answer": "option" if detected_opts else "text",
        }

    # normalize
    data.setdefault("subject", "general")
    data.setdefault("question_clean", question.strip())
    data.setdefault("is_objective", bool(detected_opts))
    data.setdefault("options", detected_opts)
    data.setdefault("asked", "")
    data.setdefault("givens", [])
    data.setdefault("expected_answer", "option" if data.get("is_objective") else "text")

    # if model forgot options but we detected
    if detected_opts and not data.get("options"):
        data["options"] = detected_opts
        data["is_objective"] = True
        data["expected_answer"] = "option"

    return data


# -------------------------
# 2) Solve Prompt (Markdown + LaTeX)
# -------------------------
def _prompt_solve(structured: Dict[str, Any], level: str, style: str, language: str) -> str:
    level = (level or "basic").lower()
    style = (style or "detailed").lower()
    language = (language or "hinglish").lower()

    is_obj = bool(structured.get("is_objective"))
    options = structured.get("options") or []

    output_rules = """
Output rules (VERY IMPORTANT):
1) Output MUST be Markdown.
2) For math, use LaTeX only inside $...$ or $$...$$.
   Examples: $Q_{enc}$, $10^{3}$, $\\Phi = \\int \\vec{E}\\cdot d\\vec{A}$.
3) DO NOT use HTML tags like <sub>, <sup>, <br>, etc.
4) Be correct and step-by-step (but not ChatGPT-like long essays).
5) Keep it exam-oriented: given -> formula -> substitution -> final.
""".strip()

    if is_obj:
        return f"""
You are QueryX, a JEE/NEET solver.

{output_rules}

This is an MCQ. Pick the correct option.
Return format:
- First line: **Answer: (A/B/C/D)** only
- Then 2-6 short bullet points as justification (very concise).

Preferences:
- Level: {level}
- Style: {style}
- Language: {language}

Question (cleaned):
{structured.get("question_clean","")}

Options:
{chr(10).join(options)}
""".strip()

    return f"""
You are QueryX, a JEE/NEET PCMB tutor.

{output_rules}

Preferences:
- Level: {level}
- Style: {style}
- Language: {language}

Question (cleaned):
{structured.get("question_clean","")}

Asked:
{structured.get("asked","")}

Givens (if any):
{json.dumps(structured.get("givens", []), ensure_ascii=False)}

Write the final answer clearly at the end as:
**Final Answer:** ...
""".strip()


# -------------------------
# 3) Verify + Auto-fix Prompt
# -------------------------
def _prompt_verify(original_question: str, structured: Dict[str, Any], draft_answer: str) -> str:
    return f"""
You are a strict verifier for JEE/NEET solutions.

You will check the draft answer for:
- calculation mistakes
- sign/unit/dimension mistakes
- wrong option (if MCQ)
- missing final answer

Return ONLY valid JSON (no markdown, no extra text):
{{
  "changed": true/false,
  "final_answer_markdown": "...",
  "issues": ["...","..."]
}}

Rules:
- If draft is correct, changed=false and final_answer_markdown = draft (but cleaned).
- If incorrect/weak, changed=true and provide corrected final answer (still concise, markdown).
- NEVER output HTML tags like <sub>/<sup>. Use LaTeX $...$ only.

Original question:
{original_question}

Structured info:
{json.dumps(structured, ensure_ascii=False)}

Draft answer:
{draft_answer}
""".strip()


def _verify_and_fix(original_question: str, structured: Dict[str, Any], draft: str, request_id: str) -> str:
    prompt = _prompt_verify(original_question, structured, draft)
    raw = _gemini_call(prompt, request_id)
    data = _safe_json_parse(raw)

    if not data or not isinstance(data, dict):
        # if verifier JSON fails, just return draft
        return draft

    final_md = (data.get("final_answer_markdown") or "").strip()
    if not final_md:
        return draft

    return _html_to_latex(final_md)


# -------------------------
# Main Pipeline
# -------------------------
def solve_pipeline(question: str, level: str, style: str, language: str, request_id: str) -> str:
    q = (question or "").strip()
    if not q:
        return "⚠️ Question empty hai."

    # 0) Local arithmetic quick win
    if _is_pure_arithmetic(q):
        try:
            val = _safe_eval_arithmetic(q)
            if abs(val - int(val)) < 1e-12:
                return str(int(val))
            return str(val)
        except Exception:
            pass

    # 1) Extract structured
    structured = _extract_structured(q, level, style, language, request_id)

    # 2) Solve
    solve_prompt = _prompt_solve(structured, level, style, language)
    draft = _gemini_call(solve_prompt, request_id)

    # 3) Verify + fix (2nd pass)
    final = _verify_and_fix(q, structured, draft, request_id)

    return final.strip()


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME, "models": MODEL_FALLBACK}

@app.get("/health")
def health():
    return {"ok": True, "models": MODEL_FALLBACK, "pil": PIL_AVAILABLE, "allowed_origins": ALLOWED_ORIGINS}

@app.post("/ask-text")
def ask_text(payload: AskTextRequest):
    request_id = str(uuid.uuid4())[:8]
    try:
        ans = solve_pipeline(payload.question, payload.level or "basic", payload.style or "detailed", payload.language or "hinglish", request_id)
        return {"answer_text": ans}
    except Exception as e:
        print(f"[{request_id}] ERROR ask-text: {type(e).__name__}: {str(e)}")
        return {"answer_text": "⚠️ Gemini error occurred. Please try again."}


@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    request_id = str(uuid.uuid4())[:8]

    if not PIL_AVAILABLE:
        return {"answer_text": "⚠️ Image support disabled (PIL not installed). requirements.txt me pillow add karo."}

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")  # type: ignore
    except Exception as e:
        print(f"[{request_id}] ERROR reading image: {type(e).__name__}: {str(e)}")
        return {"answer_text": "⚠️ Image read nahi ho pa raha. Clear photo + supported format (jpg/png) try karo."}

    # Step A: extract text from image as JSON
    extract_prompt = """
You are a strict OCR+parser.

Look at the image and extract the question.
Return ONLY valid JSON:
{
  "question": "... full cleaned question text ...",
  "options": ["(A) ...", "(B) ...", "(C) ...", "(D) ..."]   // empty if none
}

Rules:
- No markdown, no extra text.
- Keep it clean and readable.
- If options are visible, include them.
""".strip()

    try:
        raw = _gemini_call([extract_prompt, img], request_id)
        data = _safe_json_parse(raw) or {}
        extracted_q = (data.get("question") or "").strip()
        extracted_opts = data.get("options") or []

        if extracted_opts:
            extracted_q = extracted_q + "\n\nOptions:\n" + "\n".join(extracted_opts)

        if not extracted_q:
            return {"answer_text": "⚠️ Image se question extract nahi ho paya. Clear photo try karo."}

        ans = solve_pipeline(extracted_q, level, style, language, request_id)
        return {"answer_text": ans}

    except Exception as e:
        print(f"[{request_id}] ERROR ask-image: {type(e).__name__}: {str(e)}")
        return {"answer_text": "⚠️ Gemini error occurred. Please try again."}
