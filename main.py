import os
import re
import time
import uuid
import io
import traceback
from typing import Optional, List, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Optional image support (PIL)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -------------------------
# Load env
# -------------------------
load_dotenv()

# -------------------------
# Config
# -------------------------
APP_NAME = "QueryX Backend"

# API key validation
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # Isse backend start hi nahi hoga agar key missing hai
    print("CRITICAL ERROR: GEMINI_API_KEY is not set in Environment Variables!")

genai.configure(api_key=API_KEY)

# Stable models
ENV_MODEL = os.getenv("GEMINI_MODEL", "").strip()
MODEL_FALLBACK = []
if ENV_MODEL:
    MODEL_FALLBACK.append(ENV_MODEL)
MODEL_FALLBACK += ["gemini-1.5-flash", "gemini-1.5-pro"]

# Safety Settings (BLOCK_NONE taaki PCMB questions block na ho)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 1500,
}

# -------------------------
# FastAPI setup
# -------------------------
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

# -------------------------
# Models & Helpers
# -------------------------
class AskTextRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"
    style: Optional[str] = "detailed"
    language: Optional[str] = "hinglish"

def _is_pure_arithmetic(q: str) -> bool:
    q = (q or "").strip()
    return bool(q) and bool(re.match(r"^[\s0-9\.\+\-\*\/\%\(\)\^\s]+$", q))

def _safe_eval_arithmetic(expr: str) -> float:
    expr = (expr or "").strip().replace("^", "**")
    import ast
    tree = ast.parse(expr, mode="eval")
    return eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, {})

def _html_to_latex(text: str) -> str:
    if not text: return text
    text = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return text

def _build_prompt(question: str, level: str, style: str, language: str) -> str:
    return f"""
You are QueryX, an expert JEE/NEET PCMB solver.
User preferences: Level={level}, Style={style}, Language={language}.

STRICT RULES:
1. Output MUST be Markdown.
2. Math MUST be in LaTeX ($...$ or $$...$$).
3. Be direct. Start solving immediately.
4. If arithmetic, return only the number.

Question: {question}
""".strip()

def _extract_text(resp: Any) -> str:
    if not resp.candidates:
        return "⚠️ Response blocked by Safety Filters."
    try:
        return resp.text.strip()
    except Exception:
        try:
            parts = [p.text for p in resp.candidates[0].content.parts if hasattr(p, 'text')]
            return "\n".join(parts).strip()
        except:
            return ""

# -------------------------
# Core Gemini Logic
# -------------------------
def _gemini_generate(prompt: str, request_id: str, image: Optional[Any] = None) -> str:
    last_err = None
    for model_name in MODEL_FALLBACK:
        model = genai.GenerativeModel(model_name)
        for attempt in range(1, 3):
            try:
                content = [prompt, image] if image else prompt
                resp = model.generate_content(
                    content,
                    generation_config=GENERATION_CONFIG,
                    safety_settings=SAFETY_SETTINGS
                )
                text = _extract_text(resp)
                if text:
                    return _html_to_latex(text)
                raise RuntimeError("Gemini returned empty text.")
            except Exception as e:
                last_err = e
                print(f"[{request_id}] {model_name} attempt {attempt} failed: {str(e)}")
                time.sleep(1)
    
    # Isse error bubble up hokar post route tak jayega
    raise last_err if last_err else RuntimeError("Unknown API Error")

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "has_key": bool(API_KEY), "pil": PIL_AVAILABLE}

@app.post("/ask-text")
def ask_text(payload: AskTextRequest):
    request_id = str(uuid.uuid4())[:8]
    q = (payload.question or "").strip()
    if not q: return {"answer_text": "⚠️ Question empty hai."}

    if _is_pure_arithmetic(q):
        try:
            val = _safe_eval_arithmetic(q)
            return {"answer_text": str(int(val) if abs(val - int(val)) < 1e-12 else val)}
        except: pass

    prompt = _build_prompt(q, payload.level, payload.style, payload.language)
    try:
        answer = _gemini_generate(prompt, request_id)
        return {"answer_text": answer}
    except Exception as e:
        # DEBUG: Asli error ko frontend par bhejo
        err_msg = str(e)
        print(f"[{request_id}] ERROR: {traceback.format_exc()}")
        return {"answer_text": f"⚠️ DEBUG ERROR: {err_msg} | ID: {request_id}"}

@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    request_id = str(uuid.uuid4())[:8]
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        prompt = _build_prompt("Solve the question from this image.", level, style, language)
        answer = _gemini_generate(prompt, request_id, image=img)
        return {"answer_text": answer}
    except Exception as e:
        return {"answer_text": f"⚠️ DEBUG ERROR: {str(e)} | ID: {request_id}"}