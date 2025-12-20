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

# Image support (PIL)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -------------------------
# Load environment
# -------------------------
load_dotenv()

# -------------------------
# Config & Gemini Setup
# -------------------------
APP_NAME = "QueryX PCMB Solver"

# API Key - Priority: GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("CRITICAL: API Key missing in environment variables!")

genai.configure(api_key=API_KEY)

# 2025 Stable Models
MODEL_FALLBACK = ["gemini-1.5-flash", "gemini-1.5-pro"]

# Safety Settings: PCMB questions block na ho isliye BLOCK_NONE use kiya hai
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

GENERATION_CONFIG = {
    "temperature": 0.1, # JEE/NEET ke liye low temperature = high accuracy
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

# -------------------------
# FastAPI App & CORS
# -------------------------
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Models
# -------------------------
class AskRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"    # basic / advanced
    style: Optional[str] = "detailed" # short / detailed
    language: Optional[str] = "hinglish"

# -------------------------
# Helpers
# -------------------------
def _is_math_only(q: str) -> bool:
    return bool(q and re.match(r"^[\s0-9\.\+\-\*\/\%\(\)\^\s]+$", q))

def _clean_latex(text: str) -> str:
    if not text: return text
    # Convert common HTML tags to LaTeX
    text = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", text, flags=re.IGNORECASE)
    return text

def _build_prompt(q: str, level: str, style: str, lang: str) -> str:
    return f"""
You are QueryX, an expert JEE/NEET PCMB (Physics, Chemistry, Maths, Biology) solver.
User Preferences: Level={level}, Style={style}, Language={lang}.

STRICT INSTRUCTIONS:
1. Start directly with the solution. No "Sure, I can help".
2. Use Markdown for structure and $...$ or $$...$$ for ALL math/formulas.
3. Keep it exam-oriented. Explain concepts clearly.
4. If the language is 'hinglish', use a mix of Hindi (Latin script) and English.
5. If the input is simple arithmetic, return only the final result.

Question: {q}
""".strip()

# -------------------------
# Core Logic
# -------------------------
async def _generate_content(prompt: str, image: Optional[Any] = None) -> str:
    request_id = str(uuid.uuid4())[:8]
    last_err = None

    for model_name in MODEL_FALLBACK:
        try:
            model = genai.GenerativeModel(model_name)
            content = [prompt, image] if image else prompt
            
            response = model.generate_content(
                content,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS
            )
            
            if response.candidates and response.candidates[0].content.parts:
                return _clean_latex(response.text.strip())
            
            raise Exception("Response blocked or empty.")

        except Exception as e:
            last_err = str(e)
            print(f"[{request_id}] Model {model_name} failed: {last_err}")
            continue

    return f"⚠️ Error: {last_err} | ID: {request_id}"

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "billing": "active", "api_key_loaded": bool(API_KEY)}

@app.post("/ask-text")
async def ask_text(payload: AskRequest):
    q = payload.question.strip()
    if not q: return {"answer_text": "Sawal toh puchiye bhai!"}
    
    # Fast path for simple math
    if _is_math_only(q):
        try:
            # Simple eval safely
            res = eval(q.replace("^", "**"), {"__builtins__": {}})
            return {"answer_text": str(res)}
        except: pass

    prompt = _build_prompt(q, payload.level, payload.style, payload.language)
    answer = await _generate_content(prompt)
    return {"answer_text": answer}

@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish"
):
    if not PIL_AVAILABLE:
        return {"answer_text": "⚠️ Error: Pillow library missing."}
    
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        prompt = _build_prompt("Solve this question from the image.", level, style, language)
        answer = await _generate_content(prompt, image=img)
        return {"answer_text": answer}
    except Exception as e:
        return {"answer_text": f"⚠️ Image processing error: {str(e)}"}