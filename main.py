import os
import time
import json
import traceback
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

import google.generativeai as genai

# Optional image support
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# -------------------------
# Load env
# -------------------------
load_dotenv()

def get_api_key() -> Optional[str]:
    # Prefer GOOGLE_API_KEY, fallback GEMINI_API_KEY
    k = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if k:
        k = k.strip()
    return k

API_KEY = get_api_key()

# Configure Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)

# Model config
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # change if you want
TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1400"))

# Retry config
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
BASE_BACKOFF_SEC = float(os.getenv("GEMINI_BACKOFF_SEC", "1.0"))


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="QueryX Backend", version="1.0.0")

# CORS (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: set your vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Schemas
# -------------------------
class AskTextReq(BaseModel):
    question: str
    level: str = "basic"       # basic/advanced etc
    style: str = "detailed"    # detailed/short etc
    language: str = "hinglish" # english/hinglish etc

class AskResp(BaseModel):
    answer_text: str


# -------------------------
# Helpers
# -------------------------
def build_prompt(question: str, level: str, style: str, language: str) -> str:
    # Simple prompt wrapper (you can enhance later)
    return f"""
You are QueryX, a JEE/NEET tutor.
Answer the question in {language}.
Difficulty level: {level}.
Answer style: {style}.
Use correct physics/math notation. If calculations are needed, show steps briefly and final answer clearly.

Question:
{question}
""".strip()

def should_retry(err_text: str) -> bool:
    t = (err_text or "").lower()
    # Retry on quota/rate limit, transient, timeouts
    retry_signals = [
        "429", "resource_exhausted", "rate", "quota",
        "503", "unavailable", "deadline", "timeout",
        "connection reset", "temporarily", "try again"
    ]
    return any(s in t for s in retry_signals)

def gemini_generate_text(prompt: str) -> str:
    if not API_KEY:
        return "⚠️ API key missing on server. Set GOOGLE_API_KEY / GEMINI_API_KEY in Railway variables."

    model = genai.GenerativeModel(DEFAULT_MODEL)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            )

            # In 0.8.x, resp.text is standard
            text = getattr(resp, "text", None)
            if not text:
                # fallback: stringify
                text = str(resp)
            return text.strip()

        except Exception as e:
            last_err = e
            err_txt = repr(e)

            # LOG FULL ERROR (Railway logs)
            print(f"[GEMINI_ERROR] attempt={attempt}/{MAX_RETRIES} err={err_txt}")
            traceback.print_exc()

            if attempt >= MAX_RETRIES:
                break

            # Retry only if looks transient
            if should_retry(err_txt):
                sleep_s = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            else:
                break

    # Final fallback message (client-safe)
    return "⚠️ Gemini error occurred. Please try again."


def gemini_generate_image(prompt: str, image_bytes: bytes) -> str:
    if not API_KEY:
        return "⚠️ API key missing on server. Set GOOGLE_API_KEY / GEMINI_API_KEY in Railway variables."

    if not PIL_AVAILABLE:
        return "⚠️ PIL not available on server. Add pillow to requirements.txt."

    model = genai.GenerativeModel(DEFAULT_MODEL)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            resp = model.generate_content(
                [prompt, img],
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            )
            text = getattr(resp, "text", None)
            if not text:
                text = str(resp)
            return text.strip()

        except Exception as e:
            last_err = e
            err_txt = repr(e)

            print(f"[GEMINI_IMAGE_ERROR] attempt={attempt}/{MAX_RETRIES} err={err_txt}")
            traceback.print_exc()

            if attempt >= MAX_RETRIES:
                break

            if should_retry(err_txt):
                sleep_s = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                time.sleep(sleep_s)
                continue
            else:
                break

    return "⚠️ Gemini error occurred. Please try again."


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
def debug():
    # Never print actual key
    return {
        "api_key_loaded": bool(API_KEY),
        "model": DEFAULT_MODEL,
        "pillow_available": PIL_AVAILABLE,
        "retries": MAX_RETRIES,
        "backoff": BASE_BACKOFF_SEC,
    }

@app.post("/ask-text", response_model=AskResp)
def ask_text(req: AskTextReq):
    prompt = build_prompt(req.question, req.level, req.style, req.language)
    ans = gemini_generate_text(prompt)
    return {"answer_text": ans}

@app.post("/ask-image", response_model=AskResp)
async def ask_image(
    question: str = "",
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
    file: UploadFile = File(...),
):
    img_bytes = await file.read()
    prompt = build_prompt(question or "Solve the question shown in the image.", level, style, language)
    ans = gemini_generate_image(prompt, img_bytes)
    return {"answer_text": ans}
