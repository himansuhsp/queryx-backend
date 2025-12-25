import os
import re
import time
import uuid
import io
from typing import Optional, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai

# -------------------------
# ENV
# -------------------------
load_dotenv()

APP_NAME = "QueryX PCMB Solver"

API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing in environment")

genai.configure(api_key=API_KEY)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title=APP_NAME)

# -------------------------
# 🔥 CORS (FINAL, SIMPLE, WORKING)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all (safe for now)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Gemini Models
# -------------------------
MODEL_FALLBACK = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_output_tokens": 2048,
}

# -------------------------
# Image support
# -------------------------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -------------------------
# Request schema
# -------------------------
class AskRequest(BaseModel):
    question: str
    level: Optional[str] = "basic"
    style: Optional[str] = "detailed"
    language: Optional[str] = "hinglish"

# -------------------------
# Helpers
# -------------------------
MATH_RE = re.compile(r"^[0-9\.\+\-\*\/\%\(\)\^\s]+$")

def is_simple_math(q: str) -> bool:
    return bool(MATH_RE.match(q.strip()))

def safe_eval(expr: str) -> str:
    expr = expr.replace("^", "**")
    return str(eval(expr, {"__builtins__": {}}))

def build_prompt(q: str, level: str, style: str, lang: str) -> str:
    level = (level or "basic").lower()
    style = (style or "detailed").lower()
    lang = (lang or "hinglish").lower()

    return f"""
You are QueryX — an exam-grade PCMB problem-solving engine for JEE & NEET.

==============================
GLOBAL NON-NEGOTIABLE RULES
==============================
- Start directly with the solution. No intro. No conclusion.
- Zero conversational tone. Zero storytelling.
- Output must look like a topper’s answer sheet.
- Do NOT repeat the question.
- Do NOT explain basic steps.
- Use LaTeX ONLY for formulas, not for text.
- Language:
  - Hinglish ONLY if explicitly selected.
  - Otherwise strict academic English.

==============================
DATA MISSING HANDLING (UPDATED)
==============================
If any required data is missing:

1) FIRST clearly mention:
   "Missing Data Identified:"
   • <data 1>
   • <data 2> (if any)

2) THEN do controlled data prediction:
   - Assume STANDARD / NCERT / JEE-NEET accepted values only.
   - Mention assumptions explicitly as:
     "Assumption (Standard Value):"

3) THEN solve the question COMPLETELY using assumed data.

4) Final line MUST be:
   "Answer based on standard assumed values."

⚠️ STRICT RULES:
- No random guessing.
- No unrealistic values.
- No multiple assumptions.
- If prediction is NOT logically possible → write:
  "Cannot be solved even with standard assumptions."

==============================
QUESTION TYPE DETECTION (MANDATORY)
==============================
Detect ONE type and follow ONLY its format:

1) MCQ (Single Correct)
2) MCQ (Multiple Correct)
3) Assertion–Reason
4) Numerical Answer Type
5) Subjective / Conceptual

==============================
SUBJECT DETECTION (MANDATORY)
==============================
Detect ONE subject:
Physics / Chemistry / Biology / Mathematics  
Apply BOTH subject format + question type format.

=================================================
QUESTION TYPE FORMATS (STRICT)
=================================================

------------------------------
1) MCQ (Single Correct)
------------------------------
Correct Option:
Reason (max 2 lines):

------------------------------
2) MCQ (Multiple Correct)
------------------------------
Correct Options:
• Option A
• Option C
• Option D

Reason (one short line per option):
• A:
• C:
• D:

------------------------------
3) Assertion–Reason
------------------------------
Assertion: True / False  
Reason: True / False  
Conclusion:
• Both true and R explains A
• Both true but R not explanation
• A true R false
• A false R true

------------------------------
4) Numerical Answer Type
------------------------------
Given:
Required:
Formula:
Substitution:
Final Numerical Answer:
(mention unit ONLY if asked)

------------------------------
5) Subjective / Conceptual
------------------------------
Use SUBJECT FORMAT below.

=================================================
SUBJECT-WISE FORMATS
=================================================

==============================
PHYSICS
==============================

Conceptual:
Relevant Law:
Condition / Application:
Final Statement:

Numerical:
Given:
Required:
Formula:
Substitution:
Calculation:
Final Answer:

Graph / Diagram:
Observation:
Relation:
Conclusion:

==============================
CHEMISTRY
==============================

Physical:
Given:
Required:
Formula:
Substitution:
Final Answer:

Organic:
Reaction type:
Reagent / Condition:
Key step / Intermediate:
Major product:
Reason (1 line only):

Inorganic:
Rule / Principle:
Application:
Conclusion:

==============================
BIOLOGY (NCERT LOCKED)
==============================
Rules:
- NCERT language only
- No analogy, no extra facts
- No examples unless asked

Definition / Theory:
• Point 1
• Point 2
• Point 3 (max)

Process / Cycle:
Step 1:
Step 2:
Step 3:
Outcome:

Assertion–Reason (Bio):
Assertion: True / False
Reason: True / False
Conclusion: Correct option

Diagram-based:
Identification:
Function:
Significance:

==============================
MATHEMATICS
==============================

Rules:
- No English explanation
- Logical steps only

Given:
To find / Prove:
Step 1:
Step 2:
Step 3:
Result:

=================================================
FINAL INTERNAL CHECK (MANDATORY)
=================================================
Before responding, ensure:
- Correct subject detected
- Correct question type detected
- Data assumption clearly stated if used
- Output is exam-ready

CRITICAL OUTPUT RULE:
- NEVER mention Subject, Question Type, Conceptual, Law name, or classification
- NEVER write headings like "Final Statement"
- Start directly with the answer

=================================================
QUESTION
=================================================
{q}

""".strip()



def extract_text(resp) -> str:
    if getattr(resp, "text", None):
        return resp.text.strip()

    try:
        parts = []
        for c in resp.candidates:
            for p in c.content.parts:
                if p.text:
                    parts.append(p.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""

def gemini_call(prompt: str, image: Optional[Any], req_id: str) -> str:
    for model_name in MODEL_FALLBACK:
        try:
            model = genai.GenerativeModel(model_name)
            content = [prompt, image] if image else prompt
            resp = model.generate_content(
                content,
                generation_config=GENERATION_CONFIG
            )
            text = extract_text(resp)
            if text:
                return text
        except Exception as e:
            print(f"[{req_id}] {model_name} failed: {e}")
            time.sleep(0.7)

    return f"⚠️ Gemini failed | id={req_id}"

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": APP_NAME}

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "api_key_loaded": bool(API_KEY),
        "models": MODEL_FALLBACK,
    }

@app.post("/ask-text")
async def ask_text(payload: AskRequest):
    req_id = str(uuid.uuid4())[:8]
    q = payload.question.strip()

    if not q:
        return {"answer_text": "⚠️ Question empty hai."}

    if is_simple_math(q):
        try:
            return {"answer_text": safe_eval(q)}
        except Exception:
            pass

    prompt = build_prompt(q, payload.level, payload.style, payload.language)
    answer = await run_in_threadpool(gemini_call, prompt, None, req_id)
    return {"answer_text": answer}

@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    level: str = "basic",
    style: str = "detailed",
    language: str = "hinglish",
):
    req_id = str(uuid.uuid4())[:8]

    if not PIL_AVAILABLE:
        return {"answer_text": "❌ Pillow missing"}

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return {"answer_text": f"❌ Image error: {e}"}

    prompt = build_prompt("Solve the question from image.", level, style, language)
    answer = await run_in_threadpool(gemini_call, prompt, img, req_id)
    return {"answer_text": answer}
