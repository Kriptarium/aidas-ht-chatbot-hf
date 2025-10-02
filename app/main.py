
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path as _Path

from .rag import SimpleRAG
from .hf_backend import generate_answer

APP_NAME = "AIDAS-HT Chatbot (RAG + HF FLAN-T5-small)"
KB_DIR = "kb"

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
app.mount("/static", StaticFiles(directory=str(_Path(__file__).resolve().parent.parent / "frontend")), name="static")

@app.get("/", response_class=HTMLResponse)
def ui():
    index_file = _Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>AIDAS-HT Chatbot</h1><p>Frontend bulunamadı.</p>"

rag = SimpleRAG(KB_DIR)

class AskBody(BaseModel):
    user_id: str = "demo"
    question: str
    k: int = 5
    use_rag: bool = False

@app.get("/health")
def health():
    return {"status":"ok", "app":APP_NAME, "kb_docs": len(list(Path(KB_DIR).glob("*.pdf")))}

@app.post("/ask_hf")
def ask_hf(body: AskBody):
    context = rag.topk_text(body.question, k=body.k) if body.use_rag else ""
    prompt = f"Soru (Türkçe): {body.question}\n\n"
    if context:
        prompt += "Aşağıdaki bağlama dayanarak kısa ve somut bir yanıt ver:\n" + context + "\n\n"
    prompt += "Cevabı Türkçe ver."
    ans = generate_answer(prompt)
    return {"answer": ans, "mode": "HF-FLAN-T5-small", "used_context": bool(context)}

# simple quiz for continuity
class QuizAnswer(BaseModel):
    id: str
    answer: int

class QuizSubmitBody(BaseModel):
    user_id: str = "demo"
    answers: List[QuizAnswer]

QUIZ_BANK = [
    {"id":"Q1","module":"Temel","text":"Çapraz çatlaklar çoğunlukla hangi etkiyle ilişkilidir?",
     "choices":["Isıl genleşme","Kesme etkisi","Temel oturması","Boyutsal büzülme"],"correct":1},
    {"id":"Q2","module":"Form","text":"HT formunda taşıyıcı sistem alanı aşağıdakilerden hangisi değildir?",
     "choices":["Betonarme","Çelik","Yığma","PVC doğrama"],"correct":3}
]

@app.get("/quiz/start")
def quiz_start(limit: int = 5):
    return {"items": QUIZ_BANK[:limit]}

@app.post("/quiz/submit")
def quiz_submit(body: QuizSubmitBody):
    key = {q["id"]: q["correct"] for q in QUIZ_BANK}
    score, details = 0, []
    for a in body.answers:
        correct = key.get(a.id, -999)
        ok = (a.answer == correct)
        score += int(ok)
        details.append({"id": a.id, "ok": ok, "your": a.answer, "correct": correct})
    return {"score": score, "total": len(body.answers), "details": details}
