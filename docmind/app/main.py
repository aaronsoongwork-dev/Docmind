import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .agent import AgentResult, run_agent
from .auth import current_user, hash_key, new_api_key
from .chain import AskResult, ask
from .config import get_settings
from .db import ChatMessage, Document, User, get_db, init_db
from .ingest import get_store, ingest_pdf
from .models import get_llm
from .retrieval import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("docmind.api")

HISTORY_TURNS = 6   # how many previous messages of a session are fed to query rewriting / the agent


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title="DocuMind", description="Ask your documents (LangChain RAG + LangGraph agent)",
              lifespan=lifespan)
_state: dict = {}


def pipeline():
    """Lazy singletons; cleared after each ingest so BM25 sees new chunks."""
    if "retriever" not in _state:
        s = get_settings()
        _state["retriever"] = HybridRetriever(get_store(), fetch_k=s.fetch_k, top_k=s.top_k)
        _state["llm"] = get_llm()
    return _state["retriever"], _state["llm"]


def _filters(user: User, category: str | None) -> dict:
    f = {"owner": str(user.id)}          # users only ever search their own documents
    if category:
        f["category"] = category
    return f


def _history(db: Session, user: User, session_id: str | None, extra: list[str]) -> list[str]:
    if not session_id:
        return extra
    rows = db.scalars(select(ChatMessage)
                      .where(ChatMessage.user_id == user.id, ChatMessage.session_id == session_id)
                      .order_by(ChatMessage.id.desc()).limit(HISTORY_TURNS)).all()
    return [f"{m.role}: {m.content}" for m in reversed(rows)] + extra


def _save_turn(db: Session, user: User, session_id: str | None, question: str, answer: str,
               sources: list[dict] | None = None):
    if not session_id:
        return
    db.add_all([ChatMessage(user_id=user.id, session_id=session_id, role="user", content=question),
                ChatMessage(user_id=user.id, session_id=session_id, role="assistant",
                            content=answer, sources=sources)])
    db.commit()


# ---------- schemas ----------
class UserCreate(BaseModel):
    email: EmailStr


class UserCreated(BaseModel):
    id: int
    email: str
    api_key: str = Field(description="Shown ONCE. Send it as the X-API-Key header.")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    category: str | None = Field(None, description="metadata filter")
    session_id: str | None = Field(None, max_length=100, description="same id = same conversation")
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank: bool = True
    rewrite: bool = True
    history: list[str] = []


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    category: str | None = None
    session_id: str | None = Field(None, max_length=100)


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/users", response_model=UserCreated, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    key = new_api_key()
    user = User(email=body.email.lower(), api_key_hash=hash_key(key))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "That email is already registered") from None
    return UserCreated(id=user.id, email=user.email, api_key=key)


@app.post("/ingest")
def ingest(file: UploadFile = File(...), category: str = Form("general"),
           user: User = Depends(current_user), db: Session = Depends(get_db)):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # Re-uploading the same filename overwrites its chunks (same ids) instead of duplicating them.
        n = ingest_pdf(str(path), category=category, owner=str(user.id), source_name=name)
    db.add(Document(user_id=user.id, filename=name, category=category, chunk_count=n,
                    chunk_size=get_settings().chunk_size))
    db.commit()
    _state.clear()
    return {"filename": name, "chunks": n, "category": category}


@app.get("/documents")
def list_documents(user: User = Depends(current_user), db: Session = Depends(get_db)):
    docs = db.scalars(select(Document).where(Document.user_id == user.id).order_by(Document.id.desc())).all()
    return [{"id": d.id, "filename": d.filename, "category": d.category, "chunks": d.chunk_count,
             "uploaded_at": d.created_at} for d in docs]


@app.post("/ask", response_model=AskResult)
def ask_endpoint(req: AskRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not req.question.strip():
        raise HTTPException(422, "question must not be blank")
    retriever, llm = pipeline()
    try:
        result = ask(req.question, retriever, llm, filters=_filters(user, req.category), mode=req.mode,
                     rerank=req.rerank, rewrite=req.rewrite,
                     history=_history(db, user, req.session_id, req.history))
    except Exception:
        log.exception("ask failed after retries")
        raise HTTPException(502, "The language model failed to return a valid answer. Try again.") from None
    _save_turn(db, user, req.session_id, req.question, result.answer,
               [s.model_dump() for s in result.sources])
    return result


@app.post("/agent", response_model=AgentResult)
def agent_endpoint(req: AgentRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not req.question.strip():
        raise HTTPException(422, "question must not be blank")
    retriever, llm = pipeline()
    try:
        result = run_agent(req.question, retriever, llm, filters=_filters(user, req.category),
                           history=_history(db, user, req.session_id, []))
    except Exception:
        log.exception("agent failed")
        raise HTTPException(502, "The agent failed. Try again.") from None
    _save_turn(db, user, req.session_id, req.question, result.answer)
    return result


@app.get("/history/{session_id}")
def history(session_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(ChatMessage).where(ChatMessage.user_id == user.id,
                                                ChatMessage.session_id == session_id)
                      .order_by(ChatMessage.id)).all()
    return [{"role": m.role, "content": m.content, "sources": m.sources, "at": m.created_at} for m in rows]
