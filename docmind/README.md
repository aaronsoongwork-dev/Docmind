# DocuMind v2: RAG + agent + fine-tuning + auth + CI

FastAPI + LangChain + LangGraph + Chroma + SQLAlchemy (SQLite or Postgres).

## What's new in v2
| # | Feature | Where |
|---|---|---|
| 1 | Fine-tune the embedding model, report before/after | `finetune/` |
| 2 | LangGraph agent: search / calculator / ask the user | `app/agent.py`, `POST /agent` |
| 3 | Gemini vs Ollama benchmark (accuracy, failures, latency) | `benchmark/compare_llms.py` |
| 4 | Users, chat history, document records (SQLite/Postgres) | `app/db.py`, `app/auth.py`, `docker-compose.yml` |
| 5 | GitHub Actions CI (lint + tests, no secrets) | `.github/workflows/ci.yml` |

**Upgrading from v1:** keep your `.env`; copy over `app/`, `finetune/`, `benchmark/`, `tests/`, the
`requirements*.txt` files and `pyproject.toml`. Documents ingested with v1 have no `owner` metadata, so
**delete `chroma_db/` and re-upload your PDFs** (via `/ingest`, after creating a user).

## Setup
    python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
    pip install -r requirements-ml.txt                    # app + PyTorch (embeddings, reranker, fine-tuning)
    cp .env.example .env                                  # fill in GOOGLE_API_KEY
    uvicorn app.main:app --reload

## Using the API (http://localhost:8000/docs)
1. `POST /users` with `{"email": "you@example.com"}`. **Copy the `api_key`: it is shown once.**
2. Click **Authorize** (top right of /docs) and paste the key. Every other endpoint needs it.
3. `POST /ingest`: upload your CV (use category `cv`). Re-uploading the same filename replaces it.
4. `POST /ask`: `{"question": "...", "category": "cv", "session_id": "chat1"}`. Reusing `session_id`
   gives the query rewriter the previous turns. `GET /history/chat1` shows the conversation.
5. `POST /agent`: `{"question": "If I work 22 days a month, how many days is that over 6 months?"}`.
   The response lists every step (search, calculate, ...) and may come back with `needs_clarification: true`.
6. `GET /documents`: what you've ingested.
Each user only ever searches their own documents (an `owner` metadata filter is applied on every query).

## 1. Fine-tune the embedding model
    python -m finetune.generate_questions data/my_doc.pdf --per-chunk 4        # LLM writes questions per chunk
    python -m finetune.finetune_embeddings --pdf data/my_doc.pdf --questions finetune/synthetic.json
Prints hit@1 / hit@k / MRR for the base vs fine-tuned model on **held-out questions** (dense retrieval only).
Then set `EMBEDDINGS_MODEL=models/docmind-embed-ft` in `.env` and **re-ingest**: vectors from different models
are not comparable.
Honest limits: a 2-page CV gives a handful of chunks, which is far too little to fine-tune on or to trust the
numbers. Use a long document (or several; concatenate them into one PDF). The held-out questions target chunks the
model saw in training, so the result is "gain on this corpus", not on unseen documents. Include hand-written
questions in the JSON too (`{"question", "expected"}`); they're matched to chunks automatically.

## 2. Agent
`app/agent.py` is a LangGraph state machine: `decide` (LLM router, JSON-validated + retried) loops through
`search_documents` / `calculate` until it chooses `finish` or `ask_user`, with a step limit (`AGENT_MAX_STEPS`).
The calculator walks the syntax tree and accepts only numbers and `+ - * / // % **`; it never calls `eval`.

## 3. Ollama vs Gemini
    ollama pull qwen2.5:3b         # check the Ollama library for current model names
    python -m benchmark.compare_llms --pdf data/my_doc.pdf --questions benchmark/questions.json \
        --models gemini:gemini-2.5-flash ollama:qwen2.5:3b
Write ~20 questions about your own PDF first (`benchmark/questions.example.json` shows the format; include a few
`"unanswerable": true` to test hallucination). Small local models often fail the JSON schema, which shows up in the
`failed` column, a good interview talking point. Retrieval is identical for all models.

## 4. Postgres
    docker compose up --build      # app on :8000 + Postgres; tables are created at startup
Or point `DATABASE_URL` at any Postgres: `postgresql+psycopg://user:pass@host:5432/docmind`.
Tables are created with `create_all`; a real deployment would add Alembic migrations.

## 5. Tests / CI
    pip install -r requirements-dev.txt && ruff check . && pytest
No API keys or model downloads: tests use a fake LLM, fake embeddings and a temporary SQLite file. The
PyTorch fine-tuning smoke test is skipped automatically if torch isn't installed (as in CI).
To use CI: push to GitHub; the workflow runs on every push and pull request.
