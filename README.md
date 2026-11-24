# DocuMind

Ask questions about your PDFs and get answers with **source page numbers**. DocuMind is a retrieval-augmented generation (RAG) backend built with LangChain, plus a LangGraph agent, per-user accounts, embedding fine-tuning and an LLM benchmark.

<!-- Replace <user>/<repo> with your GitHub path to show the CI status:
[![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/<repo>/actions/workflows/ci.yml) -->

## Features

- **Hybrid retrieval:** dense (embedding) search plus BM25 keyword search, merged with reciprocal rank fusion.
- **Metadata filtering:** restrict a question by `category`. Every query is also scoped to the signed-in user's own documents.
- **Reranking:** a cross-encoder rescores the merged candidates before they reach the LLM.
- **Query rewriting:** the LLM turns a follow-up question into a standalone search query, using the chat history.
- **Validated answers:** the LLM must reply in JSON that a Pydantic schema checks, with retries. Citations that point to pages that were not retrieved are dropped. If nothing is retrieved, the LLM is not called.
- **LangGraph agent:** a router loops between document search, a calculator and asking you a clarifying question, with a step limit.
- **Accounts and history:** API-key auth, chat history and document records in SQLite or Postgres.
- **Embedding fine-tuning:** generate questions, fine-tune with sentence-transformers and compare base versus fine-tuned on held-out questions.
- **LLM benchmark:** compare Gemini and local Ollama models on accuracy, invalid-JSON failures and latency.
- **Swappable models:** Gemini, Ollama, Anthropic or OpenAI by changing one setting.

## How a question flows

```mermaid
flowchart LR
    Q[Question] --> RW[Query rewrite]
    RW --> D[Dense search]
    RW --> S[BM25 search]
    D --> F[RRF fusion]
    S --> F
    F --> R[Cross-encoder rerank]
    R --> L[LLM: JSON answer]
    L --> V[Validate and check citations]
    V --> A[Answer and sources]
    M[Metadata filter: owner, category] -.-> D
    M -.-> S
```

`POST /agent` wraps this pipeline as one tool in a loop:

```mermaid
flowchart LR
    DEC[decide] -->|search_documents| SEARCH[RAG pipeline] --> DEC
    DEC -->|calculate| CALC[safe calculator] --> DEC
    DEC -->|ask_user| END1[clarifying question]
    DEC -->|finish| END2[final answer]
    DEC -->|step limit| END3[give up]
```

## Tech stack

Python, FastAPI, LangChain, LangGraph, Chroma, rank_bm25, sentence-transformers, SQLAlchemy (SQLite or Postgres), Docker, GitHub Actions, pytest, ruff.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-ml.txt -r requirements-dev.txt
cp .env.example .env              # Windows: copy .env.example .env   then add GOOGLE_API_KEY
uvicorn app.main:app --reload
```

The first request downloads the embedding model (about 130 MB) and the reranker (about 1 GB). To skip the reranker, set `RERANKER_MODEL=none` in `.env`.

Open `http://localhost:8000/docs`:

1. `POST /users` with `{"email": "you@example.com"}`. Copy the `api_key`, which is shown once.
2. Click **Authorize** and paste the key.
3. `POST /ingest`: upload a PDF and give it a `category` such as `cv`.
4. `POST /ask`:
   ```json
   {"question": "Which university did I graduate from?", "category": "cv", "session_id": "chat1"}
   ```
5. `POST /agent`:
   ```json
   {"question": "If I work 22 days a month, how many days is that over 6 months?"}
   ```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/users` | Create a user, returns an API key once |
| POST | `/ingest` | Upload a PDF (needs `X-API-Key`) |
| GET | `/documents` | List your uploaded documents |
| POST | `/ask` | RAG answer with sources. Options: `category`, `session_id`, `mode` (`dense`, `sparse`, `hybrid`), `rerank`, `rewrite` |
| POST | `/agent` | Agent answer with its list of steps. May return `needs_clarification` |
| GET | `/history/{session_id}` | A saved conversation |

## Configuration

Set these in `.env` (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `ollama`, `anthropic` or `openai` |
| `LLM_MODEL` | `gemini-2.5-flash` | Model name for that provider |
| `LLM_FALLBACK_MODEL` | empty | Backup model used if the main one errors (for example a 503) |
| `GOOGLE_API_KEY` | empty | Needed for Gemini |
| `LLM_TIMEOUT` | `30` | Seconds before an LLM call times out |
| `EMBEDDINGS_MODEL` | `BAAI/bge-small-en-v1.5` | A Hugging Face id or a local fine-tuned folder |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Set to `none` to disable reranking |
| `CHUNK_SIZE` | `800` | Characters per chunk (10% overlap) |
| `TOP_K` / `FETCH_K` | `4` / `15` | Chunks sent to the LLM / candidates per retriever |
| `DATABASE_URL` | `sqlite:///./docmind.db` | Postgres: `postgresql+psycopg://user:pass@host:5432/docmind` |
| `AGENT_MAX_STEPS` | `4` | Step limit for the agent |

## Project structure

```
app/
  main.py        FastAPI endpoints
  config.py      Settings from environment / .env
  models.py      LLM, embedding and reranker factories
  ingest.py      PDF -> chunks -> metadata -> Chroma
  retrieval.py   Hybrid search, RRF, metadata filter, rerank, query rewrite
  chain.py       LCEL answer chain, JSON validation, retries, citation check
  agent.py       LangGraph agent and safe calculator
  db.py          Users, documents and chat tables
  auth.py        API-key authentication
eval/            Chunk-size and retrieval-mode experiment
finetune/        Question generation, training and before/after evaluation
benchmark/       Gemini vs Ollama comparison
tests/           Offline tests (fake LLM, fake embeddings)
```

## Experiments

### Retrieval: chunk size and method
```bash
python -m eval.run_eval data/my_doc.pdf eval/questions.json
```
Compares chunk sizes 200 and 800 across dense, sparse, hybrid and hybrid with rerank. Each question needs the text that must appear in the correct chunk (see `eval/questions.example.json`).

### Fine-tuning the embedding model
```bash
python -m finetune.generate_questions data/my_doc.pdf --per-chunk 4
python -m finetune.finetune_embeddings --pdf data/my_doc.pdf --questions finetune/synthetic.json
```
This prints hit@1, hit@k and MRR for the base model versus the fine-tuned one on held-out questions. To use the new model, set `EMBEDDINGS_MODEL=models/docmind-embed-ft`, delete `chroma_db/` and re-ingest, because vectors from different models are not comparable.

The held-out questions are new phrasings of chunks the model saw in training, so the result shows the gain on that document set, not on unseen documents. Use a long, domain-specific document. A short CV is too small to show anything.

### Gemini vs a local model
```bash
ollama pull qwen2.5:3b
python -m benchmark.compare_llms --pdf data/my_doc.pdf --questions benchmark/questions.json --models gemini:gemini-2.5-flash ollama:qwen2.5:3b
```
Prints accuracy, failures (no valid JSON after 3 tries) and latency. Retrieval is identical for every model. Include a few `"unanswerable": true` questions to test hallucination.

### My results

Fill these in after you run the experiments.

| Experiment | Setup | Result |
|---|---|---|
| Chunk size | 200 vs 800, hybrid | _to fill in_ |
| Retrieval mode | dense vs sparse vs hybrid vs hybrid + rerank | _to fill in_ |
| Fine-tuning | base vs fine-tuned, held-out questions | _to fill in_ |
| LLM comparison | Gemini vs local model | _to fill in_ |

## Docker and Postgres

```bash
docker compose up --build
```
Starts the app on port 8000 with Postgres. Tables are created at startup.

## Tests and CI

```bash
ruff check .
python -m pytest -q
```
The tests need no API keys or model downloads. They use a fake LLM, fake embeddings and a temporary SQLite file. The PyTorch training smoke test is skipped automatically when torch is not installed. GitHub Actions (`.github/workflows/ci.yml`) runs the lint check and the tests on every push and pull request.

## Design decisions

- **Reciprocal rank fusion:** cosine scores and BM25 scores are not comparable, so results are merged by rank only.
- **Retries on the whole chain:** a bad JSON reply, a timeout and a 5xx error are all retried the same way, and the API returns a clean 502 if they keep failing.
- **Calculator without `eval`:** the agent's calculator walks the syntax tree and accepts only numbers and basic operators.
- **Per-user isolation:** every query carries an `owner` metadata filter, so users cannot retrieve each other's chunks.
- **Citation check:** the LLM's cited pages are accepted only if those pages were actually retrieved.

## Limitations

- PDFs only. Scanned PDFs without text are not supported (no OCR).
- The BM25 index is rebuilt in memory after each upload, which is fine for small collections but not for large ones.
- Auth is a simple API key. There are no passwords, rate limits or key rotation.
- Database tables are created with `create_all`. A real deployment would use migrations (for example Alembic).
- There is no streaming of answers.
- The Docker and Postgres setup has not been tested end to end yet.
