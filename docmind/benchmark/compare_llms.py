"""Compare LLMs (e.g. Gemini vs a local Ollama model) on YOUR questions: accuracy, failures, latency.

  ollama pull qwen2.5:3b            # once, any small model you like
  python -m benchmark.compare_llms --pdf data/my_doc.pdf --questions benchmark/questions.json \
      --models gemini:gemini-2.5-flash ollama:qwen2.5:3b ollama:llama3.2:3b

questions.json items:
  {"question": "...", "answer": "text the correct answer must contain"}       (answerable)
  {"question": "...", "unanswerable": true}                                    (correct = says "not found")
Retrieval is identical for every model (same chunks), so differences come from the LLM alone.
"""
import argparse
import json
import statistics
import tempfile
import time

from app.chain import ask
from app.config import get_settings
from app.ingest import get_store, load_and_split
from app.models import get_llm
from app.retrieval import HybridRetriever


def split_spec(spec: str) -> tuple[str, str]:
    provider, _, model = spec.partition(":")   # split on the FIRST colon: model names contain colons too
    if not model:
        raise ValueError(f"Use provider:model, e.g. ollama:qwen2.5:3b (got {spec!r})")
    return provider, model


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))] if s else float("nan")


def is_correct(item: dict, result) -> bool:
    if item.get("unanswerable"):
        return not result.found
    return result.found and item["answer"].lower() in result.answer.lower()


def evaluate_llm(llm, retriever, questions: list[dict], rerank: bool = False) -> dict:
    correct, failures, latencies = 0, 0, []
    for item in questions:
        t0 = time.perf_counter()
        try:
            result = ask(item["question"], retriever, llm, rerank=rerank, rewrite=False)
        except Exception:      # still invalid / unreachable after the 3 retries
            failures += 1
            continue
        latencies.append(time.perf_counter() - t0)
        correct += is_correct(item, result)
    n = len(questions)
    return {"accuracy": correct / n, "failures": failures, "n": n,
            "p50_s": statistics.median(latencies) if latencies else float("nan"),
            "p95_s": percentile(latencies, 95)}


def run(pdf: str, questions: list[dict], specs: list[str], llm_factory=get_llm) -> dict:
    s = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        store = get_store(tmp, collection="bench")
        chunks = load_and_split(pdf, s.chunk_size)
        store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        retriever = HybridRetriever(store, fetch_k=s.fetch_k, top_k=s.top_k)
        return {spec: evaluate_llm(llm_factory(*split_spec(spec)), retriever, questions) for spec in specs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="benchmark/results.json")
    a = ap.parse_args()
    qs = json.load(open(a.questions, encoding="utf-8"))
    results = run(a.pdf, qs, a.models)
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\n{len(qs)} questions. 'failed' = no valid JSON answer after 3 attempts (small models do this).")
    print(f"{'model':<34}{'accuracy':>9}{'failed':>8}{'p50 s':>8}{'p95 s':>8}")
    for spec, r in results.items():
        print(f"{spec:<34}{r['accuracy']:>9.0%}{r['failures']:>8}{r['p50_s']:>8.1f}{r['p95_s']:>8.1f}")
