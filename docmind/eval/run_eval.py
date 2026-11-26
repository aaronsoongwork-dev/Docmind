"""How often is the right chunk retrieved? Compares chunk sizes x retrieval methods on YOUR questions.

  python -m eval.run_eval data/handbook.pdf eval/eval_set.json
  python -m eval.run_eval data/handbook.pdf eval/eval_set.json --chunk-sizes 200 400 800 1200 -k 4

Each question needs "expected": text that must appear in the correct chunk (unanswerable ones are skipped).
Build such a file with:  python -m eval.make_eval_set data/handbook.pdf
"""
import argparse
import json
import tempfile

from app.config import get_settings
from app.ingest import get_store, load_and_split
from app.retrieval import HybridRetriever

CONFIGS = [("dense", False), ("sparse", False), ("hybrid", False), ("hybrid", True)]


def evaluate_config(retriever, questions, mode, rerank, k):
    """Returns (hit rate @k, average characters of context that would be sent to the LLM)."""
    hits, chars = 0, 0
    for q in questions:
        docs = retriever.retrieve(q["question"], None, mode, rerank)[:k]
        hits += any(q["expected"].lower() in d.page_content.lower() for d in docs)
        chars += sum(len(d.page_content) for d in docs)
    return hits / len(questions), chars / len(questions)


def run(pdf, questions, chunk_sizes=(200, 800), k=None):
    s = get_settings()
    k = k or s.top_k
    questions = [q for q in questions if not q.get("unanswerable") and q.get("expected")]
    rows = []
    for size in chunk_sizes:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = get_store(tmp, collection=f"eval_{size}")
            chunks = load_and_split(pdf, size)
            batch_size = 1000
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
            r = HybridRetriever(store, fetch_k=s.fetch_k, top_k=k)
            for mode, rerank in CONFIGS:
                rate, avg_chars = evaluate_config(r, questions, mode, rerank, k)
                rows.append({"chunk_size": size, "chunks": len(chunks),
                             "method": f"{mode}{'+rerank' if rerank else ''}",
                             "hit_rate": rate, "avg_context_chars": avg_chars, "k": k,
                             "questions": len(questions)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("questions")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[200, 800])
    ap.add_argument("-k", type=int, default=None, help="chunks sent to the LLM (default TOP_K)")
    ap.add_argument("--out", default="eval/results.json")
    a = ap.parse_args()
    rows = run(a.pdf, json.load(open(a.questions, encoding="utf-8")), a.chunk_sizes, a.k)
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"{rows[0]['questions']} questions, hit@{rows[0]['k']}\n")
    print(f"{'chunk':>6} {'#chunks':>8}  {'method':<16}{'hit-rate':>9}{'context chars':>15}")
    for r in rows:
        print(f"{r['chunk_size']:>6} {r['chunks']:>8}  {r['method']:<16}{r['hit_rate']:>9.0%}"
              f"{r['avg_context_chars']:>15.0f}")
    print("\n'context chars' = text sent to the LLM per question. Small chunks send less text for the same k,\n"
          "so compare hit-rate AND context size, not hit-rate alone. Saved to " + a.out)


if __name__ == "__main__":
    main()
