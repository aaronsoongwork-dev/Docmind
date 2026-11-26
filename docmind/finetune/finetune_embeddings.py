"""Fine-tune the embedding model on (question, chunk) pairs and report before/after retrieval quality.

  python -m finetune.finetune_embeddings --pdf data/my_doc.pdf \
      --questions finetune/synthetic.json --out models/docmind-embed-ft --epochs 3

Then use it:  EMBEDDINGS_MODEL=models/docmind-embed-ft   (and RE-INGEST: old vectors don't match).
"""
import argparse
import json
import logging

import numpy as np

from app.config import get_settings

from .data import load_chunks, load_pairs, split_pairs

log = logging.getLogger("docmind.finetune")


def train(pairs, base_model: str, out_dir: str, epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(base_model)
    ds = Dataset.from_dict({"anchor": [q for q, _ in pairs], "positive": [p for _, p in pairs]})
    # In-batch negatives: every OTHER chunk in the batch is treated as a wrong answer. So two questions
    # about the SAME chunk must not share a batch, hence NO_DUPLICATES.
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{out_dir}_checkpoints", num_train_epochs=epochs,
        per_device_train_batch_size=min(batch_size, len(pairs)), learning_rate=lr,
        warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=10, report_to="none")
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save(out_dir)
    return out_dir


def evaluate(model_name_or_path: str, pairs, chunks: list[str], k: int = 4) -> dict:
    """Dense retrieval only (no BM25/rerank), so the number reflects the embedding model alone."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name_or_path)
    c = model.encode(chunks, normalize_embeddings=True)
    q = model.encode([x for x, _ in pairs], normalize_embeddings=True)
    ranks = []
    for scores, (_, positive) in zip(q @ c.T, pairs, strict=False):
        order = np.argsort(-scores)
        ranks.append(int(np.where(order == chunks.index(positive))[0][0]) + 1)
    r = np.array(ranks)
    return {"hit@1": float((r == 1).mean()), f"hit@{k}": float((r <= k).mean()),
            "mrr": float((1 / r).mean()), "n": len(pairs)}


def run(pdf: str, questions: str, base: str, out: str, *, test_questions: str | None = None,
        chunk_size: int = 800, epochs: int = 3, test_fraction: float = 0.3, seed: int = 42, k: int = 4) -> dict:
    """Train on `questions`; test on `test_questions` if given (a separate, reviewed file), otherwise on a
    random held-out share of `questions`."""
    chunks = load_chunks(pdf, chunk_size)
    pairs = load_pairs(chunks, questions)
    if test_questions:
        test_pairs = load_pairs(chunks, test_questions)
        if not test_pairs:
            raise ValueError("No test question could be matched to a chunk (check chunk size / 'expected' text).")
        test_texts = {q for q, _ in test_pairs}
        train_pairs = [p for p in pairs if p[0] not in test_texts]
    else:
        train_pairs, test_pairs = split_pairs(pairs, test_fraction, seed)
    before = evaluate(base, test_pairs, chunks, k)
    train(train_pairs, base, out, epochs)
    after = evaluate(out, test_pairs, chunks, k)
    return {"chunks": len(chunks), "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
            "k": k, "before": before, "after": after}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True, help="training questions (finetune.generate_questions output)")
    ap.add_argument("--test-questions", default=None,
                    help="optional separate, hand-reviewed test file (e.g. eval/eval_set.json)")
    ap.add_argument("--base", default=s.embeddings_model)
    ap.add_argument("--out", default="models/docmind-embed-ft")
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="finetune/results.json")
    ap.add_argument("-k", type=int, default=s.top_k)
    a = ap.parse_args()

    r = run(a.pdf, a.questions, a.base, a.out, test_questions=a.test_questions, chunk_size=a.chunk_size,
            epochs=a.epochs, test_fraction=a.test_fraction, seed=a.seed, k=a.k)
    json.dump(r, open(a.results, "w"), indent=1)
    print(f"\n{r['chunks']} chunks | {r['train_pairs']} train pairs | {r['test_pairs']} test questions")
    print(f"Test questions (dense retrieval only, k={a.k})")
    print(f"{'model':<28}{'hit@1':>8}{f'hit@{a.k}':>8}{'MRR':>8}")
    for name, m in [("base: " + a.base.split('/')[-1], r["before"]), ("fine-tuned", r["after"])]:
        print(f"{name:<28}{m['hit@1']:>8.0%}{m[f'hit@{a.k}']:>8.0%}{m['mrr']:>8.2f}")
    print("\nCaveat: test questions are unseen phrasings but target chunks seen in training, so this is the "
          "in-domain gain on THIS document, not a promise about new documents. Few test questions = noisy.")


if __name__ == "__main__":
    main()
