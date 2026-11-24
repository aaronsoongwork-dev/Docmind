"""Build (question, positive chunk) training pairs and split them for an honest evaluation."""
# this will turn the QUESTIONS into (question, correct chunk) pairs --> split into training and testing sets
import json
import logging
import random

from app.ingest import load_and_split

log = logging.getLogger("docmind.finetune")


def load_chunks(pdf: str, chunk_size: int) -> list[str]:
    return [c.page_content for c in load_and_split(pdf, chunk_size)]


def load_pairs(chunks: list[str], questions_json: str) -> list[tuple[str, str]]:
    """Each item is {"question", "positive"}  (synthetic: positive = the chunk text)
    or {"question", "expected"}               (hand-written: positive = first chunk containing `expected`)."""
    pairs = []
    for item in json.load(open(questions_json, encoding="utf-8")):
        if item.get("unanswerable"):
            continue
        positive = item.get("positive")
        if positive is None:
            needle = item["expected"].lower()
            positive = next((c for c in chunks if needle in c.lower()), None)
        if positive is None or positive not in chunks:
            log.warning("no matching chunk for %r (different chunk size?) - skipped", item["question"])
            continue
        pairs.append((item["question"], positive))
    return pairs


def split_pairs(pairs, test_fraction: float = 0.3, seed: int = 42):
    """Hold out whole QUESTIONS (not chunks): the test questions are phrasings the model never saw
    in training, but they target chunks it did see. This measures the in-domain gain on THIS corpus."""
    if len(pairs) < 4:
        raise ValueError(f"Need at least 4 usable pairs to split, got {len(pairs)}. "
                         "Generate more with finetune.generate_questions.")
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]
