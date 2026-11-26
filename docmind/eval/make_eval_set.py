"""Build an evaluation set from a long PDF with an LLM, then YOU review it.

  python -m eval.make_eval_set data/handbook.pdf --n 30 --unanswerable 5 --out eval/eval_set.json

Each answerable item:  {"question", "expected", "answer", "page"}
  expected = a short fragment copied word-for-word from the document (used to check retrieval)
  answer   = the key fact, 1-3 words (used to check the LLM's answer in benchmark.compare_llms)
Unanswerable items:    {"question", "unanswerable": true}   (the right reply is "not found")

Quality guard: an item is kept only if `expected` really occurs in a chunk at EVERY chunk size you plan to
test (--check-sizes), so the chunk-size comparison is fair.
Known bias: LLM-written questions tend to reuse the document's words, which favours keyword search. Review the
file and rewrite a few questions in your own words.
"""
import argparse
import json
import logging
import random
import re
import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.ingest import load_and_split
from app.models import get_llm

log = logging.getLogger("docmind.evalset")


class QA(BaseModel):
    question: str
    evidence: str
    answer: str


class Unanswerable(BaseModel):
    questions: list[str]


_qa_parser = PydanticOutputParser(pydantic_object=QA)
_un_parser = PydanticOutputParser(pydantic_object=Unanswerable)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write ONE question that the passage below answers.\n"
     "- The question must make sense on its own: name the subject, and never say 'the passage' or 'the text'.\n"
     "- Ask about a specific fact (a number, name, duration, rule). Paraphrase: do not reuse the passage's wording.\n"
     "- evidence: copy 4 to 12 consecutive words from the passage EXACTLY as written, containing the answer.\n"
     "- answer: the key fact, 1 to 3 words, copied from evidence.\n{format_instructions}"),
    ("human", "{passage}"),
]).partial(format_instructions=_qa_parser.get_format_instructions())

UNANSWERABLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Below are excerpts from a document. Write {n} questions that sound related to its topic but "
     "cannot be answered from the document (they ask for facts that are not there). "
     "Each must make sense on its own.\n{format_instructions}"),
    ("human", "{excerpts}"),
]).partial(format_instructions=_un_parser.get_format_instructions())


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def looks_like_prose(text: str, min_chars: int) -> bool:
    letters = sum(c.isalpha() for c in text)
    return len(text) >= min_chars and letters / max(len(text), 1) > 0.6   # skips tables of contents, indexes


def build(pdf: str, llm, n: int = 30, n_unanswerable: int = 5, check_sizes=(200, 800), seed: int = 42,
          sleep: float = 0.0, min_chars: int = 400) -> list[dict]:
    qa_chain = (QA_PROMPT | llm | _qa_parser).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    un_chain = (UNANSWERABLE_PROMPT | llm | _un_parser).with_retry(stop_after_attempt=3,
                                                                    wait_exponential_jitter=True)
    size_chunks = {s: [norm(c.page_content) for c in load_and_split(pdf, s)] for s in check_sizes}
    base = [c for c in load_and_split(pdf, max(check_sizes)) if looks_like_prose(c.page_content, min_chars)]
    random.Random(seed).shuffle(base)

    items, seen, tried = [], set(), 0
    for chunk in base:
        if len(items) >= n or tried >= 4 * n:
            break
        tried += 1
        try:
            qa = qa_chain.invoke({"passage": chunk.page_content})
        except Exception:
            log.warning("generation failed for a chunk, skipping", exc_info=True)
            continue
        ev, ans, q = norm(qa.evidence), norm(qa.answer), qa.question.strip()
        if not q or q.lower() in seen or not ans or ans not in ev:
            continue
        if not all(any(ev in c for c in chunks) for chunks in size_chunks.values()):
            log.info("dropped (evidence not found verbatim at every chunk size): %r", qa.evidence)
            continue
        seen.add(q.lower())
        items.append({"question": q, "expected": qa.evidence.strip(), "answer": qa.answer.strip(),
                      "page": chunk.metadata["page"]})
        if sleep:
            time.sleep(sleep)

    if n_unanswerable and base:
        excerpts = "\n---\n".join(c.page_content[:600] for c in base[:3])
        try:
            qs = un_chain.invoke({"n": n_unanswerable, "excerpts": excerpts}).questions
            items += [{"question": q.strip(), "unanswerable": True} for q in qs if q.strip()][:n_unanswerable]
        except Exception:
            log.warning("could not generate unanswerable questions", exc_info=True)
    return items


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--n", type=int, default=30, help="answerable questions")
    ap.add_argument("--unanswerable", type=int, default=5)
    ap.add_argument("--check-sizes", type=int, nargs="+", default=[200, 800])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between LLM calls (rate limits)")
    ap.add_argument("--out", default="eval/eval_set.json")
    a = ap.parse_args()
    items = build(a.pdf, get_llm(), a.n, a.unanswerable, tuple(a.check_sizes), a.seed, a.sleep)
    answerable = [i for i in items if not i.get("unanswerable")]
    if not answerable:
        raise SystemExit("No valid questions were produced (LLM errors or no usable text). Nothing was written.")
    json.dump(items, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {len(answerable)} answerable + {len(items) - len(answerable)} unanswerable questions to {a.out}")
    print("NEXT: open the file and delete bad questions, rewrite a few in your own words.")


if __name__ == "__main__":
    main()
