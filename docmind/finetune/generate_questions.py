"""Synthetic training data: ask the LLM to write questions each chunk can answer.
Hand-writing 10 questions is enough to MEASURE retrieval; fine-tuning needs hundreds.

  python -m finetune.generate_questions data/my_doc.pdf --per-chunk 4 --out finetune/synthetic.json
"""
import argparse
import json
import logging
import random
import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import get_settings
from app.models import get_llm

from .data import load_chunks

log = logging.getLogger("docmind.finetune")


class Questions(BaseModel):
    questions: list[str]


_parser = PydanticOutputParser(pydantic_object=Questions)
_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Write {n} different questions that a user could ask and that the passage below answers. "
     "Rules: each question must be answerable from THIS passage alone; vary the wording and do not copy "
     "sentences from the passage; do not mention 'the passage' or 'the text'.\n{format_instructions}"),
    ("human", "{chunk}"),
]).partial(format_instructions=_parser.get_format_instructions())


def generate(chunks: list[str], llm, per_chunk: int = 4, sleep: float = 0.0) -> list[dict]:
    chain = (_prompt | llm | _parser).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    out = []
    for i, chunk in enumerate(chunks, 1):
        try:
            qs = chain.invoke({"n": per_chunk, "chunk": chunk}).questions
        except Exception:
            log.warning("chunk %d/%d skipped: generation failed", i, len(chunks), exc_info=True)
            continue
        out += [{"question": q.strip(), "positive": chunk} for q in qs if q.strip()]
        log.info("chunk %d/%d -> %d questions", i, len(chunks), len(qs))
        if sleep:
            time.sleep(sleep)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--per-chunk", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=get_settings().chunk_size)
    ap.add_argument("--max-chunks", type=int, default=None, help="use a random sample of this many chunks")
    ap.add_argument("--min-chars", type=int, default=200, help="skip chunks shorter than this (headings etc.)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between LLM calls (rate limits)")
    ap.add_argument("--out", default="finetune/synthetic.json")
    a = ap.parse_args()
    chunks = [c for c in load_chunks(a.pdf, a.chunk_size) if len(c) >= a.min_chars]
    if a.max_chunks and a.max_chunks < len(chunks):
        chunks = random.Random(a.seed).sample(chunks, a.max_chunks)
    print(f"generating questions for {len(chunks)} chunks ...")
    data = generate(chunks, get_llm(), a.per_chunk, a.sleep)
    if not data:
        raise SystemExit("No questions were generated: the LLM failed for every chunk (see the warnings above). "
                         "Nothing was written. Try again later, set LLM_FALLBACK_MODEL in .env, or use "
                         "LLM_PROVIDER=ollama.")
    json.dump(data, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {len(data)} questions to {a.out}")
