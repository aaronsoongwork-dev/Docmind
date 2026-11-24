import os
import tempfile

# Must be set before the app is imported: no model downloads, throwaway database/vector store.
os.environ["EMBEDDINGS_PROVIDER"] = "fake"
os.environ["RERANKER_MODEL"] = "none"
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

PAGES = [
    ("hr", "Employee handbook. Annual leave is 18 days per year. Probation lasts 3 months."),
    ("it", "IT policy. Laptops must use full disk encryption. VPN token rotates every 30 days."),
    ("finance", "Finance policy. Expense claims above RM500 need director approval within 7 days."),
]


def smart_llm(rules):
    """Fake LLM. rules = [(needle, reply_or_callable)]: the first needle found in the prompt wins.
    A reply may be a str, an Exception (raised), or a function(prompt_text) -> str."""
    def _call(prompt_value):
        text = prompt_value.to_string()
        for needle, reply in rules:
            if needle in text:
                r = reply(text) if callable(reply) else reply
                if isinstance(r, Exception):
                    raise r
                return AIMessage(content=r)
        raise AssertionError("fake LLM: no rule matched prompt: " + text[:300])
    return RunnableLambda(_call)


@pytest.fixture(scope="session")
def pdf_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "handbook.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    for _, text in PAGES:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return str(p)


@pytest.fixture()
def retriever(pdf_path, tmp_path):
    from app.ingest import get_store, load_and_split
    from app.retrieval import HybridRetriever
    store = get_store(str(tmp_path / "db"), collection="test_docs")
    chunks = load_and_split(pdf_path, 800)
    for ch in chunks:                      # category per page, to test metadata filtering
        ch.metadata["category"] = PAGES[ch.metadata["page"] - 1][0]
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return HybridRetriever(store, fetch_k=10, top_k=2)
