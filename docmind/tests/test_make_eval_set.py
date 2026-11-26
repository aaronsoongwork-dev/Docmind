import json
import textwrap

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from eval.make_eval_set import build, looks_like_prose, norm
from eval.run_eval import run as run_eval
from tests.conftest import smart_llm

FILLER = ("All staff are expected to follow the published guidelines and to speak to their manager when "
          "something is unclear. Requests are reviewed in the order they arrive and every decision is recorded "
          "so that it can be checked later by the relevant department. ") * 2

PAGES = [
    "Annual leave is eighteen days per year for all full-time staff. " + FILLER,
    "Probation for new staff is described here. " + FILLER,
    "Laptops must use disk encryption and the VPN token rotates every thirty days. " + FILLER,
]


@pytest.fixture(scope="module")
def long_pdf(tmp_path_factory):
    p = tmp_path_factory.mktemp("evalset") / "policy.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    for text in PAGES:
        y = 800
        for line in textwrap.wrap(text, 90):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()
    return str(p)


def qa_reply(prompt):
    if "eighteen days" in prompt:   # a valid, verbatim evidence fragment
        return json.dumps({"question": "How much yearly holiday do permanent employees receive?",
                           "evidence": "Annual leave is eighteen days per year", "answer": "eighteen days"})
    # invented evidence that is not in the document -> must be rejected
    return json.dumps({"question": "How long is probation?", "evidence": "probation lasts ninety days",
                       "answer": "ninety days"})


def test_only_verbatim_evidence_is_kept_and_unanswerable_added(long_pdf):
    llm = smart_llm([("cannot be answered", json.dumps({"questions": ["What is the pet policy?"]})),
                     ("Write ONE question", qa_reply)])
    items = build(long_pdf, llm, n=5, n_unanswerable=1, check_sizes=(200, 800), min_chars=200)
    answerable = [i for i in items if not i.get("unanswerable")]
    assert len(answerable) == 1
    assert answerable[0]["answer"] == "eighteen days" and answerable[0]["page"] == 1
    assert [i for i in items if i.get("unanswerable")] == [{"question": "What is the pet policy?",
                                                            "unanswerable": True}]


def test_run_eval_skips_unanswerable_and_scores_hits(long_pdf):
    qs = [{"question": "annual leave eighteen days per year", "expected": "Annual leave is eighteen days"},
          {"question": "pet policy", "unanswerable": True}]
    rows = run_eval(long_pdf, qs, chunk_sizes=(200, 800), k=3)
    assert {r["method"] for r in rows} >= {"dense", "sparse", "hybrid"}
    sparse = [r for r in rows if r["method"] == "sparse"]
    assert all(r["hit_rate"] == 1.0 and r["questions"] == 1 and r["avg_context_chars"] > 0 for r in sparse)


def test_helpers():
    assert norm("  A  b\n C ") == "a b c"
    assert looks_like_prose("word " * 100, 200) and not looks_like_prose("1.2.3 .... 45 " * 40, 200)
