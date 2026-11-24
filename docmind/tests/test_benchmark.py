import json

import pytest

from benchmark.compare_llms import percentile, run, split_spec
from tests.conftest import smart_llm

QS = [{"question": "How many days of annual leave?", "answer": "18"},
      {"question": "What is the pet policy?", "unanswerable": True}]


def test_split_spec_keeps_colons_in_model_name():
    assert split_spec("ollama:qwen2.5:3b") == ("ollama", "qwen2.5:3b")
    with pytest.raises(ValueError):
        split_spec("ollama")


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([], 95) != percentile([], 95)  # NaN


def test_run_scores_accuracy_failures_and_latency(pdf_path):
    good = json.dumps({"answer": "18 days", "found": True, "cited_pages": [1]})
    nope = json.dumps({"answer": "I don't know", "found": False, "cited_pages": []})

    def factory(provider, model):
        if model == "good":      # answers the first question, refuses the second
            return smart_llm([("pet policy", nope), ("annual leave", good)])
        return smart_llm([("Question", "this is not JSON")])    # a "small model" that never obeys the schema

    res = run(pdf_path, QS, ["fake:good", "fake:bad"], llm_factory=factory)
    assert res["fake:good"]["accuracy"] == 1.0 and res["fake:good"]["failures"] == 0
    assert res["fake:bad"]["failures"] == 2 and res["fake:bad"]["accuracy"] == 0.0
