import json

import pytest

from finetune.data import load_chunks, load_pairs, split_pairs
from finetune.generate_questions import generate
from tests.conftest import smart_llm


def test_generate_questions_makes_pairs_and_skips_failures():
    replies = iter([json.dumps({"questions": ["How many days of leave?", " "]}), "garbage", "garbage", "garbage"])
    out = generate(["chunk A", "chunk B"], smart_llm([("Write 2 different questions", lambda _t: next(replies))]), 2)
    assert out == [{"question": "How many days of leave?", "positive": "chunk A"}]


def test_load_pairs_resolves_expected_text(pdf_path, tmp_path):
    chunks = load_chunks(pdf_path, 800)
    f = tmp_path / "q.json"
    f.write_text(json.dumps([
        {"question": "probation?", "expected": "3 months"},
        {"question": "synthetic", "positive": chunks[1]},
        {"question": "nope", "expected": "not in the document"},
        {"question": "trick", "unanswerable": True, "expected": "x"}]))
    pairs = load_pairs(chunks, str(f))
    assert [q for q, _ in pairs] == ["probation?", "synthetic"]


def test_split_is_deterministic_and_disjoint():
    pairs = [(f"q{i}", "c") for i in range(10)]
    a_train, a_test = split_pairs(pairs, 0.3, seed=1)
    b_train, b_test = split_pairs(pairs, 0.3, seed=1)
    assert a_test == b_test and len(a_test) == 3 and not set(a_train) & set(a_test)
    with pytest.raises(ValueError):
        split_pairs(pairs[:3])


# ---- end-to-end training smoke tests: need torch; use a tiny random model, so no downloads ----
@pytest.fixture()
def tiny_model(tmp_path, pdf_path):
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    from transformers import BertConfig, BertModel, BertTokenizerFast

    chunks = load_chunks(pdf_path, 800)
    text = " ".join(chunks) + " how many days of annual leave long is probation disk encryption required vpn token"
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(text.lower().replace(".", " ").split()))
    base = tmp_path / "tiny"
    base.mkdir()
    (base / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(str(base / "vocab.txt")).save_pretrained(base)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=128)).save_pretrained(base)
    return str(base), chunks


def test_train_and_evaluate_smoke(tmp_path, tiny_model):
    from finetune.finetune_embeddings import evaluate, train
    base, chunks = tiny_model
    pairs = [("how many days of annual leave", chunks[0]), ("how long is probation", chunks[0]),
             ("is disk encryption required", chunks[1]), ("how often does the vpn token rotate", chunks[1]),
             ("when is director approval needed", chunks[2]), ("what expense limit needs approval", chunks[2])]
    out = train(pairs, base, str(tmp_path / "ft"), epochs=2, batch_size=3)
    for path in (base, out):
        m = evaluate(path, pairs, chunks, k=2)
        assert 0 <= m["hit@1"] <= m["hit@2"] <= 1 and 0 < m["mrr"] <= 1 and m["n"] == 6


def test_run_with_separate_test_file(tmp_path, tiny_model, pdf_path):
    from finetune.finetune_embeddings import run
    base, chunks = tiny_model
    train_file, test_file = tmp_path / "train.json", tmp_path / "test.json"
    train_file.write_text(json.dumps(
        [{"question": f"question {i} about leave", "positive": chunks[i % 3]} for i in range(6)]))
    test_file.write_text(json.dumps([{"question": "holiday per year", "expected": "18 days"},
                                     {"question": "vpn rotation", "expected": "30 days"}]))
    r = run(pdf_path, str(train_file), base, str(tmp_path / "ft"), test_questions=str(test_file),
            chunk_size=800, epochs=1, k=2)
    assert r["test_pairs"] == 2 and r["train_pairs"] == 6
    for key in ("before", "after"):
        assert 0 <= r[key]["hit@1"] <= 1 and r[key]["n"] == 2
