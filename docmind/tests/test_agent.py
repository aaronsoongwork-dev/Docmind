import json

import pytest

from app.agent import run_agent, safe_calc
from tests.conftest import smart_llm

ANSWER = json.dumps({"answer": "18 days", "found": True, "cited_pages": [1]})


def decision(action, text):
    return json.dumps({"action": action, "input": text})


@pytest.mark.parametrize("expr,expected", [("2+3*4", 14), ("(120*3)/2", 180), ("-5+2", -3), ("2**10", 1024)])
def test_calculator_ok(expr, expected):
    assert safe_calc(expr) == expected


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "2**9999", "abc", "1+", "(1).real"])
def test_calculator_rejects_everything_else(bad):
    with pytest.raises(ValueError):
        safe_calc(bad)


def test_search_then_calculate_then_finish(retriever):
    def planner(text):
        if "(none yet)" in text:
            return decision("search_documents", "annual leave days")
        if "calculate(" not in text:
            return decision("calculate", "18*2")
        return decision("finish", "Two years of leave is 36 days.")

    llm = smart_llm([("planner of a document assistant", planner),
                     ("You answer questions about a document", ANSWER)])
    r = run_agent("How many leave days in two years?", retriever, llm)
    assert [s.action for s in r.steps] == ["search_documents", "calculate"]
    assert r.steps[1].observation == "36"
    assert r.answer == "Two years of leave is 36 days." and not r.needs_clarification


def test_agent_asks_for_clarification_instead_of_guessing(retriever):
    llm = smart_llm([("planner of a document assistant", decision("ask_user", "Which policy do you mean?"))])
    r = run_agent("What is the limit?", retriever, llm)
    assert r.needs_clarification and r.answer == "Which policy do you mean?" and r.steps == []


def test_calculator_error_is_fed_back_to_the_router(retriever):
    def planner(text):
        if "(none yet)" in text:
            return decision("calculate", "__import__('os')")
        return decision("finish", "Could not compute.")
    r = run_agent("hack", retriever, smart_llm([("planner of a document assistant", planner)]))
    assert r.steps[0].observation.startswith("ERROR")


def test_step_limit_stops_a_looping_agent(retriever):
    llm = smart_llm([("planner of a document assistant", decision("calculate", "1+1"))])
    r = run_agent("loop forever", retriever, llm, max_steps=3)
    assert len(r.steps) == 3 and r.answer.startswith("I couldn't finish")


def test_router_bad_json_is_retried(retriever):
    replies = iter(["nonsense", '{"action": "dance", "input": "x"}', decision("finish", "ok")])
    llm = smart_llm([("planner of a document assistant", lambda _t: next(replies))])
    assert run_agent("hi", retriever, llm).answer == "ok"
