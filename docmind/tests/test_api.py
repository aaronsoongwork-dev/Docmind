import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from tests.conftest import smart_llm

GOOD = json.dumps({"answer": "18 days", "found": True, "cited_pages": [1]})


@pytest.fixture()
def client(monkeypatch):
    llm = smart_llm([("standalone search query", "annual leave days"),
                     ("You answer questions about a document", GOOD),
                     ("planner of a document assistant",
                      json.dumps({"action": "finish", "input": "agent says hi"}))])
    monkeypatch.setattr(main, "get_llm", lambda: llm)
    main._state.clear()
    with TestClient(main.app) as c:     # `with` runs the lifespan -> creates the tables
        yield c
    main._state.clear()


def signup(client, email):
    r = client.post("/users", json={"email": email})
    assert r.status_code == 201
    return {"X-API-Key": r.json()["api_key"]}


def upload(client, headers, pdf_path, category="hr"):
    with open(pdf_path, "rb") as f:
        return client.post("/ingest", headers=headers, data={"category": category},
                           files={"file": ("handbook.pdf", f, "application/pdf")})


def test_endpoints_require_api_key(client):
    for method, url in [("post", "/ask"), ("post", "/ingest"), ("get", "/documents"), ("post", "/agent")]:
        assert getattr(client, method)(url).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "dm_wrong"}).status_code == 401


def test_duplicate_email_rejected(client):
    signup(client, "dup@example.com")
    assert client.post("/users", json={"email": "DUP@example.com"}).status_code == 409


def test_blank_question_rejected(client):
    h = signup(client, "blank@example.com")
    assert client.post("/ask", headers=h, json={"question": ""}).status_code == 422
    assert client.post("/ask", headers=h, json={"question": "   "}).status_code == 422


def test_ingest_ask_history_and_documents(client, pdf_path):
    h = signup(client, "a@example.com")
    r = upload(client, h, pdf_path)
    assert r.status_code == 200 and r.json()["chunks"] == 3

    docs = client.get("/documents", headers=h).json()
    assert docs[0]["filename"] == "handbook.pdf" and docs[0]["chunks"] == 3

    r = client.post("/ask", headers=h, json={"question": "How many days of annual leave?",
                                             "session_id": "s1", "rerank": False})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["answer"] == "18 days" and body["sources"][0]["page"] == 1

    msgs = client.get("/history/s1", headers=h).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["content"] == "18 days"


def test_reupload_overwrites_instead_of_duplicating(client, pdf_path):
    h = signup(client, "re@example.com")
    upload(client, h, pdf_path)
    upload(client, h, pdf_path)
    r = client.post("/ask", headers=h, json={"question": "annual leave", "rerank": False})
    assert len({s["chunk_id"] for s in r.json()["sources"]}) == len(r.json()["sources"])


def test_users_cannot_see_each_others_documents(client, pdf_path):
    alice, bob = signup(client, "alice@example.com"), signup(client, "bob@example.com")
    upload(client, alice, pdf_path)
    r = client.post("/ask", headers=bob, json={"question": "annual leave", "rerank": False})
    assert r.status_code == 200 and r.json()["found"] is False and r.json()["sources"] == []
    assert client.get("/documents", headers=bob).json() == []


def test_non_pdf_rejected(client):
    h = signup(client, "txt@example.com")
    r = client.post("/ingest", headers=h, files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_agent_endpoint(client, pdf_path):
    h = signup(client, "ag@example.com")
    upload(client, h, pdf_path)
    r = client.post("/agent", headers=h, json={"question": "hello"})
    assert r.status_code == 200 and r.json()["answer"] == "agent says hi"
