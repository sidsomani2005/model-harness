"""WingMan demo server: serves saved data, and live verify only passes PublicCase + answer."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from medproof import wingman
from medproof.schema import Overall, PublicCase, VerifierRun


@pytest.fixture
def server(tmp_path, cases_root):
    calls = []

    def fake_verify(case, answer, variant, target_run_id, **kw):
        calls.append(dict(case=case, answer=answer, variant=variant, tid=target_run_id, **kw))
        return VerifierRun(target_run_id=target_run_id, case_id=case.case_id, verifier_variant=variant,
                           overall=Overall.NO_ISSUE_FOUND, fake_backend=True)

    runs = tmp_path / "runs"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), wingman.make_handler(cases_root, runs, fake_verify))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", calls, runs
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_is_wingman(server):
    base, _, _ = server
    status, body = _get(base + "/")
    assert status == 200 and b"<title>WingMan</title>" in body and b"MedProof" not in body


def test_state_lists_cases_without_fabricated_verdicts(server):
    base, _, _ = server
    status, body = _get(base + "/api/state")
    data = json.loads(body)
    assert status == 200
    assert [c["original"]["case_id"] for c in data["cases"]] == ["med_01", "med_02", "med_03"]
    assert all(c["title"] and c["category"] for c in data["cases"])
    assert data["metrics"] is None and data["variants"] == []
    assert all(a["verdicts"] == {} for c in data["cases"] for a in c["original"]["answers"])


def test_live_verify_passes_only_public_case_and_answer(server):
    base, calls, runs = server
    status, body = _post(base + "/api/verify", {"case_id": "med_02", "answer": "Penicillin anaphylaxis [A004].", "variant": "haiku"})
    assert status == 200 and body["verdict"]["overall"] == "NO_ISSUE_FOUND"
    [c] = calls
    assert isinstance(c["case"], PublicCase) and c["answer"] == "Penicillin anaphylaxis [A004]."
    assert c["tid"].startswith("live__med_02__") and c["runs_root"] == runs / "live" and c["save"] is True
    assert (runs / "live" / "verifier" / c["tid"] / "haiku" / "verdict.json").is_file()
    assert not (runs / "verifier").exists()  # official tree untouched


@pytest.mark.parametrize("payload", [
    {"case_id": "med_02", "answer": "   "},
    {"case_id": "../etc", "answer": "x"},
    {"case_id": "med_02", "answer": "x", "variant": "direct-opus"},
    {"case_id": "nope", "answer": "x"},
])
def test_live_verify_rejects_bad_input(server, payload):
    base, calls, _ = server
    status, body = _post(base + "/api/verify", payload)
    assert status == 400 and body["error"] and calls == []


def test_live_verify_without_key_is_unavailable(tmp_path, cases_root, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(PermissionError):
        wingman.run_live_verify({"case_id": "med_01", "answer": "x"}, cases_root, tmp_path)
