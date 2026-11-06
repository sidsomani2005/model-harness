"""Report renders from saved runs only and escapes embedded content."""

import json
import re

from medproof import reporting
from medproof.runs import bank_item_id, save_verifier_run
from medproof.schema import Overall, VerifierRun


def _embedded(html):
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_report_without_runs_shows_no_fabricated_verdicts(tmp_path, cases_root):
    out = reporting.build_report(cases_root, tmp_path / "runs", tmp_path / "index.html")
    data = _embedded(out.read_text())
    assert data["metrics"] is None and data["variants"] == []
    ids = [c["original"]["case_id"] for c in data["cases"]]
    assert ids == ["med_01", "med_02", "med_03"]  # fixture excluded
    for c in data["cases"]:
        assert c["twin"] is not None
        for a in c["original"]["answers"] + c["twin"]["answers"]:
            assert a["kind"] == "bank" and a["verdicts"] == {}


def test_report_includes_saved_verdict_and_escapes_script(tmp_path, cases_root):
    runs = tmp_path / "runs"
    vr = VerifierRun(target_run_id=bank_item_id("med_02", "w1"), case_id="med_02", verifier_variant="haiku",
                     overall=Overall.ERROR_DETECTED, explanation="</script><script>alert(1)</script>")
    save_verifier_run(vr, runs)
    html = reporting.build_report(cases_root, runs, tmp_path / "index.html").read_text()
    assert "</script><script>alert(1)" not in html
    data = _embedded(html)
    med02 = next(c for c in data["cases"] if c["original"]["case_id"] == "med_02")
    w1 = next(a for a in med02["original"]["answers"] if a["id"].endswith("__w1"))
    assert w1["verdicts"]["haiku"]["overall"] == "ERROR_DETECTED"
    assert data["variants"] == ["haiku"]
