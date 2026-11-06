"""WingMan demo server: a local web UI over the MedProof benchmark and verifier.

    python -m medproof.wingman            # http://127.0.0.1:8765
    python -m medproof.cli wingman --port 8765

Read-only views come from saved files (cases, runs/target, runs/verifier, runs/metrics),
exactly as the static report does. The only write path is POST /api/verify, which runs
the real verifier on (PublicCase, answer text) and stores the result under runs/live/,
so a live demo never touches the official logs. Nothing is fabricated: with no API key
the page says so and only shows saved results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from medproof import reporting
from medproof.load_case import CaseFormatError, discover_cases, find_case_dir, load_private, load_public
from medproof.runs import DEFAULT_ROOT

WEB_DIR = Path(__file__).resolve().parent / "web"
LIVE_VARIANTS = ("haiku", "haiku-nosweep", "haiku-nochrono")
MAX_ANSWER_CHARS = 20_000
CATEGORY_LABELS = {
    "temporal_causality": "Temporal causality",
    "silent_contradiction": "Silent contradiction",
    "should_abstain": "Should abstain",
    "anchoring": "Diagnostic anchoring",
    "summary_hallucination": "Invented summary details",
    "stale_copy_forward": "Stale copy-forward",
}


FIX_HINT = "Start WingMan with the project environment: ./scripts/wingman.sh (or .venv/bin/python -m medproof.cli wingman)"


def api_key_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def dependency_problem() -> Optional[str]:
    """Why the live verifier cannot run in this Python, or None if it can."""
    try:
        import anthropic  # noqa: F401
        import rank_bm25  # noqa: F401
    except ImportError as exc:
        return (f"This server is running on {sys.executable}, which is missing '{exc.name}'. {FIX_HINT}")
    return None


def live_unavailable_reason() -> Optional[str]:
    problem = dependency_problem()
    if problem:
        return problem
    if not api_key_configured():
        return "No ANTHROPIC_API_KEY configured (add it to .env). Saved results are still shown."
    return None


def state_payload(cases_root: Path, runs_root: Path) -> dict[str, Any]:
    data = reporting.build_report_data(cases_root, runs_root)
    for entry, d in zip(data["cases"], discover_cases(cases_root)):
        priv = load_private(d)  # display metadata only (title, category); never sent to verify()
        entry["title"] = priv.title
        entry["category"] = CATEGORY_LABELS.get(priv.category, priv.category.replace("_", " ").capitalize())
    reason = live_unavailable_reason()
    data["status"] = {
        "api_key": reason is None,
        "live_unavailable_reason": reason,
        "live_variants": list(LIVE_VARIANTS),
        "runs_root": str(runs_root),
    }
    return data


def run_live_verify(body: dict[str, Any], cases_root: Path, runs_root: Path,
                    verify_fn: Optional[Callable] = None) -> dict[str, Any]:
    case_id = str(body.get("case_id", ""))
    answer = str(body.get("answer", ""))
    variant = str(body.get("variant") or "haiku")
    if not re.fullmatch(r"[A-Za-z0-9_]+", case_id):
        raise ValueError("invalid case_id")
    if not answer.strip():
        raise ValueError("answer is empty")
    if len(answer) > MAX_ANSWER_CHARS:
        raise ValueError(f"answer longer than {MAX_ANSWER_CHARS} characters")
    if variant not in LIVE_VARIANTS:
        raise ValueError(f"variant must be one of {LIVE_VARIANTS}")
    if verify_fn is None:
        reason = live_unavailable_reason()
        if reason:
            raise PermissionError(reason)
        from medproof.verifier.pipeline import verify as verify_fn  # real pipeline
    pc = load_public(find_case_dir(case_id, cases_root))  # PublicCase only: no key, traps or labels
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tid = f"live__{case_id}__{stamp}"
    live_root = Path(runs_root) / "live"
    vr = verify_fn(pc, answer, variant, tid, save=True, runs_root=live_root)
    out = live_root / "verifier" / tid / variant / "verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(vr.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return {"verdict": vr.model_dump(mode="json"), "saved_to": str(out)}


def make_handler(cases_root: Path, runs_root: Path, verify_fn: Optional[Callable] = None):
    lock = threading.Lock()  # one live verification at a time keeps API spend predictable

    class Handler(BaseHTTPRequestHandler):
        server_version = "WingMan/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                try:
                    self._json(HTTPStatus.OK, state_payload(cases_root, runs_root))
                except Exception as exc:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/verify":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_ANSWER_CHARS * 4:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not lock.acquire(blocking=False):
                    self._json(HTTPStatus.CONFLICT, {"error": "A verification is already running."})
                    return
                try:
                    result = run_live_verify(body, cases_root, runs_root, verify_fn)
                finally:
                    lock.release()
                self._json(HTTPStatus.OK, result)
            except PermissionError as exc:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            except (ValueError, CaseFormatError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": f"Verifier failed: {type(exc).__name__}: {exc}"})

    return Handler


def serve(cases_root: Path = Path("cases"), runs_root: Path = DEFAULT_ROOT, host: str = "127.0.0.1",
          port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(Path(cases_root), Path(runs_root)))
    print(f"WingMan running at http://{host}:{port}  (Ctrl+C to stop)")
    reason = live_unavailable_reason()
    print("Live verification:", "enabled" if reason is None else f"DISABLED. {reason}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: Optional[list[str]] = None) -> None:
    from medproof.cli import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(prog="wingman", description="WingMan demo UI (synthetic data only)")
    p.add_argument("--cases", default="cases")
    p.add_argument("--runs", default=str(DEFAULT_ROOT))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    serve(Path(a.cases), Path(a.runs), a.host, a.port)


if __name__ == "__main__":
    main()
