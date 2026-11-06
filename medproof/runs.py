"""Read/write helpers for everything under runs/ (benchmark lane).

Layout (spec section 9)::

    runs/target/<run_id>.json                              TargetRun, first answers
    runs/verifier/<target_run_id>/<variant>/verdict.json   VerifierRun
    runs/verifier/<target_run_id>/<variant>/raw.json       raw per-stage model outputs
    runs/labels/target_labels.yaml                         human labels for target answers
    runs/metrics/metrics.json, metrics.md                  computed by benchmark.py
    runs/dry_run/                                          prompt previews (gitignored)

Answer-bank items are verified under the id `bank__<case_id>__<answer_id>`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from medproof.schema import TargetRun, VerifierRun

DEFAULT_ROOT = Path("runs")
BANK_PREFIX = "bank__"


def model_short(model: str) -> str:
    parts = model.split("-")
    return parts[1] if len(parts) > 1 and parts[0] == "claude" else model.replace("/", "_")


def bank_item_id(case_id: str, answer_id: str) -> str:
    return f"{BANK_PREFIX}{case_id}__{answer_id}"


def parse_bank_item_id(item_id: str) -> tuple[str, str] | None:
    if not item_id.startswith(BANK_PREFIX):
        return None
    case_id, _, answer_id = item_id[len(BANK_PREFIX):].rpartition("__")
    return (case_id, answer_id) if case_id and answer_id else None


def _write_json_new(path: Path, data: str, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- target runs


def target_dir(root: Path = DEFAULT_ROOT) -> Path:
    return Path(root) / "target"


def save_target_run(run: TargetRun, root: Path = DEFAULT_ROOT) -> Path:
    """Target first answers are never overwritten."""
    return _write_json_new(target_dir(root) / f"{run.run_id}.json", run.model_dump_json(indent=2), overwrite=False)


def load_target_runs(root: Path = DEFAULT_ROOT) -> list[TargetRun]:
    d = target_dir(root)
    if not d.is_dir():
        return []
    return [TargetRun.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def load_target_runs_from(paths: Iterable[Path]) -> list[TargetRun]:
    out: list[TargetRun] = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        out.extend(TargetRun.model_validate_json(f.read_text(encoding="utf-8")) for f in files)
    return out


# --------------------------------------------------------------------------- verifier runs


def verifier_dir(target_run_id: str, variant: str, root: Path = DEFAULT_ROOT) -> Path:
    return Path(root) / "verifier" / target_run_id / variant


def verdict_path(target_run_id: str, variant: str, root: Path = DEFAULT_ROOT) -> Path:
    return verifier_dir(target_run_id, variant, root) / "verdict.json"


def raw_path(target_run_id: str, variant: str, root: Path = DEFAULT_ROOT) -> Path:
    return verifier_dir(target_run_id, variant, root) / "raw.json"


def code_version() -> str | None:
    """Git commit of the working tree, with "-dirty" if there are uncommitted code changes."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "medproof"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def save_verifier_run(run: VerifierRun, root: Path = DEFAULT_ROOT, overwrite: bool = False) -> Path:
    if run.code_version is None:
        run.code_version = code_version()
    return _write_json_new(
        verdict_path(run.target_run_id, run.verifier_variant, root), run.model_dump_json(indent=2), overwrite
    )


def save_raw(target_run_id: str, variant: str, payload: object, root: Path = DEFAULT_ROOT, overwrite: bool = False) -> Path:
    return _write_json_new(
        raw_path(target_run_id, variant, root), json.dumps(payload, indent=2, default=str), overwrite
    )


def load_verifier_runs(root: Path = DEFAULT_ROOT) -> list[VerifierRun]:
    base = Path(root) / "verifier"
    if not base.is_dir():
        return []
    return [
        VerifierRun.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(base.glob("*/*/verdict.json"))
    ]
