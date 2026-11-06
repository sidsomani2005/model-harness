"""Case loading (WORKPLAN Step 0; Person 2 owns and extends this file).

`load_public` is the only loader the target runner and verifier paths use.
`load_private` is for the benchmark scorer only and must never be imported by
anything under `medproof/verifier/`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from medproof.schema import PublicCase


def _source_files(case_dir: Path, meta: dict) -> list[Path]:
    listed = meta.get("source_files")
    if listed:
        return [case_dir / p for p in listed]
    return sorted((case_dir / "sources").glob("*.txt"))


def load_public(case_dir: str | Path) -> PublicCase:
    case_dir = Path(case_dir)
    meta = yaml.safe_load((case_dir / "case.yaml").read_text())
    sources: dict[str, str] = {}
    h = hashlib.sha256()
    for path in _source_files(case_dir, meta):
        text = path.read_text()
        sources[path.name] = text
        h.update(path.name.encode() + b"\0" + text.encode() + b"\0")
    h.update(meta["question"].strip().encode())
    return PublicCase(
        case_id=str(meta["id"]),
        question=meta["question"].strip(),
        sources=sources,
        source_sha256=h.hexdigest(),
    )


def load_private(case_dir: str | Path) -> dict:
    """Benchmark-side only: full case.yaml including key and scoring metadata."""
    return yaml.safe_load((Path(case_dir) / "case.yaml").read_text())
