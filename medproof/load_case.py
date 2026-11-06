"""Case loading, integrity validation and SHA-256 freezing (WORKPLAN 2.1, spec section 3).

Only two loaders exist:

- `load_public(case_dir) -> PublicCase`: case id, question and source text. This is
  the only object handed to the target runner, the online verifier and baselines.
- `load_private(case_dir) -> PrivateCase`: the benchmark-side key and answer bank.
  Only the benchmark scorer (and the CLI that feeds bank answer *text* to the
  verifier) may call it. `tests/test_no_key_leakage.py` enforces this.

Case layout (spec section 3)::

    cases/med_01/
      case.yaml            # private key, never fed to online components
      answer_bank.yaml     # private labeled answers
      sources/*.txt        # every physical line starts with a stable id, e.g. [L001]
      twin/case.yaml       # twin key (id: med_01_twin)
      twin/sources/*.txt
      freeze.json          # written by `freeze_case` after human review
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from medproof.schema import Overall, PublicCase, SourceLine, Trap

BANNER = "SYNTHETIC CASE — NOT A REAL PATIENT"
BANNER_RE = re.compile(r"SYNTHETIC CASE\s*[-\u2013\u2014]\s*NOT A REAL PATIENT")  # any dash
LINE_RE = re.compile(r"^\[([A-Z]{1,2})(\d{3})\] (\S.*)$")
CITED_ID_RE = re.compile(r"\b([A-Z]{1,2}\d{3})\b")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
FREEZE_FILE = "freeze.json"
APPROVED = "approved"

LABEL_TO_EXPECTED = {
    "correct": Overall.NO_ISSUE_FOUND,
    "wrong": Overall.ERROR_DETECTED,
    "ambiguous": Overall.INSUFFICIENT_TO_VERIFY,
}


class CaseFormatError(ValueError):
    pass


# --------------------------------------------------------------------------- sources


def parse_source_lines(file_name: str, text: str) -> list[SourceLine]:
    """Strictly parse a source file. Every physical line must be `[XX000] text`."""
    lines: list[SourceLine] = []
    seen: set[str] = set()
    prefix: Optional[str] = None
    for n, raw in enumerate(text.splitlines(), start=1):
        m = LINE_RE.match(raw)
        if not m:
            raise CaseFormatError(f"{file_name}:{n}: line does not start with a [XX000] id: {raw!r}")
        pfx, num, body = m.group(1), m.group(2), m.group(3)
        line_id = f"{pfx}{num}"
        if prefix is None:
            prefix = pfx
        elif pfx != prefix:
            raise CaseFormatError(f"{file_name}:{n}: id prefix {pfx!r} differs from file prefix {prefix!r}")
        if line_id in seen:
            raise CaseFormatError(f"{file_name}:{n}: duplicate line id {line_id}")
        if int(num) != len(lines) + 1:
            raise CaseFormatError(f"{file_name}:{n}: id {line_id} is out of sequence (expected {prefix}{len(lines) + 1:03d})")
        seen.add(line_id)
        lines.append(SourceLine(file=file_name, line_id=line_id, text=body))
    if not lines:
        raise CaseFormatError(f"{file_name}: empty source file")
    if not BANNER_RE.search(lines[0].text):
        raise CaseFormatError(f"{file_name}: first line must carry the banner {BANNER!r}")
    return lines


def public_sha256(case_id: str, question: str, sources: dict[str, str]) -> str:
    payload = json.dumps(
        {"case_id": case_id, "question": question, "sources": dict(sorted(sources.items()))},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CaseFormatError(f"missing {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CaseFormatError(f"{path}: expected a mapping")
    return data


# --------------------------------------------------------------------------- public loader


def load_public(case_dir: str | Path) -> PublicCase:
    """Return only id, question and source text. Key fields are never copied."""
    case_dir = Path(case_dir)
    meta = _read_yaml(case_dir / "case.yaml")
    case_id = meta.get("id")
    question = meta.get("question")
    source_files = meta.get("source_files") or []
    if not case_id or not question or not source_files:
        raise CaseFormatError(f"{case_dir}/case.yaml needs id, question and source_files")
    sources: dict[str, str] = {}
    for rel in source_files:
        path = case_dir / rel
        if not path.is_file():
            raise CaseFormatError(f"{case_dir}: source file {rel} not found")
        name = path.name
        if name in sources:
            raise CaseFormatError(f"{case_dir}: duplicate source file name {name}")
        text = path.read_text(encoding="utf-8")
        parse_source_lines(name, text)  # fail fast on malformed sources
        sources[name] = text
    question = " ".join(str(question).split())
    return PublicCase(
        case_id=str(case_id),
        question=question,
        sources=sources,
        source_sha256=public_sha256(str(case_id), question, sources),
    )


# --------------------------------------------------------------------------- private loader


class BankAnswer(BaseModel):
    answer_id: str
    label: Literal["correct", "wrong", "ambiguous"]
    expected_overall: Overall
    trap: Optional[Trap] = None
    style: str = ""
    rationale: str
    text: str


class PrivateCase(BaseModel):
    case_id: str
    case_dir: str
    twin_of: Optional[str] = None
    changed_fact: Optional[str] = None
    category: str = ""
    title: str = ""
    key: dict[str, Any]
    scoring_metadata: dict[str, Any] = Field(default_factory=dict)
    review_log: dict[str, Any] = Field(default_factory=dict)
    answer_bank: list[BankAnswer] = Field(default_factory=list)
    answer_bank_review_status: Optional[str] = None

    @property
    def review_status(self) -> Optional[str]:
        return self.key.get("review_status")

    @property
    def primary_trap(self) -> Optional[str]:
        return self.scoring_metadata.get("primary_trap")


def load_private(case_dir: str | Path) -> PrivateCase:
    """Benchmark-side only. Never call from verifier, baselines or target runner."""
    case_dir = Path(case_dir)
    meta = _read_yaml(case_dir / "case.yaml")
    if "key" not in meta or not isinstance(meta["key"], dict):
        raise CaseFormatError(f"{case_dir}/case.yaml has no key mapping")
    bank: list[BankAnswer] = []
    bank_status = None
    bank_path = case_dir / "answer_bank.yaml"
    if bank_path.is_file():
        raw_bank = _read_yaml(bank_path)
        bank_status = raw_bank.get("review_status")
        if raw_bank.get("case_id") != meta.get("id"):
            raise CaseFormatError(f"{bank_path}: case_id {raw_bank.get('case_id')!r} != {meta.get('id')!r}")
        try:
            bank = [BankAnswer.model_validate(a) for a in raw_bank.get("answers") or []]
        except ValidationError as exc:
            raise CaseFormatError(f"{bank_path}: {exc}") from exc
    return PrivateCase(
        case_id=str(meta["id"]),
        case_dir=str(case_dir),
        twin_of=meta.get("twin_of"),
        changed_fact=meta.get("changed_fact"),
        category=meta.get("category", ""),
        title=meta.get("title", ""),
        key=meta["key"],
        scoring_metadata=meta.get("scoring_metadata") or {},
        review_log=meta.get("review_log") or {},
        answer_bank=bank,
        answer_bank_review_status=bank_status,
    )


# --------------------------------------------------------------------------- discovery


def discover_cases(root: str | Path = "cases", include_fixture: bool = False) -> list[Path]:
    """Base case directories (those holding case.yaml), sorted. Fixtures start with '_'."""
    root = Path(root)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "case.yaml").is_file()]
    if not include_fixture:
        dirs = [p for p in dirs if not p.name.startswith("_")]
    return dirs


def twin_dir(case_dir: Path) -> Optional[Path]:
    t = case_dir / "twin"
    return t if (t / "case.yaml").is_file() else None


def case_and_twin_dirs(root: str | Path = "cases", include_fixture: bool = False) -> list[Path]:
    out: list[Path] = []
    for d in discover_cases(root, include_fixture):
        out.append(d)
        t = twin_dir(d)
        if t is not None:
            out.append(t)
    return out


def find_case_dir(case_id: str, root: str | Path = "cases") -> Path:
    for d in case_and_twin_dirs(root, include_fixture=True):
        if _read_yaml(d / "case.yaml").get("id") == case_id:
            return d
    raise CaseFormatError(f"no case with id {case_id!r} under {root}")


# --------------------------------------------------------------------------- freezing


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(case_dir: str | Path) -> dict[str, str]:
    """SHA-256 of every file that defines the case (sources, key, bank, twin)."""
    case_dir = Path(case_dir)
    out: dict[str, str] = {}
    for path in sorted(case_dir.rglob("*")):
        if path.is_file() and path.name != FREEZE_FILE and not path.name.startswith("."):
            out[path.relative_to(case_dir).as_posix()] = _hash_file(path)
    return out


def freeze_case(case_dir: str | Path) -> dict[str, Any]:
    """Write freeze.json. Refuses unless the case, its twin and its bank are approved."""
    case_dir = Path(case_dir)
    problems: list[str] = []
    targets = [case_dir] + ([t] if (t := twin_dir(case_dir)) else [])
    for d in targets:
        priv = load_private(d)
        if priv.review_status != APPROVED:
            problems.append(f"{priv.case_id}: key.review_status is {priv.review_status!r}, not 'approved'")
        for field in ("author", "blind_reviewer", "blind_answer"):
            if not priv.review_log.get(field):
                problems.append(f"{priv.case_id}: review_log.{field} is empty")
        if (d / "answer_bank.yaml").is_file() and priv.answer_bank_review_status != APPROVED:
            problems.append(f"{priv.case_id}: answer_bank review_status is {priv.answer_bank_review_status!r}")
    errors = [e for d in targets for e in validate_case_dir(d)["errors"]]
    problems.extend(errors)
    if problems:
        raise CaseFormatError("cannot freeze:\n  " + "\n  ".join(problems))
    record = {
        "case_id": load_public(case_dir).case_id,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "public_sha256": {load_public(d).case_id: load_public(d).source_sha256 for d in targets},
        "files": file_hashes(case_dir),
    }
    (case_dir / FREEZE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def freeze_status(case_dir: str | Path) -> tuple[str, list[str]]:
    """('frozen' | 'not_frozen' | 'modified_after_freeze', changed paths)."""
    case_dir = Path(case_dir)
    if case_dir.name == "twin":
        case_dir = case_dir.parent
    path = case_dir / FREEZE_FILE
    if not path.is_file():
        return "not_frozen", []
    recorded = json.loads(path.read_text(encoding="utf-8")).get("files", {})
    current = file_hashes(case_dir)
    changed = sorted(
        k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k)
    )
    return ("modified_after_freeze", changed) if changed else ("frozen", [])


# --------------------------------------------------------------------------- validation


def _cited_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for group in BRACKET_RE.findall(text):
        ids.update(CITED_ID_RE.findall(group))
    return ids


def validate_case_dir(case_dir: str | Path) -> dict[str, Any]:
    """Integrity checks for one case or twin directory. Returns errors and warnings."""
    case_dir = Path(case_dir)
    errors: list[str] = []
    warnings: list[str] = []
    is_twin = case_dir.name == "twin"
    report: dict[str, Any] = {"dir": str(case_dir), "case_id": None, "errors": errors, "warnings": warnings}
    try:
        meta = _read_yaml(case_dir / "case.yaml")
        public = load_public(case_dir)
        priv = load_private(case_dir)
    except (CaseFormatError, ValidationError, yaml.YAMLError) as exc:
        errors.append(str(exc))
        return report
    report["case_id"] = public.case_id
    report["review_status"] = priv.review_status
    report["answer_bank_review_status"] = priv.answer_bank_review_status
    report["n_bank"] = len(priv.answer_bank)

    expected_id = f"{case_dir.parent.name}_twin" if is_twin else case_dir.name
    if public.case_id != expected_id:
        errors.append(f"id {public.case_id!r} does not match directory (expected {expected_id!r})")
    if is_twin and meta.get("twin_of") != case_dir.parent.name:
        errors.append(f"twin_of must be {case_dir.parent.name!r}")
    if is_twin and not meta.get("changed_fact"):
        errors.append("twin case.yaml must describe changed_fact")

    lines: dict[str, dict[str, SourceLine]] = {}
    prefixes: dict[str, str] = {}
    for name, text in public.sources.items():
        parsed = parse_source_lines(name, text)
        pfx = re.match(r"[A-Z]+", parsed[0].line_id).group(0)
        if pfx in prefixes:
            errors.append(f"id prefix {pfx!r} used by both {prefixes[pfx]} and {name}")
        prefixes[pfx] = name
        lines[name] = {sl.line_id: sl for sl in parsed}
    all_ids = {lid for f in lines.values() for lid in f}

    listed = {Path(p).name for p in meta.get("source_files") or []}
    src_dir = case_dir / "sources"
    on_disk = {p.name for p in src_dir.iterdir() if p.is_file() and not p.name.startswith(".")} if src_dir.is_dir() else set()
    for extra in sorted(on_disk - listed):
        warnings.append(f"sources/{extra} exists but is not listed in source_files")

    key = priv.key
    for field in ("required_propositions", "unacceptable_claims", "rationale", "expected_evidence", "review_status"):
        if not key.get(field):
            errors.append(f"key.{field} is missing or empty")
    for ev in key.get("expected_evidence") or []:
        f, lid = ev.get("file"), ev.get("line_id")
        if f not in lines or lid not in lines[f]:
            errors.append(f"expected_evidence {f}:{lid} does not exist")
    trap = priv.primary_trap
    if trap not in Trap.__members__:
        errors.append(f"scoring_metadata.primary_trap {trap!r} is not a known Trap")

    for ans in priv.answer_bank:
        if ans.expected_overall != LABEL_TO_EXPECTED[ans.label]:
            errors.append(f"bank {ans.answer_id}: label {ans.label} requires expected_overall {LABEL_TO_EXPECTED[ans.label].value}")
        if ans.label == "wrong" and ans.trap is None:
            errors.append(f"bank {ans.answer_id}: wrong answers must name a trap")
        for lid in sorted(_cited_ids(ans.text) - all_ids):
            errors.append(f"bank {ans.answer_id}: cites nonexistent line {lid}")
    ids = [a.answer_id for a in priv.answer_bank]
    if len(ids) != len(set(ids)):
        errors.append("answer_bank has duplicate answer_id values")

    if not public.case_id.startswith("_"):
        n_correct = sum(a.label == "correct" for a in priv.answer_bank)
        n_wrong = sum(a.label == "wrong" for a in priv.answer_bank)
        if not is_twin and (n_correct < 2 or n_wrong < 2):
            warnings.append(f"answer bank has {n_correct} correct / {n_wrong} wrong; spec asks for 2 / 2")
        if not is_twin and twin_dir(case_dir) is None:
            warnings.append("no twin/ directory")
        if priv.review_status != APPROVED:
            warnings.append(f"review_status is {priv.review_status!r}: needs human review before freezing")

    status, changed = freeze_status(case_dir)
    report["freeze_status"] = status
    if status == "modified_after_freeze":
        errors.append("files changed after freeze: " + ", ".join(changed))
    return report
