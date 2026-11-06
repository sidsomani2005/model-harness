"""Path C: score saved verdicts against private keys and human labels (WORKPLAN 2.6, spec 4 and 7).

This module only READS verdicts that were already written to runs/verifier/. It never
calls the verifier or a model, so the key cannot influence a verdict.

Labels:
- Answer-bank items: label and trap from cases/<id>/answer_bank.yaml (constructed answers).
- Real target first answers: human labels in runs/labels/target_labels.yaml. Unlabeled
  target answers are reported as unlabeled and excluded from recall / false-alarm rates.
  `write_label_template` creates a fill-in template; nothing is ever auto-labeled.

Metrics are computed separately for `target` (real first answers) and `bank` (constructed).
`INSUFFICIENT_TO_VERIFY` is counted separately and never as a successful detection.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from medproof.load_case import case_and_twin_dirs, load_private, load_public, parse_source_lines
from medproof.runs import DEFAULT_ROOT, bank_item_id, load_target_runs, load_verifier_runs
from medproof.schema import Citation, Overall, VerifierRun

LABELS_FILE = Path("labels") / "target_labels.yaml"
LABEL_TEMPLATE_FILE = Path("labels") / "target_labels.TEMPLATE.yaml"
VALID_LABELS = {"correct", "wrong", "ambiguous"}


@dataclass
class Item:
    item_id: str
    source: str  # "target" | "bank"
    case_id: str
    base_case_id: str
    is_twin: bool
    label: Optional[str]  # correct | wrong | ambiguous | None (unlabeled) | "refusal"
    trap: Optional[str]
    answer_text: str
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- citation validity

_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_TRANSLATE).lower()
    s = re.sub(r"^\s*\[[a-z]{1,2}\d{3}\]\s*", "", s)
    return " ".join(s.split()).strip(" \"'")


def citation_is_valid(cit: Citation, lines: dict[str, dict[str, str]]) -> tuple[bool, str]:
    """Line must exist in the named file; the quote must appear in that line (after
    normalization; '...' separated fragments must appear in order)."""
    file_lines = lines.get(cit.file)
    if file_lines is None:
        return False, f"unknown file {cit.file!r}"
    text = file_lines.get(cit.line_id)
    if text is None:
        return False, f"{cit.file}:{cit.line_id} does not exist"
    quote = normalize_text(cit.quote or "")
    if not quote:
        return False, f"{cit.file}:{cit.line_id} has no quote"
    hay = normalize_text(text)
    pos = 0
    for frag in [f.strip() for f in re.split(r"\.\.\.|…", quote) if f.strip()]:
        idx = hay.find(frag, pos)
        if idx < 0:
            return False, f"quote not found in {cit.file}:{cit.line_id}"
        pos = idx + len(frag)
    return True, "ok"


# --------------------------------------------------------------------------- items


def _load_labels(runs_root: Path) -> dict[str, dict[str, Any]]:
    path = Path(runs_root) / LABELS_FILE
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = data.get("labels") or {}
    for run_id, entry in labels.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: label for {run_id} must be a mapping")
        if entry.get("label") is not None and entry["label"] not in VALID_LABELS:
            raise ValueError(f"{path}: {run_id} has invalid label {entry['label']!r}")
    return labels


def collect_items(cases_root: Path, runs_root: Path = DEFAULT_ROOT, include_fixture: bool = False) -> list[Item]:
    items: list[Item] = []
    known: dict[str, tuple[str, bool]] = {}
    for d in case_and_twin_dirs(cases_root, include_fixture=include_fixture):
        priv = load_private(d)
        base = priv.twin_of or priv.case_id
        is_twin = priv.twin_of is not None
        known[priv.case_id] = (base, is_twin)
        for ans in priv.answer_bank:
            items.append(Item(
                item_id=bank_item_id(priv.case_id, ans.answer_id), source="bank", case_id=priv.case_id,
                base_case_id=base, is_twin=is_twin, label=ans.label,
                trap=ans.trap.value if ans.trap else None, answer_text=ans.text,
                detail={"answer_id": ans.answer_id, "style": ans.style, "rationale": ans.rationale},
            ))
    labels = _load_labels(runs_root)
    for run in load_target_runs(runs_root):
        if run.case_id not in known:
            continue
        if run.response_model and run.response_model != run.model:
            continue  # answered by a different model than requested: invalid for the official dataset
        base, is_twin = known[run.case_id]
        entry = labels.get(run.run_id) or {}
        label = "refusal" if run.stop_reason == "refusal" else entry.get("label")
        items.append(Item(
            item_id=run.run_id, source="target", case_id=run.case_id, base_case_id=base, is_twin=is_twin,
            label=label, trap=entry.get("trap"), answer_text=run.answer_raw,
            detail={"model": run.model, "timestamp_utc": run.timestamp_utc, "stop_reason": run.stop_reason,
                    "labeled_by": entry.get("labeled_by"), "label_notes": entry.get("notes")},
        ))
    return items


def write_label_template(cases_root: Path, runs_root: Path = DEFAULT_ROOT) -> Path:
    """Template listing unlabeled target first answers. Humans fill it in and save it
    as runs/labels/target_labels.yaml. Never auto-filled."""
    labels = _load_labels(runs_root)
    pending = {
        it.item_id: {"case_id": it.case_id, "label": None, "trap": None, "labeled_by": [], "notes": ""}
        for it in collect_items(cases_root, runs_root)
        if it.source == "target" and it.label is None and it.item_id not in labels
    }
    path = Path(runs_root) / LABEL_TEMPLATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Fill in after reading each target first answer against the case key.\n"
        "# label: correct | wrong | ambiguous ; trap: a Trap name or null ; labeled_by: reviewer names.\n"
        "# Copy completed entries into runs/labels/target_labels.yaml under `labels:`.\n"
    )
    path.write_text(header + yaml.safe_dump({"labels": pending}, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- metrics


def _pct(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def _p90(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def _source_lines(cases_root: Path, include_fixture: bool) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for d in case_and_twin_dirs(cases_root, include_fixture=include_fixture):
        pc = load_public(d)
        out[pc.case_id] = {
            name: {sl.line_id: sl.text for sl in parse_source_lines(name, text)} for name, text in pc.sources.items()
        }
    return out


def score_group(items: list[Item], verdicts: dict[tuple[str, str], VerifierRun], variant: str,
                lines: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, int]] = {}
    per_case: dict[str, dict[str, Any]] = {}
    costs: list[float] = []
    latencies: list[float] = []
    cit_total = cit_valid = 0
    invalid_examples: list[str] = []
    trap_checked = trap_match = trap_unreported = 0
    missing = unlabeled = refusals = stub = 0
    for it in items:
        if it.label == "refusal":
            refusals += 1
            continue
        if it.label is None:
            unlabeled += 1
            continue
        vr = verdicts.get((it.item_id, variant))
        if vr is None:
            missing += 1
            continue
        if any("stub" in r for r in vr.needs_review_reasons) or vr.verifier_variant.startswith("stub"):
            stub += 1
            continue  # counted, but never included in recall / false-alarm / cost
        overall = vr.overall.value
        counts = by_label.setdefault(it.label, {})
        counts[overall] = counts.get(overall, 0) + 1
        pc = per_case.setdefault(it.case_id, {})
        pc_counts = pc.setdefault(it.label, {})
        pc_counts[overall] = pc_counts.get(overall, 0) + 1
        costs.append(vr.estimated_cost_usd)
        latencies.append(vr.latency_ms)
        for cv in vr.claim_verdicts:
            for cit in cv.citations:
                cit_total += 1
                ok, why = citation_is_valid(cit, lines.get(it.case_id, {}))
                if ok:
                    cit_valid += 1
                elif len(invalid_examples) < 10:
                    invalid_examples.append(f"{it.item_id}: {why}")
        if it.label == "wrong" and overall == Overall.ERROR_DETECTED.value and it.trap:
            trap_checked += 1
            found = {t.value for t in vr.traps_found}
            if not found:
                trap_unreported += 1
            elif it.trap in found:
                trap_match += 1

    def block(label: str, hit: str) -> dict[str, Any]:
        c = by_label.get(label, {})
        n = sum(c.values())
        return {"n": n, "counts": c, "rate": _pct(c.get(hit, 0), n), "k": c.get(hit, 0)}

    wrong = block("wrong", Overall.ERROR_DETECTED.value)
    correct = block("correct", Overall.ERROR_DETECTED.value)
    ambiguous = block("ambiguous", Overall.INSUFFICIENT_TO_VERIFY.value)
    correct_insuff = by_label.get("correct", {}).get(Overall.INSUFFICIENT_TO_VERIFY.value, 0)
    wrong_insuff = by_label.get("wrong", {}).get(Overall.INSUFFICIENT_TO_VERIFY.value, 0)
    return {
        "variant": variant,
        "n_scored": sum(sum(c.values()) for c in by_label.values()),
        "n_case_clusters": len({it.base_case_id for it in items if (it.item_id, variant) in verdicts}),
        "error_recall": {"k": wrong["k"], "n": wrong["n"], "rate": wrong["rate"], "counts": wrong["counts"]},
        "false_alarm_rate": {"k": correct["k"], "n": correct["n"], "rate": correct["rate"], "counts": correct["counts"]},
        "abstention": {
            "ambiguous_insufficient": {"k": ambiguous["k"], "n": ambiguous["n"], "rate": ambiguous["rate"]},
            "false_abstentions_on_correct": correct_insuff,
            "insufficient_on_wrong": wrong_insuff,
        },
        "trap_attribution": {"checked": trap_checked, "matched": trap_match, "no_trap_reported": trap_unreported,
                             "rate": _pct(trap_match, trap_checked),
                             "note": "automatic match on traps_found; confirm by teammate review"},
        "evidence_validity": {"citations": cit_total, "valid": cit_valid, "rate": _pct(cit_valid, cit_total),
                              "invalid_examples": invalid_examples,
                              "note": "line exists and quote matches the line; whether it supports the "
                                      "explanation needs human review"},
        "cost_usd": {"mean": statistics.fmean(costs) if costs else None,
                     "median": statistics.median(costs) if costs else None, "n": len(costs)},
        "latency_ms": {"median": statistics.median(latencies) if latencies else None,
                       "p90": _p90(latencies), "n": len(latencies)},
        "missing_verdicts": missing,
        "unlabeled": unlabeled,
        "target_refusals": refusals,
        "stub_verdicts": stub,
        "per_case": per_case,
    }


def evaluate(cases_root: Path, runs_root: Path = DEFAULT_ROOT, include_fixture: bool = False) -> dict[str, Any]:
    items = collect_items(cases_root, runs_root, include_fixture)
    item_ids = {it.item_id for it in items}
    verdicts = {(v.target_run_id, v.verifier_variant): v for v in load_verifier_runs(runs_root)
                if v.target_run_id in item_ids}
    variants = sorted({v for _, v in verdicts})
    lines = _source_lines(cases_root, include_fixture)
    tables: dict[str, dict[str, Any]] = {}
    for source in ("target", "bank"):
        group = [it for it in items if it.source == source]
        tables[source] = {
            "n_items": len(group),
            "description": "real target first answers (human-labeled)" if source == "target"
                           else "constructed answer bank (team-written, independently labeled)",
            "variants": {v: score_group(group, verdicts, v, lines) for v in variants},
        }
    target_runs = [
        {"run_id": it.item_id, "case_id": it.case_id, "is_twin": it.is_twin, "label": it.label,
         "trap": it.trap, **it.detail}
        for it in items if it.source == "target"
    ]
    item_rows = [
        {"item_id": it.item_id, "source": it.source, "case_id": it.case_id, "is_twin": it.is_twin,
         "label": it.label, "trap": it.trap,
         "verdicts": {v: verdicts[(it.item_id, v)].overall.value for v in variants if (it.item_id, v) in verdicts}}
        for it in items
    ]
    return {
        "variants": variants,
        "tables": tables,
        "target_runs": target_runs,
        "items": item_rows,
        "notes": [
            "Small-N, preliminary. Items are clustered by case; variants of one case are not independent scenarios.",
            "Rates exclude unlabeled target answers, refusals and missing verdicts; those are counted separately.",
            "INSUFFICIENT_TO_VERIFY is never counted as a detection.",
        ],
    }


def _fmt_rate(block: dict[str, Any]) -> str:
    if not block["n"]:
        return "n/a"
    return f"{block['k']}/{block['n']} ({block['rate']:.0%})"


def _fmt_num(x: Optional[float], spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def to_markdown(metrics: dict[str, Any]) -> str:
    out = ["# WingMan measured results", "", "Generated by `python -m medproof.cli evaluate` from files under `runs/`. "
           "No numbers are hand-edited.", ""]
    for note in metrics["notes"]:
        out.append(f"- {note}")
    for source, table in metrics["tables"].items():
        out += ["", f"## {source}: {table['description']}", "", f"Items: {table['n_items']}", ""]
        if not table["variants"]:
            out.append("No verdicts saved yet.")
            continue
        out.append("| variant | error recall | false-alarm rate | insufficient on wrong | false abstentions | "
                   "citation validity | mean cost (USD) | median latency (ms) | p90 latency (ms) | missing | unlabeled | refusals | stub |")
        out.append("|" + "---|" * 13)
        for v, m in table["variants"].items():
            ev = m["evidence_validity"]
            out.append(
                f"| {v} | {_fmt_rate(m['error_recall'])} | {_fmt_rate(m['false_alarm_rate'])} | "
                f"{m['abstention']['insufficient_on_wrong']} | {m['abstention']['false_abstentions_on_correct']} | "
                f"{ev['valid']}/{ev['citations']} | {_fmt_num(m['cost_usd']['mean'], '.6f')} | "
                f"{_fmt_num(m['latency_ms']['median'], '.0f')} | {_fmt_num(m['latency_ms']['p90'], '.0f')} | "
                f"{m['missing_verdicts']} | {m['unlabeled']} | {m['target_refusals']} | {m['stub_verdicts']} |"
            )
    return "\n".join(out) + "\n"


def write_metrics(metrics: dict[str, Any], runs_root: Path = DEFAULT_ROOT) -> tuple[Path, Path]:
    d = Path(runs_root) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    j, m = d / "metrics.json", d / "metrics.md"
    j.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    m.write_text(to_markdown(metrics), encoding="utf-8")
    return j, m
