"""MedProof command line (WORKPLAN 2.8, spec section 9).

    python -m medproof.cli validate-cases
    python -m medproof.cli freeze --case med_01
    python -m medproof.cli run-targets --model claude-sonnet-5-5 --first-only [--dry-run] [--yes]
    python -m medproof.cli verify --variant haiku --runs runs/target/
    python -m medproof.cli baselines --models haiku,opus --runs runs/target/
    python -m medproof.cli evaluate --case-set cases/ --runs runs/
    python -m medproof.cli report --output report/index.html
    python -m medproof.cli demo --case med_01 --offline
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from medproof import benchmark, reporting
from medproof.load_case import (
    CaseFormatError,
    case_and_twin_dirs,
    discover_cases,
    find_case_dir,
    freeze_case,
    freeze_status,
    load_private,
    load_public,
    parse_source_lines,
    validate_case_dir,
)
from medproof.runs import (
    DEFAULT_ROOT,
    bank_item_id,
    load_target_runs,
    load_target_runs_from,
    save_verifier_run,
    verdict_path,
    verifier_dir,
)
from medproof.schema import Overall, PublicCase, VerifierRun

console = Console()


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# --------------------------------------------------------------------------- validate / freeze


def cmd_validate(args) -> int:
    dirs = case_and_twin_dirs(args.cases, include_fixture=True)
    table = Table(title="Case validation")
    for col in ("case", "errors", "warnings", "review", "bank", "freeze"):
        table.add_column(col)
    n_err = 0
    for d in dirs:
        r = validate_case_dir(d)
        n_err += len(r["errors"])
        table.add_row(
            str(r["case_id"] or d), str(len(r["errors"])), str(len(r["warnings"])),
            str(r.get("review_status")), str(r.get("n_bank")), str(r.get("freeze_status")),
        )
    console.print(table)
    for d in dirs:
        r = validate_case_dir(d)
        for e in r["errors"]:
            console.print(f"[red]ERROR[/red] {r['case_id'] or d}: {e}")
        for w in r["warnings"]:
            console.print(f"[yellow]warn[/yellow]  {r['case_id'] or d}: {w}")
    console.print("[green]validate-cases passed[/green]" if n_err == 0 else f"[red]{n_err} error(s)[/red]")
    return 0 if n_err == 0 else 1


def cmd_freeze(args) -> int:
    case_dir = Path(args.cases) / args.case
    try:
        record = freeze_case(case_dir)
    except CaseFormatError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(f"[green]Frozen {record['case_id']}[/green] -> {case_dir / 'freeze.json'}")
    for cid, sha in record["public_sha256"].items():
        console.print(f"  {cid}: {sha}")
    return 0


# --------------------------------------------------------------------------- run-targets


def cmd_run_targets(args) -> int:
    from medproof.targets.anthropic_runner import run_targets

    model = args.model or os.environ.get("TARGET_MODEL", "claude-sonnet-5-5")
    base_dirs = discover_cases(args.cases)
    if args.case:
        base_dirs = [d for d in base_dirs if d.name in set(args.case)]
    selected: list[tuple[PublicCase, bool, Path]] = []
    for d in base_dirs:
        selected.append((load_public(d), False, d))
        t = d / "twin"
        if not args.no_twins and (t / "case.yaml").is_file():
            selected.append((load_public(t), True, t))
    if not selected:
        console.print("[red]No cases selected.[/red]")
        return 1
    if not args.dry_run:
        unfrozen = [(pc.case_id, freeze_status(d)[0]) for pc, _, d in selected if freeze_status(d)[0] != "frozen"]
        if unfrozen:
            console.print("[red]Refusing official run: cases must be frozen first (spec section 3).[/red]")
            for cid, st in unfrozen:
                console.print(f"  {cid}: {st}")
            return 1
        console.print(Panel.fit(
            f"OFFICIAL target run with {model}\n"
            + "\n".join(f"  {pc.case_id}{' (twin)' if tw else ''}" for pc, tw, _ in selected)
            + "\nEach case is called once in a fresh conversation. Existing first runs are skipped.",
            title="run-targets"))
        if not args.yes:
            console.print("[yellow]Not running. Re-run with --yes to make the official calls.[/yellow]")
            return 2
    outcomes = run_targets([(pc, tw) for pc, tw, _ in selected], model=model, runs_root=Path(args.runs), dry_run=args.dry_run)
    bad = 0
    for o in outcomes:
        color = {"saved": "green", "dry_run": "cyan", "skipped_existing": "yellow", "refusal": "magenta"}.get(o.status, "red")
        bad += o.status == "error"
        console.print(f"[{color}]{o.status:17}[/{color}] {o.case_id}  {o.detail}  {o.path or ''}")
    return 1 if bad else 0


# --------------------------------------------------------------------------- verify / baselines


@dataclass
class WorkItem:
    item_id: str
    case: PublicCase
    answer: str
    refusal: bool


def _work_items(args) -> list[WorkItem]:
    items: list[WorkItem] = []
    public_cache: dict[str, PublicCase] = {}

    def public_for(case_id: str) -> PublicCase:
        if case_id not in public_cache:
            public_cache[case_id] = load_public(find_case_dir(case_id, args.cases))
        return public_cache[case_id]

    case_filter = set(args.case or [])
    if not args.no_targets:
        runs = load_target_runs_from([Path(p) for p in args.runs_paths]) if args.runs_paths else load_target_runs(Path(args.runs_root))
        for run in runs:
            if run.case_id.startswith("_") and not args.include_fixture:
                continue
            if case_filter and run.case_id.removesuffix("_twin") not in case_filter:
                continue
            try:
                pc = public_for(run.case_id)
            except CaseFormatError as exc:
                console.print(f"[yellow]skip {run.run_id}: {exc}[/yellow]")
                continue
            if pc.source_sha256 != run.case_version_sha256:
                console.print(f"[yellow]skip {run.run_id}: case sources changed since this target run[/yellow]")
                continue
            items.append(WorkItem(run.run_id, pc, run.answer_raw, run.stop_reason == "refusal"))
    if not args.no_bank:
        for d in case_and_twin_dirs(args.cases, include_fixture=args.include_fixture):
            priv = load_private(d)  # benchmark side: only the answer TEXT is passed on
            if case_filter and priv.case_id.removesuffix("_twin") not in case_filter:
                continue
            for ans in priv.answer_bank:
                items.append(WorkItem(bank_item_id(priv.case_id, ans.answer_id), public_for(priv.case_id), ans.text, False))
    return items


def _run_variant(args, variant: str, fn: Callable[[WorkItem], VerifierRun]) -> int:
    items = _work_items(args)
    if not items:
        console.print("[yellow]Nothing to verify (no target runs or bank items selected).[/yellow]")
        return 0
    errors = 0
    for it in items:
        path = verdict_path(it.item_id, variant, Path(args.runs_root))
        if path.exists() and not args.overwrite:
            console.print(f"[dim]exists  {it.item_id} [{variant}][/dim]")
            continue
        if it.refusal:
            vr = VerifierRun(target_run_id=it.item_id, verifier_variant=variant, overall=Overall.TARGET_REFUSAL,
                             case_id=it.case.case_id, explanation="Target refused; no answer to verify.")
        else:
            try:
                vr = fn(it)
            except Exception as exc:  # report and continue; nothing is saved for this item
                errors += 1
                console.print(f"[red]error   {it.item_id} [{variant}]: {type(exc).__name__}: {exc}[/red]")
                continue
        if vr.target_run_id != it.item_id or vr.verifier_variant != variant:
            vr = vr.model_copy(update={"target_run_id": it.item_id, "verifier_variant": variant})
        if vr.case_id is None:
            vr = vr.model_copy(update={"case_id": it.case.case_id})
        save_verifier_run(vr, Path(args.runs_root), overwrite=args.overwrite)
        console.print(f"saved   {it.item_id} [{variant}] -> {vr.overall.value}")
    return 1 if errors else 0


def _point_verifier_at(runs_root: Path) -> None:
    """The verifier writes raw.json under MEDPROOF_RUNS_DIR (read at import), so set both."""
    os.environ["MEDPROOF_RUNS_DIR"] = str(runs_root)
    import medproof.verifier.pipeline as pipeline

    if hasattr(pipeline, "RUNS_DIR"):
        pipeline.RUNS_DIR = Path(runs_root)


def cmd_verify(args) -> int:
    _point_verifier_at(Path(args.runs_root))
    from medproof.verifier.pipeline import verify

    return _run_variant(args, args.variant, lambda it: verify(it.case, it.answer, variant=args.variant, target_run_id=it.item_id))


BASELINE_ALIASES = {"haiku": "direct-haiku", "opus": "direct-opus", "direct-haiku": "direct-haiku", "direct-opus": "direct-opus"}


def cmd_baselines(args) -> int:
    from medproof.baselines import run_baseline

    rc = 0
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        variant = BASELINE_ALIASES.get(name)
        if variant is None:
            console.print(f"[red]unknown baseline {name!r}; use haiku and/or opus[/red]")
            return 1
        rc |= _run_variant(args, variant, lambda it, v=variant: run_baseline(
            it.case, it.answer, v, it.item_id, runs_root=Path(args.runs_root), overwrite=args.overwrite))
    return rc


# --------------------------------------------------------------------------- evaluate / report


def cmd_evaluate(args) -> int:
    metrics = benchmark.evaluate(Path(args.case_set), Path(args.runs), include_fixture=args.include_fixture)
    j, m = benchmark.write_metrics(metrics, Path(args.runs))
    console.print(benchmark.to_markdown(metrics))
    console.print(f"wrote {j} and {m}")
    unlabeled = [r for r in metrics["target_runs"] if r["label"] is None]
    if unlabeled:
        t = benchmark.write_label_template(Path(args.case_set), Path(args.runs))
        console.print(f"[yellow]{len(unlabeled)} target answer(s) unlabeled; fill in {t} and save as "
                      f"{Path(args.runs) / benchmark.LABELS_FILE}[/yellow]")
    return 0


def cmd_report(args) -> int:
    out = reporting.build_report(Path(args.cases), Path(args.runs), Path(args.output))
    console.print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- demo


def _line_lookup(pc: PublicCase) -> dict[tuple[str, str], str]:
    return {(name, sl.line_id): sl.text for name, text in pc.sources.items() for sl in parse_source_lines(name, text)}


def _print_verdict(vr: VerifierRun, pc: PublicCase, label: str) -> None:
    lines = _line_lookup(pc)
    color = {"ERROR_DETECTED": "red", "NO_ISSUE_FOUND": "green"}.get(vr.overall.value, "yellow")
    console.print(f"\n[bold {color}]{vr.overall.value}[/bold {color}]  ({label}; variant {vr.verifier_variant})")
    if vr.explanation:
        console.print(vr.explanation)
    for r in vr.needs_review_reasons:
        console.print(f"[yellow]needs review:[/yellow] {r}")
    for cv in vr.claim_verdicts:
        console.print(f"\n  [bold]{cv.status.value}[/bold] ({cv.severity.value}) {cv.claim_text}")
        if cv.why:
            console.print(f"    why: {cv.why}")
        if cv.uncertainty:
            console.print(f"    unknown: {cv.uncertainty}")
        for c in cv.citations:
            src = lines.get((c.file, c.line_id), "[line not found in source]")
            console.print(f"    {c.file}:{c.line_id}  {src}")
    for om in vr.omissions:
        tag = "confirmed" if om.confirmed else "unconfirmed"
        console.print(f"\n  [bold]OMITTED EVIDENCE[/bold] ({om.kind}, {'material' if om.material else 'minor'}, {tag}) {om.description}")
        for c in om.citations:
            src = lines.get((c.file, c.line_id), "[line not found in source]")
            console.print(f"    {c.file}:{c.line_id}  {src}")
    if vr.fake_backend:
        console.print("[yellow]fake_backend=True: offline test output, not a real verdict[/yellow]")
    console.print(f"\n  calls={vr.total_model_calls} tokens={vr.input_tokens}/{vr.output_tokens} "
                  f"cost=${vr.estimated_cost_usd:.6f} latency={vr.latency_ms}ms")


def cmd_demo(args) -> int:
    case_id = f"{args.case}_twin" if args.twin else args.case
    try:
        case_dir = find_case_dir(case_id, args.cases)
    except CaseFormatError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    pc = load_public(case_dir)
    runs_root = Path(args.runs)
    if args.bank_answer:
        item_id, label = bank_item_id(case_id, args.bank_answer), "CONSTRUCTED answer-bank item (not a target output)"
        priv = load_private(case_dir)
        match = [a for a in priv.answer_bank if a.answer_id == args.bank_answer]
        if not match:
            console.print(f"[red]no bank answer {args.bank_answer!r} for {case_id}[/red]")
            return 1
        answer = match[0].text
    else:
        runs = [r for r in load_target_runs(runs_root) if r.case_id == case_id]
        if not runs:
            console.print(f"[red]No saved target first answer for {case_id}.[/red] Run the official target pass, "
                          f"or replay a constructed item with --bank-answer <id> (it will be labeled as constructed).")
            return 1
        run = runs[0]
        item_id, answer = run.run_id, run.answer_raw
        label = f"REAL target first answer: {run.model} at {run.timestamp_utc}"
    console.print(Panel(pc.question, title=f"{case_id}  |  question"))
    console.print(Panel(answer or "(empty)", title=label))
    if args.live:
        _point_verifier_at(runs_root)
        from medproof.verifier.pipeline import verify

        try:
            vr = verify(pc, answer, variant=args.variant, target_run_id=item_id)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = verifier_dir(item_id, args.variant, runs_root) / f"live_{ts}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(vr.model_dump_json(indent=2) + "\n", encoding="utf-8")
            _print_verdict(vr, pc, f"LIVE verifier run, saved to {out}")
            return 0
        except Exception as exc:
            console.print(f"[yellow]Live verification failed ({type(exc).__name__}: {exc}); replaying saved verdict.[/yellow]")
    path = verdict_path(item_id, args.variant, runs_root)
    if not path.is_file():
        console.print(f"[red]No saved verdict at {path}. Run `verify --variant {args.variant}` first.[/red]")
        return 1
    _print_verdict(VerifierRun.model_validate_json(path.read_text(encoding="utf-8")), pc, f"REPLAY of saved verdict {path}")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medproof", description="MedProof benchmark and verifier CLI (synthetic data only)")
    p.add_argument("--cases", default="cases", help="cases directory")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("validate-cases", help="check case integrity, review and freeze status")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("freeze", help="SHA-256 freeze an approved case (and its twin and bank)")
    s.add_argument("--case", required=True)
    s.set_defaults(func=cmd_freeze)

    s = sub.add_parser("run-targets", help="blind first-attempt target run (official runs are final)")
    s.add_argument("--model", default=None)
    s.add_argument("--first-only", action="store_true", help="accepted for spec compatibility; first-only is always enforced")
    s.add_argument("--dry-run", action="store_true", help="write prompts to runs/dry_run without calling the API")
    s.add_argument("--yes", action="store_true", help="confirm the official run")
    s.add_argument("--case", action="append", help="limit to a base case id (repeatable)")
    s.add_argument("--no-twins", action="store_true")
    s.add_argument("--runs", default=str(DEFAULT_ROOT))
    s.set_defaults(func=cmd_run_targets)

    def add_item_args(s):
        s.add_argument("--runs", dest="runs_paths", action="append", help="target run dir/file (default runs/target)")
        s.add_argument("--runs-root", default=str(DEFAULT_ROOT))
        s.add_argument("--case", action="append", help="limit to a base case id (repeatable)")
        s.add_argument("--no-bank", action="store_true", help="skip answer-bank items")
        s.add_argument("--no-targets", action="store_true", help="skip target first answers")
        s.add_argument("--include-fixture", action="store_true")
        s.add_argument("--overwrite", action="store_true", help="replace existing verdicts")

    s = sub.add_parser("verify", help="run the online verifier on target answers and bank items")
    s.add_argument("--variant", default="haiku")
    add_item_args(s)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("baselines", help="run direct whole-answer judges")
    s.add_argument("--models", default="haiku")
    add_item_args(s)
    s.set_defaults(func=cmd_baselines)

    s = sub.add_parser("evaluate", help="score saved verdicts against keys and labels")
    s.add_argument("--case-set", default="cases")
    s.add_argument("--runs", default=str(DEFAULT_ROOT))
    s.add_argument("--include-fixture", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("report", help="build report/index.html from saved runs")
    s.add_argument("--output", default="report/index.html")
    s.add_argument("--runs", default=str(DEFAULT_ROOT))
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("demo", help="replay a saved case verdict (offline) or run the verifier live")
    s.add_argument("--case", required=True)
    s.add_argument("--twin", action="store_true")
    s.add_argument("--variant", default="haiku")
    s.add_argument("--bank-answer", default=None, help="replay a constructed answer-bank item instead of the target answer")
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="replay saved logs only (default)")
    mode.add_argument("--live", action="store_true", help="run verify() now, fall back to replay on failure")
    s.add_argument("--runs", default=str(DEFAULT_ROOT))
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
