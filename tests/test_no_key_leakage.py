"""The online path (verifier, LLM wrapper, source index, Laya adapter, baselines and
target runner) must never see the private key. Spec sections 3, 4 and 13."""

import ast
import inspect
from pathlib import Path

from medproof.load_case import case_and_twin_dirs, load_private, load_public
from medproof.schema import PublicCase
from medproof.verifier import pipeline

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "medproof"

# Modules that must have no access to case.yaml at all.
VERIFIER_SIDE = [PKG / "verifier", PKG / "llm.py", PKG / "source_index.py", PKG / "laya_adapter.py"]
# Online modules that may receive a PublicCase but never load cases themselves.
ONLINE_SIDE = VERIFIER_SIDE + [PKG / "baselines.py", PKG / "targets"]

FORBIDDEN_STRINGS = ("case.yaml", "answer_bank", "load_private", "target_labels", "freeze.json", "twin/")
FORBIDDEN_MODULES = ("medproof.load_case", "medproof.benchmark", "medproof.reporting", "medproof.cli")


def _py_files(paths):
    for p in paths:
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.is_file():
            yield p


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            for a in node.names:
                yield f"{node.module}.{a.name}"


def test_online_modules_never_import_case_loaders_or_scorer():
    files = list(_py_files(ONLINE_SIDE))
    assert files, "expected online modules to exist"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports(tree):
            assert not any(name == m or name.startswith(m + ".") for m in FORBIDDEN_MODULES), (
                f"{path.relative_to(ROOT)} imports {name}"
            )


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def test_online_modules_never_reference_private_files():
    for path in _py_files(ONLINE_SIDE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                text = node.value
            elif isinstance(node, ast.Name):
                text = node.id
            elif isinstance(node, ast.Attribute):
                text = node.attr
            if text:
                for bad in FORBIDDEN_STRINGS:
                    assert bad not in text, f"{path.relative_to(ROOT)} references {bad!r}"


def test_public_case_has_only_public_fields():
    assert set(PublicCase.model_fields) == {"case_id", "question", "sources", "source_sha256"}


def test_verify_signature_takes_only_public_case_and_answer():
    params = list(inspect.signature(pipeline.verify).parameters)
    assert params == ["case", "answer", "variant", "target_run_id"]


def _key_strings(priv):
    out = []
    key = priv.key
    for field in ("required_propositions", "unacceptable_claims", "permissible_uncertainty", "alternative_correct_answers"):
        out.extend(str(x) for x in key.get(field) or [])
    if key.get("rationale"):
        out.append(str(key["rationale"]))
    out.extend(a.rationale for a in priv.answer_bank)
    out.extend(a.text for a in priv.answer_bank)
    return [" ".join(s.split()) for s in out if len(s.strip()) > 20]


def test_public_case_contains_no_key_or_bank_text(cases_root):
    dirs = case_and_twin_dirs(cases_root, include_fixture=True)
    assert dirs
    for d in dirs:
        public_blob = " ".join(load_public(d).model_dump_json().split())
        priv = load_private(d)
        for s in _key_strings(priv):
            assert s not in public_blob, f"{priv.case_id}: private text leaked into PublicCase: {s[:60]!r}"
        trap = priv.primary_trap
        assert trap and trap not in public_blob, f"{priv.case_id}: trap name leaked"
