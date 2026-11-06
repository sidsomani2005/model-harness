"""Every source line has a stable id; every cited id exists; frozen cases are unchanged."""

import pytest

from medproof.load_case import (
    BANNER,
    BANNER_RE,
    CaseFormatError,
    case_and_twin_dirs,
    load_public,
    parse_source_lines,
    public_sha256,
    validate_case_dir,
)


def test_all_cases_and_twins_pass_integrity(cases_root):
    dirs = case_and_twin_dirs(cases_root, include_fixture=True)
    assert dirs
    failures = {str(d): r["errors"] for d in dirs if (r := validate_case_dir(d))["errors"]}
    assert not failures, failures


def test_every_file_carries_synthetic_banner(cases_root):
    for d in case_and_twin_dirs(cases_root, include_fixture=True):
        for name, text in load_public(d).sources.items():
            assert BANNER_RE.search(text.splitlines()[0]), f"{d}/{name}"


@pytest.mark.parametrize(
    "text",
    [
        f"[L001] {BANNER}\nno id here",
        f"[L001] {BANNER}\n[L001] duplicate",
        f"[L001] {BANNER}\n[L003] gap",
        f"[L001] {BANNER}\n[M002] mixed prefix",
        f"[L001] {BANNER}\n",  # trailing newline is fine; next case is blank line
        "[L001] missing banner",
    ],
)
def test_parser_rejects_malformed_sources(text):
    if text.endswith("\n") and text.count("\n") == 1:
        parse_source_lines("x.txt", text)  # valid: single line + newline
        return
    with pytest.raises(CaseFormatError):
        parse_source_lines("x.txt", text)


def test_blank_line_rejected():
    with pytest.raises(CaseFormatError):
        parse_source_lines("x.txt", f"[L001] {BANNER}\n\n[L002] after blank")


def test_public_hash_is_order_independent_and_content_sensitive():
    a = public_sha256("c", "q", {"a.txt": "1", "b.txt": "2"})
    b = public_sha256("c", "q", {"b.txt": "2", "a.txt": "1"})
    c = public_sha256("c", "q", {"a.txt": "1", "b.txt": "3"})
    assert a == b and a != c
