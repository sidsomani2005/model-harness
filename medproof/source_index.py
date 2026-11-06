"""Stage 0: deterministic chart ingest, line-ID resolution, timestamps, and BM25 retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

from rank_bm25 import BM25Okapi

from medproof.schema import PublicCase, SourceLine

LINE_RE = re.compile(r"^\s*\[([A-Za-z]{1,4})(\d{1,5})\]\s?(.*)$")
# A citation as a target or checker may write it: "L007", "[L7]", "lab_results.txt:L007", "lab_results.txt L007"
CITE_RE = re.compile(r"(?:([\w.\-]+\.txt)\s*[:#,]?\s*)?\[?\b([A-Za-z]{1,4})(\d{1,5})\b\]?")

_DT_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2}):(\d{2})\b"), "ymdhm"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})\b"), "mdyhm"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "mdy"),
]

NEGATION_CUES = re.compile(
    r"\b(no|not|denies|denied|without|negative|nkda|none|ruled out|rule out|r/o|absent|never|unremarkable)\b",
    re.I,
)
STATUS_CUES = re.compile(
    r"\b(suspected|possible|probable|likely|presumed|preliminary|pending|rule out|r/o|"
    r"differential|unclear|unconfirmed|awaiting|consider|versus|vs\.?|copied|carried forward)\b",
    re.I,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set(
    "a an the of and or to in on at for with by from is was were be been are this that these those "
    "it its as per patient pt record chart".split()
)


def norm_line_id(prefix: str, number: str) -> tuple[str, int]:
    return prefix.upper(), int(number)


def parse_timestamp(text: str) -> str | None:
    for rx, kind in _DT_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        try:
            if kind == "ymdhm":
                dt = datetime(g[0], g[1], g[2], g[3], g[4])
            elif kind == "ymd":
                dt = datetime(g[0], g[1], g[2])
            elif kind == "mdyhm":
                dt = datetime(g[2], g[0], g[1], g[3], g[4])
            else:
                dt = datetime(g[2], g[0], g[1])
        except ValueError:
            continue
        return dt.isoformat(timespec="minutes") if kind in ("ymdhm", "mdyhm") else dt.date().isoformat()
    return None


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class SourceIndex:
    """Indexed view over a PublicCase's source files."""

    SHORT_CHART_CHARS = 16_000  # below this, sweeps send the whole chart

    def __init__(self, case: PublicCase):
        self.case = case
        self.lines: list[SourceLine] = []
        self._by_key: dict[tuple[str, str, int], SourceLine] = {}  # (file, PREFIX, n)
        self._by_short: dict[tuple[str, int], list[SourceLine]] = defaultdict(list)  # (PREFIX, n)
        self._file_lines: dict[str, list[SourceLine]] = defaultdict(list)
        self.malformed: list[tuple[str, int, str]] = []

        for fname, text in case.sources.items():
            current_ts: str | None = None
            for i, raw in enumerate(text.splitlines()):
                if not raw.strip():
                    continue
                m = LINE_RE.match(raw)
                if not m:
                    self.malformed.append((fname, i, raw))
                    continue
                prefix, num, body = m.groups()
                ts = parse_timestamp(body)
                inherited = False
                if ts is None and current_ts is not None:
                    ts, inherited = current_ts, True
                elif ts is not None and _looks_like_header(body):
                    current_ts = ts
                line = SourceLine(
                    file=fname, line_id=f"{prefix.upper()}{num}", text=body.strip(),
                    timestamp=ts, timestamp_inherited=inherited, line_no=len(self._file_lines[fname]),
                )
                key = norm_line_id(prefix, num)
                self.lines.append(line)
                self._by_key[(fname, *key)] = line
                self._by_short[key].append(line)
                self._file_lines[fname].append(line)

        self._bm25 = BM25Okapi([tokenize(f"{l.file} {l.text}") or ["_"] for l in self.lines])

    # ------------------------------------------------------------ lookup

    def resolve(self, ref: str, file_hint: str | None = None) -> SourceLine | None:
        """Resolve a citation such as 'L7', '[L007]', 'lab_results.txt:L007'. None if absent or ambiguous."""
        m = CITE_RE.search(ref.strip())
        if not m:
            return None
        fname, prefix, num = m.groups()
        fname = fname or file_hint
        key = norm_line_id(prefix, num)
        if fname:
            hit = self._by_key.get((fname, *key))
            if hit:
                return hit
            # tolerate a wrong/abbreviated file hint if the line ID itself is unique
        cands = self._by_short.get(key, [])
        return cands[0] if len(cands) == 1 else None

    def extract_citations(self, text: str) -> tuple[list[SourceLine], list[str]]:
        """All line-ID-looking references in free text -> (resolved lines, unresolved refs)."""
        found: list[SourceLine] = []
        missing: list[str] = []
        seen: set[tuple[str, str]] = set()
        for m in CITE_RE.finditer(text):
            fname, prefix, num = m.groups()
            # only treat bracketed or file-qualified or known-prefix tokens as citations
            known_prefix = any(k[0] == prefix.upper() for k in self._by_short)
            if not known_prefix:
                continue
            line = self.resolve(m.group(0), fname)
            if line:
                if (line.file, line.line_id) not in seen:
                    seen.add((line.file, line.line_id))
                    found.append(line)
            else:
                missing.append(m.group(0).strip())
        return found, missing

    def neighbors(self, line: SourceLine, window: int = 2) -> list[SourceLine]:
        fl = self._file_lines[line.file]
        lo, hi = max(0, line.line_no - window), min(len(fl), line.line_no + window + 1)
        return fl[lo:hi]

    def file_header(self, fname: str, n: int = 2) -> list[SourceLine]:
        return self._file_lines[fname][:n]

    # ------------------------------------------------------------ retrieval

    def search(self, query: str, k: int = 6, exclude: set[tuple[str, str]] | None = None) -> list[SourceLine]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(self.lines)), key=lambda i: scores[i], reverse=True)
        out: list[SourceLine] = []
        for i in order:
            if scores[i] <= 0:
                break
            l = self.lines[i]
            if exclude and (l.file, l.line_id) in exclude:
                continue
            out.append(l)
            if len(out) >= k:
                break
        return out

    def cue_lines(self) -> list[SourceLine]:
        """Lines carrying negation or evidence-status language (useful for sweeps)."""
        return [l for l in self.lines if NEGATION_CUES.search(l.text) or STATUS_CUES.search(l.text)]

    def timeline(self) -> list[SourceLine]:
        """Timestamped lines in chronological order (explicit timestamps first on ties)."""
        dated = [l for l in self.lines if l.timestamp]
        return sorted(dated, key=lambda l: (l.timestamp, l.timestamp_inherited, l.file, l.line_no))

    # ------------------------------------------------------------ rendering

    @property
    def total_chars(self) -> int:
        return sum(len(l.text) for l in self.lines)

    @property
    def is_short(self) -> bool:
        return self.total_chars <= self.SHORT_CHART_CHARS

    @staticmethod
    def render(lines: list[SourceLine], with_ts: bool = True) -> str:
        out = []
        for l in lines:
            ts = ""
            if with_ts and l.timestamp and l.timestamp_inherited:
                ts = f"  (note dated {l.timestamp})"
            out.append(f"{l.file} [{l.line_id}] {l.text}{ts}")
        return "\n".join(out)

    def render_full(self) -> str:
        parts = []
        for fname, fl in self._file_lines.items():
            parts.append(f"=== {fname} ===\n" + "\n".join(f"[{l.line_id}] {l.text}" for l in fl))
        return "\n\n".join(parts)


def _looks_like_header(body: str) -> bool:
    """A dated line that opens a note/section, so following undated lines inherit its date."""
    b = body.lower()
    return any(w in b for w in ("note", "summary", "report", "consult", "h&p", "discharge", "updated", "admission"))


def dedupe(lines: list[SourceLine]) -> list[SourceLine]:
    seen: set[tuple[str, str]] = set()
    out = []
    for l in lines:
        k = (l.file, l.line_id)
        if k not in seen:
            seen.add(k)
            out.append(l)
    return out
