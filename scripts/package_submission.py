"""Build submission/medproof_submission.zip from git-tracked files plus generated
run logs and the report. Never includes .env, .venv or scratch output.

Usage: python scripts/package_submission.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "medproof_submission.zip"
GENERATED = ["runs/target", "runs/verifier", "runs/labels", "runs/metrics", "report/index.html"]
NEVER = (".env", ".venv/", "runs/dry_run/", "runs/scratch/", "submission/", "__pycache__/")


def main() -> int:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    files = {Path(f) for f in tracked if (ROOT / f).is_file()}
    for g in GENERATED:
        p = ROOT / g
        if p.is_file():
            files.add(p.relative_to(ROOT))
        elif p.is_dir():
            files.update(q.relative_to(ROOT) for q in p.rglob("*") if q.is_file() and not q.name.startswith("live_"))
    files = sorted(f for f in files if not any(f.as_posix() == n.rstrip("/") or f.as_posix().startswith(n) for n in NEVER))
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("WARNING: uncommitted changes are included from the working tree:\n" + dirty)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f"medproof/{f.as_posix()}")
    print(f"wrote {OUT} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
