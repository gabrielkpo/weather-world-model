"""Compare the JSON outputs of two runs, ignoring provenance, timing and memory keys.

Usage: python .github/scripts/compare_runs.py DIR_A DIR_B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Provenance, timing (any key with "seconds") and memory use vary from run to run
VOLATILE = {"date", "code_commit", "code_dirty", "peak_rss_mb"}


def strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE and "seconds" not in k}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main(a: Path, b: Path) -> int:
    files_a = sorted(p.relative_to(a) for p in a.rglob("*.json"))
    files_b = sorted(p.relative_to(b) for p in b.rglob("*.json"))
    if not files_a or files_a != files_b:
        print(f"different file sets: {files_a} vs {files_b}")
        return 1
    for f in files_a:
        if strip(json.loads((a / f).read_text())) != strip(json.loads((b / f).read_text())):
            print(f"{f} differs between {a} and {b}")
            return 1
    print(f"{len(files_a)} file(s) identical: {a} vs {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
