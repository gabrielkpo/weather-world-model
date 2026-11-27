"""Fail if a notebook in notebooks/ is committed with outputs or execution counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    bad = []
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    for path in notebooks:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        for i, cell in enumerate(cells):
            has_output = cell.get("outputs") or cell.get("execution_count") is not None
            if cell["cell_type"] == "code" and has_output:
                bad.append(f"{path} cell {i}")
    for item in bad:
        print(f"output or execution count left in {item}")
    print(f"{len(notebooks)} notebook(s) checked")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
