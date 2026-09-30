"""Reading .env, without a dependency and without surprises.

The project keeps credentials in a gitignored .env, which is the right place for them
— but nothing in Python reads that file by itself, so a key sitting in .env and a key
the code can see are two different things. This closes that gap for the command-line
entry points.

Two rules it follows deliberately. A variable already set in the real environment always
wins, so an explicit `GEMINI_API_KEY=... command` overrides the file rather than being
silently ignored. And nothing here ever logs or echoes a value — the loader reports which
*names* it set, never what they were set to.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_env(start: Path | None = None) -> Path | None:
    """The nearest .env at or above ``start``, so the CLI works from any subdirectory."""
    here = (start or Path.cwd()).resolve()
    for d in [here, *here.parents]:
        p = d / ".env"
        if p.is_file():
            return p
    return None


def load_env(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Load .env into os.environ. Returns the names that were set, never the values."""
    p = Path(path) if path else find_env()
    if p is None or not Path(p).is_file():
        return []

    applied: list[str] = []
    for raw in Path(p).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        val = val.strip()
        # Quotes are how people write values containing spaces; they are not the value.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = val
        applied.append(key)
    return applied
