"""
Read-only access to what other streams already write to disk. The API adds no
data of its own, so Unity reads exactly the same plays whether it loads them from
StreamingAssets or from here.

PLAYS_DIR (default: Hindsight's StreamingAssets/plays) is scanned recursively for
  index.json               one per game (Hindsight schemaVersion 1)
  {gameId}_{playId}.json   one per play
To serve several games, export each into its own subfolder; the exporter clears
its output folder on every run.

DECISIONS_PATH (default: packages/counterfactual/reports/decisions.jsonl) is the
play model's per-play reports, one JSON object per line.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

from .schema import PlayIndex

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYS_DIR = REPO_ROOT / "packages/replay-engine/hindsight/Assets/StreamingAssets/plays"
DEFAULT_DECISIONS = REPO_ROOT / "packages/counterfactual/reports/decisions.jsonl"
PLAY_FILE = re.compile(r"^(\d+)_(\d+)\.json$")


class PlayStore:
    def __init__(self, plays_dir: Optional[str] = None, decisions_path: Optional[str] = None):
        self.plays_dir = Path(plays_dir or os.environ.get("PLAYS_DIR") or DEFAULT_PLAYS_DIR)
        self.decisions_path = Path(decisions_path or os.environ.get("DECISIONS_PATH") or DEFAULT_DECISIONS)
        self.games: Dict[int, PlayIndex] = {}
        self.play_files: Dict[str, Path] = {}
        self.reports: Dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        games: Dict[int, PlayIndex] = {}
        files: Dict[str, Path] = {}
        if self.plays_dir.is_dir():
            for path in sorted(self.plays_dir.rglob("*.json")):
                if path.name == "index.json":
                    with open(path, encoding="utf-8") as f:
                        index = PlayIndex.model_validate(json.load(f))
                    games[index.gameId] = index
                elif PLAY_FILE.match(path.name):
                    files[path.stem] = path

        reports: Dict[str, dict] = {}
        if self.decisions_path.is_file():
            with open(self.decisions_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("game_id") is not None and r.get("play_id") is not None:
                            reports[f"{int(r['game_id'])}_{int(r['play_id'])}"] = r

        self.games, self.play_files, self.reports = games, files, reports

    def play_path(self, play_id: str) -> Optional[Path]:
        return self.play_files.get(play_id)

    def report(self, play_id: str) -> Optional[dict]:
        return self.reports.get(play_id)
