"""How wrong is it, in yards.

This is the module the rest of the stream is tuned against, so it is worth being
pedantic about two things it would be easy to get quietly wrong.

**Identity is assigned per frame, not once.** A tracker that swaps two receivers
halfway through a play has not made a small error, but a single global assignment
of tracks to players would charge it only the distance between two men who were
near each other anyway. Matching frame by frame, and separately counting how often
the assignment *changed*, is the standard MOT treatment and it is the one that makes
a swap visible as a swap.

**Misses are not free.** A pipeline that emits only the six players it was sure
about would post a superb RMSE. Coverage and miss counts are reported beside the
error for exactly that reason, and the headline number is deliberately a tuple.

Everything is in yards. Pixels are not a unit of accuracy: three pixels at the near
sideline and three at the far hash are different distances, so no figure here is
ever quoted in them.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..contract import PlayTrack, shortest_deg

MOVING_YPS = 1.5
"""Below this, the direction of travel is not a fact about the player.

A lineman set in his stance has a measured ``dir`` in the release and a held one in
our output, and comparing them produces a large angle that means nothing — the
velocity vector being differenced is almost zero, so its argument is noise. Quoting a
single direction error over all samples therefore says more about how many linemen
were in the play than about the pipeline. Both figures are reported; the moving one
is the one to tune against.
"""

GATE_YARDS = 3.0
"""Beyond this, a predicted track and a real player are not the same person.

Set by what a football field affords: teammates line up about a yard apart, so a
gate much larger than a couple of yards will happily marry a guard to the centre
next to him and report a tidy error for it.
"""


@dataclass
class PlayerScore:
    player_id: str
    frames_truth: int
    frames_matched: int
    rmse_yards: float
    p95_yards: float
    speed_mae: float
    dir_mae_deg: float
    o_mae_deg: float
    switches: int

    @property
    def coverage(self) -> float:
        return self.frames_matched / self.frames_truth if self.frames_truth else 0.0


@dataclass
class Score:
    """One play, scored."""

    play: str
    frames: int
    truth_players: int

    median_yards: float
    rmse_yards: float
    p95_yards: float
    max_yards: float

    coverage: float
    misses: int
    false_positives: int
    id_switches: int
    mota: float

    speed_mae: float
    dir_mae_deg: float
    dir_mae_moving_deg: float
    o_mae_deg: float
    o_measured_fraction: float

    per_player: dict[str, PlayerScore] = dc_field(default_factory=dict)
    notes: list[str] = dc_field(default_factory=list)

    # ------------------------------------------------------------------ verdict

    def meets(self, median: float = 1.0, p95: float = 2.0, switches: int = 5) -> bool:
        """The bar from the plan. Deliberately three numbers, not one: a pipeline
        can pass on precision and fail on identity, and those need different fixes."""
        return (
            self.median_yards <= median
            and self.p95_yards <= p95
            and self.id_switches <= switches
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "play": self.play, "frames": self.frames, "players": self.truth_players,
            "median_yd": round(self.median_yards, 3), "rmse_yd": round(self.rmse_yards, 3),
            "p95_yd": round(self.p95_yards, 3), "max_yd": round(self.max_yards, 3),
            "coverage": round(self.coverage, 3), "misses": self.misses,
            "false_pos": self.false_positives, "id_switches": self.id_switches,
            "mota": round(self.mota, 3), "speed_mae": round(self.speed_mae, 3),
            "dir_mae_deg": round(self.dir_mae_deg, 1),
            "dir_moving_deg": round(self.dir_mae_moving_deg, 1),
            "o_mae_deg": round(self.o_mae_deg, 1),
            "o_measured": round(self.o_measured_fraction, 3),
        }

    def table(self) -> str:
        w = "  {:<22} {}"
        lines = [
            f"{self.play} — {self.frames} frames, {self.truth_players} players tracked",
            "",
            w.format("position median", f"{self.median_yards:.2f} yd"),
            w.format("position rmse", f"{self.rmse_yards:.2f} yd"),
            w.format("position p95", f"{self.p95_yards:.2f} yd"),
            w.format("position worst", f"{self.max_yards:.2f} yd"),
            "",
            w.format("coverage", f"{self.coverage * 100:.1f}% of player-frames"),
            w.format("misses", str(self.misses)),
            w.format("false positives", str(self.false_positives)),
            w.format("id switches", str(self.id_switches)),
            w.format("MOTA", f"{self.mota:.3f}"),
            "",
            w.format("speed mae", f"{self.speed_mae:.2f} yd/s"),
            w.format("direction mae", f"{self.dir_mae_deg:.1f}° (all samples)"),
            w.format("direction mae, moving", f"{self.dir_mae_moving_deg:.1f}° (above {MOVING_YPS} yd/s)"),
            w.format("orientation mae", f"{self.o_mae_deg:.1f}°"),
            w.format("orientation measured", f"{self.o_measured_fraction * 100:.0f}% of samples"),
        ]
        if self.notes:
            lines += ["", "  notes:"] + [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


def score(
    pred: PlayTrack,
    truth: PlayTrack,
    gate_yards: float = GATE_YARDS,
    play: str | None = None,
) -> Score:
    """Score a predicted play against ground truth on a shared clock.

    Frames are paired by time, not by index: a pipeline that resampled 59.94 fps
    video onto the 10 Hz grid will not have the same frame numbering as the release,
    and pairing on index would produce a beautiful-looking error that is really a
    half-frame of lag.
    """
    notes: list[str] = []

    pairs = _pair_frames(pred, truth)
    if not pairs:
        raise ValueError("no frames in common — check that both tracks put t = 0 at the snap")

    errors: list[float] = []
    speed_err: list[float] = []
    dir_err: list[float] = []
    dir_err_moving: list[float] = []
    o_err: list[float] = []
    o_measured = 0
    o_total = 0

    misses = 0
    false_positives = 0
    matched_total = 0
    truth_total = 0

    # truth player id -> the predicted track it was matched to on the previous frame
    last_assignment: dict[str, str] = {}
    switches = 0
    per_player_raw: dict[str, dict[str, list]] = {}

    for pf, tf in pairs:
        t_ids = list(tf.players.keys())
        p_ids = list(pf.players.keys())
        truth_total += len(t_ids)

        if not t_ids:
            false_positives += len(p_ids)
            continue
        if not p_ids:
            misses += len(t_ids)
            continue

        tp = np.array([[tf.players[i].x, tf.players[i].y] for i in t_ids])
        pp = np.array([[pf.players[i].x, pf.players[i].y] for i in p_ids])
        cost = np.linalg.norm(tp[:, None, :] - pp[None, :, :], axis=-1)

        # Anything past the gate must never be chosen, even if it is the only option
        # left; a large finite cost would still be picked by the assignment.
        gated = np.where(cost <= gate_yards, cost, 1e6)
        ri, ci = linear_sum_assignment(gated)

        frame_matched = set()
        for r, c in zip(ri, ci):
            if gated[r, c] >= 1e6:
                continue
            tid, pid = t_ids[r], p_ids[c]
            d = float(cost[r, c])
            errors.append(d)
            frame_matched.add(tid)
            matched_total += 1

            tpl, ppl = tf.players[tid], pf.players[pid]
            speed_err.append(abs(tpl.s - ppl.s))
            dir_err.append(abs(shortest_deg(tpl.dir, ppl.dir)))
            if tpl.s >= MOVING_YPS:
                dir_err_moving.append(abs(shortest_deg(tpl.dir, ppl.dir)))
            o_err.append(abs(shortest_deg(tpl.o, ppl.o)))
            o_total += 1
            if ppl.o_source == "measured":
                o_measured += 1

            slot = per_player_raw.setdefault(
                tid, {"err": [], "speed": [], "dir": [], "o": [], "switch": 0}
            )
            slot["err"].append(d)
            slot["speed"].append(abs(tpl.s - ppl.s))
            slot["dir"].append(abs(shortest_deg(tpl.dir, ppl.dir)))
            slot["o"].append(abs(shortest_deg(tpl.o, ppl.o)))

            prev = last_assignment.get(tid)
            if prev is not None and prev != pid:
                switches += 1
                slot["switch"] += 1
            last_assignment[tid] = pid

        misses += len(t_ids) - len(frame_matched)
        false_positives += len(p_ids) - len(frame_matched)

    if not errors:
        notes.append(
            f"nothing matched within {gate_yards} yards — the output is not merely "
            "imprecise, it is somewhere else. Suspect the registration, the play "
            "direction, or a coordinate convention."
        )

    # With nothing matched there is no error distribution to summarise. Reporting NaN
    # is correct; computing it from an all-NaN array just prints warnings that make a
    # real failure harder to read than the note already attached above.
    e = np.asarray(errors, dtype=float)
    stats = (
        dict(median=float(np.median(e)), rmse=float(np.sqrt(np.mean(e ** 2))),
             p95=float(np.percentile(e, 95)), worst=float(np.max(e)))
        if len(e)
        else dict(median=float("nan"), rmse=float("nan"), p95=float("nan"), worst=float("nan"))
    )
    per_player = {
        tid: PlayerScore(
            player_id=tid,
            frames_truth=sum(1 for _, tf in pairs if tid in tf.players),
            frames_matched=len(v["err"]),
            rmse_yards=float(np.sqrt(np.mean(np.square(v["err"])))) if v["err"] else float("nan"),
            p95_yards=float(np.percentile(v["err"], 95)) if v["err"] else float("nan"),
            speed_mae=float(np.mean(v["speed"])) if v["speed"] else float("nan"),
            dir_mae_deg=float(np.mean(v["dir"])) if v["dir"] else float("nan"),
            o_mae_deg=float(np.mean(v["o"])) if v["o"] else float("nan"),
            switches=v["switch"],
        )
        for tid, v in per_player_raw.items()
    }

    unseen = [p for p in truth.roster if p not in per_player]
    if unseen:
        notes.append(f"{len(unseen)} player(s) never matched on any frame: {', '.join(sorted(unseen)[:6])}")

    if truth.meta.get("play_direction_confidence", 1.0) < 0.3:
        notes.append(
            "ground truth's play direction was inferred from only "
            f"{truth.meta.get('play_direction_confidence', 0) * 10:.1f} yards of ball "
            "movement — if the error is near-symmetric about midfield, it is mirrored, not wrong"
        )

    mota = 1.0 - ((misses + false_positives + switches) / truth_total) if truth_total else 0.0

    return Score(
        play=play or str(truth.meta.get("play_id", "?")),
        frames=len(pairs),
        truth_players=len(truth.roster),
        median_yards=stats["median"],
        rmse_yards=stats["rmse"],
        p95_yards=stats["p95"],
        max_yards=stats["worst"],
        coverage=matched_total / truth_total if truth_total else 0.0,
        misses=misses,
        false_positives=false_positives,
        id_switches=switches,
        mota=mota,
        speed_mae=float(np.mean(speed_err)) if speed_err else float("nan"),
        dir_mae_deg=float(np.mean(dir_err)) if dir_err else float("nan"),
        dir_mae_moving_deg=float(np.mean(dir_err_moving)) if dir_err_moving else float("nan"),
        o_mae_deg=float(np.mean(o_err)) if o_err else float("nan"),
        o_measured_fraction=(o_measured / o_total) if o_total else 0.0,
        per_player=per_player,
        notes=notes,
    )


def _pair_frames(pred: PlayTrack, truth: PlayTrack, tol: float = 0.05):
    """Pair frames on time, nearest within half a tick."""
    out = []
    p_times = np.array([f.t for f in pred.frames]) if pred.frames else np.asarray([])
    for tf in truth.frames:
        if not len(p_times):
            break
        k = int(np.argmin(np.abs(p_times - tf.t)))
        if abs(p_times[k] - tf.t) <= tol:
            out.append((pred.frames[k], tf))
    return out
