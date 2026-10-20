"""``whatif`` — the command that puts a counterfactual next to the play it disputes.

At 0% coverage before this file. The parts worth pinning: that it writes the
simulated play where the renderer will find it, that the simulated play lands
beside the real one rather than at the end of the game, that it never overwrites a
real play, and that ``--worst`` ranks by the thing it claims to rank by.
"""

import json

import pytest

from cf_support import HINDSIGHT_PLAYS, MODEL
from gridiron_counterfactual import __main__ as cli
from gridiron_tracking import hindsight

pytestmark = pytest.mark.skipif(
    not HINDSIGHT_PLAYS.exists() or not MODEL.exists(),
    reason="needs the hindsight subtree and the fitted model",
)


@pytest.fixture
def plays(tmp_path):
    """A few real completed passes, copied somewhere writable."""
    pytest.importorskip("sklearn")
    out = tmp_path / "plays"
    out.mkdir()
    taken = 0
    for path in hindsight.plays_in(HINDSIGHT_PLAYS):
        if path.name.startswith("999000001_"):
            continue
        doc = json.loads(path.read_text())
        if doc.get("passResult") != "C":
            continue
        (out / path.name).write_text(path.read_text())
        taken += 1
        if taken == 4:
            break
    assert taken, "no completed passes to work with"
    return out


def real_ids(plays):
    return {p.stem for p in plays.glob("*.json") if p.name != "index.json"}


def test_no_write_leaves_the_folder_alone(plays, capsys):
    before = real_ids(plays)
    rc = cli.main(["whatif", "--worst", "2", "--plays", str(plays), "--no-write"])
    assert rc == 0
    assert real_ids(plays) == before
    out = capsys.readouterr().out
    assert "separation" in out and "call" in out


def test_a_simulation_is_written_next_to_the_play_it_argues_with(plays):
    before = real_ids(plays)
    assert cli.main(["whatif", "--worst", "1", "--plays", str(plays)]) == 0

    added = real_ids(plays) - before
    assert len(added) == 1
    sim_id = added.pop()
    game, play = sim_id.split("_")

    doc = json.loads((plays / f"{sim_id}.json").read_text())
    assert doc["simulated"] is True
    assert doc["description"].startswith("SIMULATED")
    # Same quarter and drive, next free play id: the navigator groups on those, so
    # this is what puts the forecast one keypress from the throw.
    real = [i for i in before if i.startswith(game)]
    assert real
    nearest = min(real, key=lambda i: abs(int(i.split("_")[1]) - int(play)))
    nearest_doc = json.loads((plays / f"{nearest}.json").read_text())
    assert doc["quarter"] == nearest_doc["quarter"]
    assert doc["driveNumber"] == nearest_doc["driveNumber"]
    assert int(play) > int(nearest.split("_")[1])


def test_a_simulation_never_overwrites_a_real_play(plays):
    """It takes the next free id, and 'free' has to mean free."""
    before = real_ids(plays)
    cli.main(["whatif", "--all-passes", "--plays", str(plays)])

    for stem in before:
        doc = json.loads((plays / f"{stem}.json").read_text())
        assert doc.get("simulated") is not True, f"{stem} was overwritten"


def test_rerunning_does_not_pile_up_simulations(plays):
    """Each run writes a sim; the second run must not treat the first as a play
    to review, or every pass breeds a forecast of a forecast."""
    cli.main(["whatif", "--all-passes", "--plays", str(plays)])
    after_one = real_ids(plays)
    cli.main(["whatif", "--all-passes", "--plays", str(plays)])
    after_two = real_ids(plays)

    sims_one = {i for i in after_one if json.loads((plays / f"{i}.json").read_text()).get("simulated")}
    sims_two = {i for i in after_two if json.loads((plays / f"{i}.json").read_text()).get("simulated")}
    assert len(sims_two) == len(sims_one), (
        f"simulations grew from {len(sims_one)} to {len(sims_two)} on a re-run"
    )


def test_worst_ranks_by_the_advantage_it_claims_to_rank_by(plays, capsys):
    cli.main(["whatif", "--all-passes", "--plays", str(plays), "--no-write"])
    everything = capsys.readouterr().out

    cli.main(["whatif", "--worst", "1", "--plays", str(plays), "--no-write"])
    top = capsys.readouterr().out

    def advantages(text):
        return [float(line.split("advantage")[1].split()[0])
                for line in text.splitlines() if "advantage" in line]

    all_adv, top_adv = advantages(everything), advantages(top)
    assert all_adv and len(top_adv) == 1
    assert top_adv[0] == pytest.approx(max(all_adv))


def test_the_index_is_rewritten_to_include_the_simulation(plays):
    cli.main(["whatif", "--worst", "1", "--plays", str(plays)])
    index = json.loads((plays / "index.json").read_text())
    ids = {p["id"] for p in index["plays"]}
    assert len(ids) == len(real_ids(plays))
    ordered = [p["playId"] for p in index["plays"]]
    assert ordered == sorted(ordered), "the client needs plays sorted by playId"


def test_a_named_play_is_reviewed_and_others_are_not(plays, capsys):
    one = sorted(real_ids(plays))[0]
    cli.main(["whatif", one, "--plays", str(plays), "--no-write"])
    out = capsys.readouterr().out
    assert one.split("_")[1] in out


def test_a_missing_folder_is_reported_not_crashed(tmp_path, capsys):
    rc = cli.main(["whatif", "--worst", "1", "--plays", str(tmp_path / "nope")])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_naming_nothing_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["whatif"])


def test_a_simulated_play_is_never_itself_reviewed(plays, capsys):
    """Mutation-found gap: removing the skip survived, because reusing the id
    masked it. Two independent guards need two tests — a forecast of a forecast is
    wrong even when it does not multiply."""
    cli.main(["whatif", "--all-passes", "--plays", str(plays)])
    sims = [p for p in plays.glob("*.json")
            if p.name != "index.json" and json.loads(p.read_text()).get("simulated")]
    assert sims

    capsys.readouterr()
    cli.main(["whatif", "--all-passes", "--plays", str(plays), "--no-write"])
    out = capsys.readouterr().out

    # A simulation carries its source play's passResult, so --all-passes picks it
    # up; review_play then refuses it for having no events and the run prints a
    # skip. Neither the count of reviews nor the file list shows the difference —
    # the skip line does, and it is the only observable the guard has.
    # Matched on a digit boundary: play 2022090800_1230 contains the id
    # 2022090800_123 as a prefix, and a plain substring test calls that a hit.
    import re
    for sim in sims:
        assert not re.search(rf"\b{re.escape(sim.stem)}(?!\d)", out), (
            f"{sim.stem} is a forecast and was handed to the reviewer; it should "
            f"never have been loaded"
        )
    assert "skipped" not in out.lower()


def test_load_skips_a_simulation_even_when_it_looks_like_a_pass(plays):
    """The skip tested directly, not through its side effects.

    A simulation written today carries no passResult, so --all-passes filters it
    out before the skip is reached and removing the skip changes nothing
    observable. That is an accident of which keys _write_simulation copies: add
    passResult to that list and the guard becomes the only thing standing between
    a re-run and a forecast of a forecast.
    """
    cli.main(["whatif", "--worst", "1", "--plays", str(plays)])
    sim = next(p for p in plays.glob("*.json")
               if p.name != "index.json" and json.loads(p.read_text()).get("simulated"))

    doc = json.loads(sim.read_text())
    doc["passResult"] = "C"                      # the accident, made deliberate
    sim.write_text(json.dumps(doc))

    loaded = cli._load(plays, wanted=None, passes_only=True)
    stems = {p.stem for p, _ in loaded}
    assert sim.stem not in stems, "a forecast was loaded for review"

    # And it is skipped for being simulated, not for lacking a pass result.
    assert any(not t.meta.get("simulated") for _, t in loaded)
