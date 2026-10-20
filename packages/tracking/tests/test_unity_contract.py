"""What the exporter writes against what the Unity client actually declares.

Unity's ``JsonUtility`` silently ignores any JSON field the target class does not
declare. No error, no warning — the value is simply not there at runtime. That is
how ``simulated``, ``oSource`` and ``provenance`` were sent for a week and did
nothing, and why a model's forecast drew identically to measured tracking.

Nobody on the Python side can compile the C#. What can be checked, and is checked
here, is the contract between them: every field the exporter writes is either
declared in ``PlayData.cs`` or is on a short list of things known to be inert, with
a reason. A field that drifts off that list fails this test rather than quietly
doing nothing in the viewer.

The parsing is deliberately crude — a regex over field declarations — because the
alternative is a C# parser, and the failure mode of the regex is a false alarm
somebody reads, not a silent pass.
"""

import json
import re

import pytest

from tracking_support import HINDSIGHT_PLAYS, REPO

PLAY_DATA_CS = REPO / "packages/replay-engine/hindsight/Assets/Scripts/Data/PlayData.cs"

pytestmark = pytest.mark.skipif(
    not PLAY_DATA_CS.exists(),
    reason="packages/replay-engine/hindsight is not merged into this tree",
)

# Written on purpose and known not to reach the client. Each one needs a reason,
# because "it is in the list" must never become the explanation.
INERT = {
    "provenance": (
        "a nested object with arbitrary keys. JsonUtility supports neither "
        "dictionaries nor untyped nesting, so reading it would mean a DTO per "
        "block. simulated and oSource carry the same warning in a shape the client "
        "can actually read; this stays in the file for humans and for any other "
        "consumer."
    ),
}


def declared_fields() -> dict[str, set[str]]:
    """``class name -> public field names``, with C#'s ``@`` escapes stripped."""
    source = PLAY_DATA_CS.read_text()
    out: dict[str, set[str]] = {}
    for name, body in re.findall(r"public class (\w+)\s*\{(.*?)\n    \}", source, re.S):
        fields = re.findall(r"public\s+[\w\[\]<>?]+\s+@?(\w+)\s*;", body)
        out[name] = set(fields)
    return out


def an_exported_play() -> dict:
    """Whatever the exporter most recently wrote, preferring a vision clip.

    A clip exercises the fields a release play does not: synthetic ids, trackId,
    and an oSource that is not ``measured``.
    """
    files = sorted(HINDSIGHT_PLAYS.glob("999000001_*.json"))
    if not files:
        files = [p for p in HINDSIGHT_PLAYS.glob("*.json") if p.name != "index.json"]
    if not files:
        pytest.skip("no exported plays")
    return json.loads(files[0].read_text())


def test_the_dtos_parse_at_all():
    """If this fails the rest of the file is vacuously true, which is worse than red."""
    classes = declared_fields()
    for name in ("PlayData", "PlayerInfo", "FrameData", "PlayerState", "BallState"):
        assert name in classes, f"{name} not found in PlayData.cs"
        assert classes[name], f"{name} parsed with no fields"


def test_every_field_the_exporter_writes_is_declared_or_known_inert():
    classes = declared_fields()
    doc = an_exported_play()

    undeclared = set(doc) - classes["PlayData"] - set(INERT)
    assert not undeclared, (
        f"the exporter writes {sorted(undeclared)}, which PlayData.cs does not "
        f"declare. JsonUtility will drop them without a word. Declare them, or add "
        f"them to INERT with a reason."
    )


def test_the_two_honesty_fields_are_declared_and_sent():
    """The ones that stop a forecast drawing like a measurement."""
    classes = declared_fields()
    doc = an_exported_play()

    assert "simulated" in classes["PlayData"]
    assert "simulated" in doc
    assert isinstance(doc["simulated"], bool)

    assert "oSource" in classes["PlayerState"]
    sources = {p["oSource"] for f in doc["frames"] for p in f["players"]}
    assert sources, "no player sample carries an oSource"
    assert sources <= {"measured", "from_dir", "from_velocity", "assumed", "authored"}


def test_provenance_is_sent_and_is_not_readable_by_the_client():
    """Pins the thing the handoff note has to say.

    It is written — a person reading the file should see it — and it cannot be
    deserialised by JsonUtility. Asserting both stops someone being told to 'just
    read provenance' and losing an afternoon to it.
    """
    doc = an_exported_play()
    assert "provenance" in doc
    assert isinstance(doc["provenance"], dict)
    assert isinstance(doc["provenance"]["coordinates"], dict), (
        "provenance is nested; if it ever flattens to scalars this test should be "
        "replaced by a DTO and the handoff note updated"
    )
    assert "provenance" not in declared_fields()["PlayData"]


def test_every_player_frame_field_the_client_reads_is_present():
    classes = declared_fields()
    doc = an_exported_play()
    sample = doc["frames"][0]["players"][0]

    # Everything PlayerState declares must actually arrive, or the client gets a
    # silent zero — a player at (0, 0) facing downfield, drawn with total confidence.
    missing = classes["PlayerState"] - set(sample)
    assert not missing, f"PlayerState declares {sorted(missing)} and the file has none"


def test_frames_carry_the_fields_the_timeline_needs():
    classes = declared_fields()
    doc = an_exported_play()
    frame = doc["frames"][0]

    for field in ("frameId", "frameType", "players"):
        assert field in frame
    assert frame["frameType"] in ("BEFORE_SNAP", "SNAP", "AFTER_SNAP")
    # `event` is declared as `@event` in C#; the JSON key has no escape.
    assert "event" in classes["FrameData"]


def test_the_roster_carries_what_the_views_are_built_from():
    classes = declared_fields()
    doc = an_exported_play()
    player = doc["players"][0]

    missing = classes["PlayerInfo"] - set(player)
    assert not missing, f"PlayerInfo declares {sorted(missing)} and the file has none"
    assert player["side"] in ("offense", "defense", "?") or player["side"]


def test_ids_are_unique_because_the_client_matches_on_them():
    """Its own notes say players are matched by nflId, not array position."""
    doc = an_exported_play()
    ids = [p["nflId"] for p in doc["players"]]
    assert len(set(ids)) == len(ids), "two players share an nflId; they will merge"
    for frame in doc["frames"]:
        frame_ids = [p["nflId"] for p in frame["players"]]
        assert len(set(frame_ids)) == len(frame_ids)


# ------------------------------------------- the plays that are committed to the repo


def every_play():
    return [p for p in HINDSIGHT_PLAYS.glob("*.json") if p.name != "index.json"]


def test_the_index_and_the_folder_agree():
    """The client reads index.json and then fetches each play by its id. An entry
    with no file is a dead click; a file with no entry is invisible."""
    index = json.loads((HINDSIGHT_PLAYS / "index.json").read_text())
    listed = {p["id"] for p in index["plays"]}
    on_disk = {p.stem for p in every_play()}

    assert not (listed - on_disk), f"index points at missing files: {sorted(listed - on_disk)}"
    assert not (on_disk - listed), f"files the navigator cannot reach: {sorted(on_disk - listed)}"


def test_the_index_is_sorted_by_play_id():
    """Its own notes: plays must be grouped contiguously by quarter and drive,
    which holds when they are sorted by playId."""
    index = json.loads((HINDSIGHT_PLAYS / "index.json").read_text())
    ids = [p["playId"] for p in index["plays"]]
    assert ids == sorted(ids)


def test_every_committed_play_is_well_formed():
    for path in every_play():
        doc = json.loads(path.read_text())
        assert doc["schemaVersion"] == 1, path.name
        assert doc["frames"], path.name
        assert doc["players"], path.name
        ids = [p["nflId"] for p in doc["players"]]
        assert len(set(ids)) == len(ids), f"{path.name} has two players sharing an id"
        frame_ids = [f["frameId"] for f in doc["frames"]]
        assert frame_ids == list(range(1, len(frame_ids) + 1)), (
            f"{path.name} frameIds are not 1..N contiguous"
        )
        assert 1 <= doc["snapFrameId"] <= len(doc["frames"]), path.name


def test_every_simulation_points_at_a_play_that_exists():
    """A forecast with no play to argue with is a panel with nothing on one side."""
    on_disk = {p.stem for p in every_play()}
    sims = 0
    for path in every_play():
        doc = json.loads(path.read_text())
        if not doc.get("simulated"):
            continue
        sims += 1
        assert doc.get("simulatedFrom") is not None, f"{path.name} says nothing about its source"
        source = f"{doc['gameId']}_{doc['simulatedFrom']}"
        assert source in on_disk, f"{path.name} argues with {source}, which is gone"
        assert source != path.stem
    assert sims, "no simulations committed; this test is no longer meaningful"


def test_no_committed_play_claims_a_measurement_it_did_not_make():
    """A simulated play must carry the flag, and a measured one must not."""
    for path in every_play():
        doc = json.loads(path.read_text())
        sources = {p.get("oSource", "") for f in doc["frames"] for p in f["players"]}
        inferred = sources - {"measured", ""}
        if doc.get("simulated"):
            assert inferred, f"{path.name} is simulated but every angle claims measured"
        elif path.name.startswith("999000001_"):
            assert inferred, f"{path.name} came from video; nothing measured those angles"
        else:
            assert sources <= {"measured", ""}, (
                f"{path.name} is release tracking but carries {inferred}"
            )
