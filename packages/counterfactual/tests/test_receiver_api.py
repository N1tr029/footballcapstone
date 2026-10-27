"""The receiver what-if's HTTP endpoints, on records shaped exactly as its exporter writes them."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from gridiron_counterfactual.receiver import api  # noqa: E402
from gridiron_counterfactual.receiver.export import PLAYS, RECORDS  # noqa: E402


def _receiver(name, nfl_id, p, epa):
    return {"player_id": str(nfl_id), "name": name, "route": "GO", "depth": 10.0, "separation": 3.0,
            "p_complete": p, "exp_epa": epa, "available": True}


def _record(play_id, flagged, gap):
    chosen = _receiver("Chosen", 10, 0.2, -0.5)
    alt = _receiver("Option", 11, 0.6, -0.5 + gap) if gap else None
    return {"id": f"2022090800_{play_id}", "play_id": f"2022090800-{play_id}", "game_id": 2022090800,
            "play_id_number": play_id, "kind": "target_choice", "t": 2.5,
            "player": {"player_id": "1", "name": "QB"}, "flagged": flagged, "chosen": chosen,
            "alternative": alt, "epa_gap": {"expected": gap, "low": gap / 2, "high": gap * 1.5},
            "win_prob_gap": {"expected": gap * 0.03, "low": gap * 0.01, "high": gap * 0.05},
            "confidence": 1.0, "options": [chosen] + ([alt] if alt else []),
            "what_happened": {"pass_result": "I", "epa": -0.9}, "assumptions": ["an assumption"]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    records = tmp_path / "receiver_whatif.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in
                               [_record(56, True, 1.2), _record(77, False, 0.1), _record(88, True, 2.0)])
                       + "not json\n")
    plays = tmp_path / "plays"
    plays.mkdir()
    (plays / "2022090800_56.json").write_text(json.dumps(
        {"schemaVersion": 1, "gameId": 2022090800, "playId": 56, "simulated": True, "simulatedFrom": 56,
         "simulationStartFrameId": 1, "frames": [{"frameId": 1}]}))
    monkeypatch.setattr(api, "store", api.ReceiverStore(records, plays))
    return TestClient(api.app)


def test_health_and_a_bad_line_does_not_empty_the_rest(client):
    assert client.get("/health").json() == {"status": "ok", "records": 3, "simulatedPlays": 1}


def test_list_puts_the_largest_flagged_gap_first(client):
    plays = client.get("/receiver-whatif").json()["plays"]
    assert [p["playId"] for p in plays] == ["2022090800_88", "2022090800_56", "2022090800_77"]
    assert plays[0] == {"playId": "2022090800_88", "flagged": True, "passer": "QB", "chosen": "Chosen",
                        "alternative": "Option", "gapPoints": 2.0, "gapWinProbability": 0.06,
                        "hasSimulated": False}
    flagged = client.get("/receiver-whatif", params={"flagged": True}).json()["plays"]
    assert {p["playId"] for p in flagged} == {"2022090800_56", "2022090800_88"}


def test_one_record_and_its_simulated_play(client):
    body = client.get("/plays/2022090800_56.json/receiver-whatif").json()
    assert body["alternative"]["name"] == "Option" and body["hasSimulated"] is True and body["assumptions"]
    sim = client.get("/plays/2022090800_56/receiver-whatif/simulated").json()
    assert sim["simulated"] is True and sim["simulatedFrom"] == 56


def test_missing_ones_say_how_to_get_them(client):
    assert "receiver export" in client.get("/plays/2022090800_99/receiver-whatif").json()["detail"]
    r = client.get("/plays/2022090800_77/receiver-whatif/simulated")
    assert r.status_code == 404 and "flagged" in r.json()["detail"]


def test_the_router_mounts_on_another_app_unchanged(client):
    from fastapi import FastAPI

    other = FastAPI()
    other.include_router(api.router)
    assert TestClient(other).get("/receiver-whatif").status_code == 200


def test_the_committed_export_is_servable():
    if not RECORDS.is_file():
        pytest.skip("no committed export")
    s = api.ReceiverStore()
    assert s.records
    for pid, path in s.plays.items():
        assert s.records[pid]["flagged"]
        doc = json.loads(path.read_text())
        assert doc["simulated"] is True and doc["simulationStartFrameId"] <= len(doc["frames"])
