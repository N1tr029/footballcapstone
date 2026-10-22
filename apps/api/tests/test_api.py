import json

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.store import DEFAULT_DECISIONS, DEFAULT_PLAYS_DIR, PlayStore


def _write_game(folder, game_id, play_ids):
    folder.mkdir(parents=True, exist_ok=True)
    plays = []
    for n, pid in enumerate(play_ids):
        plays.append({
            "id": f"{game_id}_{pid}", "gameId": game_id, "playId": pid, "quarter": 1,
            "gameClock": "15:00", "driveNumber": 1, "possessionTeam": "BUF", "down": 1,
            "yardsToGo": 10, "playType": "pass", "passResult": "C", "yardsGained": 6,
            "touchdown": False, "description": f"play {n}",
        })
        detail = {
            "schemaVersion": 1, "gameId": game_id, "playId": pid, "homeTeam": "LA",
            "visitorTeam": "BUF", "description": f"play {n}", "quarter": 1, "gameClock": "15:00",
            "driveNumber": 1, "down": 1, "yardsToGo": 10, "possessionTeam": "BUF",
            "defensiveTeam": "LA", "lineOfScrimmage": 35.0, "firstDownLine": 45.0,
            "passResult": "C", "frameRate": 10, "snapFrameId": 1,
            "players": [{"nflId": 1, "name": "QB", "jerseyNumber": 17, "club": "BUF",
                         "position": "QB", "side": "offense"}],
            "frames": [{"frameId": 1, "frameType": "SNAP", "event": "ball_snap",
                        "ball": {"x": 35.0, "y": 26.6, "s": 0.0},
                        "players": [{"nflId": 1, "x": 30.1, "y": 26.4, "o": 88.2, "dir": 91.0, "s": 0.4}]}],
        }
        (folder / f"{game_id}_{pid}.json").write_text(json.dumps(detail))
    index = {"schemaVersion": 1, "gameId": game_id, "homeTeam": "LA", "visitorTeam": "BUF", "plays": plays}
    (folder / "index.json").write_text(json.dumps(index))


REPORT = {
    "game_id": 2022090800, "play_id": 56, "decision": "consider_alternative",
    "status": "modeled_alternative_more_open",
    "probabilities": {"target_was_reasonable": 0.25, "consider_alternative": 0.75},
    "confidence": "moderate",
    "evidence": {"targeted_receiver_id": 1, "best_alternative_receiver_id": 2, "open_threshold_yards": 3.0},
    "caveat": "Retrospective heuristic.", "play_description": "play 0", "pass_result": "C",
    "yards_gained": 6, "down": 1, "yards_to_go": 10,
}


def _write_simulated(folder, game_id, play_id, simulated_from):
    """What `gridiron_counterfactual whatif` writes beside a real play."""
    real = json.loads((folder / f"{game_id}_{simulated_from}.json").read_text())
    real.update({"playId": play_id, "simulated": True, "simulatedFrom": simulated_from,
                 "description": "SIMULATED — model's forecast of this throw (pass). This did not happen."})
    real["frames"][0]["players"][0]["x"] = 40.0
    (folder / f"{game_id}_{play_id}.json").write_text(json.dumps(real))


@pytest.fixture
def client(tmp_path, monkeypatch):
    _write_game(tmp_path / "plays", 2022090800, [56, 77])
    _write_simulated(tmp_path / "plays", 2022090800, 57, simulated_from=56)
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(REPORT) + "\n")
    monkeypatch.setattr(main, "store", PlayStore(str(tmp_path / "plays"), str(decisions)))
    return TestClient(main.app)


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "schemaVersion": 1, "games": 1, "plays": 3, "reviews": 1, "simulated": 1}


def test_index_matches_webplaysource_paths(client):
    a = client.get("/plays").json()
    b = client.get("/plays/index.json").json()
    assert a == b
    assert [p["id"] for p in a["plays"]] == ["2022090800_56", "2022090800_77"]


def test_index_is_served_byte_for_byte(client, tmp_path):
    # A sparse entry, like the ones `whatif` writes. Re-serializing it would add
    # playType/yardsGained/touchdown as null, which Unity's JsonUtility can't read.
    path = tmp_path / "plays" / "index.json"
    index = json.loads(path.read_text())
    index["plays"].append({"id": "2022090800_57", "gameId": 2022090800, "playId": 57, "description": "SIMULATED"})
    path.write_text(json.dumps(index))
    main.store.reload()
    r = client.get("/plays")
    assert r.content == path.read_bytes()
    assert b"null" not in r.content


def test_play_served_with_and_without_json_suffix(client):
    a = client.get("/plays/2022090800_56")
    b = client.get("/plays/2022090800_56.json")
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()
    assert a.json()["frames"][0]["players"][0]["o"] == 88.2


def test_unknown_play_404(client):
    assert client.get("/plays/1_1").status_code == 404


def test_several_games_require_game_id(client, tmp_path, monkeypatch):
    _write_game(tmp_path / "plays" / "other", 2022091100, [5])
    main.store.reload()
    assert client.get("/plays").status_code == 400
    assert client.get("/plays?gameId=2022091100").json()["gameId"] == 2022091100
    assert len(client.get("/games").json()["games"]) == 2


def test_recommendation_uses_model_contract(client):
    rec = client.get("/plays/2022090800_56/recommendation").json()
    assert rec["call"]["type"] == "pass"
    assert rec["choice_probabilities"]["consider_alternative"] == 0.75
    assert rec["win_prob_delta"] is None and rec["assumptions"]
    assert client.get("/plays/2022090800_77/recommendation").status_code == 404
    assert client.get("/recommendations").json() == {"plays": ["2022090800_56"]}


def test_optimal_returns_the_simulated_play(client):
    r = client.get("/plays/2022090800_56/optimal")
    assert r.status_code == 200
    body = r.json()
    assert body["simulated"] is True and body["simulatedFrom"] == 56
    assert body["frames"][0]["players"][0]["x"] == 40.0
    assert client.get("/plays/2022090800_56.json/optimal").status_code == 200


def test_optimal_missing_explains_how_to_make_it(client):
    r = client.get("/plays/2022090800_77/optimal")
    assert r.status_code == 404
    assert "whatif 2022090800_77" in r.json()["detail"]
    assert client.get("/plays/1_1/optimal").status_code == 404


def test_compare_returns_both(client):
    body = client.get("/plays/2022090800_56/compare").json()
    assert body["hasOptimal"] is True
    assert body["current"]["frames"][0]["players"][0]["x"] == 30.1
    assert body["optimal"]["frames"][0]["players"][0]["x"] == 40.0
    assert body["recommendation"]["call"]["type"] == "pass"

    none = client.get("/plays/2022090800_77/compare").json()
    assert none["hasOptimal"] is False and none["optimal"] is None and none["recommendation"] is None


def test_list_optimal(client):
    assert client.get("/optimal").json() == {
        "plays": [{"playId": "2022090800_56", "optimalPlayId": "2022090800_57"}]
    }


def test_index_with_only_id_fields_still_loads(tmp_path):
    folder = tmp_path / "plays"
    folder.mkdir()
    (folder / "index.json").write_text(json.dumps({
        "schemaVersion": 1, "gameId": 1, "homeTeam": "A", "visitorTeam": "B",
        "plays": [{"id": "1_2", "gameId": 1, "playId": 2, "description": "SIMULATED"}],
    }))
    assert PlayStore(str(folder), str(tmp_path / "none.jsonl")).games[1].plays[0].id == "1_2"


def test_a_broken_index_does_not_stop_the_others(tmp_path):
    _write_game(tmp_path / "plays" / "good", 2022090800, [56])
    bad = tmp_path / "plays" / "bad"
    bad.mkdir()
    (bad / "index.json").write_text('{"schemaVersion": 1}')
    s = PlayStore(str(tmp_path / "plays"), str(tmp_path / "none.jsonl"))
    assert list(s.games) == [2022090800]


def test_cors_header_for_webgl(client):
    r = client.get("/plays", headers={"Origin": "http://localhost:5000"})
    assert r.headers.get("access-control-allow-origin") == "*"


@pytest.mark.skipif(not DEFAULT_PLAYS_DIR.is_dir(), reason="Hindsight export not in checkout")
def test_real_repo_data_loads():
    s = PlayStore()
    assert s.games and s.play_files
    index = next(iter(s.games.values()))
    for entry in index.plays:
        assert entry.id in s.play_files, f"index lists {entry.id} but no file exists"
    if DEFAULT_DECISIONS.is_file():
        assert s.reports
    for real, sim in s.simulated_of.items():
        assert real in s.play_files, f"{sim} was simulated from {real}, which is missing"
