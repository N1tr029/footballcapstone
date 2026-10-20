# tests/fixtures — pinned copies of the Unity app's side of the contract

The Unity replay app lives in [`EllisJeremy/hindsight-frontend`](https://github.com/EllisJeremy/hindsight-frontend)
and is no longer part of this repository. Two things it owns are still the only honest
way to test the exporter, so they are pinned here:

| path | what it is | the test that needs it |
|---|---|---|
| `hindsight-plays/` | 122 real exported plays plus `index.json`, byte-for-byte as the app ships them | `test_hindsight.py`, `test_producers_agree.py`, `test_unity_contract.py`, `counterfactual/tests/test_review.py` |
| `hindsight-contract/PlayData.cs` | the client's DTOs, so `test_unity_contract.py` can still check that every field the exporter writes is one Unity declares | `test_unity_contract.py` |

Both are **read-only fixtures**. The CLIs that write plays (`tracking ... unity`,
`counterfactual whatif`) default to `data/unity-plays/`, which is ignored, so a run
cannot edit what is committed here.

`PlayData.cs` is a snapshot and can go stale: it was taken from hindsight-frontend at
`525c4676` (2026-10-19). If the app adds a field, re-copy it — the test's failure mode
is a false alarm somebody reads, not a silent pass, which is the point.
