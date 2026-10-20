"""The labelling harness: the store, the dataset it writes, and the split.

The split is the part worth testing. Frames four apart from one camera are nearly
the same picture, so a random train/val split puts near-duplicates in both halves
and the validation score measures memorisation rather than whether the detector
finds a football on footage it has not seen.
"""

import json

import pytest

from gridiron_tracking import labelling


def shots(clips=("a", "b"), per_clip=6, w=1280, h=720, tmp=None):
    out = []
    for clip in clips:
        for i in range(per_clip):
            frame = 100 + i * 4
            out.append(labelling.Shot(
                key=f"{clip}/{frame}", clip=clip, frame=frame,
                path=str((tmp / f"{clip}_{frame:05d}.jpg") if tmp else f"{clip}_{frame}.jpg"),
                width=w, height=h))
    return out


def test_a_label_without_a_ball_is_still_a_label():
    """Frames with no visible ball are the negatives that stop a fine-tune
    re-learning the down-and-distance graphic."""
    none = labelling.Label(key="a/100")
    some = labelling.Label(key="a/104", x=640.0, y=360.0)
    assert not none.visible
    assert some.visible


def test_the_store_round_trips(tmp_path):
    store = labelling.LabelSet()
    store.labels["a/100"] = labelling.Label(key="a/100", x=10.0, y=20.0, size=18)
    store.labels["a/104"] = labelling.Label(key="a/104")
    path = store.save(tmp_path / "labels.json")

    back = labelling.LabelSet.load(path)
    assert back.counts() == {"labelled": 2, "with_a_ball": 1, "without": 1}
    assert back.labels["a/100"].x == 10.0
    assert back.labels["a/100"].size == 18
    assert not back.labels["a/104"].visible


def test_a_missing_store_is_empty_rather_than_an_error(tmp_path):
    assert labelling.LabelSet.load(tmp_path / "nothing.json").labels == {}


def test_the_dataset_holds_out_whole_clips(tmp_path):
    """Not random frames: four frames apart from one camera is the same picture."""
    frames = tmp_path / "frames"
    frames.mkdir()
    all_shots = shots(tmp=frames)
    for s in all_shots:
        (frames / f"{s.clip}_{s.frame:05d}.jpg").write_bytes(b"not really a jpeg")

    store = labelling.LabelSet()
    for s in all_shots:
        store.labels[s.key] = labelling.Label(key=s.key, x=640.0, y=360.0)

    data = labelling.to_yolo(all_shots, store, out_dir=tmp_path / "ds",
                             val_fraction=0.5, log=lambda *a: None)

    train = sorted(p.stem for p in (tmp_path / "ds/images/train").glob("*.jpg"))
    val = sorted(p.stem for p in (tmp_path / "ds/images/val").glob("*.jpg"))
    assert train and val
    # No clip appears in both halves.
    assert {n.rsplit("_", 1)[0] for n in train}.isdisjoint(
        {n.rsplit("_", 1)[0] for n in val})
    assert data.exists()
    assert "football" in data.read_text()


def test_boxes_are_written_normalised_and_empties_are_kept(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    all_shots = shots(clips=("a",), per_clip=2, tmp=frames)
    for s in all_shots:
        (frames / f"{s.clip}_{s.frame:05d}.jpg").write_bytes(b"x")

    store = labelling.LabelSet()
    store.labels[all_shots[0].key] = labelling.Label(
        key=all_shots[0].key, x=640.0, y=360.0, size=20)
    store.labels[all_shots[1].key] = labelling.Label(key=all_shots[1].key)

    labelling.to_yolo(all_shots, store, out_dir=tmp_path / "ds",
                      val_fraction=0.0, log=lambda *a: None)

    texts = sorted((tmp_path / "ds/labels/train").glob("*.txt"))
    assert len(texts) == 2, "the frame with no ball must still produce a file"

    bodies = [t.read_text().strip() for t in texts]
    box = next(b for b in bodies if b)
    assert "" in bodies, "an empty label file is how YOLO is told 'no ball here'"

    cls, cx, cy, bw, bh = box.split()
    assert cls == "0"
    # The format writes six decimals, so compare to that and not to float exactness.
    assert float(cx) == pytest.approx(0.5, abs=1e-6)       # 640 of 1280
    assert float(cy) == pytest.approx(0.5, abs=1e-6)       # 360 of 720
    assert float(bw) == pytest.approx(20 / 1280, abs=1e-6)
    assert float(bh) == pytest.approx(20 / 720, abs=1e-6)


def test_exporting_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nothing labelled"):
        labelling.to_yolo(shots(), labelling.LabelSet(), out_dir=tmp_path / "ds",
                          log=lambda *a: None)


def test_labels_for_frames_that_are_gone_are_ignored(tmp_path):
    """Re-extracting with a different stride leaves orphans in the store."""
    frames = tmp_path / "frames"
    frames.mkdir()
    all_shots = shots(clips=("a",), per_clip=2, tmp=frames)
    for s in all_shots:
        (frames / f"{s.clip}_{s.frame:05d}.jpg").write_bytes(b"x")

    store = labelling.LabelSet()
    store.labels[all_shots[0].key] = labelling.Label(key=all_shots[0].key, x=1.0, y=2.0)
    store.labels["a/99999"] = labelling.Label(key="a/99999", x=3.0, y=4.0)

    labelling.to_yolo(all_shots, store, out_dir=tmp_path / "ds",
                      val_fraction=0.0, log=lambda *a: None)
    assert len(list((tmp_path / "ds/images/train").glob("*.jpg"))) == 1
