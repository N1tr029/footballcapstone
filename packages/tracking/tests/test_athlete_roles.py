"""Officials cannot fill missing player slots in a supposedly complete formation."""
from types import SimpleNamespace
import numpy as np

from gridiron_tracking.detect import Detection
from gridiron_tracking import athlete_roles


def body(x):return Detection(frame=0,x1=x,y1=0,x2=x+3,y2=10,score=.9)


def test_officials_are_removed_before_building_the_roster(monkeypatch,tmp_path):
    people=[body(1),body(5),body(9)]
    monkeypatch.setattr(athlete_roles,"classify",lambda *a,**k:{"roles":["player","official","player"]})
    result,evidence=athlete_roles.refine(np.zeros((20,20,3),np.uint8),people,None,0,cache_dir=tmp_path)
    assert result==[people[0],people[2]] and evidence["excludedOfficials"]==1
    assert evidence["verifiedPlayers"]==2


def test_tiling_refines_only_the_formation_and_reuses_known_uniform_roles(monkeypatch,tmp_path):
    from gridiron_tracking.detect import yolo
    initial=[body(1),body(5),body(9)];extra=[body(13),body(17)];seen=[]
    def classify(image,people,**kwargs):
        seen.append(people)
        return {"roles":["player","official","player"] if len(people)==3 else ["player","other"]}
    class Detector:
        def __init__(self,**kwargs):assert kwargs["tiles"]==2
        def detect(self,frame,image):return initial+extra
    monkeypatch.setattr(athlete_roles,"classify",classify);monkeypatch.setattr(yolo,"YoloDetector",Detector)
    spec=SimpleNamespace(weights="unused",conf=.25,imgsz=1280,device="cpu")
    registration=SimpleNamespace(spec=SimpleNamespace(width=53.3),to_field=lambda f,p:np.asarray(p))
    result,evidence=athlete_roles.refine(np.zeros((30,30,3),np.uint8),initial,registration,0,
        cache_dir=tmp_path,detector_spec=spec)
    assert result==[initial[0],initial[2],extra[0]]
    assert len(seen)==2 and seen[1]==extra
    assert evidence["excludedOfficials"]==1 and evidence["excludedOther"]==1


def test_tiled_detections_must_obey_the_same_sideline_cuts(monkeypatch,tmp_path):
    from gridiron_tracking.detect import yolo
    initial=[body(1)];bench=body(17);calls=[]
    def classify(image,people,**kwargs):
        calls.extend(people);return {"roles":["player"]*len(people)}
    class Detector:
        def __init__(self,**kwargs):pass
        def detect(self,frame,image):return initial+[bench]
    monkeypatch.setattr(athlete_roles,"classify",classify);monkeypatch.setattr(yolo,"YoloDetector",Detector)
    spec=SimpleNamespace(weights="unused",conf=.25,imgsz=1280,device="cpu")
    registration=SimpleNamespace(spec=SimpleNamespace(width=53.3),
        to_field=lambda f,p:np.array([[40.,25.],[40.,52.]]))
    result,_=athlete_roles.refine(np.zeros((30,30,3),np.uint8),initial,registration,0,
        cache_dir=tmp_path,detector_spec=spec,field_cuts=(5,49))
    assert result==initial and bench not in calls


def test_a_truncated_head_cannot_seed_another_ground_position(monkeypatch,tmp_path):
    from gridiron_tracking.detect import yolo
    initial=[body(1)];head=Detection(frame=0,x1=5,y1=20,x2=15,y2=30,score=.9);seen=[]
    def classify(image,people,**kwargs):
        seen.extend(people);return {"roles":["player"]*len(people)}
    class Detector:
        def __init__(self,**kwargs):pass
        def detect(self,frame,image):return initial+[head]
    monkeypatch.setattr(athlete_roles,"classify",classify);monkeypatch.setattr(yolo,"YoloDetector",Detector)
    spec=SimpleNamespace(weights="unused",conf=.25,imgsz=1280,device="cpu")
    registration=SimpleNamespace(spec=SimpleNamespace(width=53.3),
        to_field=lambda f,p:np.array([[40.,25.],[40.,25.]]))
    result,_=athlete_roles.refine(np.zeros((30,30,3),np.uint8),initial,registration,0,
        cache_dir=tmp_path,detector_spec=spec)
    assert result==initial and head not in seen
