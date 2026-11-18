"""Word-only field calibration needs two rows, temporal evidence and paint."""
import numpy as np
import pytest

from gridiron_tracking.registration import word_calibration as words,vision
from gridiron_tracking.registration.homography import Homography


def labels():
    h=Homography(np.array([[15.,3.,-350.],[.2,-8.,530.],[0,.005,1.]]))
    return [{"number":n,"row":r,"pixel":h.to_image([[10+n,y]])[0].tolist()}
        for r,y in (("near",10.),("far",43.3333333333)) for n in (20,30,40)]


def accept_paint(monkeypatch):
    monkeypatch.setattr(vision,"sanity_problems",lambda *a,**k:[])
    monkeypatch.setattr(vision,"snap_to_paint",lambda image,h,spec:(h,2.5))
    monkeypatch.setattr(vision,"turf_overlap",lambda *a:.9)


def test_word_boxes_fit_without_long_line_intersections_and_fail_bad_paint(monkeypatch):
    accept_paint(monkeypatch);image=np.zeros((720,1280,3),np.uint8)
    result=words.fit(image,labels(),"nfl")
    assert result and result[0]==2.5
    assert result[1].residual_px<1
    assert words.fit(image,[p for p in labels() if p["row"]=="near"],"nfl") is None
    monkeypatch.setattr(vision,"snap_to_paint",lambda image,h,spec:(h,1.1))
    assert words.fit(image,labels(),"nfl") is None


def test_single_frame_cannot_confirm_its_own_number_labels(monkeypatch,tmp_path):
    from gridiron_tracking import jerseys
    from gridiron_tracking.registration import paint_auto
    accept_paint(monkeypatch)
    monkeypatch.setattr(jerseys,"native_binary",lambda _:"unused")
    monkeypatch.setattr(words,"numbers",lambda *a:labels())
    monkeypatch.setattr(paint_auto,"image_registration",lambda *a:np.eye(3))
    frame=np.zeros((720,1280,3),np.uint8)
    assert words.calibrate({0:frame},{},fps=30,cache_dir=tmp_path,level="nfl") is None
    result=words.calibrate({f:frame for f in range(13)},{},fps=30,cache_dir=tmp_path,level="nfl")
    assert result and result.frame==0 and result.paint==2.5
    assert {e["frame"] for e in result.evidence["numbers"]}=={0,12}
