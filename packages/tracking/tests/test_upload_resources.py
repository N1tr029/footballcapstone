"""Automatic uploads use available acceleration and release inactive models."""
import gc
import sys
import weakref
from functools import lru_cache
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("cuda,mps,expected",[(True,True,"cuda"),(False,True,"mps"),(False,False,"cpu")])
def test_detector_automatically_selects_available_hardware(monkeypatch,cuda,mps,expected):
    from gridiron_tracking.detect.yolo import YoloDetector
    monkeypatch.setitem(sys.modules,"ultralytics",SimpleNamespace(YOLO=lambda _:object()))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:mps))))
    assert YoloDetector().device==expected
    assert YoloDetector(device="cpu").device=="cpu"


def test_field_reader_weights_are_released_before_tracking(monkeypatch):
    from gridiron_tracking.registration import local
    class Weights:pass
    @lru_cache(maxsize=1)
    def load(name):return Weights()
    @lru_cache(maxsize=1)
    def tokenizer(name):return object()
    weights=load("local-model");ref=weakref.ref(weights);del weights
    tokenizer("local-model");calls=[]
    monkeypatch.setattr(local,"_load",load);monkeypatch.setattr(local,"_constraint_tokenizer",tokenizer)
    monkeypatch.setitem(sys.modules,"mlx.core",SimpleNamespace(clear_cache=lambda:calls.append("cleared")))
    gc.collect();assert ref() is not None
    local.release()
    assert ref() is None and not tokenizer.cache_info().currsize and calls==["cleared"]
