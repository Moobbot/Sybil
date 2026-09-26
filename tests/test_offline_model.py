"""The model must build without the network: a hospital machine may have none.

SybilNet used to create its video encoder with `r3d_18(pretrained=True)`, which downloads torchvision's
Kinetics-400 weights (133 MB) into the container every time the container is created — then Sybil's
checkpoint replaced every one of those parameters (load_state_dict, strict). Without internet the model
did not load at all (measured: URLError on both load paths). The encoder is now built without
pretrained weights; the scores are unchanged (known-answer test, and a parity run with the old image).

Skipped without torch/torchvision (CI).
"""
import os
import sys
from argparse import Namespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _no_download(*args, **kwargs):
    raise AssertionError(f"SybilNet tried to download weights: {args[:1]}")


def test_sybilnet_builds_without_downloading(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torchvision.models._api as tv_api

    # Every way torch/torchvision fetch weights (torchvision keeps its own reference to the function).
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _no_download)
    monkeypatch.setattr(torch.hub, "download_url_to_file", _no_download)
    if hasattr(tv_api, "load_state_dict_from_url"):
        monkeypatch.setattr(tv_api, "load_state_dict_from_url", _no_download)

    from sybil.models.sybil import SybilNet

    net = SybilNet(Namespace(dropout=0.0, max_followup=6))
    # The encoder is there, with parameters to be filled by the checkpoint.
    assert any(name.startswith("image_encoder.") for name, _ in net.named_parameters())
