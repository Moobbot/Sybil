"""P5f — known answer: the reference CT case must give the scores recorded since P0.

The case (291 slices, one series) is patient data and is NOT distributed with the code.
Run inside the service image, with the checkpoints, against a local copy of the case:

    docker run --rm -v <case>:/case:ro -v <checkpoints>:/app/sybil_checkpoints \
        -e SYBIL_KNOWN_ANSWER_DIR=/case <sybil image> python -m pytest tests/test_known_answer.py

Skipped without the variable (CI) or without torch.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASE = os.environ.get("SYBIL_KNOWN_ANSWER_DIR")
pytestmark = pytest.mark.skipif(not CASE, reason="SYBIL_KNOWN_ANSWER_DIR not set")

# Cumulative risk, years 1-6. Identical in P0, P3, P4 and P4d (weights w.efe9a4e71ca6).
EXPECTED = [
    0.010900917281180979,
    0.02014553619432984,
    0.03623114203667281,
    0.04591641456665786,
    0.05377927794134665,
    0.08312354442405184,
]


def test_reference_case_scores(tmp_path):
    pytest.importorskip("torch")
    from call_model import LOAD_INFO, load_model, predict

    model = load_model()
    assert LOAD_INFO["fallback"] is False, "checkpoints did not load: a score mismatch would be meaningless"
    pred_dict, _, attention_info = predict(CASE, str(tmp_path), model)
    got = pred_dict["predictions"][0]
    assert len(got) == len(EXPECTED)
    for year, (g, e) in enumerate(zip(got, EXPECTED), start=1):
        assert math.isclose(g, e, rel_tol=1e-6), f"year {year}: {g} != {e}"
    assert attention_info["total_images"] == 291
