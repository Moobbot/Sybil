"""P4c — model_info.py: model version identity.

Run: cd Sybil && python -m pytest tests/test_model_info.py -q
(model_info.py is identical in CVD-Risk-Estimator — so is this test).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_info as mi  # noqa: E402


def _write(p, data: bytes):
    with open(p, "wb") as f:
        f.write(data)
    return str(p)


def test_digest_does_not_depend_on_order(tmp_path):
    a = _write(tmp_path / "a.ckpt", b"aaa")
    b = _write(tmp_path / "b.ckpt", b"bbb")
    assert mi.weights_digest([a, b]) == mi.weights_digest([b, a])


def test_changing_one_byte_changes_digest(tmp_path):
    a = _write(tmp_path / "a.ckpt", b"aaa")
    d1 = mi.weights_digest([a])
    _write(tmp_path / "a.ckpt", b"aab")
    assert mi.weights_digest([a]) != d1


def test_renaming_a_file_changes_digest(tmp_path):
    # the file name is part of the identity (e.g. iter 700 vs 800) — same content, different name => different digest
    a = _write(tmp_path / "x-00700.ptm", b"w")
    b = _write(tmp_path / "x-00800.ptm", b"w")
    assert mi.weights_digest([a]) != mi.weights_digest([b])


def test_missing_file_gives_none(tmp_path):
    a = _write(tmp_path / "a.ckpt", b"aaa")
    assert mi.weights_digest([a, str(tmp_path / "missing.ckpt")]) is None
    assert mi.weights_digest([]) is None


def test_describe_weights_returns_file_names_not_paths(tmp_path):
    a = _write(tmp_path / "a.ckpt", b"aaa")
    ws = mi.describe_weights([a])
    assert ws == [{"file": "a.ckpt", "bytes": 3, "sha256": mi.file_sha256(a)}]
    assert str(tmp_path) not in repr(ws)


def test_build_version_full_and_with_flags():
    v = mi.build_version("sybil", "src.1a2b3c4d", "0123456789abcdef" * 4, [])
    assert v == "sybil@src.1a2b3c4d+w.0123456789ab"
    v2 = mi.build_version("cvd", "tri2dnet-iter700", "f" * 64, ["det.none"])
    assert v2 == "cvd@tri2dnet-iter700+w.ffffffffffff+det.none"


def test_build_version_without_digest_is_unknown():
    assert mi.build_version("sybil", "src.x", None, ["fallback"]) == "sybil@src.x+w.unknown+fallback"


def test_source_digest_is_stable_and_catches_code_changes(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(pkg / "a.py", b"x = 1\n")
    _write(pkg / "b.py", b"y = 2\n")
    _write(pkg / "note.txt", b"not counted")
    d1 = mi.source_digest([str(pkg)])
    assert d1 == mi.source_digest([str(pkg)])
    _write(pkg / "note.txt", b"changing a non-.py file -> no change")
    assert mi.source_digest([str(pkg)]) == d1
    _write(pkg / "a.py", b"x = 2\n")
    assert mi.source_digest([str(pkg)]) != d1


def test_source_digest_accepts_single_files(tmp_path):
    f = _write(tmp_path / "call_model.py", b"def f(): pass\n")
    assert mi.source_digest([f]) is not None
    assert mi.source_digest([str(tmp_path / "missing.py")]) is None


def test_build_info_digest_matches_weights_digest_without_paths(tmp_path):
    a = _write(tmp_path / "a.ckpt", b"aaa")
    b = _write(tmp_path / "b.json", b"{}")
    info = mi.build_info("sybil", "src.x", [b, a], [], "cpu")
    assert info["version"] == mi.build_version("sybil", "src.x", mi.weights_digest([a, b]), [])
    assert info["contract"] == 1 and info["loaded"] is True
    assert [w["file"] for w in info["weights"]] == ["a.ckpt", "b.json"]
    assert str(tmp_path) not in repr(info)


def test_build_info_missing_file_gives_w_unknown(tmp_path):
    a = _write(tmp_path / "a.ckpt", b"aaa")
    info = mi.build_info("cvd", "iter700.src.x", [a, str(tmp_path / "missing.pt")], [], "cuda")
    assert "+w.unknown" in info["version"]


def test_source_digest_does_not_depend_on_line_endings(tmp_path):
    # Builds on Windows (autocrlf) and Linux must give the SAME src — no false "code changed".
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write(a / "m.py", b"x = 1\ny = 2\n")
    _write(b / "m.py", b"x = 1\r\ny = 2\r\n")
    assert mi.source_digest([str(a)]) == mi.source_digest([str(b)])
