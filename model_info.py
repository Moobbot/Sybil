"""Model version identity (P4c) — a SHARED file, identical in Sybil and
CVD-Risk-Estimator (like inference_gate.py).

Purpose: every diagnosis result must know EXACTLY which weights + inference code
produced it. The identity is computed from the CONTENT of the loaded files, not from
names/configuration, because both services can silently switch model paths (Sybil:
the except branch when loading checkpoints; CVD: when the heart detector fails to
load it falls back to the "simple" method).

    <key>@<code>+w.<digest12>[+<flag>...]
    e.g. sybil@src.1a2b3c4d+w.3f9a1c2b7d10
         cvd@iter700.src.5e6f7a8b+w.8be0c41a92f3
         cvd@iter700.src.5e6f7a8b+w.8be0c41a92f3+det.simple   (PER-CASE flag, appended by routes.py)

A file that cannot be read => "w.unknown" — never made up.
"""
import hashlib
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional

CONTRACT_VERSION = 1
_CHUNK = 1 << 20


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def weights_digest(paths: Iterable[str]) -> Optional[str]:
    """sha256 of the SORTED list of (file name, content sha256).

    The file name is included because it carries meaning (e.g. iter 700 vs 800).
    Any missing file or an empty list => None (if we do not know, we say so).
    """
    paths = list(paths)
    if not paths:
        return None
    try:
        entries = sorted((os.path.basename(p), file_sha256(p)) for p in paths)
    except OSError:
        return None
    h = hashlib.sha256()
    for name, digest in entries:
        h.update(f"{name}\0{digest}\n".encode())
    return h.hexdigest()


def describe_weights(paths: Iterable[str]) -> List[dict]:
    """Return file NAMES only (no absolute paths). Weight file names are hashes/iterations
    and contain no PHI."""
    out = []
    for p in sorted(paths, key=os.path.basename):
        try:
            out.append({"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": file_sha256(p)})
        except OSError:
            out.append({"file": os.path.basename(p), "bytes": None, "sha256": None})
    return out


def source_digest(roots: Iterable[str]) -> Optional[str]:
    """sha256 of the INFERENCE CODE (.py) — catches preprocessing changes that leave
    the weights unchanged. `roots` holds directories (walked recursively) and/or single files."""
    files = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
        elif os.path.isdir(root):
            for d, _, names in os.walk(root):
                if "__pycache__" in d:
                    continue
                files.extend(os.path.join(d, n) for n in names if n.endswith(".py"))
        else:
            return None
    if not files:
        return None
    base = os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f in files])
    h = hashlib.sha256()
    try:
        for f in sorted(files, key=lambda p: os.path.relpath(os.path.abspath(p), base)):
            rel = os.path.relpath(os.path.abspath(f), base).replace(os.sep, "/")
            with open(f, "rb") as fh:
                # Normalise CRLF -> LF: the same source built on Windows (autocrlf)
                # and on Linux must give the SAME identity, not a false "code changed".
                content = fh.read().replace(b"\r\n", b"\n")
            h.update(rel.encode() + b"\0" + hashlib.sha256(content).hexdigest().encode() + b"\n")
    except OSError:
        return None
    return h.hexdigest()


def build_version(key: str, code: str, digest: Optional[str], flags: Iterable[str]) -> str:
    w = f"w.{digest[:12]}" if digest else "w.unknown"
    return "+".join([f"{key}@{code}", w, *flags])


def build_info(
    key: str,
    code: str,
    weight_paths: List[str],
    flags: List[str],
    device: Optional[str],
    fallback: bool = False,
) -> dict:
    """Body returned by GET /info. Computed ONCE when the model loads; each file is
    hashed ONCE (describe_weights) and the digest is derived from that result."""
    weights = describe_weights(weight_paths)
    digest = None
    if weights and all(w["sha256"] for w in weights):
        h = hashlib.sha256()
        for w in sorted(weights, key=lambda w: w["file"]):
            h.update(f"{w['file']}\0{w['sha256']}\n".encode())
        digest = h.hexdigest()
    return {
        "contract": CONTRACT_VERSION,
        "model": key,
        "loaded": True,
        "version": build_version(key, code, digest, flags),
        "code_version": code,
        "weights": weights,
        "flags": list(flags),
        "fallback": fallback,
        "device": device,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
