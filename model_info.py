"""Dinh danh phien ban mo hinh (P4c) — file DUNG CHUNG, giong het o Sybil va
CVD-Risk-Estimator (nhu inference_gate.py).

Muc dich: moi ket qua chan doan phai biet DUNG bo trong so + ma suy luan nao da
sinh ra no. Dinh danh tinh tu NOI DUNG file da nap, khong tu ten/cau hinh, vi
ca hai service deu co duong doi mo hinh tham lang (Sybil: nhanh except khi nap
checkpoint; CVD: detector tim nap hong thi roi ve cach "simple").

    <key>@<code>+w.<digest12>[+<co>...]
    vd: sybil@src.1a2b3c4d+w.3f9a1c2b7d10
        cvd@iter700.src.5e6f7a8b+w.8be0c41a92f3
        cvd@iter700.src.5e6f7a8b+w.8be0c41a92f3+det.simple   (co THEO TUNG CA, do routes.py noi vao)

Khong doc duoc file => "w.unknown" — khong bao gio bia.
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
    """sha256 cua danh sach (ten file, sha256 noi dung) DA SAP XEP.

    Ten file tinh vao vi no mang y nghia (vd iter 700 vs 800). Thieu file nao
    hoac danh sach rong => None (khong biet thi noi khong biet).
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
    """Chi tra TEN file (khong duong dan tuyet doi). Ten trong so la hash/iter,
    khong chua PHI."""
    out = []
    for p in sorted(paths, key=os.path.basename):
        try:
            out.append({"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": file_sha256(p)})
        except OSError:
            out.append({"file": os.path.basename(p), "bytes": None, "sha256": None})
    return out


def source_digest(roots: Iterable[str]) -> Optional[str]:
    """sha256 cua MA SUY LUAN (.py) — bat duoc thay doi tien xu ly ma khong doi
    trong so. `roots` gom thu muc (duyet de quy) va/hoac file le."""
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
                # Chuan hoa CRLF -> LF: cung ma nguon build tren Windows (autocrlf)
                # va Linux phai ra CUNG dinh danh, khong bao "doi ma" gia.
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
    """Noi dung tra ve o GET /info. Tinh MOT lan luc nap model; moi file chi
    hash MOT lan (describe_weights), digest tinh lai tu ket qua do."""
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
