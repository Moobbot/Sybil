"""Run inference for ONE case at a time; later cases wait their turn.

The web server handles requests on several threads (Flask's dev server, FastAPI's
threadpool): two simultaneous requests are two parallel inferences, peak RAM
doubles and the machine runs out of memory (OOM). This lock forces inferences to
run one after another. Later requests are not rejected, they only wait.

`status()` reports the state to /health without waiting for the lock, so /health
still answers while a case is running.

Watchdog: if a case runs longer than INFERENCE_MAX_SECONDS it is considered HUNG
(CUDA hang, stalled I/O...). The lock would then be held forever and every later
case would queue for nothing. /health reports `stuck`, and the process exits so
that `restart: unless-stopped` in compose brings the service back up — Docker does
NOT restart a container just because it is "unhealthy".
"""

import functools
import os
import sys
import threading
import time

# Longest valid case measured: CVD on CPU, 748 s (P0). 30 minutes is generous.
# A reasoned default; override it with the environment variable on different hardware.
INFERENCE_MAX_SECONDS = int(os.getenv("INFERENCE_MAX_SECONDS", "1800"))

_run_lock = threading.Lock()
_info_lock = threading.Lock()
_info = {"busy": False, "started_at": None, "waiting": 0}


def serialized(fn):
    """Wrap an inference function so that its calls run one after another."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _info_lock:
            _info["waiting"] += 1
        _run_lock.acquire()
        try:
            with _info_lock:
                _info["waiting"] -= 1
                _info["busy"] = True
                _info["started_at"] = time.time()
            return fn(*args, **kwargs)
        finally:
            with _info_lock:
                _info["busy"] = False
                _info["started_at"] = None
            _run_lock.release()

    return wrapper


def status():
    """Current state; does not wait for the inference lock."""
    with _info_lock:
        started = _info["started_at"]
        running = round(time.time() - started, 1) if started else 0
        return {
            "busy": _info["busy"],
            "waiting": _info["waiting"],
            "running_seconds": running,
            "stuck": running > INFERENCE_MAX_SECONDS,
        }


def _watchdog(check_every=30):
    while True:
        time.sleep(check_every)
        st = status()
        if st["stuck"]:
            print(
                f"[watchdog] inference has run {st['running_seconds']}s > "
                f"{INFERENCE_MAX_SECONDS}s — considered HUNG, exiting so Docker restarts the service",
                file=sys.stderr,
                flush=True,
            )
            os._exit(1)


def start_watchdog():
    """Call ONCE when the service starts."""
    threading.Thread(target=_watchdog, name="inference-watchdog", daemon=True).start()
