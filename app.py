from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

OUTPUT_DIR = Path("output")
STATUS_PATH = OUTPUT_DIR / "status.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_runner_thread: threading.Thread | None = None


def _write_status(payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_status() -> dict:
    if not STATUS_PATH.exists():
        return {"state": "idle", "updated_at": int(time.time())}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "unknown", "updated_at": int(time.time())}


def _is_running() -> bool:
    return _runner_thread is not None and _runner_thread.is_alive()


def _run_screener(mode: str) -> None:
    command = ["python", "main.py", "--mode", mode]
    _write_status(
        {
            "state": "running",
            "mode": mode,
            "pid": os.getpid(),
            "command": " ".join(command),
            "started_at": int(time.time()),
            "updated_at": int(time.time()),
        }
    )

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        _write_status(
            {
                "state": "completed" if result.returncode == 0 else "failed",
                "mode": mode,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
                "finished_at": int(time.time()),
                "updated_at": int(time.time()),
            }
        )
    except Exception as exc:
        _write_status(
            {
                "state": "failed",
                "mode": mode,
                "error": str(exc),
                "finished_at": int(time.time()),
                "updated_at": int(time.time()),
            }
        )


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/run")
def run(mode: str = "ALL") -> dict:
    global _runner_thread

    normalized_mode = mode.strip().upper()
    allowed = {"ALL", "SP500", "DOW30", "NASDAQ_ONLY", "OTHERLISTED_ONLY"}
    if normalized_mode not in allowed:
        raise HTTPException(status_code=400, detail={"error": "invalid mode", "allowed": sorted(allowed)})

    with _lock:
        if _is_running():
            return {"started": False, "reason": "running", "status": _read_status()}

        _runner_thread = threading.Thread(target=_run_screener, args=(normalized_mode,), daemon=True)
        _runner_thread.start()

    return {"started": True, "mode": normalized_mode, "status": _read_status()}


@app.get("/status")
def status() -> dict:
    status_payload = _read_status()
    status_payload["running"] = _is_running()
    return status_payload


@app.get("/results/bull")
def results_bull() -> FileResponse:
    bull_path = OUTPUT_DIR / "bull.csv"
    if not bull_path.exists():
        raise HTTPException(status_code=404, detail="bull.csv not found")
    return FileResponse(path=bull_path, media_type="text/csv", filename="bull.csv")


@app.get("/results/bear")
def results_bear() -> FileResponse:
    bear_path = OUTPUT_DIR / "bear.csv"
    if not bear_path.exists():
        raise HTTPException(status_code=404, detail="bear.csv not found")
    return FileResponse(path=bear_path, media_type="text/csv", filename="bear.csv")
