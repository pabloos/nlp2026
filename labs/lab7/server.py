"""
Proxy Ollama para el laboratorio.

Arquitectura:
  - Cola FIFO asyncio: serializa las llamadas al modelo, evita saturar Ollama
  - Caché exacto: hash(model + system + prompt) → respuesta; los alumnos que
    ejecuten el mismo prompt obtienen la respuesta cacheada en <5 ms
  - Auth: cabecera X-API-Key compartida con el aula
  - /health: expone profundidad de cola, entradas en caché y estadísticas

Uso:
    pip install fastapi uvicorn httpx
    uvicorn server:app --host 0.0.0.0 --port 8000

Variables de entorno:
    OLLAMA_URL      URL local de Ollama           (default: http://localhost:11434)
    LAB_API_KEY     Clave compartida con alumnos  (default: lab7-2026)
    WORKER_COUNT    Workers concurrentes          (default: 1)
    MAX_QUEUE_SIZE  Máx. peticiones en espera     (default: 60)
"""
import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
API_KEY        = os.getenv("LAB_API_KEY",    "lab7-2026")
WORKER_COUNT   = int(os.getenv("WORKER_COUNT",   "1"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "60"))

app   = FastAPI(title="Lab7 Proxy")
_cache: dict[str, dict] = {}
_stats: dict[str, int]  = defaultdict(int)
_queue: asyncio.Queue   = None  # inicializado en startup


# ── Auth ──────────────────────────────────────────────────────────────────────

def _check_key(request: Request):
    if request.headers.get("X-API-Key", "") != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_key(body: dict) -> str:
    raw = "\x00".join([body.get("model", ""), body.get("system", ""), body.get("prompt", "")])
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Worker ────────────────────────────────────────────────────────────────────

async def _worker(worker_id: int):
    log.info(f"Worker {worker_id} listo")
    while True:
        ck, body, fut = await _queue.get()
        try:
            if ck in _cache:
                _stats["hits"] += 1
                log.info(f"W{worker_id} CACHE HIT  {ck[:8]}…")
                fut.set_result({**_cache[ck], "_cached": True})
            else:
                _stats["misses"] += 1
                t0 = time.time()
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(f"{OLLAMA_URL}/api/generate", json=body)
                r.raise_for_status()
                data = r.json()
                elapsed = round(time.time() - t0, 2)
                data["_elapsed"] = elapsed
                _cache[ck] = data
                log.info(f"W{worker_id} OLLAMA     {ck[:8]}…  {elapsed}s  cola={_queue.qsize()}")
                fut.set_result(data)
        except Exception as e:
            _stats["errors"] += 1
            log.error(f"W{worker_id} ERROR: {e}")
            if not fut.done():
                fut.set_exception(HTTPException(status_code=502, detail=str(e)))
        finally:
            _queue.task_done()


@app.on_event("startup")
async def startup():
    global _queue
    _queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    for i in range(WORKER_COUNT):
        asyncio.create_task(_worker(i))
    log.info(
        f"Proxy listo → Ollama: {OLLAMA_URL} | "
        f"workers: {WORKER_COUNT} | cola max: {MAX_QUEUE_SIZE}"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/generate", dependencies=[Depends(_check_key)])
async def generate(request: Request):
    body = await request.json()
    body["stream"] = False
    ck = _cache_key(body)

    # Fast path: hit en caché antes de encolar
    if ck in _cache:
        _stats["hits"] += 1
        return JSONResponse({**_cache[ck], "_cached": True})

    loop = asyncio.get_event_loop()
    fut  = loop.create_future()
    try:
        _queue.put_nowait((ck, body, fut))
        _stats["queued"] += 1
    except asyncio.QueueFull:
        _stats["rejected"] += 1
        raise HTTPException(
            status_code=503,
            detail=f"Cola llena ({_queue.qsize()} peticiones en espera). Reintenta en unos segundos."
        )

    return JSONResponse(await fut)


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "ollama_url":    OLLAMA_URL,
        "queue_depth":   _queue.qsize() if _queue else 0,
        "cache_entries": len(_cache),
        "stats":         dict(_stats),
    }


@app.delete("/cache", dependencies=[Depends(_check_key)])
async def clear_cache():
    n = len(_cache)
    _cache.clear()
    _stats.clear()
    log.info(f"Cache vaciado ({n} entradas)")
    return {"cleared": n}
