"""Сервер демо: WebSocket со стримом шагов генерации + раздача статического сайта.

Запуск с GPU:   python server.py                    (код авторов EAGLE; видно дерево на каждом шаге)
                python server.py --engine sglang    (SGLang, как в NeMo; дерево драфтера не видно)
Без GPU:        python server.py --mock
Модели задаются флагами --target и --drafter (оба можно повторять). По умолчанию пул из четырёх:
  [target] Llama-3.1-8B-Instruct, [target] DeepSeek-R1-Distill-Llama-8B (если хватает видеопамяти),
  [draft] authors (официальный драфтер EAGLE-3 под Llama) и [draft] ours (наш драфтер под Llama).
Браузер выбирает любые две модели пула.

Сайт отвечает сразу, модели грузятся в фоне: /api/info отдаёт state loading → ready (или failed).
Движок меняется со страницы (POST /api/engine): сервер ставит зависимости другого движка
(setup_engine.sh, SGLang ~5 ГБ), освобождает GPU и перезапускает себя тем же процессом с --engine другого.
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "site")
MAX_NEW_TOKENS = 1024
ENGINES = {"eagle": "код авторов EAGLE", "sglang": "SGLang"}
DEFAULT_TARGETS = ["llama=Llama-3.1-8B-Instruct=meta-llama/Llama-3.1-8B-Instruct",
                   "dsl=DeepSeek-R1-Distill-Llama-8B=deepseek-ai/DeepSeek-R1-Distill-Llama-8B=optional"]
DEFAULT_DRAFTERS = ["authors=authors=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B=llama",
                    "ours=ours=Funmagster/EAGLE3-Llama-3.1-8B-Instruct-L16-22-29-fcnorm=llama"]

parser = argparse.ArgumentParser()
parser.add_argument("--mock", action="store_true", help="без GPU, синтетические шаги")
parser.add_argument("--engine", choices=list(ENGINES), default="eagle",
                    help="eagle: код авторов с деревом; sglang: серверы SGLang, как NeMo serve/bench_sglang")
parser.add_argument("--eagle-python", default="", help="python окружения кода авторов (для смены движка со страницы)")
parser.add_argument("--sglang-python", default="", help="python окружения SGLang (для смены движка со страницы)")
parser.add_argument("--max-servers", type=int, default=0, help="SGLang: сколько моделей держать на GPU (0 = все сразу)")
parser.add_argument("--target", action="append", metavar="ID=ПОДПИСЬ=ПУТЬ[=optional]",
                    help="можно несколько раз; optional - грузить, только если хватает видеопамяти")
parser.add_argument("--drafter", action="append", metavar="ID=ПОДПИСЬ=ПУТЬ=ID_ТАРГЕТА", help="можно несколько раз")
parser.add_argument("--videos", default="", help="папка с фоновым видео и музыкой страницы демо (берётся первое по имени)")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

PYTHONS = {"eagle": args.eagle_python, "sglang": args.sglang_python}
PYTHONS[args.engine] = PYTHONS[args.engine] or sys.executable
engine = None
# phase: loading → ready | failed; switching - ставим зависимости другого движка и перезапускаемся
state = {"phase": "loading", "engine": args.engine, "message": "загружаю модели", "since": time.time()}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def build_engine(name):
    if args.mock:
        from mock import MockEngine
        if os.environ.get("MOCK_LOAD_S"):  # имитация загрузки моделей, чтобы проверить страницу
            time.sleep(float(os.environ["MOCK_LOAD_S"]))
        return MockEngine(sglang=name == "sglang")
    if name == "sglang":
        from engine_sglang import DrafterSpec, Engine, EngineConfig, TargetSpec
    else:
        from engine import DrafterSpec, Engine, EngineConfig, TargetSpec

    def _target(s):
        tid, label, path, *flag = s.split("=", 3)
        return TargetSpec(tid, label, path, optional=bool(flag and flag[0] == "optional"))
    targets = [_target(s) for s in args.target or DEFAULT_TARGETS]
    drafters = [DrafterSpec(*s.split("=", 3)) for s in args.drafter or DEFAULT_DRAFTERS]
    log(f"Движок {name}: загружаю таргеты {[t.path for t in targets]} и драфтеры {[d.path for d in drafters]}")
    cfg = EngineConfig(targets=targets, drafters=drafters)
    if name == "sglang":
        cfg.max_servers = args.max_servers
    return Engine(cfg)


def load():
    global engine
    try:
        engine = build_engine(args.engine)
        state.update(phase="ready", message="", since=time.time())
        log(f"Сервер готов: движок {args.engine}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        state.update(phase="failed", message=f"не загрузился движок {ENGINES[args.engine]}: {e!r}", since=time.time())


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gpu_lock = threading.Lock()
waiting = 0


def _status():
    return {"state": state["phase"], "engine_id": state["engine"], "engines": ENGINES,
            "switch": {k: state.get(k) for k in ("to", "message", "error")} | {"elapsed": int(time.time() - state["since"])}}


@app.get("/api/info")
def info():
    base = engine.info() if engine is not None and state["phase"] != "switching" else {
        "models": [], "gpu": "", "mock": args.mock, "engine": {"name": ENGINES[state["engine"]], "version": ""}}
    return {**base, **_status(), "busy": gpu_lock.locked(), "waiting": waiting}


@app.post("/api/engine")
def switch_engine(req: dict):
    to = req.get("engine")
    if to not in ENGINES:
        return {"ok": False, "error": f"нет движка {to}", **_status()}
    if state["phase"] in ("switching", "loading"):
        return {"ok": False, "error": "сервер уже загружается, подождите", **_status()}
    if to == state["engine"] and state["phase"] == "ready":
        return {"ok": True, **_status()}
    state.update(phase="switching", to=to, error=None, message="жду конца текущей генерации", since=time.time())
    threading.Thread(target=_switch, args=(to,), daemon=True).start()
    return {"ok": True, **_status()}


def _switch(to):
    """Ставит зависимости движка to и перезапускает сервер им же (тот же PID, тот же лог, тот же порт)."""
    gpu_lock.acquire()  # новые генерации не стартуют; текущая доигрывает
    log(f"Смена движка: {state['engine']} → {to}")
    try:
        if not args.mock:
            state["message"] = "ставлю зависимости " + ENGINES[to]
            # для установки нужен python кода авторов: из окружения SGLang ставить код авторов нельзя
            py = PYTHONS["eagle"] = PYTHONS["eagle"] or shutil.which("python3") or sys.executable
            proc = subprocess.Popen(["bash", os.path.join(HERE, "setup_engine.sh"), to, py],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tail = []
            for line in proc.stdout:
                sys.stderr.write(line)
                line = line.strip()
                if line:
                    tail = (tail + [line])[-20:]
                    if not line.startswith(("━", " ")) and len(line) < 160:
                        state["message"] = line
            if proc.wait():
                raise RuntimeError("установка не удалась: " + " / ".join(tail[-3:]))
            if to == "sglang":
                PYTHONS["sglang"] = PYTHONS["sglang"] or os.path.join(os.environ.get("DEMO_ROOT", "/content"), "sgl", "bin", "python")
        else:
            state["message"] = "имитация: ставлю зависимости " + ENGINES[to]
            time.sleep(float(os.environ.get("MOCK_SWITCH_S", "4")))
    except Exception as e:  # noqa: BLE001
        log(f"Смена движка не удалась: {e}")
        state.update(phase="ready", error=str(e), message="", since=time.time())
        gpu_lock.release()
        return
    state["message"] = "освобождаю GPU и перезапускаю сервер"
    if hasattr(engine, "shutdown"):
        try:
            engine.shutdown()
        except Exception as e:  # noqa: BLE001
            log(f"shutdown: {e!r}")
    argv, skip = [], False
    for a in sys.argv[1:]:  # те же флаги, кроме --engine
        if skip:
            skip = False; continue
        if a == "--engine":
            skip = True; continue
        if not a.startswith("--engine="):
            argv.append(a)
    for k, v in PYTHONS.items():  # оба окружения переживают перезапуск
        if v and f"--{k}-python" not in argv:
            argv += [f"--{k}-python", v]
    py = PYTHONS[to] or sys.executable
    sys.stdout.flush(); sys.stderr.flush()
    os.execv(py, [py, os.path.abspath(__file__), "--engine", to] + argv)


@app.websocket("/api/run")
async def run(ws: WebSocket):
    global waiting
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    try:
        req = json.loads(await ws.receive_text())
        if engine is None or state["phase"] != "ready":
            await ws.send_json({"type": "error", "message": "сервер меняет движок, подождите"}); return
        ready = {m["id"] for m in engine.info()["models"] if "error" not in m}
        lanes = req.get("lanes")
        if not isinstance(lanes, list) or not 1 <= len(lanes) <= 2 or len(set(lanes)) != len(lanes):
            await ws.send_json({"type": "error", "message": "нужны одна или две разные модели"}); return
        if missing := [m for m in lanes if m not in ready]:
            await ws.send_json({"type": "error", "message": f"нет модели {', '.join(map(str, missing))}"}); return
        prompt = str(req.get("prompt", ""))[:4000].strip() or "Привет!"
        max_new = max(1, min(int(req.get("max_new_tokens", 256)), MAX_NEW_TOKENS))
        temperature = max(0.0, min(float(req.get("temperature", 0.0)), 1.5))
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"type": "error", "message": f"плохой запрос: {e}"}); return

    def emit(ev):
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def work():
        global waiting
        waiting += 1
        if gpu_lock.locked():
            emit({"type": "queue", "waiting": waiting - 1})
        with gpu_lock:
            waiting -= 1
            try:
                if not stop.is_set():
                    engine.run(lanes, prompt, max_new, temperature, emit, cancelled=stop.is_set)
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "message": repr(e)})
            finally:
                emit(None)

    threading.Thread(target=work, daemon=True).start()

    async def watch_client():
        try:
            while True:
                msg = await ws.receive_text()
                if json.loads(msg).get("type") == "cancel":
                    stop.set()
        except Exception:  # noqa: BLE001
            stop.set()

    watcher = asyncio.create_task(watch_client())
    try:
        while (ev := await queue.get()) is not None:
            await ws.send_json(ev)
        await ws.close()
    except WebSocketDisconnect:
        stop.set()
    finally:
        watcher.cancel()


VIDEO_EXT = (".mp4", ".webm", ".m4v", ".mov", ".ogv")
AUDIO_EXT = (".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac")


@app.get("/api/videos")
def videos():
    """Фон страницы демо: первое видео по имени и первая музыка по имени (играет по кругу вместо звука видео)."""
    if not args.videos or not os.path.isdir(args.videos):
        return {"videos": [], "audio": []}
    names = []
    for dp, _, fn in os.walk(args.videos):
        for f in fn:
            if not f.startswith("."):
                names.append(os.path.relpath(os.path.join(dp, f), args.videos).replace(os.sep, "/"))
    return {"videos": sorted(n for n in names if n.lower().endswith(VIDEO_EXT)),
            "audio": sorted(n for n in names if n.lower().endswith(AUDIO_EXT))}


if args.videos and os.path.isdir(args.videos):
    app.mount("/videos", StaticFiles(directory=args.videos), name="videos")
if os.path.isdir(SITE):
    app.mount("/", StaticFiles(directory=SITE, html=True), name="site")

if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
