"""Движок демо на SGLang: тот же запуск, что NeMo AutoModel делает для замеров (serve/bench_sglang),
и те же флаги сервера, что big-training/code/final_eval.py.

Каждая модель пула - свой процесс `sglang.launch_server` на своём порту:
  <id таргета>  - сервер таргета без драфтера (таргетов может быть несколько, например Llama-3.1-8B-Instruct и DSL);
  <id драфтера> - сервер своего таргета с --speculative-algorithm EAGLE3 и деревом статьи
                  (8 шагов = глубина 7, top-k 10, 60 узлов).

SGLang не отдаёт дерево драфтера, поэтому шаги приходят без дерева: только сколько токенов принято за проход
таргета (по счётчику spec_verify_ct в потоке ответа).

Гонка: две модели нельзя гнать на GPU одновременно, иначе они делят GPU и замер нечестный. Поэтому дорожки
идут по очереди, каждая со своим временем; браузер проигрывает обе по часам GPU и начинает показ,
когда пошла вторая дорожка (первая к этому моменту уже записана).

Версия SGLang - 0.5.9, как в замерах команды. Наш драфтер везёт в своём репозитории пакет sglang_eagle31: он добавляет
fc_norm (без него SGLang молча выбрасывает эти веса) и чинит баг 0.5.9, из-за которого CUDA-графы снимают слои таргета
по умолчанию 2/16/29 вместо слоёв драфтера. Если пакет есть в каталоге драфтера, движок подключает его ко всем
серверам через SGLANG_EXTERNAL_MODEL_PACKAGE (на драфтеры без fc_norm и таргет он не влияет).

Сколько серверов держать на GPU сразу, считается по памяти: 8B в fp16 ~16 ГБ на сервер. На 80 ГБ помещаются
все четыре, на 40 ГБ только один, и модели перезагружаются при смене пары (минута-две на каждую).
"""
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

TREE = {"num_steps": 8, "topk": 10, "draft_tokens": 60}   # дерево статьи; num_steps = глубина + 1
GB_PER_SERVER = 20                                        # веса 16 + драфтер + KV на 2048 токенов + CUDA-графы
BASE_PORT = 30000
PLUGIN = "sglang_eagle31"   # пакет из репозитория нашего драфтера: fc_norm + починка слоёв в CUDA-графах


@dataclass
class TargetSpec:
    id: str
    label: str
    path: str  # репозиторий HF или каталог таргет-модели
    optional: bool = False  # грузится последним и только если хватает видеопамяти, иначе пропускается


@dataclass
class DrafterSpec:
    id: str
    label: str
    path: str
    target: str = ""  # id таргета, под который обучен драфтер (пусто = первый таргет)


@dataclass
class EngineConfig:
    targets: list = field(default_factory=list)
    drafters: list = field(default_factory=list)
    max_length: int = 2048
    max_servers: int = 0          # 0 = все модели сразу; меньше - модели меняются на GPU по очереди
    workdir: str = "/tmp/eagle3-sglang"


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, mem = [x.strip() for x in out.split(",")]
        return name, int(mem) / 1024
    except Exception:  # noqa: BLE001
        return "cpu", 0.0


def _draft_meta(dc, n_layers):
    """Слои таргета в счёте авторов (i = вход слоя i) и переключатели EAGLE-3.1 из config.json драфтера."""
    if dc.get("eagle_capture_layers") is not None:
        layers = [int(x) for x in dc["eagle_capture_layers"]]
    else:
        ids = (dc.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
        layers = [int(x) + 1 for x in ids] if ids else [2, n_layers // 2, n_layers - 3]
    return {"layers": layers, "fc_norm": bool(dc.get("fc_norm") or dc.get("use_aux_norm")),
            "norm_output": bool(dc.get("norm_output"))}


def _sglang_view(src, dst, n_layers):
    """Каталог драфтера в том виде, какой нужен SGLang (как orchestrate.sglang_draft в репозитории команды).

    Веса - ссылки на скачанные файлы. В config.json: архитектура LlamaForCausalLMEagle3 (у авторов там
    LlamaForCausalLM, и SGLang падает на словаре драфтера 32K) и слои таргета в счёте SGLang (выход слоя = i - 1),
    если они заданы по-нашему через eagle_capture_layers.
    """
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isfile(s) and f != "config.json" and not os.path.exists(d):
            os.symlink(os.path.realpath(s), d)
    with open(os.path.join(src, "config.json")) as f:
        dc = json.load(f)
    dc["architectures"] = ["LlamaForCausalLMEagle3"]
    if dc.get("eagle_capture_layers") is not None:
        ec = dict(dc.get("eagle_config") or {})
        ec["eagle_aux_hidden_state_layer_ids"] = [int(i) - 1 for i in dc["eagle_capture_layers"]]
        ec["use_aux_hidden_state"] = True
        dc["eagle_config"] = ec
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(dc, f, indent=1)
    return dc


class _Server:
    def __init__(self, model_id, cmd, port, log, env=None):
        self.id, self.cmd, self.port, self.log_path, self.env = model_id, cmd, port, log, env
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self.pid_file = log + ".pid"

    def start(self, timeout=1200):
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=self.env)
        with open(self.pid_file, "w") as f:  # чтобы следующий запуск демо добил серверы, если этот упадёт
            f.write(str(self.proc.pid))
        until = time.time() + timeout
        while time.time() < until:
            if self.proc.poll() is not None:
                raise RuntimeError(f"сервер SGLang {self.id} упал, лог: {self.log_path}\n" + _tail(self.log_path))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(3)
        raise RuntimeError(f"сервер SGLang {self.id} не поднялся, лог: {self.log_path}")

    def stop(self):
        """Остановить сервер со всеми его процессами (планировщик SGLang держит GPU-память в дочернем процессе)."""
        if self.proc:
            pg = self.proc.pid
            try:
                os.killpg(pg, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                pass
            for _ in range(30):  # ждём, пока выйдут и дочерние процессы, затем добиваем
                if not _group_alive(pg):
                    break
                time.sleep(1)
            try:
                os.killpg(pg, signal.SIGKILL)
            except OSError:
                pass
        self.proc = None
        _forget(self.pid_file)

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def _group_alive(pg):
    try:
        os.killpg(pg, 0)
        return True
    except OSError:
        return False


def _forget(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _kill_stale(workdir):
    """Серверы SGLang от прошлого запуска демо (ячейку перезапустили, процесс упал) держат GPU: добиваем их."""
    for f in os.listdir(workdir):
        if f.endswith(".pid"):
            try:
                pg = int(open(os.path.join(workdir, f)).read().strip())
                os.killpg(pg, signal.SIGKILL)
                print(f"[sglang] остановил старый сервер {f[:-4]} (pid {pg})", file=sys.stderr)
            except (OSError, ValueError):
                pass
            _forget(os.path.join(workdir, f))


def _gpu_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        used, total = [float(x) / 1024 for x in out.split(",")]
        return used, total
    except Exception:  # noqa: BLE001
        return None


def _tail(path, n=40):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        os.makedirs(cfg.workdir, exist_ok=True)
        _kill_stale(cfg.workdir)
        atexit.register(self.shutdown)
        self.targets, bad_targets = {}, {}
        for t in cfg.targets:
            try:  # закрытая Llama без HF_TOKEN не скачается: тогда выпадают она и её драфтеры, остальное работает
                tdir = t.path if os.path.isdir(t.path) else \
                    snapshot_download(t.path, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
                tok = AutoTokenizer.from_pretrained(tdir)
                with open(os.path.join(tdir, "config.json")) as f:
                    n_layers = json.load(f)["num_hidden_layers"]
                self.targets[t.id] = {"spec": t, "dir": tdir, "tokenizer": tok, "stop_ids": self._stop_ids(tok),
                                      "n_layers": n_layers}
            except Exception as e:  # noqa: BLE001
                print(f"[sglang] таргет {t.id} ({t.path}) не скачался: {e!r}", file=sys.stderr)
                bad_targets[t.id] = f"таргет {t.path} не скачался (нужен HF_TOKEN с доступом?): {e!r}"
        paths = {t.id: t.path for t in cfg.targets}
        self.gpu, mem = _gpu()
        n = len(cfg.targets) + len(cfg.drafters)
        fits = int((mem - 2) // GB_PER_SERVER) if mem else n
        if not cfg.max_servers and fits < n - sum(1 for t in cfg.targets if t.optional):
            # каждый сервер SGLang держит свою копию таргета, поэтому все модели сразу нужны ~20 ГБ на модель
            raise RuntimeError(f"все {n} модели сразу на {self.gpu} ({mem:.0f} ГБ) не помещаются: SGLang нужно "
                               f"~{GB_PER_SERVER} ГБ на модель, то есть {n * GB_PER_SERVER + 2} ГБ. Возьмите H100 или A100 80 ГБ, "
                               "или движок eagle (одна копия таргета на все драфтеры, помещается в 40 ГБ), "
                               "или --max-servers 1 (модели будут перезагружаться)")
        opt_targets = {t.id for t in cfg.targets if t.optional}
        n_opt = len(opt_targets) + sum(1 for d in cfg.drafters if (d.target or cfg.targets[0].id) in opt_targets)
        self.capacity = cfg.max_servers or min(n, max(fits, n - n_opt))
        # --mem-fraction-static считается для каждого сервера отдельно при старте (см. _frac)
        self.need_gb = GB_PER_SERVER
        self.version = _sglang_version()

        self.models, self.servers, self.failed, self.lru, self.target_of = [], {}, {}, [], {}
        self.plugin = {}
        common = [sys.executable, "-m", "sglang.launch_server", "--dtype", "float16",
                  "--host", "127.0.0.1", "--mem-fraction-static", "0.8", "--disable-radix-cache",
                  "--attention-backend", "triton", "--sampling-backend", "pytorch", "--random-seed", "0",
                  "--max-running-requests", "1", "--cuda-graph-max-bs", "1",
                  "--context-length", str(cfg.max_length)]
        first = cfg.targets[0].id if cfg.targets else ""
        entries = [(t.id, t.label, None, t.id) for t in cfg.targets] + \
                  [(d.id, d.label, d.path, d.target or first) for d in cfg.drafters]
        opt = {t.id for t in cfg.targets if t.optional}
        entries.sort(key=lambda e: e[3] in opt)   # необязательные модели поднимаются последними
        self.optional = {e[0] for e in entries if e[3] in opt}
        for i, (mid, label, path, tid) in enumerate(entries):
            port = BASE_PORT + i
            meta = {"id": mid, "label": label, "kind": "target" if path is None else "drafter", "target": tid,
                    "path": path or paths.get(tid, "")}
            try:
                if tid in bad_targets:
                    raise RuntimeError(bad_targets[tid])
                if tid not in self.targets:
                    raise ValueError(f"нет таргета {tid}")
                t = self.targets[tid]
                cmd = common + ["--model-path", t["dir"], "--port", str(port)]
                if path is not None:
                    src = path if os.path.isdir(path) else snapshot_download(path)
                    if os.path.isdir(os.path.join(src, PLUGIN)) and PLUGIN not in self.plugin:
                        self.plugin[PLUGIN] = os.path.abspath(src)
                    dc = _sglang_view(src, os.path.join(cfg.workdir, "drafts", mid), t["n_layers"])
                    meta.update(_draft_meta(dc, t["n_layers"]))
                    if meta["norm_output"] and _ver(self.version) < (0, 5, 12):
                        raise ValueError(f"norm_output (EAGLE-3.1) нужен SGLang >= 0.5.12, стоит {self.version}")
                    cmd += ["--speculative-algorithm", "EAGLE3",
                            "--speculative-draft-model-path", os.path.join(cfg.workdir, "drafts", mid),
                            "--speculative-num-steps", str(TREE["num_steps"]),
                            "--speculative-eagle-topk", str(TREE["topk"]),
                            "--speculative-num-draft-tokens", str(TREE["draft_tokens"])]
                self.servers[mid] = _Server(mid, cmd, port, os.path.join(cfg.workdir, f"server_{mid}.log"))
                self.target_of[mid] = tid
            except Exception as e:  # noqa: BLE001  битая ссылка на драфтер не роняет демо
                print(f"[sglang] {mid} ({path or tid}) не подготовлен: {e!r}", file=sys.stderr)
                meta["error"] = repr(e)
                self.failed[mid] = repr(e)
            self.models.append(meta)
        env = self._server_env()
        for srv in self.servers.values():
            srv.env = env

        # сразу поднимаем столько серверов, сколько влезает, и прогреваем каждый
        for mid in self.pool()[self.capacity:]:
            if mid in self.optional:   # по расчёту не влезает: не грузим и не вытесняем ради неё другие
                self._mark_failed(mid, RuntimeError(f"не загружена: не хватает видеопамяти, на {self.gpu} помещается "
                                                    f"{self.capacity} модели по ~{GB_PER_SERVER} ГБ"))
        for mid in self.pool()[: self.capacity]:
            if mid in self.optional and not self._room():
                self._mark_failed(mid, RuntimeError(self._no_room_msg()))
                continue
            try:
                self._ensure([mid], lambda ev: None)
                for _ in self._gen(mid, self.encode(mid, "Hi"), 16, 0.0, lambda: False, lambda ev: None):
                    pass
            except Exception as e:  # noqa: BLE001
                print(f"[sglang] {mid} не запустился: {e}", file=sys.stderr)
                self._mark_failed(mid, e)

    def _server_env(self):
        env = dict(os.environ)
        # SGLang собирает часть ядер ninja, а pip ставит его рядом с python окружения
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"   # в конфиге драфтера 2048, у таргета больше
        if PLUGIN in self.plugin:
            env["PYTHONPATH"] = self.plugin[PLUGIN] + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = PLUGIN
            print(f"[sglang] подключаю {PLUGIN} из {self.plugin[PLUGIN]}", file=sys.stderr)
        return env

    def _mark_failed(self, mid, e):
        self.failed[mid] = repr(e)
        for m in self.models:
            if m["id"] == mid:
                m["error"] = repr(e)
        srv = self.servers.pop(mid, None)
        if srv:
            srv.stop()
        if mid in self.lru:
            self.lru.remove(mid)

    def pool(self):
        return [m["id"] for m in self.models if "error" not in m]

    def info(self):
        return {"target": self.cfg.targets[0].path, "targets": [t.path for t in self.cfg.targets], "gpu": self.gpu, "models": self.models, "mock": False,
                "engine": {"name": "SGLang", "version": self.version, "servers_on_gpu": self.capacity, "tree": False},
                "tree": {"total_token": TREE["draft_tokens"], "depth": TREE["num_steps"] - 1, "top_k": TREE["topk"]}}

    # ---------- токенизация ----------
    @staticmethod
    def _stop_ids(tok):
        ids = {tok.eos_token_id}
        for t in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>"):
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != tok.unk_token_id:
                ids.add(i)
        return sorted(i for i in ids if i is not None)

    def encode(self, mid, prompt):
        tok = self.targets[self.target_of[mid]]["tokenizer"]
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False).input_ids

    # ---------- серверы на GPU ----------
    def _ensure(self, lanes, status):
        """Поднять серверы нужных моделей, выгрузив давно не нужные, если GPU не вмещает все."""
        for mid in lanes:
            srv = self.servers[mid]
            if srv.alive:
                self.lru.remove(mid); self.lru.append(mid)
                continue
            while len(self.lru) >= self.capacity:
                victim = next((m for m in self.lru if m not in lanes), self.lru[0])
                self.lru.remove(victim)
                self.servers[victim].stop()
            status({"type": "loading", "lane": mid, "message": "загружаю модель на GPU"})
            self._wait_free()
            self._frac(srv, self.capacity - len(self.lru))
            srv.start()
            self.lru.append(mid)

    def _frac(self, srv, remaining):
        """SGLang берёт под веса и KV-кэш долю --mem-fraction-static от памяти, свободной в момент его старта,
        а не от всей памяти GPU. Поэтому одна доля на все серверы не годится: каждый следующий видит меньше
        свободной памяти, и третьему не хватает даже на веса (Not enough memory). Делим свободную память
        поровну между серверами, которые ещё предстоит поднять, и пересчитываем долю под текущий остаток."""
        g = _gpu_used()
        if not g:
            return
        used, total = g
        free = total - used
        budget = (free - 1.0) / max(1, remaining)       # ГБ на этот сервер
        static = budget - 3.0                            # веса + KV; ~3 ГБ остаются на CUDA-графы и активации
        frac = max(0.05, min(0.95, static / free))
        i = srv.cmd.index("--mem-fraction-static")
        srv.cmd[i + 1] = f"{frac:.3f}"
        print(f"[sglang] {srv.id}: свободно {free:.1f} ГБ, серверу {budget:.1f} ГБ, --mem-fraction-static {frac:.3f}",
              file=sys.stderr)

    def _room(self):
        """Хватит ли видеопамяти ещё на один сервер (необязательные модели без этого не грузятся)."""
        g = _gpu_used()
        return g is None or g[1] - g[0] >= GB_PER_SERVER + 1

    def _no_room_msg(self):
        g = _gpu_used()
        free = f"свободно {g[1] - g[0]:.0f} ГБ" if g else "мало свободной памяти"
        return f"не загружена: не хватает видеопамяти ({free}, нужно ~{GB_PER_SERVER} ГБ)"

    def _wait_free(self, timeout=120):
        """Ждём, пока выгруженный сервер действительно отдаст память: иначе новый падает с CUDA out of memory."""
        until = time.time() + timeout
        while (g := _gpu_used()) is not None:
            used, total = g
            if total - used >= self.need_gb:
                return
            if time.time() > until:
                raise RuntimeError(f"на GPU занято {used:.1f} из {total:.1f} ГБ, серверу нужно {self.need_gb:.0f} ГБ. "
                                   "Память держит другой процесс: перезапустите среду Colab (Runtime → Restart session)")
            time.sleep(2)

    def shutdown(self):
        for srv in self.servers.values():
            if srv.proc:
                srv.stop()

    # ---------- генерация ----------
    def run(self, lanes, prompt, max_new_tokens, temperature, emit, cancelled=lambda: False):
        lanes = [lanes] if isinstance(lanes, str) else list(lanes)
        ids = {lane: self.encode(lane, prompt) for lane in lanes}
        for lane in lanes:
            emit({"type": "start", "lane": lane, "prompt_tokens": len(ids[lane])})

        def one(lane):
            if cancelled():
                emit({"type": "done", "lane": lane, "tokens": 0, "steps": 0, "time_s": 0.0, "tok_s": 0.0,
                      "tau": 0.0, "stop": "cancel"})
                return
            try:
                for ev in self._gen(lane, ids[lane], max_new_tokens, temperature, cancelled, emit):
                    ev["lane"] = lane
                    emit(ev)
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "lane": lane, "message": repr(e)})

        if self.capacity < len(lanes):
            # GPU держит одну модель (--max-servers 1): дорожки по очереди, перед каждой поднимаем её сервер
            for lane in sorted(lanes, key=lambda m: m not in self.targets):
                try:
                    self._ensure([lane], emit)
                except Exception as e:  # noqa: BLE001
                    emit({"type": "error", "lane": lane, "message": repr(e)}); continue
                one(lane)
            return
        # обе модели на GPU: настоящая гонка, запросы к двум серверам идут одновременно и делят GPU
        try:
            self._ensure(lanes, emit)
        except Exception as e:  # noqa: BLE001
            for lane in lanes:
                emit({"type": "error", "lane": lane, "message": repr(e)})
            return
        threads = [threading.Thread(target=one, args=(lane,), daemon=True) for lane in lanes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _gen(self, lane, ids, max_new_tokens, temperature, cancelled, emit):
        srv = self.servers[lane]
        params = {"temperature": temperature, "max_new_tokens": max_new_tokens, "stop_token_ids": self.targets[self.target_of[lane]]["stop_ids"],
                  "top_p": 1.0, "top_k": -1, "min_p": 0.0, "sampling_seed": 0}
        body = json.dumps({"input_ids": ids, "sampling_params": params, "stream": True}).encode()
        req = urllib.request.Request(srv.url + "/generate", data=body, headers={"Content-Type": "application/json"})
        text, ct, vct, steps, total, pend_dt, pend_n, pend_text = "", 0, 0, 0, 0.0, 0.0, 0, ""
        reason = "length"
        t0 = last = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                out = json.loads(data)
                now = time.perf_counter()
                meta = out.get("meta_info", {})
                new_ct = int(meta.get("completion_tokens", ct))
                new_vct = int(meta.get("spec_verify_ct") or 0)
                full = out.get("text", "")
                delta = full[len(text):] if full.startswith(text) else full   # SGLang шлёт весь текст целиком
                text = text + delta
                pend_dt += now - last; last = now
                pend_n += new_ct - ct; pend_text += delta
                rounds = new_vct - vct if lane not in self.targets else new_ct - ct
                ct, vct = new_ct, new_vct
                fr = meta.get("finish_reason")
                if fr:
                    reason = "eos" if (fr.get("type") if isinstance(fr, dict) else fr) == "stop" else "length"
                # первый кусок - prefill (у драфтера без проверки дерева): прибавляем его к следующему шагу
                if pend_n and rounds > 0:
                    steps += rounds
                    total += pend_dt
                    ev = {"type": "step", "i": steps, "n_new": pend_n, "delta": pend_text, "dt_ms": pend_dt * 1000}
                    if rounds > 1:
                        ev["steps"] = rounds   # кусок потока объединил несколько проходов таргета
                    yield ev
                    pend_dt, pend_n, pend_text = 0.0, 0, ""
                if cancelled():
                    reason = "cancel"
                    break
        if pend_n:  # хвост без проверки дерева (ответ кончился на prefill)
            steps += 1; total += pend_dt
            yield {"type": "step", "i": steps, "n_new": pend_n, "delta": pend_text, "dt_ms": pend_dt * 1000}
        total = max(total, 1e-9)
        yield {"type": "done", "tokens": ct, "steps": steps, "time_s": total, "tok_s": ct / total,
               "tau": ct / steps if steps else 0.0, "stop": reason, "wall_s": time.perf_counter() - t0}


def _sglang_version():
    try:
        from importlib.metadata import version
        return version("sglang")
    except Exception:  # noqa: BLE001
        return "?"


def _ver(v):
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except ValueError:
        return (99,)
