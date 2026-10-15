"""Пошаговая генерация на коде авторов EAGLE с выдачей дерева драфтера на каждом шаге.

Пул моделей на одной таргет-модели (веса таргета загружаются один раз):
  target        - обычная генерация таргета с KV-кэшем, без драфтера;
  <id драфтера> - спекулятивная генерация с драфтером EAGLE-3 (их может быть несколько).
Гонка: любые две модели из пула, шаги чередуются по часам GPU (race.py).

Драфтеры EAGLE-3.1 (fc_norm, norm_output) и драфтеры с другими слоями таргета работают на пропатченном
коде авторов (eagle_patch.py). Слои читаются из EAGLE_CAPTURE_LAYERS, и движок выставляет её перед каждым
шагом каждой дорожки, потому что переменная одна на процесс, а драфтеры в гонке разные.
Веса принимаются и в раскладке авторов (midlayer.*), и в раскладке NeMo / vLLM (model.layers.0.*).

Каждый шаг отдаётся колбэком emit(dict). Время шага меряется только по GPU-работе
(с torch.cuda.synchronize), без декодирования текста и отправки по сети.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoConfig, AutoTokenizer

from eagle.model.ea_model import EaModel
from eagle.model.utils import (
    evaluate_posterior,
    initialize_tree,
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)


OPT_TARGET_GB = 17   # 8B в fp16 с запасом


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
    path: str  # каталог или репозиторий HF с config.json и model.safetensors/pytorch_model.bin
    target: str = ""  # id таргета, под который обучен драфтер (пусто = первый таргет)


@dataclass
class EngineConfig:
    targets: list = field(default_factory=list)
    drafters: list = field(default_factory=list)
    total_token: int = 60
    depth: int = 7  # для EAGLE-3 нужна глубина 7, а не 5 (см. отчёт по воспроизведению)
    top_k: int = 10
    max_length: int = 2048


def _resolve(path, name):
    local = os.path.join(path, name)
    return local if os.path.exists(local) else hf_hub_download(path, name)


# раскладка NeMo / vLLM (EAGLE-3.1) -> имена весов в коде авторов
_NEMO_TO_AUTHORS = (("model.layers.0.", "midlayer."), ("model.fc_norm.", "fc_norm."), ("model.fc.", "fc."),
                    ("model.norm.", "norm."))


def _load_state(path, device):
    try:
        state = torch.load(_resolve(path, "pytorch_model.bin"), map_location=device)
    except Exception:
        from safetensors.torch import load_file
        state = load_file(_resolve(path, "model.safetensors"))
    out = {}
    for k, v in state.items():
        for old, new in _NEMO_TO_AUTHORS:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def _capture_layers(dc):
    """Слои таргета в счёте авторов (i = вход слоя i). NeMo/SGLang/vLLM пишут выход слоя, это i - 1."""
    if dc.get("eagle_capture_layers") is not None:
        return [int(x) for x in dc["eagle_capture_layers"]]
    ids = (dc.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    return [int(x) + 1 for x in ids] if ids else None


class _ScoreSpy:
    """Перехватывает финальный torch.topk в topK_genrate, чтобы достать вероятности узлов дерева."""

    def __init__(self, ea_layer):
        self.ea_layer = ea_layer
        self.values = None
        self.indices = None

    def __enter__(self):
        self._real = torch.topk
        total = self.ea_layer.total_tokens

        def spy(inp, k, *args, **kwargs):
            out = self._real(inp, k, *args, **kwargs)
            if inp.dim() == 1 and k == total:
                self.values, self.indices = out.values, out.indices
            return out

        torch.topk = spy
        return self

    def __exit__(self, *exc):
        torch.topk = self._real

    def node_probs(self):
        """Вероятности путей в порядке узлов дерева (узел 0 - корень, p=1)."""
        if self.values is None:
            return None
        order = torch.argsort(self.indices)
        probs = torch.exp(self.values[order].float()).tolist()
        return [1.0] + probs


class Engine:
    """Несколько таргетов (у каждого свои веса, токенизатор и KV-кэш) и драфтеры поверх них.

    Модель пула - это таргет без драфтера (id таргета) или драфтер (id драфтера) на своём таргете.
    Генераторы дорожек чередуются, поэтому перед каждым шагом движок переключается на таргет этой дорожки
    (_bind), так же как выставляет слои захвата драфтера (_use_layers).
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.targets, self.bad_targets = {}, {}
        for t in sorted(cfg.targets, key=lambda t: t.optional):   # необязательные - последними
            if t.optional and not self._room(len(cfg.drafters)):
                free = torch.cuda.mem_get_info()[0] / 2**30
                self.bad_targets[t.id] = f"не загружена: не хватает видеопамяти (свободно {free:.0f} ГБ, нужно ~{OPT_TARGET_GB} ГБ)"
                print(f"[engine] таргет {t.id}: {self.bad_targets[t.id]}", file=sys.stderr)
                continue
            try:  # закрытая Llama без HF_TOKEN не скачается: тогда выпадают она и её драфтеры, остальное работает
                base = self._load_target(t.path)
                tok = AutoTokenizer.from_pretrained(t.path, use_fast=False)
                self.targets[t.id] = {"spec": t, "base": base, "tokenizer": tok, "stop_ids": self._stop_ids(tok),
                                      "bkv": None}
            except Exception as e:  # noqa: BLE001
                print(f"[engine] таргет {t.id} ({t.path}) не загрузился: {e!r}", file=sys.stderr)
                self.bad_targets[t.id] = f"таргет {t.path} не загрузился (нужен HF_TOKEN с доступом?): {e!r}"
        if not self.targets:
            raise RuntimeError("не загрузился ни один таргет: " + "; ".join(self.bad_targets.values()))
        self.models, self.layers, self.meta, self.failed, self.target_of = {}, {}, {}, {}, {}
        first = cfg.targets[0].id
        for spec in cfg.drafters:
            try:
                tid = spec.target or first
                if tid in self.bad_targets:
                    raise RuntimeError(self.bad_targets[tid])
                if tid not in self.targets:
                    raise ValueError(f"нет таргета {tid}")
                base = self.targets[tid]["base"]
                device = base.model.layers[-1].self_attn.q_proj.weight.device
                n_layers = base.config.num_hidden_layers
                conf = _resolve(spec.path, "config.json")
                with open(conf) as f:
                    dc = json.load(f)
                layers = _capture_layers(dc)
                if layers is not None:  # проверяем до загрузки: иначе ошибка всплывёт посреди гонки
                    if len(layers) != 3 or sorted(set(layers)) != layers or not 0 <= layers[0] < layers[-1] < n_layers:
                        raise ValueError(f"eagle_capture_layers {layers} не подходят таргету из {n_layers} слоёв")
                state = _load_state(spec.path, device)
                if dc.get("fc_norm") and not any(k.startswith("fc_norm.") for k in state):
                    raise ValueError("в конфиге fc_norm, а в весах его нет")  # иначе код авторов молча пропустит
                model = EaModel(True, base, self.targets[tid]["spec"].path, conf,
                                cfg.total_token, cfg.depth, cfg.top_k, 1.0, state)
                if dc.get("norm_output") and not getattr(model.ea_layer, "norm_output", False):
                    raise ValueError("norm_output есть в конфиге, но код EAGLE не пропатчен (eagle_patch.py)")
                self.models[spec.id] = model
                self.target_of[spec.id] = tid
                self.layers[spec.id] = ",".join(map(str, layers)) if layers else None
                self.meta[spec.id] = {"layers": layers or [2, n_layers // 2, n_layers - 3],
                                      "fc_norm": bool(dc.get("fc_norm", False)),
                                      "norm_output": bool(dc.get("norm_output", False)), "path": spec.path,
                                      "target": tid}
            except Exception as e:  # noqa: BLE001  одна битая ссылка не должна ронять демо
                print(f"[engine] драфтер {spec.id} ({spec.path}) не загрузился: {e!r}", file=sys.stderr)
                self.failed[spec.id] = repr(e)
        self.specs = {s.id: s for s in cfg.drafters if s.id in self.models}
        for tid in self.targets:
            self.target_of[tid] = tid
        self._use_target(first if first in self.targets else next(iter(self.targets)))
        self.warmup()

    @staticmethod
    def _load_target(path):
        from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
        arch = AutoConfig.from_pretrained(path).architectures[0]
        if arch == "LlamaForCausalLM":
            cls = KVLlama
        elif arch == "Qwen3ForCausalLM":
            from eagle.model.modeling_qwen3_kv import Qwen3ForCausalLM as cls
        else:
            raise ValueError(f"таргет {arch} не поддержан демо")
        return cls.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto")

    @staticmethod
    def _room(n_drafters):
        """Свободно ли на GPU столько, чтобы взять ещё одну 8B-модель и оставить место драфтерам и KV-кэшу."""
        if not torch.cuda.is_available():
            return True
        free = torch.cuda.mem_get_info()[0] / 2**30
        return free >= OPT_TARGET_GB + 1.5 * n_drafters + 3

    def _use_target(self, tid):
        t = self.targets[tid]
        self.tid, self.base, self.tokenizer, self.stop_ids = tid, t["base"], t["tokenizer"], t["stop_ids"]

    def _bind(self, tid, gen):
        """Генератор дорожки, который перед каждым своим шагом переключает движок на свой таргет."""
        while True:
            self._use_target(tid)
            try:
                ev = next(gen)
            except StopIteration:
                return
            yield ev

    def warmup(self):
        """Первый запуск CUDA-ядер медленный: прогоняем каждую модель пула, чтобы не портить первый замер."""
        for mode in self.pool():
            self.run(mode, "Hi", 16, 0.0, lambda ev: None)

    def pool(self):
        return [*self.targets, *self.specs]

    # ---------- общее ----------
    def info(self):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        models = [{"id": tid, "label": t["spec"].label, "kind": "target", "path": t["spec"].path, "target": tid}
                  for tid, t in self.targets.items()]
        models += [{"id": s.id, "label": s.label, "kind": "drafter", **self.meta[s.id]} for s in self.specs.values()]
        models += [{"id": s.id, "label": s.label, "kind": "drafter", "path": s.path, "target": s.target,
                    "error": self.failed[s.id]} for s in self.cfg.drafters if s.id in self.failed]
        models[:0] = [{"id": t.id, "label": t.label, "kind": "target", "path": t.path, "target": t.id,
                       "error": self.bad_targets[t.id]} for t in self.cfg.targets if t.id in self.bad_targets]
        return {"target": self.cfg.targets[0].path, "targets": [t.path for t in self.cfg.targets],
                "gpu": gpu, "models": models, "mock": False,
                "engine": {"name": "код авторов EAGLE", "version": "SafeAILab/EAGLE cb7e084", "tree": True},
                "tree": {"total_token": self.cfg.total_token, "depth": self.cfg.depth, "top_k": self.cfg.top_k}}

    @staticmethod
    def _stop_ids(tok):
        ids = {tok.eos_token_id}
        for t in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>"):
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != tok.unk_token_id:
                ids.add(i)
        return {i for i in ids if i is not None}

    def encode(self, prompt):
        tok = self.tokenizer
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        return ids.to(self.base.device)

    def _piece(self, tid):
        return self.tokenizer.decode([tid], skip_special_tokens=False)

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _base_kv(self):
        from eagle.model.kv_cache import initialize_past_key_values
        t = self.targets[self.tid]
        if t["bkv"] is None:
            t["bkv"] = initialize_past_key_values(self.base, max_length=self.cfg.max_length)
        t["bkv"][2].zero_()
        return t["bkv"][0]

    def _eagle_kv(self, model):
        from eagle.model.kv_cache import initialize_past_key_values
        if not hasattr(model, "past_key_values"):
            model.past_key_values, model.past_key_values_data, model.current_length_data = \
                initialize_past_key_values(model.base_model, max_length=self.cfg.max_length)
        model.current_length_data.zero_()
        return model.past_key_values, model.past_key_values_data, model.current_length_data

    def run(self, lanes, prompt, max_new_tokens, temperature, emit, cancelled=lambda: False):
        """lanes: одна модель пула или список из двух разных моделей (гонка)."""
        from race import interleave
        lanes = [lanes] if isinstance(lanes, str) else list(lanes)
        gens = {}
        for lane in lanes:
            tid = self.target_of[lane]
            self._use_target(tid)
            input_ids = self.encode(prompt)  # у разных таргетов свои токенизаторы и шаблоны чата
            emit({"type": "start", "lane": lane, "prompt_tokens": int(input_ids.shape[1])})
            gen = (self._gen_baseline(input_ids, max_new_tokens, temperature, cancelled) if lane in self.targets
                   else self._gen_eagle(lane, input_ids, max_new_tokens, temperature, cancelled))
            gens[lane] = self._bind(tid, gen)
        with torch.inference_mode():
            interleave(gens, emit)

    def _done(self, out, steps, gpu_time, reason):
        n = len(out)
        return {"type": "done", "tokens": n, "steps": steps, "time_s": gpu_time,
                "tok_s": n / gpu_time if gpu_time > 0 else 0.0,
                "tau": n / steps if steps else 0.0, "stop": reason}

    def _cut_stop(self, new_ids):
        for k, t in enumerate(new_ids):
            if t in self.stop_ids:
                return new_ids[:k], True
        return new_ids, False

    # ---------- обычная генерация (генератор событий, по шагу за next) ----------
    def _gen_baseline(self, input_ids, max_new_tokens, temperature, cancelled):
        lp = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None
        pkv = self._base_kv()
        inner = self.base.model
        out, text, steps, gpu, reason = [], "", 0, 0.0, "length"
        outputs = None
        while True:
            # общий таргет: дерево EAGLE не должно попасть в маску обычной генерации
            inner.tree_mask = None; inner.tree_mode = None
            self._sync(); t0 = time.perf_counter()
            if outputs is None:  # prefill входит в первый шаг
                outputs = self.base(input_ids, past_key_values=pkv, use_cache=True)
            logits = outputs.logits[:, -1]
            if lp is not None:
                nxt = torch.multinomial(torch.softmax(lp(None, logits), dim=-1), 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            outputs = self.base(nxt, past_key_values=pkv, use_cache=True)
            self._sync(); dt = time.perf_counter() - t0; gpu += dt

            tid = int(nxt[0, 0]); steps += 1
            if tid in self.stop_ids:
                reason = "eos"; break
            out.append(tid)
            new_text = self.tokenizer.decode(out, skip_special_tokens=True)
            yield {"type": "step", "i": steps, "n_new": 1, "delta": new_text[len(text):], "dt_ms": dt * 1000}
            text = new_text
            if len(out) >= max_new_tokens:
                break
            if cancelled():
                reason = "cancel"; break
        yield self._done(out, steps, gpu, reason)

    # ---------- EAGLE ----------
    def _use_layers(self, lane):
        """Слои таргета для драфтера этой дорожки: переменная общая, ставим перед каждым шагом."""
        if self.layers[lane]:
            os.environ["EAGLE_CAPTURE_LAYERS"] = self.layers[lane]
        else:
            os.environ.pop("EAGLE_CAPTURE_LAYERS", None)

    def _gen_eagle(self, lane, input_ids, max_new_tokens, temperature, cancelled):
        model = self.models[lane]
        lp = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None
        padding = torch.full((1, 1), -1, dtype=torch.long, device=input_ids.device)
        input_ids = input_ids.clone()
        model.ea_layer.reset_kv()
        pkv, pkv_data, cur_len = self._eagle_kv(model)
        prompt_len = input_ids.shape[1]
        out, text, steps, gpu, reason = [], "", 0, 0.0, "length"

        self._use_layers(lane)
        reset_tree_mode(model)
        self._sync(); t0 = time.perf_counter()
        with _ScoreSpy(model.ea_layer) as spy:
            draft_tokens, retrieve_indices, tree_mask, tree_pos, _, _, _ = initialize_tree(input_ids, model, pkv, lp)
        probs = spy.node_probs()
        self._sync(); prefill = time.perf_counter() - t0  # prefill входит в первый шаг

        while True:
            self._use_layers(lane)
            self._sync(); t0 = time.perf_counter()
            model.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_new, _ = tree_decoding(model, draft_tokens, pkv, tree_pos, input_ids, retrieve_indices)
            tree_ids = draft_tokens[0].tolist()
            tree_ri = retrieve_indices.tolist()
            tree_probs = probs
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best, acc_len, sample_p = evaluate_posterior(logits, candidates, lp)
            best, acc_len = int(best), int(acc_len)
            with _ScoreSpy(model.ea_layer) as spy:
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_pos, _, _, _ = update_inference_inputs(
                    input_ids, candidates, best, acc_len, retrieve_indices, lp, 0,
                    pkv_data, cur_len, model, hidden_new, sample_p,
                )
            probs = spy.node_probs()
            self._sync(); dt = time.perf_counter() - t0 + prefill; prefill = 0.0
            gpu += dt
            steps += 1

            accepted = [i for i in tree_ri[best][: acc_len + 1]]
            new_ids, hit_stop = self._cut_stop([tree_ids[i] for i in accepted])
            new_ids = new_ids[: max_new_tokens - len(out)]
            out.extend(new_ids)
            new_text = self.tokenizer.decode(out, skip_special_tokens=True)
            yield {"type": "step", "i": steps, "n_new": len(new_ids), "delta": new_text[len(text):],
                   "dt_ms": dt * 1000, "tree": self._tree(tree_ids, tree_ri, tree_probs), "accepted": accepted}
            text = new_text
            if hit_stop:
                reason = "eos"; break
            if len(out) >= max_new_tokens or input_ids.shape[1] - prompt_len >= max_new_tokens:
                break
            if input_ids.shape[1] > self.cfg.max_length - self.cfg.total_token - 20:
                break
            if cancelled():
                reason = "cancel"; break
        yield self._done(out, steps, gpu, reason)

    def _tree(self, ids, retrieve, probs):
        n = len(ids)
        parent = [-1] * n
        for row in retrieve:
            prev = None
            for idx in row:
                if idx < 0:
                    break
                if prev is not None:
                    parent[idx] = prev
                prev = idx
        return {"t": [self._piece(i) for i in ids], "par": parent,
                "p": [round(x, 4) for x in probs] if probs and len(probs) == n else None}
