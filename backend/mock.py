"""Имитация движка без GPU: тот же протокол событий, синтетические деревья и тайминги.

Нужна, чтобы проверять интерфейс и записывать пример для витрины, когда Colab не запущен.
Цифры здесь не настоящие.
"""
import random
import re
import time

THINK = (
    "Хорошо, пользователь просит объяснить спекулятивный декодинг простыми словами. "
    "Нужно сравнить с обычной генерацией и упомянуть, почему это ускоряет модель без потери качества.\n"
    "</think>\n\n"
)
ANSWER = (
    "**Спекулятивный декодинг** ускоряет генерацию большой языковой модели, не меняя её ответы.\n\n"
    "1. Маленькая быстрая модель-драфтер предлагает сразу несколько следующих токенов, "
    "причём не одной цепочкой, а деревом вариантов.\n"
    "2. Большая модель проверяет всё дерево за один проход и принимает самый длинный "
    "префикс, который совпадает с тем, что она сгенерировала бы сама.\n"
    "3. Если драфтер угадал, за один шаг большой модели получается сразу 4–6 токенов вместо одного.\n\n"
    "EAGLE-3 строит драфтер поверх скрытых состояний трёх слоёв таргет-модели, поэтому "
    "угадывает чаще и ускоряет генерацию в 3–4 раза.\n\n"
    "**Почему ответ не меняется.** Таргет проверяет каждый предложенный токен и оставляет только те, "
    "которые выбрал бы сам. Если драфтер ошибся, ветка обрезается на первой ошибке, а таргет добавляет "
    "свой правильный токен. Поэтому при температуре 0 текст получается ровно таким же, как без драфтера, "
    "просто за меньшее число проходов большой модели.\n\n"
    "**Где ускорение меньше.** Драфтер угадывает хуже на редких языках, в коде с необычными именами и "
    "в длинных рассуждениях. Ещё ускорение зависит от железа: на быстрых GPU узким местом становится "
    "запуск ядер, и выигрыш от дерева частично съедается накладными расходами.\n\n"
    "**Итог.** Спекулятивный декодинг - это способ получить тот же ответ быстрее: маленькая модель "
    "угадывает, большая проверяет, а пользователь видит текст в несколько раз раньше."
)
# Llama-3.1-8B-Instruct отвечает сразу, DeepSeek-R1-Distill сначала рассуждает (THINK)
TOKENS = {"llama": re.findall(r"\s*\S+|\s+", ANSWER), "dsl": re.findall(r"\s*\S+|\s+", THINK + ANSWER)}
VOCAB = ["модель", " токен", " и", " в", " за", " один", " шаг", ",", ".", " это", " не", " большой",
         " дерево", " быстро", " проверяет", " драфтер", " вариант", " сразу", " ответ", " the", " of"]


# пул как в ноутбуке; p - вероятность принять очередной токен (цифры не настоящие)
POOL = [
    {"id": "llama", "label": "Llama-3.1-8B-Instruct", "kind": "target", "target": "llama",
     "path": "meta-llama/Llama-3.1-8B-Instruct"},
    {"id": "authors", "label": "authors", "kind": "drafter", "target": "llama", "layers": [2, 16, 29],
     "fc_norm": False, "norm_output": False, "path": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "p": 0.925},
    {"id": "ours", "label": "ours", "kind": "drafter", "target": "llama", "layers": [16, 22, 29],
     "fc_norm": True, "norm_output": False, "path": "Funmagster/EAGLE3-Llama-3.1-8B-Instruct-L16-22-29-fcnorm",
     "p": 0.935},
    {"id": "dsl", "label": "DeepSeek-R1-Distill-Llama-8B", "kind": "target", "target": "dsl",
     "path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},
]
TARGET_OF = {m["id"]: m["target"] for m in POOL}
POOL_KIND = {m["id"]: m["kind"] for m in POOL}


class MockEngine:
    def __init__(self, total_token=60, depth=7, top_k=10, seed=None, sglang=False):
        """sglang=True ведёт себя как engine_sglang: без деревьев, дорожки меряются по очереди."""
        self.total_token, self.depth, self.top_k, self.sglang = total_token, depth, top_k, sglang
        self.rng = random.Random(seed)
        self.accept = {m["id"]: m.get("p") for m in POOL}

    def info(self):
        return {"target": POOL[0]["path"], "targets": [m["path"] for m in POOL if m["kind"] == "target"], "gpu": "нет, имитация",
                "models": [{k: v for k, v in m.items() if k != "p"} for m in POOL],
                "mock": True, "engine": {"name": "имитация SGLang" if self.sglang else "имитация", "version": "без GPU",
                                         "tree": not self.sglang},
                "tree": {"total_token": self.total_token, "depth": self.depth, "top_k": self.top_k}}

    def run(self, lanes, prompt, max_new_tokens, temperature, emit, cancelled=lambda: False, pace=True):
        from race import interleave
        lanes = [lanes] if isinstance(lanes, str) else list(lanes)
        for lane in lanes:
            emit({"type": "start", "lane": lane, "prompt_tokens": 24 + len(prompt) // 4})
        gens = {lane: self._gen(lane, max_new_tokens, cancelled, pace) for lane in lanes}
        if self.sglang:  # SGLang дерево не отдаёт
            def strip(g):
                for ev in g:
                    if ev["type"] == "step":
                        ev.pop("tree", None); ev.pop("accepted", None)
                    yield ev
            gens = {lane: strip(g) for lane, g in gens.items()}
        interleave(gens, emit)

    def _gen(self, mode, max_new_tokens, cancelled, pace):
        toks = TOKENS[TARGET_OF[mode]][:max_new_tokens]
        pos, steps, gpu = 0, 0, 0.0
        while pos < len(toks) and not cancelled():
            steps += 1
            if POOL_KIND[mode] == "target":
                dt = self.rng.uniform(0.024, 0.029)
                n = 1
                ev = {"type": "step", "i": steps, "n_new": 1, "delta": toks[pos], "dt_ms": dt * 1000}
            else:
                dt = self.rng.uniform(0.041, 0.050)
                acc = min(self._accept_len(self.accept[mode]), self.depth)
                n = min(acc + 1, len(toks) - pos)
                tree, accepted = self._tree(toks[pos:pos + self.depth + 1], n - 1)
                ev = {"type": "step", "i": steps, "n_new": n, "delta": "".join(toks[pos:pos + n]),
                      "dt_ms": dt * 1000, "tree": tree, "accepted": accepted}
            if pace:  # две дорожки идут по очереди, поэтому на стене каждая вдвое медленнее, как на GPU
                time.sleep(dt)
            gpu += dt
            pos += n
            yield ev
        yield {"type": "done", "tokens": pos, "steps": steps, "time_s": gpu, "tok_s": pos / gpu if gpu else 0,
               "tau": pos / steps if steps else 0, "stop": "cancel" if cancelled() else "eos"}

    def _accept_len(self, p):
        k = 0
        while k < self.depth and self.rng.random() < p:
            k += 1
        return k

    def _tree(self, truth, acc):
        """Дерево в формате движка: узел 0 - корень, родитель всегда раньше ребёнка."""
        rng = self.rng
        t, par, p, depth = [truth[0]], [-1], [1.0], [0]
        chain = [0]
        prob = 1.0
        # правильная цепочка: принятые узлы + один отвергнутый «почти правильный» хвост
        for d in range(1, min(acc + 2, self.depth + 1)):
            prob *= rng.uniform(0.75, 0.97)
            tok = truth[d] if d <= acc and d < len(truth) else rng.choice(VOCAB)
            t.append(tok); par.append(chain[-1]); p.append(prob); depth.append(d)
            chain.append(len(t) - 1)
        # отвлекающие ветки: как в EAGLE, на каждой глубине около top_k узлов
        quotas = [10, 10, 9, 8, 8, 7, 7][: self.depth]
        for d in range(1, self.depth + 1):
            have = sum(1 for x in depth if x == d)
            prev = sorted((i for i in range(len(t)) if depth[i] == d - 1), key=lambda i: -p[i])[:4]
            for _ in range(max(0, quotas[d - 1] - have)):
                if not prev or len(t) >= self.total_token:
                    break
                parent = prev[min(int(rng.expovariate(1.2)), len(prev) - 1)]
                t.append(rng.choice(VOCAB)); par.append(parent); depth.append(d)
                p.append(round(p[parent] * rng.uniform(0.02, 0.4), 4))
        order = sorted(range(len(t)), key=lambda i: (depth[i], -p[i]))
        remap = {old: new for new, old in enumerate(order)}
        tree = {"t": [t[i] for i in order], "par": [remap[par[i]] if par[i] >= 0 else -1 for i in order],
                "p": [round(p[i], 4) for i in order]}
        return tree, [remap[i] for i in chain[: acc + 1]]
