"""Гонка двух режимов на одном GPU без деления ресурсов.

Шаги режимов выполняются строго по очереди, и каждый раз вперёд идёт тот режим,
у которого меньше накопленное время GPU. Так оба потока событий приходят в браузер
синхронно по «часам GPU», а замер каждого шага остаётся чистым.
"""


def interleave(gens, emit):
    clock = {m: 0.0 for m in gens}
    gens = dict(gens)
    while gens:
        lane = min(gens, key=clock.get)
        try:
            ev = next(gens[lane])
        except StopIteration:
            del gens[lane]
            continue
        ev["lane"] = lane
        if ev["type"] == "step":
            clock[lane] += ev["dt_ms"]
        emit(ev)
        if ev["type"] in ("done", "error"):
            gens.pop(lane, None)
