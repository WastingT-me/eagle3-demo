import base64, gzip, io, json, os, sys, tarfile
# Пересобирает ноутбук: код демо (backend/ и site/) пакуется в него архивом.
# Сборка воспроизводимая: при тех же файлах ноутбук совпадает байт в байт, и git видит только настоящие правки.
root = sys.argv[1] if len(sys.argv) > 1 else "."
buf = io.BytesIO()

MEDIA = (".mp4", ".webm", ".m4v", ".mov", ".ogv", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac")


def clean(ti):
    ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
    ti.mode = 0o755 if ti.name.endswith(".sh") else 0o644
    return ti

with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
    for sub in ("backend", "site"):
        for dp, dn, fn in os.walk(os.path.join(root, sub)):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                # фон (видео и музыка) в ноутбук не кладём: на Colab он берётся с Google Диска
                if f.endswith(".pyc") or f.startswith(".") or f.lower().endswith(MEDIA):
                    continue
                full = os.path.join(dp, f)
                tar.add(full, arcname=os.path.join("eagle3-demo", os.path.relpath(full, root)), filter=clean)
b64 = base64.b64encode(buf.getvalue()).decode()

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
md("""
# Демо EAGLE-3

Ноутбук поднимает демо-сервер на GPU Colab и открывает его в интернет через Cloudflare Tunnel.
Пул моделей, на странице выбираются любые две, все держатся на GPU сразу:
- **[target] Llama-3.1-8B-Instruct** генерирует без драфтера;
- **[draft] authors**, официальный драфтер EAGLE-3 под Llama-3.1-8B-Instruct (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`);
- **[draft] ours**, наш драфтер под Llama-3.1-8B-Instruct (`Funmagster/EAGLE3-Llama-3.1-8B-Instruct-L16-22-29-fcnorm`, слои 16, 22, 29 и fc_norm);
- **[draft] authors from scratch**, драфтер в раскладке авторов (слои 2, 16, 29, без fc_norm), который мы обучили с нуля (`Funmagster/EAGLE3-Llama-3.1-8B-Instruct-L2-16-29-baseline`);
- **[target] DeepSeek-R1-Distill-Llama-8B** (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`), модель с рассуждением. Грузится последней и только если хватает видеопамяти: с движком EAGLE она помещается и на 80 ГБ; с SGLang (~20 ГБ на каждую модель) её не будет, пока все пять не влезают.

Движок по умолчанию **код авторов EAGLE** (`ENGINE = "eagle"`): одна копия таргета на все драфтеры, помещается в 40 ГБ, и на странице видно дерево драфтера на каждом шаге.
На странице демо движок можно сменить на **SGLang 0.5.9** (как в NeMo AutoModel и в замерах команды) и обратно. Смена занимает порядка 10 минут: сервер ставит SGLang (~5 ГБ, отдельное окружение на Python 3.12), освобождает GPU и перезагружает модели. SGLang держит свою копию таргета на каждую модель (~20 ГБ): на 80 ГБ сразу живут три модели, остальные загружаются при выборе (около минуты). Дерево драфтера SGLang не отдаёт.

1. **Runtime → Change runtime type → H100** или A100 (40 или 80 ГБ). Для показа включите **Background execution**, чтобы сессия жила при закрытой вкладке.
2. **Нужен `HF_TOKEN`**: Llama-3.1-8B-Instruct закрыта лицензией Meta. Примите лицензию на странице модели на Hugging Face, затем слева значок ключа → секрет **`HF_TOKEN`** и доступ ноутбука к нему. Без токена демо не запустится.
3. **Фоновое видео и музыка (необязательно).** Положите видео (mp4 или webm) и музыку (mp3) в папку **`videos`** в корне своего Google Диска. Ноутбук попросит доступ к Диску, скопирует папку на диск Colab, и страница демо покажет видео фоном, а музыку будет играть вместо звука видео; и видео, и музыка идут по кругу. Если файлов несколько, берётся первый по имени. Без mp3 звучит само видео. Если папки нет, фон обычный.
4. **Runtime → Run all.** Первый запуск ставит зависимости и качает ~18 ГБ весов, это 10–15 минут.
5. В последней ячейке появится ссылка на демо.

Сессия Colab живёт до 24 часов; после перезапуска адрес туннеля меняется.
"""),
code("""
# Таргеты: (id, подпись, репозиторий HF или путь). Драфтеры: (id, подпись, репозиторий, id своего таргета).
# На странице перед подписью стоит [target] или [draft].
# Драфтер принимается в раскладке авторов (config.json + model.safetensors / pytorch_model.bin) или NeMo / vLLM.
# Слои таргета берутся из "eagle_capture_layers" (счёт авторов) или "eagle_config.eagle_aux_hidden_state_layer_ids"
# (счёт SGLang/NeMo, на 1 меньше); "fc_norm": true включает нормировку EAGLE-3.1.
# Движок при запуске: "eagle" - код авторов, с деревом драфтера; "sglang" - как в NeMo, без дерева.
# На странице демо его можно сменить, это порядка 10 минут.
ENGINE = "eagle"
TARGETS = [
    ("llama", "Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    # "optional": грузится последней и только если хватает видеопамяти (~20 ГБ на SGLang), иначе пропускается
    ("dsl",   "DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "optional"),
]
DRAFTERS = [
    ("authors", "authors",               "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",                     "llama"),
    ("ours",    "ours",                  "Funmagster/EAGLE3-Llama-3.1-8B-Instruct-L16-22-29-fcnorm", "llama"),
    ("scratch", "authors from scratch",  "Funmagster/EAGLE3-Llama-3.1-8B-Instruct-L2-16-29-baseline",   "llama"),
]
# Фон страницы демо: папка на Google Диске с видео и mp3 (пусто = без фона), берётся первое по имени; оба по кругу.
VIDEOS_DRIVE_DIR = "videos"
# Постоянная ссылка: сайт на GitHub Pages находит этот сервер через секретный гист.
# BACKEND_GIST - id гиста (конец его адреса); нужен секрет Colab GH_TOKEN (токен GitHub classic с правом gist).
# Пусто - только временная ссылка trycloudflare. Без секрета GH_TOKEN ноутбук напишет, что адрес не записан, и демо работает по временной ссылке.
# В тот же гист сервер кладёт последнюю сессию GPU (файл eagle3-replay.json): её играет страница, когда Colab выключен.
BACKEND_GIST = "e7b489608c0c3c8eee1d15498a86872e"
PAGES_URL = "https://wastingt-me.github.io/eagle3-demo"

import os
from google.colab import userdata
try:  # Llama-3.1-8B-Instruct закрыта лицензией: без токена она не скачается; сервер наследует токен из окружения
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN найден")
except Exception:
    print("Секрета HF_TOKEN нет: Llama-3.1-8B-Instruct закрыта лицензией и не скачается, добавьте секрет HF_TOKEN")
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
code(f"""
# Код демо (сервер + сайт), упакован в ноутбук
import base64, io, tarfile
BUNDLE = "{b64}"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall("/content", filter="data")
print("распаковано в /content/eagle3-demo")
"""),
code("""
# Зависимости движка (тот же скрипт сервер зовёт при смене движка со страницы)
import sys
EAGLE_PY, SGLANG_PY = sys.executable, "/content/sgl/bin/python"
!bash /content/eagle3-demo/backend/setup_engine.sh {ENGINE} {EAGLE_PY}
PY = SGLANG_PY if ENGINE == "sglang" else EAGLE_PY
"""),
code("""
# Запуск сервера: загрузка моделей и прогрев, несколько минут. Сайт отвечает сразу, генерация - когда модели готовы
import subprocess, time, urllib.request, json
try:  # ячейку запускают повторно: старый сервер демо и его серверы SGLang держат GPU, останавливаем их
    srv.terminate(); srv.wait(timeout=90)
except NameError:
    pass
except Exception:
    srv.kill()
# фон демо (видео и музыка): подключаем Google Диск и копируем папку на диск Colab (с Диска видео тормозит)
VIDEOS = ""
if VIDEOS_DRIVE_DIR:
    import os, shutil
    from google.colab import drive
    try:
        drive.mount("/content/drive")
        src = os.path.join("/content/drive/MyDrive", VIDEOS_DRIVE_DIR)
        if os.path.isdir(src):
            shutil.copytree(src, "/content/videos", dirs_exist_ok=True)
            VIDEOS = "/content/videos"
            fs = [f.lower() for _, _, fs in os.walk(VIDEOS) for f in fs]
            nv = sum(f.endswith((".mp4", ".webm", ".m4v", ".mov")) for f in fs)
            na = sum(f.endswith((".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".flac")) for f in fs)
            print(f"фон: видео {nv}, музыка {na} (берётся первое по имени; без музыки звучит видео)")
        else:
            print(f"папки {src} нет: демо без фонового видео")
    except Exception as e:
        print("Google Диск не подключён, демо без фонового видео:", e)
args = [PY, "server.py", "--engine", ENGINE, "--eagle-python", EAGLE_PY, "--sglang-python", SGLANG_PY] + (["--videos", VIDEOS] if VIDEOS else [])
if BACKEND_GIST:  # последняя сессия GPU уходит в гист для оффлайн-режима страницы
    try:
        from google.colab import userdata
        os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")
        args += ["--gist", BACKEND_GIST]
    except Exception as e:
        print("нет секрета GH_TOKEN: оффлайн-страница не получит запись этой сессии:", e)
for t in TARGETS:
    args += ["--target", "=".join(t)]
for d in DRAFTERS:
    args += ["--drafter", "=".join(d)]
log = open("/content/server.log", "w")
srv = subprocess.Popen(args, cwd="/content/eagle3-demo/backend", stdout=log, stderr=subprocess.STDOUT)

def ready():
    try:
        i = json.load(urllib.request.urlopen("http://localhost:8000/api/info", timeout=3))
    except Exception:
        return None
    if i.get("state") == "failed":
        print(open("/content/server.log").read()[-4000:])
        raise RuntimeError(i["switch"]["message"])
    return i if i.get("state") == "ready" else None

t0 = time.time()
while (info := ready()) is None:
    if srv.poll() is not None:
        print(open("/content/server.log").read()[-4000:])
        raise RuntimeError("сервер упал, лог выше")
    time.sleep(15)
    print(f"{int(time.time()-t0)} с: загружаю…", open("/content/server.log").read().strip().splitlines()[-1:] )
print("Сервер готов:", info["gpu"], "·", info["engine"]["name"], info["engine"]["version"])
for m in info["models"]:
    tag = "[target]" if m["kind"] == "target" else "[draft]"
    print(" ", "✗" if "error" in m else "✓", tag, m["label"], "·", m.get("error") or m.get("path", ""))
"""),
code("""
# Туннель Cloudflare: публичный https-адрес без регистрации, новый при каждом запуске
import re, subprocess, threading
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared && chmod +x /content/cloudflared
try: tun.terminate()  # туннель от прошлого запуска ячейки
except NameError: pass
tun = subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://localhost:8000"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tun.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        url = m.group(0); break
# дочитываем вывод туннеля в фоне, иначе он остановится на полном буфере
threading.Thread(target=lambda: [None for _ in tun.stdout], daemon=True).start()
print("Демо (временная ссылка):", url + "/demo.html")
if BACKEND_GIST:  # пишем текущий адрес туннеля в гист: по нему постоянная страница на GitHub Pages находит сервер
    import json, time, urllib.request
    try:
        body = {"files": {"eagle3-backend.json": {"content": json.dumps(
            {"url": url, "updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())})}}}
        req = urllib.request.Request("https://api.github.com/gists/" + BACKEND_GIST, method="PATCH",
            data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + userdata.get("GH_TOKEN"),
                                                     "Accept": "application/vnd.github+json"})
        urllib.request.urlopen(req, timeout=20)
        print("Постоянная ссылка:", PAGES_URL.rstrip("/") + "/demo.html")
    except Exception as e:
        print("Не удалось записать адрес в гист (нужен секрет GH_TOKEN с правом gist):", e)
"""),
md("""
Если демо перестало открываться, перезапустите две последние ячейки. Лог сервера: `/content/server.log`.

Чтобы остановить всё и освободить GPU: **Runtime → Disconnect and delete runtime**.
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(os.path.join(root, "colab", "eagle3_demo.ipynb"), "w"), ensure_ascii=False, indent=1)
print(len(b64), "bytes b64")
