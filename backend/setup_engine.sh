#!/usr/bin/env bash
# Ставит зависимости движка демо. Зовут ноутбук (ячейка зависимостей) и сервер при смене движка со страницы.
#   bash setup_engine.sh eagle  PYTHON   код авторов EAGLE на проверенном коммите + патч, в окружение PYTHON
#   bash setup_engine.sh sglang          SGLang 0.5.9 в отдельное окружение $ROOT/sgl на Python 3.12
# Повторный запуск быстрый: готовое не переставляется. Ничего не спрашивает (stdin закрыт).
set -euo pipefail
exec < /dev/null
ENGINE=${1:?eagle или sglang}
PY=${2:-python3}
ROOT=${DEMO_ROOT:-/content}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

if [ "$ENGINE" = sglang ]; then
  # Отдельное окружение на Python 3.12: SGLang тянет outlines_core 0.1.26, у которого нет готовых сборок
  # под Python новее 3.12, и на Python Colab pip собирает его из исходников на Rust и падает.
  if "$ROOT/sgl/bin/python" -c "import sglang, fastapi, uvicorn; assert sglang.__version__ == '0.5.9'" 2>/dev/null; then
    echo "SGLang 0.5.9 уже установлен"; exit 0
  fi
  echo "ставлю uv"
  "$PY" -m pip install -q uv
  UV=$("$PY" -c "import uv; print(uv.find_uv_bin())")
  test -x "$ROOT/sgl/bin/python" || "$UV" venv -q --python 3.12 "$ROOT/sgl"
  # ~5 ГБ (torch, CUDA, flashinfer): 5-15 минут; без -q видно, что установка идёт
  echo "ставлю SGLang 0.5.9 (~5 ГБ)"
  "$UV" pip install --python "$ROOT/sgl/bin/python" "sglang[all]==0.5.9" fastapi "uvicorn[standard]" huggingface_hub
  "$ROOT/sgl/bin/python" -c "import sglang; print('SGLang', sglang.__version__)"
elif [ "$ENGINE" = eagle ]; then
  if [ ! -d EAGLE ]; then
    echo "качаю код авторов EAGLE"
    git clone -q https://github.com/SafeAILab/EAGLE.git && git -C EAGLE checkout -q cb7e084
  fi
  # патч под EAGLE-3.1 (fc_norm, norm_output) и слои из конфига; повторный запуск ничего не меняет
  "$PY" "$HERE/eagle_patch.py" EAGLE
  echo "ставлю зависимости кода авторов"
  "$PY" -m pip install -q -e EAGLE --no-deps
  "$PY" -m pip install -q "transformers==4.53.1" accelerate safetensors fastapi "uvicorn[standard]"
else
  echo "неизвестный движок: $ENGINE" >&2; exit 2
fi
echo "готово: $ENGINE"
