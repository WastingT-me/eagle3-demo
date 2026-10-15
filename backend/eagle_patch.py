"""Патч кода авторов EAGLE для наших драфтеров (на месте, повторный запуск ничего не меняет).

    python eagle_patch.py /content/EAGLE

Те же правки, что big-training/code/patch_eagle.py в репозитории команды, только для пакета eagle (генерация):
  слои таргета    modeling_llama_kv.py берёт слои из EAGLE_CAPTURE_LAYERS ("16,22,29"; i = вход слоя i),
                  без переменной остаются 2, N/2, N-3, как у авторов;
  fc_norm         если в config.json драфтера "fc_norm": true, перед fc стоит RMSNorm на каждый из трёх слоёв;
  norm_output     если "norm_output": true (второй переключатель EAGLE-3.1), финальная RMSNorm драфтера
                  применяется к его выходу, и на следующий шаг драфта идёт уже нормированное состояние;
  проверка весов  для драфтеров с "eagle_capture_layers" в конфиге все веса обязаны загрузиться.
Каждая правка - точная замена текста, поэтому другая версия EAGLE упадёт сразу, а не молча.
"""
import sys
from pathlib import Path

MARK = "# eagle3-demo patched"

CAPTURE_HELPER = '''import os as _os


def capture_layers(n_layers):
    """Target states fed to the draft (index i = input of layer i). EAGLE-3's default: 2, N/2, N-3."""
    spec = _os.environ.get("EAGLE_CAPTURE_LAYERS")
    if not spec:
        return [2, n_layers // 2, n_layers - 3]
    layers = [int(x) for x in spec.split(",")]
    if len(layers) != 3 or sorted(set(layers)) != layers or not 0 <= layers[0] < layers[-1] < n_layers:
        raise ValueError(f"EAGLE_CAPTURE_LAYERS must be 3 increasing layer ids, got {spec}")
    return layers


'''
CAPTURE_OLD = '            if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2:\n'
CAPTURE_NEW = '            if idx in capture_layers(len(self.layers)):\n'
MODULE_START = '""" PyTorch LLaMA model."""\n'


def fc_norm_apply(indent):
    return (f'{indent}if hasattr(self, "fc_norm"):   # EAGLE-3.1 fc_norm: each captured state normalised on its own\n'
            f'{indent}    parts = hidden_states.chunk(3, dim=-1)\n'
            f'{indent}    hidden_states = torch.cat([norm(part) for norm, part in zip(self.fc_norm, parts)], dim=-1)\n')


def fc_norm_modules(indent):
    return (f'{indent}if getattr(config, "fc_norm", False):\n'
            f'{indent}    self.fc_norm = nn.ModuleList(\n'
            f'{indent}        [LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(3)])\n')


def replace_once(source, old, new, where):
    if source.count(old) != 1:
        raise ValueError(f'{where}: pinned EAGLE source changed near {old.strip()[:70]!r}')
    return source.replace(old, new, 1)


def patch_capture(s):
    w = 'model/modeling_llama_kv.py'
    s = replace_once(s, MODULE_START, MODULE_START + CAPTURE_HELPER, w)
    return replace_once(s, CAPTURE_OLD, CAPTURE_NEW, w)


def patch_cnets(s):
    w = 'model/cnets.py'
    fc = ('            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)\n'
          '        self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n')
    s = replace_once(s, fc, fc + fc_norm_modules('        ')
                     + '        self.norm_output = bool(getattr(config, "norm_output", False))   # EAGLE-3.1\n', w)
    # norm_output: выход слоя драфтера сразу проходит финальную норму (её же видит lm_head и следующий шаг)
    out = '        hidden_states = layer_outputs[0]\n'
    s = replace_once(s, out, out + '        if self.norm_output:\n            hidden_states = self.norm(hidden_states)\n', w)
    # ...поэтому перед lm_head норму второй раз не применяем
    if s.count('self.lm_head(self.norm(') != 2:
        raise ValueError(f'{w}: pinned EAGLE source changed near lm_head(self.norm(')
    s = s.replace('self.lm_head(self.norm(', 'self.lm_head(self._head_norm(')
    s = replace_once(s, '    def reset_kv(self):\n',
                     '    def _head_norm(self, x):\n        return x if self.norm_output else self.norm(x)\n\n'
                     '    def reset_kv(self):\n', w)
    apply = ('        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:\n'
             '            hidden_states = self.fc(hidden_states)\n')
    new = ('        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:\n'
           + fc_norm_apply('            ') + '            hidden_states = self.fc(hidden_states)\n')
    return replace_once(s, apply, new, w)


def patch_ea_model(s):
    # в отличие от patch_eagle.py здесь не трогаем EAGLE_CAPTURE_LAYERS: демо держит несколько драфтеров
    # с разными слоями и выставляет переменную перед каждым шагом (engine.py)
    w = 'model/ea_model.py'
    old = '        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)\n'
    new = old + ('        if hasattr(config, "eagle_capture_layers"):   # our drafts: every weight must be used\n'
                 '            missing = [k for k in load_.missing_keys\n'
                 '                       if not k.startswith("embed_tokens") and not k.endswith("inv_freq")]\n'
                 '            unexpected = [k for k in load_.unexpected_keys   # d2t/t2d are dropped above\n'
                 '                          if not (k in ("d2t", "t2d") and config.vocab_size == config.draft_vocab_size)]\n'
                 '            if missing or unexpected:\n'
                 '                raise ValueError(f"Draft weights do not match the model: missing {missing}, "\n'
                 '                                 f"unexpected {unexpected}")\n')
    return replace_once(s, old, new, w)


def main(repo):
    model = Path(repo) / 'eagle' / 'model'
    for name, fn in (('modeling_llama_kv.py', patch_capture), ('cnets.py', patch_cnets),
                     ('ea_model.py', patch_ea_model)):
        path = model / name
        src = path.read_text(encoding='utf-8')
        if MARK in src:
            continue
        out = fn(src) + f'\n{MARK}\n'
        compile(out, str(path), 'exec')
        path.write_text(out, encoding='utf-8')
    print('EAGLE пропатчен:', model)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
