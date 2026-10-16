/* Демо: генерация двух моделей из пула по часам GPU.
   Пул: таргеты без драфтера ([target]) и драфтеры EAGLE-3 ([draft]), у каждого драфтера свой таргет. Обе модели генерируют одновременно
   (SGLang: два сервера на одном GPU; движок авторов: шаги чередуются), браузер показывает их на общих часах. */
(function () {
  const $ = id => document.getElementById(id);
  const SLOTS = [0, 1];
  const HIST_MAX = 10;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* приватный режим */ } },
  };

  let backend = null, info = null, replay = null, socket = null, running = false;
  let models = [], byId = {}, playable = new Set();
  let session = null;  // запись: все модели, прогнанные на одном запросе с одними настройками
  const selects = SLOTS.map(i => $("m" + i));

  // ---------- модели ----------
  const tag = m => (m.kind === "target" ? "[target] " : "[draft] ") + m.label;
  const targetOf = m => (m && (m.kind === "target" ? m : byId[m.target])) || null;
  function describe(m) {
    if (!m) return "";
    if (m.kind === "target") return "без драфтера: один токен за проход таргета";
    if (m.error) return "не загрузился на сервере";
    const t = targetOf(m), parts = [];
    if (t) parts.push("таргет " + t.label);
    if (m.layers) parts.push("слои таргета " + m.layers.join(", "));
    if (m.fc_norm) parts.push("fc_norm");
    if (m.norm_output) parts.push("norm_output");
    return parts.join(" · ");
  }

  function setModels(list, canPlay) {
    models = list; byId = Object.fromEntries(list.map(m => [m.id, m]));
    playable = new Set(list.filter(canPlay).map(m => m.id));
    const ids = list.map(m => m.id).filter(id => playable.has(id));
    let saved = []; try { saved = JSON.parse(store.get("eagle3-pair") || "[]"); } catch { saved = []; }
    const pick = [];
    const pref = [saved[0], "ours", ...ids.filter(id => byId[id].kind === "drafter"), ...ids];
    pick[0] = pref.find(id => playable.has(id));
    const t0 = targetOf(byId[pick[0]]);
    pick[1] = [saved[1], t0 && t0.id, ...ids].find(id => playable.has(id) && id !== pick[0]);
    selects.forEach((sel, i) => {
      sel.textContent = "";
      list.forEach(m => {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = tag(m) + (playable.has(m.id) ? "" : m.error ? " (не загрузился)" : " (нет в записи)");
        o.disabled = !playable.has(m.id);
        sel.appendChild(o);
      });
      if (pick[i]) sel.value = pick[i];
      sel.dataset.prev = sel.value;
    });
    $("run").disabled = !(pick[0] && pick[1]);
    SLOTS.forEach(i => { if (!slots[i].events.length) slots[i].label(selects[i].value); });
    renderTabs();
  }

  selects.forEach((sel, i) => sel.addEventListener("change", () => {
    const other = selects[1 - i];
    if (other.value === sel.value) { other.value = sel.dataset.prev; other.dataset.prev = other.value; }
    sel.dataset.prev = sel.value;
    store.set("eagle3-pair", JSON.stringify(selects.map(s => s.value)));
    SLOTS.forEach(k => { if (!slots[k].events.length) slots[k].label(selects[k].value); });
    renderTabs();
  }));

  // ---------- дорожка ----------
  function Lane(slot) {
    const root = document.querySelector(`.lane[data-slot="${slot}"]`);
    const q = k => root.querySelector(`[data-k="${k}"]`);
    const lane = { slot, root, q, events: [], trees: [], hist: [], model: null };
    lane.label = function (id) {
      const m = byId[id];
      lane.model = id;
      q("name").textContent = m ? tag(m) : "Модель " + (slot + 1);
      q("meta").textContent = describe(m);
      root.dataset.kind = m ? m.kind : "";
    };
    lane.reset = function (maxTokens, id) {
      lane.label(id);
      // DeepSeek-R1-Distill сначала рассуждает до </think>: этот кусок показываем приглушённым
      const t = targetOf(byId[id]);
      const thinking = /R1-Distill|thinking/i.test((t && (t.path || t.label)) || "");
      Object.assign(lane, { events: [], queue: [], trees: [], hist: [], tokens: 0, steps: 0, cum: 0, raw: "", recv: 0,
        inThink: thinking, sawThinkEnd: false, done: null, maxTokens });
      const out = q("out");
      out.innerHTML = '<span class="think"></span><span class="answer"></span>';
      lane.think = out.firstChild; lane.answer = out.lastChild;
      ["tps", "tau", "steps", "time"].forEach(k => (q(k).textContent = "—"));
      q("bar").style.width = "0";
      q("state").textContent = "";
      root.classList.remove("won");
    };
    lane.push = ev => { lane.events.push(ev); lane.queue.push(ev); };
    return lane;
  }
  const slots = SLOTS.map(Lane);
  const isDrafter = lane => byId[lane.model] && byId[lane.model].kind === "drafter";

  const fmt = (x, d = 1) => x.toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
  const plural = n => { const a = n % 10, b = n % 100; return a === 1 && b !== 11 ? "токен" : a >= 2 && a <= 4 && (b < 12 || b > 14) ? "токена" : "токенов"; };

  function appendText(lane, delta, n, stepNo) {
    if (!delta) return;
    const before = lane.raw.length;
    lane.raw += delta;
    if (before === 0 && /^\s*<think>/.test(delta)) lane.inThink = true;
    let thinkPart = "", answerPart = "";
    if (lane.inThink) {
      const end = lane.raw.indexOf("</think>");
      if (end >= 0) {
        const cut = end - before;
        thinkPart = cut > 0 ? delta.slice(0, cut) : "";
        answerPart = delta.slice(Math.max(0, cut + 8)).replace(/^\s+/, "");
        lane.inThink = false; lane.sawThinkEnd = true;
      } else thinkPart = delta;
    } else answerPart = delta;
    thinkPart = thinkPart.replace(/^\s*<think>\s*/, "");
    const add = (host, text) => {
      if (!text) return;
      const s = document.createElement("span");
      s.className = "c fresh" + (stepNo % 2 ? " alt" : "");
      s.textContent = text;
      if (n > 1) s.title = `${n} ${plural(n)} за один шаг`;
      host.appendChild(s);
      setTimeout(() => s.classList.remove("fresh"), 600);
    };
    add(lane.think, thinkPart);
    add(lane.answer, answerPart);
    const out = lane.q("out");
    if (out.scrollHeight - out.scrollTop - out.clientHeight < 80) out.scrollTop = out.scrollHeight;
  }

  function showStats(lane) {
    const s = lane.cum / 1000;
    lane.q("tps").textContent = s > 0 ? fmt(lane.tokens / s, 0) : "—";
    lane.q("tau").textContent = lane.steps ? fmt(lane.tokens / lane.steps, 2) : "—";
    lane.q("steps").textContent = lane.steps;
    lane.q("time").textContent = fmt(s, 2) + " с";
    lane.q("bar").style.width = Math.min(100, (100 * lane.tokens) / lane.maxTokens) + "%";
  }

  function apply(lane, ev) {
    if (ev.type === "queue") lane.q("state").textContent = `ждёт GPU, впереди ${ev.waiting}`;
    else if (ev.type === "start") lane.q("state").textContent = "генерирует…";
    else if (ev.type === "error") { lane.q("state").textContent = "ошибка: " + ev.message; lane.done = ev; }
    else if (ev.type === "step") {
      if (!lane.steps) lane.q("state").textContent = "генерирует…";
      lane.steps += ev.steps || 1; lane.tokens += ev.n_new; lane.cum += ev.dt_ms;
      appendText(lane, ev.delta, ev.n_new, lane.steps);
      if (isDrafter(lane)) {
        // кусок потока SGLang может объединить несколько проходов таргета: в гистограмму идут только одиночные
        if (!ev.steps || ev.steps === 1) lane.hist.push(ev.n_new);
        if (ev.tree) { lane.trees.push({ tree: ev.tree, accepted: ev.accepted, n: ev.n_new }); onTree(lane); }
        else if (lane.slot === treeState.slot) { drawHist(lane.hist); showAccepted(ev.n_new); }
      }
      showStats(lane);
    } else if (ev.type === "done") {
      lane.done = ev; lane.tokens = ev.tokens; lane.steps = ev.steps; lane.cum = ev.time_s * 1000;
      if (!lane.sawThinkEnd && lane.think.textContent) lane.think.className = "answer";
      showStats(lane);
      lane.q("state").textContent = ev.stop === "cancel" ? "остановлено" : ev.stop === "length" ? "финиш · лимит токенов" : "финиш";
      verdict();
    }
  }

  // ---------- проигрыватель: общие часы GPU для обеих дорожек ----------
  const clock = { t: 0, last: 0 };
  const speed = () => parseFloat($("speed").value) || 1;

  function drainControl(lane) {
    while (lane.queue.length && lane.queue[0].type !== "step") apply(lane, lane.queue.shift());
  }

  function tick(now) {
    const dt = Math.min(now - clock.last, 100);
    clock.last = now;
    let target = clock.t + dt * speed();
    for (const lane of slots) {
      drainControl(lane);
      // дорожка ещё считает, а событий нет: дальше её часов показывать нельзя
      if (!lane.done && !lane.queue.length) target = Math.min(target, Math.max(lane.cum, clock.t));
    }
    clock.t = Math.max(clock.t, target);
    let busy = false;
    for (const lane of slots) {
      while (lane.queue.length && lane.queue[0].type === "step" && lane.cum + lane.queue[0].dt_ms <= clock.t)
        apply(lane, lane.queue.shift());
      drainControl(lane);
      if (!lane.done || lane.queue.length) busy = true;
    }
    if (busy) requestAnimationFrame(tick);
    else finish();
  }

  function start(pair) {
    const maxT = parseInt($("max-tokens").value, 10);
    slots.forEach((lane, i) => lane.reset(maxT, pair[i]));
    $("verdict").hidden = true;
    treeState.slot = slots.find(isDrafter).slot;
    treeState.follow = true; treeState.idx = 0;
    renderTabs(); renderTree();
    clock.t = 0; clock.last = performance.now();
    running = true; syncButtons();
    requestAnimationFrame(tick);
  }

  function finish() {
    running = false;
    socket = null;
    if (backend && slots.every(l => l.done && l.done.type === "done" && l.done.stop !== "cancel")) {
      const key = [$("prompt").value, $("max-tokens").value, $("temperature").value].join("|");
      if (!session || session.key !== key)
        session = { key, prompt: $("prompt").value, info, max_new_tokens: +$("max-tokens").value,
          temperature: +$("temperature").value, runs: {} };
      session.recorded = new Date().toISOString();
      slots.forEach(l => (session.runs[l.model] = l.events.slice()));
    }
    syncButtons();
  }

  function syncButtons() {
    $("run").disabled = running || !(selects[0].value && selects[1].value);
    selects.forEach(s => (s.disabled = running));
    $("engine").disabled = running || !!waiting;
    $("stop").hidden = !running || !backend;
    $("again").hidden = running || !slots[0].events.length;
    const n = session ? Object.keys(session.runs).length : 0;
    $("save").hidden = running || !n;
    $("save").textContent = `Скачать запись (${n} ${n === 1 ? "модель" : n < 5 ? "модели" : "моделей"})`;
  }

  function verdict() {
    const [a, b] = slots;
    if (!a.done || !b.done || a.done.type !== "done" || b.done.type !== "done") return;
    const [fast, slow] = a.done.tok_s >= b.done.tok_s ? [a, b] : [b, a];
    fast.root.classList.add("won");
    $("speedup").textContent = fmt(fast.done.tok_s / Math.max(slow.done.tok_s, 1e-9), 2) + "×";
    $("verdict-text").textContent = "быстрее: " + tag(byId[fast.model]);
    $("verdict").hidden = false;
  }

  // ---------- дерево и гистограмма ----------
  const treeState = { slot: 0, follow: true, idx: 0 };
  const shown = () => slots[treeState.slot];

  function renderTabs() {
    const host = $("tree-tabs");
    host.textContent = "";
    const drafters = slots.filter(l => (l.model ? isDrafter(l) : byId[selects[l.slot].value] && byId[selects[l.slot].value].kind === "drafter"));
    if (!drafters.some(l => l.slot === treeState.slot) && drafters.length) treeState.slot = drafters[0].slot;
    const go = l => {
      treeState.slot = l.slot; treeState.follow = true; treeState.idx = Math.max(0, (l.trees || []).length - 1);
      renderTabs(); renderTree(false);
    };
    // два драфтера - две страницы в одной рамке: дерево и распределение каждого, листаются стрелками или вкладками
    const cur = Math.max(0, drafters.findIndex(l => l.slot === treeState.slot));
    const arrow = (txt, d, label) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "pg"; b.textContent = txt; b.setAttribute("aria-label", label);
      b.addEventListener("click", () => go(drafters[(cur + d + drafters.length) % drafters.length]));
      return b;
    };
    if (drafters.length > 1) host.appendChild(arrow("‹", -1, "Предыдущий драфтер"));
    drafters.forEach(l => {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.slot = l.slot;
      b.textContent = byId[l.model] ? tag(byId[l.model]) : "Модель " + (l.slot + 1);
      b.setAttribute("aria-pressed", l.slot === treeState.slot);
      b.addEventListener("click", () => go(l));
      host.appendChild(b);
    });
    if (drafters.length > 1) {
      host.appendChild(arrow("›", 1, "Следующий драфтер"));
      const n = document.createElement("span"); n.className = "pg-n mono"; n.textContent = `${cur + 1} / ${drafters.length}`;
      host.appendChild(n);
    }
  }

  function onTree(lane) {
    if (lane.slot !== treeState.slot) return;
    const n = lane.trees.length;
    $("step").max = n - 1; $("step").disabled = false;
    if (treeState.follow) { treeState.idx = n - 1; $("step").value = n - 1; renderTree(true); }
    else $("step-label").textContent = `${treeState.idx + 1} из ${n}`;
    drawHist(lane.hist);
  }
  function showAccepted(n) {
    $("k-acc").textContent = `${n} ${plural(n)}`;
    $("k-acc-sub").textContent = "за последний проход таргета";
  }
  function renderTree(animate) {
    const lane = shown(), item = lane.trees && lane.trees[treeState.idx];
    if (!item) {
      const noTree = info && info.engine && info.engine.tree === false;
      document.querySelector(".scrub").hidden = noTree;
      document.querySelector(".tree-head .legend").hidden = noTree;
      EagleTree.render($("tree"), null, [], { empty: noTree
        ? "SGLang не отдаёт дерево драфтера, поэтому здесь только число принятых токенов за шаг. Дерево видно с движком кода авторов (ENGINE = \"eagle\" в ноутбуке)."
        : undefined });
      $("step-label").textContent = "—"; $("k-acc").textContent = "—"; $("k-acc-sub").textContent = "";
      $("step").disabled = true; $("step").max = 0;
      drawHist(lane.hist || []);
      return;
    }
    EagleTree.render($("tree"), item.tree, item.accepted, { animate: animate && speed() < 1 });
    $("step").max = lane.trees.length - 1; $("step").disabled = false; $("step").value = treeState.idx;
    $("step-label").textContent = `${treeState.idx + 1} из ${lane.trees.length}`;
    $("k-acc").textContent = `${item.accepted.length - 1} из ${item.tree.t.length - 1}`;
    $("k-acc-sub").textContent = `узлов драфтера; вместе с токеном таргета шаг дал ${item.n} ${plural(item.n)}`;
    drawHist(lane.hist);
  }
  $("step").addEventListener("input", e => {
    treeState.idx = +e.target.value;
    treeState.follow = treeState.idx === shown().trees.length - 1;
    renderTree(false);
  });

  function drawHist(values) {
    const svg = $("hist"), W = 260, H = 150, B = 18, T = 26, NS = "http://www.w3.org/2000/svg";
    svg.textContent = "";
    const mk = (name, attrs, text) => {
      const e = document.createElementNS(NS, name);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (text !== undefined) e.textContent = text;
      svg.appendChild(e); return e;
    };
    // столбцы с градиентом от фиолетового к бирюзовому, как полосы лидерборда
    const lg = mk("linearGradient", { id: "hgrad", x1: 0, y1: 1, x2: 0, y2: 0 });
    [["0", "#8b5cf6"], ["1", "#67e8f9"]].forEach(([o, c]) => {
      const st = document.createElementNS(NS, "stop"); st.setAttribute("offset", o); st.setAttribute("stop-color", c); lg.appendChild(st);
    });
    const counts = new Array(HIST_MAX + 1).fill(0);
    values.forEach(v => counts[Math.max(1, Math.min(v, HIST_MAX))]++);
    const top = Math.max(1, ...counts), bw = (W - 4) / HIST_MAX, h0 = H - B, hMax = H - B - T;
    for (let k = 1; k <= HIST_MAX; k++) {
      const h = (hMax * counts[k]) / top, x = 2 + (k - 1) * bw;
      const r = mk("rect", { x: x + 3, y: h0 - h, width: bw - 6, height: counts[k] ? Math.max(h, 2) : 0, rx: 5, class: "hbar" });
      const tt = document.createElementNS(NS, "title"); tt.textContent = `${k} ${plural(k)} за шаг: ${counts[k]} раз`; r.appendChild(tt);
      mk("text", { x: x + bw / 2, y: H - 4, "text-anchor": "middle", class: "hlab" }, k);
      if (counts[k]) mk("text", { x: x + bw / 2, y: h0 - h - 4, "text-anchor": "middle", class: "hlab" }, counts[k]);
    }
    mk("line", { x1: 2, x2: W - 2, y1: h0, y2: h0, class: "hbase" });
    if (values.length) {
      const mean = values.reduce((s, v) => s + v, 0) / values.length;
      const x = 2 + (Math.min(mean, HIST_MAX) - 0.5) * bw;
      mk("line", { x1: x, x2: x, y1: 14, y2: h0, class: "hmean" });
      mk("text", { x: Math.min(Math.max(x, 50), W - 50), y: 10, "text-anchor": "middle", class: "hmean-lab" }, `среднее τ = ${fmt(mean, 2)}`);
    }
    $("tau-mean").textContent = values.length ? fmt(values.reduce((s, v) => s + v, 0) / values.length, 2) : "—";
  }

  // ---------- живой режим ----------
  const pair = () => selects.map(s => s.value);

  function runLive() {
    const p = pair();
    const req = { lanes: p, prompt: $("prompt").value, max_new_tokens: +$("max-tokens").value, temperature: +$("temperature").value };
    start(p);
    const ws = new WebSocket(backend.replace(/^http/, "ws") + "/api/run");
    socket = ws;
    ws.onopen = () => ws.send(JSON.stringify(req));
    ws.onmessage = e => {
      const ev = JSON.parse(e.data);
      const lane = slots.find(l => l.model === ev.lane);
      if (lane && ev.type === "loading") { lane.q("state").textContent = ev.message + "…"; return; }
      if (lane) {
        lane.push(ev);
      }
      else if (ev.type === "queue" || ev.type === "error") slots.forEach(l => l.push({ ...ev }));
    };
    ws.onclose = () => slots.forEach(l => {
      if (!l.events.some(ev => ev.type === "done" || ev.type === "error")) l.push({ type: "error", message: "соединение прервано" });
    });
  }

  // любые две модели из записи: у каждой свои часы GPU, поэтому пара собирается из отдельных прогонов
  function playRecord(rec, p, keepPrompt) {
    if (!keepPrompt && rec.prompt) $("prompt").value = rec.prompt;
    if (rec.max_new_tokens) $("max-tokens").value = String(rec.max_new_tokens);
    if (rec.temperature !== undefined) $("temperature").value = String(rec.temperature);
    start(p);
    slots.forEach((l, i) => {
      const evs = rec.runs[p[i]];
      if (evs) evs.forEach(ev => l.push({ ...ev }));
      else l.push({ type: "error", message: "этой модели нет в записи" });
    });
  }

  // ---------- кнопки ----------
  $("run").addEventListener("click", () => (backend ? runLive() : replay && playRecord(replay, pair())));
  $("again").addEventListener("click", () => {
    const p = slots.map(l => l.model);
    playRecord({ runs: Object.fromEntries(slots.map(l => [l.model, l.events.slice()])) }, p, true);
  });
  $("stop").addEventListener("click", () => { try { socket && socket.send(JSON.stringify({ type: "cancel" })); } catch { /* закрыт */ } });
  $("save").addEventListener("click", () => {
    if (!session) return;
    const { key, ...rec } = session;
    const blob = new Blob([JSON.stringify(rec)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "eagle3-run.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $("chips").addEventListener("click", e => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    $("prompt").value = chip.dataset.prompt;
    document.querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", c === chip));
  });
  $("srv-form").addEventListener("submit", e => {
    e.preventDefault();
    const url = $("srv-url").value.trim().replace(/\/+$/, "");
    if (url) connect([url]);
  });

  // ---------- фоновое видео и музыка ----------
  // Файлы лежат на демо-сервере (папка videos с Google Диска, см. ноутбук), иначе в site/assets/ (имена в assets/background.js).
  // Музыка - первый mp3 (или другой аудиофайл) из той же папки, по кругу; видео тогда всегда без звука.
  // Если аудиофайла нет, звучит само видео.
  const bgVideo = $("bg-video"), bgAudio = $("bg-audio");
  const snd = () => (bgAudio.src ? bgAudio : bgVideo);   // что сейчас даёт звук
  const fileUrl = (base, name) => base + "/videos/" + name.split("/").map(encodeURIComponent).join("/");
  async function loadBackground(base) {
    const local = window.BACKGROUND || {};
    const asset = name => (name ? "assets/" + name.split("/").map(encodeURIComponent).join("/") : "");
    let src = asset(local.video), music = asset(local.music);
    if (base) {
      try {
        const res = await (await fetch(base + "/api/videos")).json();
        if ((res.videos || [])[0]) src = fileUrl(base, res.videos[0]);
        if ((res.audio || [])[0]) music = fileUrl(base, res.audio[0]);
      } catch { /* у сервера видео нет */ }
    }
    const v = bgVideo;
    if (v.dataset.src === src && bgAudio.dataset.src === music) return;
    v.dataset.src = src; bgAudio.dataset.src = music;
    v.loop = true; bgAudio.loop = true;
    v.onloadeddata = () => { document.body.classList.add("has-bg"); syncSound(); };
    v.onerror = () => { document.body.classList.remove("has-bg"); v.removeAttribute("src"); syncSound(); };
    if (src) v.src = src; else { v.removeAttribute("src"); document.body.classList.remove("has-bg"); }
    if (music) { bgAudio.src = music; v.muted = true; } else bgAudio.removeAttribute("src");
    syncSound();
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    v.play().catch(() => { v.muted = true; v.play().catch(() => {}); });
    // со звуком браузер разрешает запуск только после действия пользователя:
    // пробуем сразу, иначе ждём первого клика или клавиши
    const a = snd();
    a.muted = false;
    a.play().catch(() => { a.muted = true; if (a === v) v.play().catch(() => {}); }).finally(syncSound);
  }
  function syncSound() {
    const a = snd(), b = $("sound-btn");
    const on = !a.muted && !(a === bgAudio && a.paused);
    b.hidden = !document.body.classList.contains("has-bg") && !bgAudio.src;
    b.textContent = on ? "Звук: вкл" : "Звук: выкл";
    b.setAttribute("aria-pressed", String(on));
  }
  function setSound(on) {
    const a = snd();
    a.muted = !on;
    if (on && a.paused) a.play().catch(() => { a.muted = true; }).finally(syncSound);
    syncSound();
  }
  $("sound-btn").addEventListener("click", e => {
    e.stopPropagation();
    const a = snd();
    userSound = true;
    setSound(a.muted || (a === bgAudio && a.paused));
  });
  let userSound = false;   // пользователь сам выбрал звук кнопкой - больше не трогаем
  const unmute = () => {
    const a = snd();
    if (!userSound && a.src && (a.muted || a.paused)) setSound(true);
    removeEventListener("pointerdown", unmute); removeEventListener("keydown", unmute);
  };
  addEventListener("pointerdown", unmute); addEventListener("keydown", unmute);
  [bgVideo, bgAudio].forEach(el => ["volumechange", "play", "pause"].forEach(t => el.addEventListener(t, syncSound)));

  // ---------- команда ----------
  (function renderTeam() {
    const hues = ["var(--accept)", "var(--target)", "var(--win)", "var(--reject)"];
    const host = $("team");
    (window.TEAM || []).forEach((p, i) => {
      const card = document.createElement("div");
      card.className = "person" + (p.video ? " on-video" : "");
      if (p.photo) {
        const ava = document.createElement("div");
        ava.className = "ava"; ava.style.background = hues[i % hues.length];
        const img = document.createElement("img"); img.src = p.photo; img.alt = ""; ava.appendChild(img);
        card.append(ava);
      }
      const name = document.createElement("b"); name.textContent = p.name;
      const role = document.createElement("span"); role.className = "role"; role.textContent = p.role;
      card.append(name);
      if (p.role) card.append(role);
      if (p.about) { const about = document.createElement("p"); about.textContent = p.about; card.append(about); }
      host.appendChild(card);
      if (p.video) {   // подпись под человеком на фоновом видео
        const cap = $("video-person");
        cap.append(name.cloneNode(true));
        if (p.role) cap.append(role.cloneNode(true));
        cap.hidden = false;
        // ПК: подпись появляется, когда до заголовка «Команда» докрутили (он на экране или выше).
        // Телефон и планшет: когда пустой экран после команды (mentor-gap) открыт больше чем наполовину.
        const head = document.querySelector(".team-sec h2"), gap = $("mentor-gap");
        const narrow = matchMedia("(max-width: 1199.98px)");
        const sync = () => {
          let on;
          if (narrow.matches) {
            on = gap.offsetHeight > 0 && gap.getBoundingClientRect().top < innerHeight * .5;
            // подпись стоит внизу экрана, а в самом конце страницы - над подвалом со ссылками
            const foot = document.querySelector("footer"), seen = foot ? innerHeight - foot.getBoundingClientRect().top : 0;
            cap.style.bottom = Math.max(76, seen + 14) + "px";
          } else {
            on = !head || head.getBoundingClientRect().top < innerHeight;
            cap.style.bottom = "";
          }
          cap.classList.toggle("shown", on);
          document.body.classList.toggle("mentor-view", on && narrow.matches);
        };
        addEventListener("scroll", sync, { passive: true }); addEventListener("resize", sync); sync();
      }
    });
  })();

  // ---------- конфигурация стенда ----------
  function showConfig(inf) {
    const t = inf.tree || {}, eng = inf.engine || {};
    const targets = (inf.models || []).filter(m => m.kind === "target");
    $("cfg-target").textContent = targets.length ? targets.map(m => m.label).join(", ") : "—";
    $("cfg-engine").textContent = eng.name ? `${eng.name}${eng.version ? " · " + eng.version : ""}` : "—";
    $("cfg-gpu").textContent = inf.gpu || "—";
    $("cfg-tree").textContent = t.total_token ? `${t.total_token} узлов · глубина ${t.depth} · top-k ${t.top_k}` : "—";
    const body = $("cfg-models");
    body.textContent = "";
    (inf.models || []).forEach(m => {
      const tr = document.createElement("tr");
      const norms = m.kind === "target" ? "—" : [m.fc_norm && "fc_norm", m.norm_output && "norm_output"].filter(Boolean).join(", ") || "нет";
      const own = (inf.models || []).find(x => x.id === m.target);
      const cells = [tag(m), m.kind === "target" ? "—" : (own ? own.label : m.target || "—"), m.path || "—",
        m.kind === "target" ? "—" : (m.layers || []).join(", "), norms];
      cells.forEach((c, i) => {
        const td = document.createElement("td");
        td.textContent = c;
        if (i === 2) td.className = "mono";
        tr.appendChild(td);
      });
      if (m.error) { tr.className = "off"; tr.title = "не загрузился: " + m.error; tr.cells[0].textContent += " (не загрузился)"; }
      body.appendChild(tr);
    });
  }

  // ---------- подключение ----------
  async function probe(url) {
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 5000);
    try {
      const r = await fetch(url + "/api/info", { signal: ctl.signal });
      return r.ok ? await r.json() : null;
    } catch { return null; } finally { clearTimeout(t); }
  }
  function setPill(kind, text) { $("srv-pill").className = "pill " + kind; $("srv-text").textContent = text; }

  async function connect(candidates) {
    setPill("", "Проверяю демо-сервер…");
    for (const url of candidates) {
      if (!url) continue;
      const res = await probe(url);
      if (res && (res.state === "loading" || res.state === "switching")) { backend = url; waitEngine(url, res); return; }
      if (res && res.state === "failed") {
        backend = null; info = res;
        setPill("", res.switch && res.switch.message || "Движок не загрузился, см. лог сервера");
        showEngine(url, res);
        await loadReplay(true);
        return;
      }
      if (res && res.models && res.models.filter(m => !m.error).length >= 2) {
        backend = url; info = res; store.set("eagle3-backend", url);
        const n = res.models.filter(m => !m.error).length;
        setPill("live", `${res.mock ? "Имитация без GPU" : "Онлайн"} · ${n} ${n < 5 ? "модели" : "моделей"} · ${res.gpu}`);
        showConfig(res);
        showEngine(url, res);
        $("srv-url").value = url;
        $("run").textContent = "Запустить генерацию";
        setModels(res.models, m => !m.error);
        loadBackground(url);
        return;
      }
    }
    backend = null;
    await loadReplay();
    loadBackground(null);
  }

  // ---------- смена движка ----------
  // Сервер ставит зависимости другого движка и перезапускается с ним: порядка 10 минут без гонок.
  let engineUrl = null, engineInfo = null, waiting = null;
  const ENGINE_NOTE = {
    eagle: "Код авторов EAGLE показывает дерево драфтера на каждом шаге, модели генерируют по очереди шагами.",
    sglang: "SGLang быстрее и ближе к замерам команды, но дерево драфтера не показывает.",
  };
  const engineName = (res, id) => (res && res.engines && res.engines[id]) || id;
  function showEngine(url, res) {
    engineUrl = url; engineInfo = res;
    const box = $("engine-box");
    box.hidden = !(res && res.engines && res.engine_id);
    if (box.hidden) return;
    $("engine").value = res.engine_id;
    const err = res.switch && res.switch.error;
    if (err) warn(`Движок не сменился: ${err}. Остался ${engineName(res, res.engine_id)}.`, false);
    else $("engine-warn").hidden = true;
  }
  function warn(text, ask) {
    $("engine-warn-text").textContent = text;
    $("engine-yes").hidden = !ask;
    $("engine-no").textContent = ask ? "Отмена" : "Понятно";
    $("engine-warn").hidden = false;
  }
  $("engine").addEventListener("change", () => {
    const to = $("engine").value;
    if (!engineInfo || to === engineInfo.engine_id) { $("engine-warn").hidden = true; return; }
    warn(`Сменить движок на ${engineName(engineInfo, to)}? Сервер поставит зависимости и перезагрузит модели: ` +
      `это порядка 10 минут, и всё это время генерация недоступна. ${ENGINE_NOTE[to] || ""}`, true);
  });
  $("engine-no").addEventListener("click", () => {
    $("engine-warn").hidden = true;
    if (engineInfo && engineInfo.engine_id) $("engine").value = engineInfo.engine_id;
  });
  $("engine-yes").addEventListener("click", async () => {
    const to = $("engine").value, url = engineUrl;
    $("engine-warn").hidden = true;
    let res = null;
    try {
      const r = await fetch(url + "/api/engine", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ engine: to }) });
      res = await r.json();
    } catch { res = null; }
    if (!res || !res.ok) { warn("Сервер не принял смену движка" + (res && res.error ? ": " + res.error : ""), false); $("engine").value = engineInfo.engine_id; return; }
    backend = url;
    waitEngine(url, res);
  });

  function waitEngine(url, res) {
    const to = res.state === "switching" ? res.switch.to : res.engine_id;
    const t0 = performance.now() - 1000 * ((res.switch && res.switch.elapsed) || 0);
    waiting = { url, to };
    $("run").disabled = true; $("engine").disabled = true;
    $("engine-box").hidden = false; $("engine").value = to;
    $("engine-warn").hidden = true;
    let last = res;
    const mmss = () => { const s = Math.floor((performance.now() - t0) / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
    const show = () => {
      const sw = !last || last.state === "switching" || (res.state === "switching");
      const what = sw ? `Меняю движок на ${engineName(last || res, to)} · ${mmss()} из ~10 мин` : `Загружаю модели (${engineName(last, to)}) · ${mmss()}`;
      const msg = !last ? "перезапускаю сервер" : last.state === "switching" ? (last.switch && last.switch.message) || "" : sw ? "загружаю модели" : "";
      setPill("switching", what + (msg ? " · " + msg : ""));
    };
    show();
    const timer = setInterval(show, 1000);
    (async function poll() {
      for (;;) {
        await new Promise(r => setTimeout(r, 4000));
        if (!waiting || waiting.url !== url) break;
        const r = await probe(url);
        if (r) last = r; else if (last) last = null;
        if (r && (r.state === "ready" || r.state === "failed")) break;
      }
      clearInterval(timer);
      waiting = null;
      connect([url]);
    })();
  }

  async function loadReplay(keepPill) {
    try { replay = await (await fetch("replay/sample.json")).json(); } catch { replay = null; }
    if (replay && replay.info && replay.info.models) {
      info = replay.info;
      const n = Object.keys(replay.runs).length;
      if (!keepPill) setPill("replay", (replay.info.mock ? "Оффлайн · синтетический пример" : "Оффлайн · запись живого запуска") + ` · ${n} модели`);
      $("run").textContent = "Показать запись генерации";
      showConfig(replay.info);
      setModels(replay.info.models, m => !!replay.runs[m.id]);
    } else {
      setPill("", "Сервер не запущен, записи нет");
      $("run").disabled = true;
    }
  }

  const param = new URLSearchParams(location.search).get("backend");
  const same = location.protocol.startsWith("http") ? location.origin : null;
  // постоянная ссылка: страница на GitHub Pages находит сервер Colab по <meta name="eagle3-backend">,
  // где "gist:ID" - гист, в который ноутбук при каждом запуске пишет текущий адрес туннеля (или сам адрес)
  async function fixedBackend() {
    const v = ((document.querySelector('meta[name="eagle3-backend"]') || {}).content || "").trim();
    if (!v.startsWith("gist:")) return v;
    try {
      const r = await fetch("https://api.github.com/gists/" + v.slice(5), { cache: "no-store" });
      const f = (await r.json()).files["eagle3-backend.json"];
      return JSON.parse(f.content).url || "";
    } catch { return ""; }
  }
  const clean = u => u && u.replace(/\/+$/, "");
  fixedBackend().then(fixed => connect([clean(param), same, clean(fixed), store.get("eagle3-backend")]));
  renderTree();
})();
