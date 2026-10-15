/* Рисует дерево драфтера EAGLE: корень слева (токен таргет-модели), глубина вправо.
   tree = {t: [строки токенов], par: [индекс родителя, -1 у корня], p: [вероятность пути] | null}
   accepted = индексы узлов принятого пути (включая корень). */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const COL = 112, ROW = 22, BOX_H = 18, PAD = 12, TOP = 22;

  function label(tok) {
    let s = tok.replace(/\n/g, "↵").replace(/\t/g, "⇥");
    if (s.startsWith(" ")) s = "␣" + s.slice(1);
    if (s === "") s = "∅";
    return s.length > 11 ? s.slice(0, 10) + "…" : s;
  }

  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function layout(tree, accepted) {
    const n = tree.t.length, kids = Array.from({ length: n }, () => []);
    const depth = new Array(n).fill(0);
    for (let i = 1; i < n; i++) if (tree.par[i] >= 0) kids[tree.par[i]].push(i);
    const acc = new Set(accepted);
    const p = tree.p || tree.t.map(() => 1);
    // принятая ветка сверху, остальные по убыванию вероятности
    kids.forEach(k => k.sort((a, b) => (acc.has(b) - acc.has(a)) || (p[b] - p[a])));
    const y = new Array(n).fill(0);
    let leaf = 0, maxD = 0;
    (function walk(i, d) {
      depth[i] = d; maxD = Math.max(maxD, d);
      if (!kids[i].length) { y[i] = leaf++; return; }
      kids[i].forEach(c => walk(c, d + 1));
      y[i] = (y[kids[i][0]] + y[kids[i][kids[i].length - 1]]) / 2;
    })(0, 0);
    return { kids, depth, y, leaves: Math.max(leaf, 1), maxD, acc, p };
  }

  function render(svgHost, tree, accepted, opts = {}) {
    svgHost.textContent = "";
    if (!tree) {
      const d = document.createElement("div");
      d.className = "tree-empty";
      d.textContent = opts.empty || "Дерево появится, когда начнётся генерация.";
      svgHost.appendChild(d);
      return;
    }
    const L = layout(tree, accepted);
    const W = PAD * 2 + (L.maxD + 1) * COL, H = TOP + PAD + L.leaves * ROW;
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Дерево драфтера: ${tree.t.length} узлов, принято ${accepted.length - 1}` });
    if (opts.animate) svg.classList.add("anim");
    for (let d = 0; d <= L.maxD; d++) {
      const t = el("text", { x: PAD + d * COL, y: 14, class: "axis" }, svg);
      t.textContent = d === 0 ? "таргет" : `глубина ${d}`;
    }
    const X = i => PAD + L.depth[i] * COL, Y = i => TOP + L.y[i] * ROW;
    const widths = tree.t.map(s => Math.min(COL - 14, 14 + label(s).length * 7.3));
    const edges = el("g", {}, svg), nodes = el("g", {}, svg);
    const accList = accepted.slice();
    const accSet = L.acc;
    // первый отвергнутый узел на продолжении принятого пути
    const last = accList[accList.length - 1];
    const miss = L.kids[last] ? L.kids[last][0] : undefined;
    for (let i = 1; i < tree.t.length; i++) {
      const pr = tree.par[i]; if (pr < 0) continue;
      const x1 = X(pr) + widths[pr], y1 = Y(pr) + BOX_H / 2, x2 = X(i), y2 = Y(i) + BOX_H / 2, mx = (x1 + x2) / 2;
      el("path", { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
        class: "edge" + (accSet.has(i) && accSet.has(pr) ? " acc" : "") }, edges);
    }
    const pMax = Math.max(...L.p.slice(1), 1e-9);
    for (let i = 0; i < tree.t.length; i++) {
      const cls = i === 0 ? "node root" : accSet.has(i) ? "node acc" : i === miss ? "node first-miss" : "node";
      const g = el("g", { class: cls, transform: `translate(${X(i)},${Y(i)})` }, nodes);
      if (opts.animate) g.style.animationDelay = `${L.depth[i] * 35}ms`;
      if (i > 0 && !accSet.has(i)) g.setAttribute("opacity", (0.35 + 0.65 * Math.pow(L.p[i] / pMax, 0.4)).toFixed(2));
      el("rect", { width: widths[i], height: BOX_H, rx: 4 }, g);
      const t = el("text", { x: 7, y: 13 }, g);
      t.textContent = label(tree.t[i]);
      const tip = el("title", {}, g);
      const role = i === 0 ? "токен таргет-модели" : accSet.has(i) ? "принят" : "отвергнут";
      tip.textContent = `${JSON.stringify(tree.t[i])} · ${role}` + (tree.p && i > 0 ? ` · p пути ${(tree.p[i] * 100).toFixed(1)}%` : "");
    }
    if (opts.fit) { svg.removeAttribute("width"); svg.removeAttribute("height"); }
    svgHost.appendChild(svg);
    if (!opts.fit) {
      // держим в кадре корень и принятую ветку
      const ys = accList.map(Y);
      const mid = (Math.min(...ys) + Math.max(...ys)) / 2;
      svgHost.scrollTop = Math.max(0, mid - svgHost.clientHeight / 2);
      svgHost.scrollLeft = 0;
    }
  }

  window.EagleTree = { render };
})();
