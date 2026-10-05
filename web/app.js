(function () {
  "use strict";
  const M = window.CurveMath;
  const app = document.getElementById("app");
  const D = window.SITE_DATA;
  const NAMES = { rf: "Random Forest", lgbm: "LightGBM", xgb: "XGBoost" };
  const ORDER = ["lgbm", "xgb", "rf"];           // fixed slot order: series 1,2,3 never re-assigned
  const COLORS = { lgbm: "var(--s1)", xgb: "var(--s2)", rf: "var(--s3)" };
  const SHAPES = { lgbm: "circle", xgb: "square", rf: "diamond" };   // secondary encoding (not color alone)
  const NS = "http://www.w3.org/2000/svg";

  document.getElementById("theme").addEventListener("click", () => {
    const r = document.documentElement, cur = r.getAttribute("data-theme");
    const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    r.setAttribute("data-theme", dark ? "light" : "dark");
  });

  if (!D || !D.subsets || !Object.keys(D.subsets).length) {
    app.innerHTML = '<div class="empty">No score data yet. Run <code>python -m rulpm.train_baseline --subset FD001 --task fail</code> then <code>python -m rulpm.build_site</code>.</div>';
    return;
  }

  const state = { subset: Object.keys(D.subsets)[0], threshold: 0.5, visible: new Set(ORDER) };
  const el = (tag, attrs = {}, text) => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; };
  const sv = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const f2 = (v) => (v == null ? "–" : v.toFixed(2)), f3 = (v) => (v == null ? "–" : v.toFixed(3));

  // ---- layout -------------------------------------------------------------
  const controls = el("div", { class: "controls" });
  const subSel = el("select", { id: "subset" });
  Object.keys(D.subsets).forEach((s) => subSel.appendChild(el("option", { value: s }, s)));
  controls.appendChild(Object.assign(el("label"), { innerHTML: "Subset " })).appendChild(subSel);
  const slider = el("div", { class: "slider" });
  const range = el("input", { type: "range", min: "0", max: "1", step: "0.01", value: "0.5", id: "thr", "aria-label": "Decision threshold" });
  const tval = el("span", { class: "tval", "aria-live": "polite" }, "0.50");
  slider.appendChild(el("label", { for: "thr" }, "Threshold"));
  slider.appendChild(range); slider.appendChild(tval);
  controls.appendChild(slider);
  const tunedBtn = el("button", { type: "button" }, "Use validation-tuned threshold");
  controls.appendChild(tunedBtn);
  app.appendChild(controls);

  const legend = el("div", { class: "legend", role: "group", "aria-label": "Models" });
  app.appendChild(legend);
  const charts = el("div", { class: "charts" });
  app.appendChild(charts);
  const tableWrap = el("div", { class: "tablewrap" });
  app.appendChild(tableWrap);
  const verify = el("div", { class: "verify" });
  app.appendChild(verify);
  app.appendChild(el("p", { class: "note" }, "Test set: each engine's last observed cycle. The 'validation-tuned' threshold was chosen on out-of-fold training predictions (highest threshold meeting the target recall), never on these test engines. Small test sets make model differences noisy."));

  function shapeEl(kind, cx, cy, r, color, ring) {
    let e;
    if (kind === "circle") e = sv("circle", { cx, cy, r });
    else if (kind === "square") e = sv("rect", { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 1 });
    else e = sv("path", { d: `M${cx},${cy - r * 1.25} L${cx + r * 1.25},${cy} L${cx},${cy + r * 1.25} L${cx - r * 1.25},${cy} Z` });
    e.setAttribute("fill", color);
    if (ring) { e.setAttribute("stroke", "var(--panel)"); e.setAttribute("stroke-width", "2"); }
    return e;
  }

  legend.replaceChildren(...ORDER.map((m) => {
    const lab = el("label");
    const cb = el("input", { type: "checkbox", checked: "", "data-m": m, "aria-label": NAMES[m] });
    cb.addEventListener("change", () => { cb.checked ? state.visible.add(m) : state.visible.delete(m); render(); });
    const s = sv("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
    s.appendChild(shapeEl(SHAPES[m], 8, 8, 5, COLORS[m], false));
    lab.append(cb, s, document.createTextNode(NAMES[m]));
    return lab;
  }));

  // ---- chart --------------------------------------------------------------
  const W = 420, H = 360, L = 48, R = 14, T = 12, B = 42;
  const px = (v) => L + v * (W - L - R), py = (v) => H - B - v * (H - T - B);

  function makeChart(id, title, hint, xlab, ylab) {
    const card = el("div", { class: "card" });
    card.append(el("h2", {}, title), el("p", { class: "hint" }, hint));
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0", "aria-label": `${title}. Drag to change the threshold; arrow keys also work.`, id });
    const tip = el("div", { class: "tip" });
    card.append(svg, tip);
    charts.appendChild(card);
    return { svg, tip, card, xlab, ylab };
  }
  const rocC = makeChart("roc", "ROC curve", "True-positive rate vs false-positive rate", "False-positive rate", "True-positive rate (recall)");
  const prC = makeChart("pr", "Precision-recall curve", "Precision vs recall (better when failures are rare)", "Recall", "Precision");

  function drawFrame(c) {
    const s = c.svg; s.replaceChildren();
    for (let i = 0; i <= 5; i++) {
      const v = i / 5;
      s.append(sv("line", { x1: px(v), x2: px(v), y1: py(0), y2: py(1), stroke: "var(--grid)", "stroke-width": 1 }),
               sv("line", { x1: px(0), x2: px(1), y1: py(v), y2: py(v), stroke: "var(--grid)", "stroke-width": 1 }));
      const tx = sv("text", { x: px(v), y: py(0) + 16, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)" }); tx.textContent = v.toFixed(1);
      const ty = sv("text", { x: px(0) - 8, y: py(v) + 4, "text-anchor": "end", "font-size": 11, fill: "var(--muted)" }); ty.textContent = v.toFixed(1);
      s.append(tx, ty);
    }
    const xl = sv("text", { x: (px(0) + px(1)) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: "var(--text2)" }); xl.textContent = c.xlab;
    const yl = sv("text", { x: 12, y: (py(0) + py(1)) / 2, "text-anchor": "middle", "font-size": 12, fill: "var(--text2)", transform: `rotate(-90 12 ${(py(0) + py(1)) / 2})` }); yl.textContent = c.ylab;
    s.append(xl, yl);
  }

  function entries() { return ORDER.filter((m) => D.subsets[state.subset][m]).map((m) => ({ m, d: D.subsets[state.subset][m] })); }

  function render() {
    const es = entries().map((e) => ({ ...e, cv: M.curves(e.d.y_true, e.d.proba) }));
    state.es = es;
    // charts
    drawFrame(rocC); drawFrame(prC);
    const base = es.length && es[0].cv.P ? es[0].cv.P / (es[0].cv.P + es[0].cv.N) : 0;
    rocC.svg.append(sv("line", { x1: px(0), y1: py(0), x2: px(1), y2: py(1), stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "4 4" }));
    prC.svg.append(sv("line", { x1: px(0), y1: py(base), x2: px(1), y2: py(base), stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "4 4" }));
    es.forEach(({ m, d, cv }) => {
      if (!state.visible.has(m)) return;
      const rocPath = cv.roc.map((q, i) => `${i ? "L" : "M"}${px(q.fpr).toFixed(1)},${py(q.tpr).toFixed(1)}`).join(" ");
      const prPath = cv.pr.map((q, i) => `${i ? "L" : "M"}${px(q.recall).toFixed(1)},${py(q.precision).toFixed(1)}`).join(" ");
      [[rocC, rocPath], [prC, prPath]].forEach(([c, path]) => c.svg.append(sv("path", { d: path, fill: "none", stroke: COLORS[m], "stroke-width": 2, "stroke-linejoin": "round" })));
    });
    es.forEach(({ m, d }) => {
      if (!state.visible.has(m)) return;
      const a = M.atThreshold(d.y_true, d.proba, state.threshold);
      const fpr = a.fp + a.tn ? a.fp / (a.fp + a.tn) : 0;
      rocC.svg.append(shapeEl(SHAPES[m], px(fpr), py(a.recall), 5, COLORS[m], true));
      if (a.precision != null) prC.svg.append(shapeEl(SHAPES[m], px(a.recall), py(a.precision), 5, COLORS[m], true));
    });
    renderTable(es); renderVerify(es);
  }

  function renderTable(es) {
    const rows = es.filter((e) => state.visible.has(e.m)).map(({ m, d, cv }) => {
      const a = M.atThreshold(d.y_true, d.proba, state.threshold);
      return `<tr><td><span class="dot" style="background:${COLORS[m]}"></span>${NAMES[m]}</td>
        <td>${f3(M.rocAuc(cv.roc))}</td><td>${f3(M.averagePrecision(cv.pr))}</td>
        <td>${f2(a.recall)}</td><td>${f2(a.precision)}</td><td>${a.flagged}</td><td>${a.fn}</td><td>${a.fp}</td></tr>`;
    }).join("");
    tableWrap.innerHTML = `<table><caption class="note" style="text-align:left;caption-side:top;padding:8px 10px 0">Metrics at threshold ${f2(state.threshold)} (${es[0] ? es[0].d.n : 0} test engines, ${es[0] ? es[0].d.positives : 0} failing within 30 cycles)</caption>
      <thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Recall</th><th>Precision</th><th>Flagged</th><th>Missed failures</th><th>False alarms</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderVerify(es) {
    const lines = es.map(({ m, d, cv }) => {
      const a = M.rocAuc(cv.roc), b = M.averagePrecision(cv.pr);
      const okA = d.python.roc_auc == null || Math.abs(a - d.python.roc_auc) < 1e-6;
      const okB = d.python.pr_auc == null || Math.abs(b - d.python.pr_auc) < 1e-6;
      return `<li>${NAMES[m]}: browser ROC-AUC ${f3(a)} vs Python ${f3(d.python.roc_auc)} &middot; PR-AUC ${f3(b)} vs Python ${f3(d.python.pr_auc)}
        <span class="${okA && okB ? "ok" : "bad"}">${okA && okB ? "✓ match" : "✗ MISMATCH"}</span>
        &middot; <a href="${d.file}" style="color:var(--accent)">scores</a> <code>sha256 ${d.sha256.slice(0, 12)}…</code></li>`;
    }).join("");
    verify.innerHTML = `<strong>Verify it yourself.</strong> This page recomputes both AUC values from the raw scores in the browser and compares them with the numbers Python printed. Check a file hash with <code>certutil -hashfile &lt;file&gt; SHA256</code> (Windows) or <code>shasum -a 256 &lt;file&gt;</code>. Rebuild everything from the public NASA files with <code>python -m rulpm.train_baseline --subset ${state.subset} --task fail</code>.<ul>${lines}</ul>`;
  }

  // ---- threshold control --------------------------------------------------
  function setThreshold(v) {
    state.threshold = Math.min(1, Math.max(0, Math.round(v * 100) / 100));
    range.value = state.threshold; tval.textContent = state.threshold.toFixed(2); render();
  }
  range.addEventListener("input", () => setThreshold(parseFloat(range.value)));
  subSel.addEventListener("change", () => { state.subset = subSel.value; render(); });
  tunedBtn.addEventListener("click", () => {
    const es = state.es.filter((e) => state.visible.has(e.m) && e.d.python.tuned_threshold != null);
    if (es.length) setThreshold(es[0].d.python.tuned_threshold);
  });

  function pointerPick(c, kind) {
    const nearest = (ev) => {
      const r = c.svg.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width * W - L) / (W - L - R), y = ((H - B) - (ev.clientY - r.top) / r.height * H) / (H - T - B);
      let best = null;
      state.es.filter((e) => state.visible.has(e.m)).forEach(({ m, cv }) => {
        (kind === "roc" ? cv.roc : cv.pr).forEach((q) => {
          const qx = kind === "roc" ? q.fpr : q.recall, qy = kind === "roc" ? q.tpr : q.precision;
          const dd = (qx - x) ** 2 + (qy - y) ** 2;
          if (!best || dd < best.dd) best = { dd, m, q, qx, qy };
        });
      });
      return best;
    };
    let dragging = false;
    c.svg.addEventListener("pointerdown", (ev) => { dragging = true; c.svg.setPointerCapture(ev.pointerId); apply(ev, true); });
    c.svg.addEventListener("pointerup", () => { dragging = false; });
    c.svg.addEventListener("pointerleave", () => { c.tip.style.display = "none"; });
    c.svg.addEventListener("pointermove", (ev) => apply(ev, dragging));
    function apply(ev, set) {
      const b = nearest(ev); if (!b) return;
      const thr = isFinite(b.q.thr) ? b.q.thr : 1;
      const cr = c.card.getBoundingClientRect();
      c.tip.style.display = "block";
      c.tip.style.left = Math.min(ev.clientX - cr.left + 12, cr.width - 190) + "px";
      c.tip.style.top = ev.clientY - cr.top + 12 + "px";
      c.tip.textContent = `${NAMES[b.m]} · threshold ${thr.toFixed(2)} · ${kind === "roc" ? "FPR" : "recall"} ${b.qx.toFixed(2)} · ${kind === "roc" ? "TPR" : "precision"} ${b.qy.toFixed(2)}`;
      if (set) setThreshold(thr);
    }
    c.svg.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowRight" || ev.key === "ArrowUp") { setThreshold(state.threshold + 0.01); ev.preventDefault(); }
      if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") { setThreshold(state.threshold - 0.01); ev.preventDefault(); }
    });
  }
  pointerPick(rocC, "roc"); pointerPick(prC, "pr");
  render();
})();
