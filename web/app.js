// Early-warning audit explorer. All numbers are precomputed aggregates in data.json.
"use strict";

const S = { year: "2014J", kind: "boosting", budget: "0.2", week: 4, every: "4", rounds: 3, auditWeek: "4",
            trace: "no activity in the last week" };
const TRACES = ["no activity in the last week", "missed an assessment deadline", "never used the forum",
                "never took a quiz", "no activity before the start"];
const TRACE_LABEL = { "no activity in the last week": "Quiet week", "missed an assessment deadline": "Missed deadline",
                      "never used the forum": "No forum", "never took a quiz": "No quiz", "no activity before the start": "Late start" };
const DIM_LABEL = { deprivation: "Area deprivation", disability: "Disability", education: "Prior education" };
let DATA = null;

const pct = (x, d = 0) => (x == null || Number.isNaN(x)) ? "–" : `${(100 * x).toFixed(d)}%`;
const pts = x => `${x >= 0 ? "+" : "−"}${Math.abs(Math.round(100 * x))}`;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $ = id => document.getElementById(id);

function groupLabel(dim, g) {
  if (dim === "deprivation") return g === "unknown" ? "Not recorded" : g === "0-20%" ? "0-20% most deprived" : g === "80-100%" ? "80-100% least deprived" : g;
  if (dim === "disability") return g === "declared" ? "Declared" : "None declared";
  return g.charAt(0).toUpperCase() + g.slice(1);
}
const labelMargin = labels => Math.ceil(Math.max(...labels.map(l => l.length)) * 6.6) + 14;

function plotStyle() {
  return { background: "transparent", color: css("--ink2"), fontSize: "12px",
           fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif" };
}

function tiles(el, items) {
  el.innerHTML = items.map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
}

function mount(el, node) { el.replaceChildren(node); }

// ---------- 1. reach by week ----------
function renderReach() {
  const Y = DATA.years[S.year];
  const w = Y.weeks[S.week];
  const m = w.models[S.kind];
  const [precision, , reached] = m.b[S.budget];
  $("week-out").textContent = S.week === 0 ? "0 (before the start)" : S.week;
  const items = [
    [pct(w.gone), "of struggling students already gone"],
    [pct(precision), "of calls go to someone struggling"],
    [pct(reached), "of struggling students reached"],
  ];
  tiles($("reach-tiles"), items);

  const parts = ["Already gone", "Reached", "Missed"];
  const rows = [], summary = [];
  for (const wk of Y.weeks) {
    const r = wk.models[S.kind].b[S.budget][2];
    const vals = [wk.gone, r, 1 - wk.gone - r];
    parts.forEach((p, i) => rows.push({ week: wk.week, part: p, v: vals[i] }));
    summary.push({ week: wk.week, gone: wk.gone, reached: r, missed: 1 - wk.gone - r, auc: wk.models[S.kind].auc });
  }
  const el = $("reach-chart");
  mount(el, Plot.plot({
    width: el.clientWidth, height: 300, marginLeft: 46, style: plotStyle(),
    x: { label: "Week →", domain: [0, 20], ticks: 11 },
    y: { label: "↑ Struggling students", domain: [0, 1], tickFormat: "%", grid: true },
    color: { domain: parts, range: [css("--gone"), css("--blue"), css("--peach")], legend: true },
    marks: [
      Plot.areaY(rows, Plot.stackY({ x: "week", y: "v", fill: "part", order: parts })),
      Plot.ruleX([S.week], { stroke: css("--ink"), strokeWidth: 1.5 }),
      Plot.tip(summary, Plot.pointerX({ x: "week", y: d => d.gone + d.reached,
        title: d => `Week ${d.week}\nAlready gone ${pct(d.gone)}\nReached ${pct(d.reached)}\nMissed ${pct(d.missed)}` })),
    ],
  }));
}

// ---------- 2. repeated contacts ----------
function maxRounds(every) { return Math.min(6, Math.floor(20 / every) + 1); }

function renderCampaign() {
  const Y = DATA.years[S.year];
  const grid = Y.grid[S.budget][S.kind];
  const every = +S.every;
  const max = maxRounds(every);
  const slider = $("rounds");
  slider.max = max;
  if (S.rounds > max) S.rounds = max;
  slider.value = S.rounds;
  const wk = Array.from({ length: S.rounds }, (_, i) => i * every);
  $("rounds-out").textContent = `${S.rounds} (week ${wk.join(", ")})`;
  const [reached, contacted, precision] = grid[`${every}x${S.rounds}`];
  const random = Y.grid[S.budget].random;
  const [rReached] = random[`${every}x${S.rounds}`];
  tiles($("campaign-tiles"), [
    [pct(reached), "of struggling students reached"],
    [pct(rReached), "if tutors called at random"],
    [pct(contacted), "of all students called"],
  ]);

  const shades = { 1: css("--seq1"), 2: css("--seq2"), 4: css("--seq3"), 8: css("--seq4") };
  const pts_ = [];
  for (const e of [1, 2, 4, 8]) {
    for (let n = 1; n <= maxRounds(e); n++) {
      const g = grid[`${e}x${n}`];
      if (g) pts_.push({ every: e, n, reached: g[0], contacted: g[1], precision: g[2],
                         label: e === 1 ? "every week" : `every ${e} weeks` });
    }
  }
  const sel = pts_.find(p => p.every === every && p.n === S.rounds);
  const rand = Object.entries(random).map(([k, v]) => ({ k, reached: v[0], contacted: v[1] }))
    .sort((a, b) => a.contacted - b.contacted);
  const el = $("campaign-chart");
  mount(el, Plot.plot({
    width: el.clientWidth, height: 300, marginLeft: 46, style: plotStyle(),
    x: { label: "Students called →", domain: [0, 1], tickFormat: "%", grid: true },
    y: { label: "↑ Struggling students reached", domain: [0, 1], tickFormat: "%", grid: true },
    color: { domain: ["every week", "every 2 weeks", "every 4 weeks", "every 8 weeks"],
             range: [shades[1], shades[2], shades[4], shades[8]], legend: true },
    marks: [
      Plot.line(rand, { x: "contacted", y: "reached", stroke: css("--muted"), strokeDasharray: "3,3" }),
      Plot.text([rand[Math.floor(rand.length / 3)]], { x: "contacted", y: "reached", text: () => "random", fill: css("--muted"), dy: 16, dx: 6, textAnchor: "start" }),
      Plot.line(pts_, { x: "contacted", y: "reached", stroke: "label", z: "label", strokeWidth: 2 }),
      Plot.dot(pts_, { x: "contacted", y: "reached", fill: "label", r: 4, stroke: css("--bg"),
        tip: true, title: d => `${d.label}, ${d.n} round${d.n > 1 ? "s" : ""}\nReached ${pct(d.reached)}\nCalled ${pct(d.contacted)}` }),
      Plot.dot([sel], { x: "contacted", y: "reached", r: 9, stroke: css("--ink"), strokeWidth: 2, fill: "none" }),
    ],
  }));
}

// ---------- 3. who is missed ----------
function renderProfile() {
  const prof = DATA.years[S.year].audits[S.auditWeek].profile;
  const rows = [
    ["passed", "Passed", ""],
    ["at risk, flagged", "Flagged", ""],
    ["failed, not flagged", "Missed, will fail", "missed"],
    ["withdrew later, not flagged", "Missed, will drop out", "missed"],
  ];
  const cols = [["active_last_week", "Active last week"], ["missed_a_deadline", "Missed a deadline"],
                ["failed_a_mark", "Failed a mark"]];
  const bar = v => `<div class="bar"><span style="width:${(100 * (v ?? 0)).toFixed(1)}%"></span><em>${pct(v)}</em></div>`;
  const body = rows.map(([k, name, cls]) => {
    const r = prof[k];
    return `<tr class="${cls}"><td>${name}</td>${cols.map(([c]) => `<td class="num">${bar(r[c])}</td>`).join("")}</tr>`;
  }).join("");
  $("profile").innerHTML = `<div class="scroll"><table class="profile"><thead><tr><th></th>${cols.map(([, h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
}

// ---------- 4. groups ----------
function groupPlots(el, rows, build) {
  el.replaceChildren();
  const margin = labelMargin(rows.map(r => r.label));   // one margin for every panel, so the scales line up
  for (const dim of ["deprivation", "disability", "education"]) {
    const sub = rows.filter(r => r.dimension === dim);
    if (!sub.length) continue;
    const h = document.createElement("div");
    h.className = "note"; h.textContent = DIM_LABEL[dim];
    el.append(h, build(sub, dim, el.clientWidth, margin));
  }
}

function renderGroups() {
  const A = DATA.years[S.year].audits[S.auditWeek];
  const subs = A.subgroups.map(r => ({ ...r, label: groupLabel(r.dimension, r.group) }));
  const first = subs.filter(r => r.dimension === "deprivation");
  const overall = first.reduce((a, r) => a + r.miss * r.n * r.at_risk, 0) / first.reduce((a, r) => a + r.n * r.at_risk, 0);
  const named = subs.filter(r => r.group !== "unknown");
  const spread = Math.max(...named.map(r => Math.abs(r.miss - overall)));
  const poor = A.calibration.find(r => r.group === "0-20%"), rich = A.calibration.find(r => r.group === "80-100%");
  tiles($("miss-tiles"), [
    [pct(overall), `of struggling students missed, every group within ${Math.ceil(100 * spread)} points`],
    [`${pct(rich.observed)} → ${pct(poor.observed)}`, "real risk, richest to poorest areas"],
    [`${pct(rich.predicted)} → ${pct(poor.predicted)}`, "what the warning predicts"],
  ]);
  const cal = A.calibration.filter(r => r.dimension !== "all").map(r => ({ ...r, label: groupLabel(r.dimension, r.group) }));
  groupPlots($("calib-chart"), cal, (sub, dim, width, margin) => Plot.plot({
    width, height: 34 * sub.length + 34, marginLeft: margin, style: plotStyle(),
    x: { domain: [0.25, 0.65], tickFormat: "%", grid: true, label: null },
    y: { domain: sub.map(r => r.label), label: null },
    color: { domain: ["predicted", "real"], range: [css("--blue"), css("--orange")], legend: dim === "deprivation" },
    marks: [
      Plot.link(sub, { x1: "predicted", x2: "observed", y1: "label", y2: "label", stroke: css("--gone"), strokeWidth: 4 }),
      Plot.dot(sub, { x: "predicted", y: "label", fill: () => "predicted", r: 5, stroke: css("--bg") }),
      Plot.dot(sub, { x: "observed", y: "label", fill: () => "real", r: 5, stroke: css("--bg"),
        tip: true, title: d => `${d.label}\nPredicted ${pct(d.predicted)}\nReal ${pct(d.observed)}` }),
    ],
  }));
}

// ---------- 5. missing traces ----------
function renderTraces() {
  const A = DATA.years[S.year].audits[S.auditWeek];
  const rows = A.traces.filter(t => t.trace === S.trace && t.dimension === "disability")
    .map(t => ({ ...t, label: groupLabel(t.dimension, t.group) }));
  const dis = rows.find(r => r.dimension === "disability" && r.group === "declared");
  $("trace-note").textContent = dis
    ? `Extra risk: ${pts(dis.difference)} points with a declared disability, ${pts(dis.difference - dis.gap)} for everyone else.` : "";
  groupPlots($("trace-chart"), rows, (sub, dim, width, margin) => Plot.plot({
    width, height: 34 * sub.length + 46, marginLeft: margin, style: plotStyle(),
    x: { domain: [0, 0.6], tickFormat: d => `+${Math.round(100 * d)}`, grid: true, label: "Extra risk, points →" },
    y: { domain: sub.map(r => r.label), label: null },
    marks: [
      Plot.link(sub, { x1: d => d.difference_ci[0], x2: d => d.difference_ci[1], y1: "label", y2: "label", stroke: css("--orange"), strokeOpacity: 0.45, strokeWidth: 3 }),
      Plot.dot(sub, { x: "difference", y: "label", fill: css("--orange"), r: 5, stroke: css("--bg"),
        tip: true, title: d => `${d.label}\nWithout it ${pct(d.risk_absent)} struggle\nWith it ${pct(d.risk_present)}` }),
    ],
  }));
}

// ---------- wiring ----------
function renderAll() { renderReach(); renderCampaign(); renderProfile(); renderGroups(); renderTraces(); }

function setPressed() {
  document.querySelectorAll(".seg").forEach(seg => {
    seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.value === String(S[seg.dataset.key]))));
  });
}

function init() {
  const traceSeg = document.querySelector('.seg[data-key="trace"]');
  traceSeg.innerHTML = TRACES.map(t => `<button data-value="${t}">${TRACE_LABEL[t]}</button>`).join("");
  document.querySelectorAll(".seg").forEach(seg => seg.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    S[seg.dataset.key] = b.dataset.value;
    setPressed(); renderAll();
  }));
  $("week").addEventListener("input", e => { S.week = +e.target.value; renderReach(); });
  $("rounds").addEventListener("input", e => { S.rounds = +e.target.value; renderCampaign(); });
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderAll, 150); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  setPressed(); renderAll();
}

const theme = new URLSearchParams(location.search).get("theme");
if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;

fetch("data.json").then(r => r.json()).then(d => { DATA = d; init(); })
  .catch(err => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="note">Could not load the data (${err}).</p>`); });
