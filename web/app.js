// Early-warning audit explorer. All numbers are precomputed aggregates in data.json.
"use strict";

const S = { year: "2014J", kind: "boosting", budget: "0.2", week: 4, every: "4", rounds: 3, auditWeek: "4",
            trace: "no activity in the last week" };
const TRACES = ["no activity in the last week", "missed an assessment deadline", "never used the forum",
                "never took a quiz", "no activity before the start"];
const DIM_LABEL = { deprivation: "Deprivation of home area", disability: "Disability", education: "Prior education" };
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
  $("week-out").textContent = S.week === 0 ? "0 (before the course starts)" : S.week;
  const items = [
    [w.n.toLocaleString(), "students still enrolled"],
    [pct(w.gone), "of the students who will fail or withdraw have already left"],
    [pct(precision), `of the ${Math.round(+S.budget * w.n).toLocaleString()} students contacted go on to fail or withdraw (random: ${pct(w.base)})`],
    [pct(reached), "of all the students who will fail or withdraw are reached this week"],
  ];
  if (S.kind === "marks") items.push([pct(w.mark), "of enrolled students have a mark back"]);
  tiles($("reach-tiles"), items);

  const parts = ["Already withdrawn", "Flagged this week", "Still enrolled, not flagged"];
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
    x: { label: "End of week →", domain: [0, 20], ticks: 11 },
    y: { label: `Share of the ${Y.eventual.toLocaleString()} who fail or withdraw`, domain: [0, 1], tickFormat: "%", grid: true },
    color: { domain: parts, range: [css("--gone"), css("--blue"), css("--peach")], legend: true },
    marks: [
      Plot.areaY(rows, Plot.stackY({ x: "week", y: "v", fill: "part", order: parts })),
      Plot.ruleX([S.week], { stroke: css("--ink"), strokeWidth: 1.5 }),
      Plot.tip(summary, Plot.pointerX({ x: "week", y: d => d.gone + d.reached,
        title: d => `Week ${d.week}\nAlready withdrawn ${pct(d.gone)}\nFlagged this week ${pct(d.reached)}\nNot flagged ${pct(d.missed)}\nAUC ${d.auc.toFixed(2)}` })),
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
  $("rounds-out").textContent = S.rounds;
  const [reached, contacted, precision] = grid[`${every}x${S.rounds}`];
  const random = Y.grid[S.budget].random;
  const [rReached] = random[`${every}x${S.rounds}`];
  const weeks = Array.from({ length: S.rounds }, (_, i) => i * every);
  const listed = weeks.length === 1 ? `week ${weeks[0]}` : `weeks ${weeks.slice(0, -1).join(", ")} and ${weeks.at(-1)}`;
  tiles($("campaign-tiles"), [
    [pct(reached), `of the students who will fail or withdraw are reached (contacts at ${listed})`],
    [pct(contacted), `of the ${Y.cohort.toLocaleString()} students are contacted at least once`],
    [pct(precision), "of the students contacted needed it"],
    [pct(rReached), "reached by the same rounds with students picked at random"],
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
    x: { label: "Students contacted, share of the cohort →", domain: [0, 1], tickFormat: "%", grid: true },
    y: { label: "↑ Students who fail or withdraw, reached", domain: [0, 1], tickFormat: "%", grid: true },
    color: { domain: ["every week", "every 2 weeks", "every 4 weeks", "every 8 weeks"],
             range: [shades[1], shades[2], shades[4], shades[8]], legend: true },
    marks: [
      Plot.line(rand, { x: "contacted", y: "reached", stroke: css("--muted"), strokeDasharray: "3,3" }),
      Plot.text([rand[Math.floor(rand.length / 3)]], { x: "contacted", y: "reached", text: () => "students picked at random", fill: css("--muted"), dy: 16, dx: 6, textAnchor: "start" }),
      Plot.line(pts_, { x: "contacted", y: "reached", stroke: "label", z: "label", strokeWidth: 2 }),
      Plot.dot(pts_, { x: "contacted", y: "reached", fill: "label", r: 4, stroke: css("--bg"),
        tip: true, title: d => `${d.label}, ${d.n} round${d.n > 1 ? "s" : ""}\nReached ${pct(d.reached)}\nContacted ${pct(d.contacted)}\nNeeded it ${pct(d.precision)}` }),
      Plot.dot([sel], { x: "contacted", y: "reached", r: 9, stroke: css("--ink"), strokeWidth: 2, fill: "none" }),
    ],
  }));
}

// ---------- 3. who is missed ----------
function renderProfile() {
  const prof = DATA.years[S.year].audits[S.auditWeek].profile;
  const rows = [
    ["passed", "Went on to pass", ""],
    ["at risk, flagged", "At risk, flagged", ""],
    ["failed, not flagged", "Will fail, not flagged", "missed"],
    ["withdrew later, not flagged", "Will withdraw later, not flagged", "missed"],
  ];
  const cols = [["active_last_week", "Active last week"], ["median_clicks_pct", "Median click rank in course"],
                ["missed_a_deadline", "Missed a deadline"], ["failed_a_mark", "Failed a mark (of those with one back)"]];
  const bar = v => `<div class="bar"><span style="width:${(100 * (v ?? 0)).toFixed(1)}%"></span><em>${pct(v)}</em></div>`;
  const body = rows.map(([k, name, cls]) => {
    const r = prof[k];
    return `<tr class="${cls}"><td>${name}<br><span class="note">${r.n.toLocaleString()} students</span></td>${cols.map(([c]) => `<td class="num">${bar(r[c])}</td>`).join("")}</tr>`;
  }).join("");
  $("profile").innerHTML = `<div class="scroll"><table class="profile"><thead><tr><th>At the end of week ${S.auditWeek}</th>${cols.map(([, h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>
    <p class="note">Sections 3 to 5 use the clicks-and-timing warning with a 20% contact list, whatever the settings above. The two "not flagged" rows are the students it misses.</p>`;
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
  groupPlots($("miss-chart"), subs, (sub, dim, width, margin) => Plot.plot({
    width, height: 34 * sub.length + 34, marginLeft: margin, style: plotStyle(),
    x: { domain: [0.4, 0.9], tickFormat: "%", grid: true, label: null },
    y: { domain: sub.map(r => r.label), label: null },
    marks: [
      Plot.ruleX([overall], { stroke: css("--muted"), strokeDasharray: "3,3" }),
      Plot.link(sub, { x1: d => d.miss_ci[0], x2: d => d.miss_ci[1], y1: "label", y2: "label", stroke: css("--blue"), strokeOpacity: 0.45, strokeWidth: 3 }),
      Plot.dot(sub, { x: "miss", y: "label", fill: css("--blue"), r: 5, stroke: css("--bg"),
        tip: true, title: d => `${d.label}\n${pct(d.miss)} of at-risk students missed [${pct(d.miss_ci[0])}, ${pct(d.miss_ci[1])}]\n${d.n.toLocaleString()} students, ${pct(d.at_risk)} at risk` }),
    ],
  }));
  const cal = A.calibration.filter(r => r.dimension !== "all").map(r => ({ ...r, label: groupLabel(r.dimension, r.group) }));
  groupPlots($("calib-chart"), cal, (sub, dim, width, margin) => Plot.plot({
    width, height: 34 * sub.length + 34, marginLeft: margin, style: plotStyle(),
    x: { domain: [0.25, 0.65], tickFormat: "%", grid: true, label: null },
    y: { domain: sub.map(r => r.label), label: null },
    color: { domain: ["predicted", "actual"], range: [css("--blue"), css("--orange")], legend: dim === "deprivation" },
    marks: [
      Plot.link(sub, { x1: "predicted", x2: "observed", y1: "label", y2: "label", stroke: css("--gone"), strokeWidth: 4 }),
      Plot.dot(sub, { x: "predicted", y: "label", fill: () => "predicted", r: 5, stroke: css("--bg") }),
      Plot.dot(sub, { x: "observed", y: "label", fill: () => "actual", r: 5, stroke: css("--bg"),
        tip: true, title: d => `${d.label}\nPredicted ${pct(d.predicted)}\nActually failed or withdrew ${pct(d.observed)}` }),
    ],
  }));
}

// ---------- 5. missing traces ----------
function renderTraces() {
  const A = DATA.years[S.year].audits[S.auditWeek];
  const rows = A.traces.filter(t => t.trace === S.trace).map(t => ({ ...t, label: groupLabel(t.dimension, t.group) }));
  const card = A.scorecard.find(c => c.trace === S.trace);
  const dis = rows.find(r => r.dimension === "disability" && r.group === "declared");
  let note = "";
  if (card) note += `Across all ${S.year === "2014J" ? 22 : 15} course runs, students without this trace are ${pts(card.difference)} points more likely to fail or withdraw (${pts(card.run_min)} to ${pts(card.run_max)} inside individual runs). `;
  if (dis) note += `In the test year it adds ${pts(dis.difference)} points for students who declared a disability, ${pts(-dis.gap)} points ${dis.gap < 0 ? "less" : "more"} than for everyone else [${pts(-dis.gap_ci[1])}, ${pts(-dis.gap_ci[0])}].`;
  $("trace-note").textContent = note;
  groupPlots($("trace-chart"), rows, (sub, dim, width, margin) => Plot.plot({
    width, height: 34 * sub.length + 46, marginLeft: margin, style: plotStyle(),
    x: { domain: [0, 0.6], tickFormat: d => `+${Math.round(100 * d)}`, grid: true, label: "Extra risk, percentage points →" },
    y: { domain: sub.map(r => r.label), label: null },
    marks: [
      Plot.link(sub, { x1: d => d.difference_ci[0], x2: d => d.difference_ci[1], y1: "label", y2: "label", stroke: css("--orange"), strokeOpacity: 0.45, strokeWidth: 3 }),
      Plot.dot(sub, { x: "difference", y: "label", fill: css("--orange"), r: 5, stroke: css("--bg"),
        tip: true, title: d => `${d.label}\nWithout the trace ${pct(d.risk_absent)} fail or withdraw, with it ${pct(d.risk_present)}\n${pct(d.share_absent)} of the group lack it` }),
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
  traceSeg.innerHTML = TRACES.map(t => `<button data-value="${t}">${t.charAt(0).toUpperCase() + t.slice(1)}</button>`).join("");
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
