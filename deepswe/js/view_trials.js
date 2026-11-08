// ============================================================
// view_trials.js — trial table + trial detail (trajectory viewer)
// ============================================================
import { el, clear, parseQuery, setQuery, navigate, escapeHtml, renderDiff } from "./dom.js";
import {
  loadArtifact, SOURCE_DEEPSWE, SOURCE_CROSSBENCH,
  modelKey, modelLabel, modelMatches, sourceLabel,
  fmtCost, fmtInt, fmtCost4, fmtDuration,
} from "./data.js";
import { dataHeader } from "./view_common.js";

export async function renderTrials(mount) {
  const q = parseQuery();
  const source = q.source === SOURCE_CROSSBENCH ? SOURCE_CROSSBENCH : SOURCE_DEEPSWE;
  const isCross = source === SOURCE_CROSSBENCH;
  const dataset = q.dataset || "all";
  const outcome = q.outcome || "all";
  const modelFilter = q.model || "all";
  const critique = q.critique || "all";
  const search = q.q || "";

  const root = el("div.wrap");
  root.appendChild(dataHeader("trials"));
  mount.appendChild(root);

  const status = el("div.muted", { style: { marginTop: "1rem", fontSize: "12px" } }, "Loading trial index…");
  root.appendChild(status);

  let data;
  try { data = await loadArtifact("trials"); }
  catch (e) { status.textContent = `Couldn't load the trial index: ${e}`; return; }

  // model options scoped to current source/dataset
  const modelMap = new Map();
  for (const t of data.rows) {
    if (isCross ? t.eval_scope !== "cross-bench" : (t.eval_scope !== "full" || t.source !== SOURCE_DEEPSWE)) continue;
    if (isCross && dataset !== "all" && t.source !== dataset) continue;
    const k = modelKey(t.model, t.reasoning_effort);
    modelMap.set(k, { key: k, label: modelLabel(t.model, t.reasoning_effort) });
  }
  const modelOptions = Array.from(modelMap.values()).sort((a, b) => a.label.localeCompare(b.label));

  // filter
  const e = search.trim().toLowerCase();
  const filtered = data.rows.filter((t) => {
    if (isCross) { if (t.eval_scope !== "cross-bench") return false; if (dataset !== "all" && t.source !== dataset) return false; }
    else if (t.eval_scope !== "full" || t.source !== SOURCE_DEEPSWE) return false;
    if (modelFilter !== "all" && !modelMatches(modelFilter, t.model, t.reasoning_effort)) return false;
    if (outcome === "pass" && !t.passed) return false;
    if (outcome === "fail" && (t.passed || t.errored)) return false;
    if (outcome === "error" && !t.errored) return false;
    if (isCross && critique !== "all") {
      const r = t.critique?.rating ?? null;
      if (critique === "good" && r !== "good") return false;
      if (critique === "bad" && r !== "bad") return false;
    }
    if (e && ![t.task_name, t.model, modelLabel(t.model, t.reasoning_effort)].join(" ").toLowerCase().includes(e)) return false;
    return true;
  });

  status.remove();

  // ---- filter bar ----
  const bar = el("div.filter-bar");
  root.appendChild(bar);

  const topRow = el("div", { style: { display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "flex-end" } });
  const searchInput = el("input", { id: "trials-search", type: "search", placeholder: "Task or model", value: search, style: { minWidth: "18rem" } });
  let debounce;
  searchInput.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => setQuery({ q: searchInput.value || null }), 200); });
  topRow.appendChild(field("Search", searchInput));
  topRow.appendChild(field("Source", select(source, [
    { value: SOURCE_DEEPSWE, label: "DeepSWE" },
    { value: SOURCE_CROSSBENCH, label: "Cross-benchmark analysis" },
  ], (v) => setQuery({ source: v === SOURCE_DEEPSWE ? null : v, dataset: null, critique: null, model: null }, { replace: false }))));
  const countEl = el("div.count-right.tabular", {}, `${filtered.length} matching`);
  topRow.appendChild(countEl);
  bar.appendChild(topRow);

  const filterGrid = el("div.filter-grid", { style: { marginTop: ".75rem" } });
  if (isCross) {
    filterGrid.appendChild(field("Dataset", select(dataset, [
      { value: "all", label: "All datasets" }, { value: "deep-swe", label: "DeepSWE" }, { value: "swebenchpro", label: "SWE-Bench Pro" },
    ], (v) => setQuery({ dataset: v === "all" ? null : v }))));
  }
  filterGrid.appendChild(field("Outcome", select(outcome, [
    { value: "all", label: "All outcomes" }, { value: "pass", label: "Passed" }, { value: "fail", label: "Failed" }, { value: "error", label: "Errored" },
  ], (v) => setQuery({ outcome: v === "all" ? null : v }))));
  filterGrid.appendChild(field("Model", select(modelFilter, [
    { value: "all", label: "All models" }, ...modelOptions.map((m) => ({ value: m.key, label: m.label })),
  ], (v) => setQuery({ model: v === "all" ? null : v }))));
  if (isCross) {
    filterGrid.appendChild(field("Trajectory analysis", select(critique, [
      { value: "all", label: "All analysis" }, { value: "bad", label: "Verifier disagrees" }, { value: "good", label: "Verifier agrees" },
    ], (v) => setQuery({ critique: v === "all" ? null : v }))));
  }
  bar.appendChild(filterGrid);

  // ---- sortable table ----
  let sort = null; // {key, direction}
  const tableWrap = el("div.dtable-wrap");
  root.appendChild(tableWrap);

  const renderTable = () => {
    clear(tableWrap);
    const sorted = sort ? [...filtered].sort((a, b) => compareTrials(a, b, sort.key, sort.direction)) : filtered;
    const table = el("table.dtable");
    const cols = [
      isCross ? { key: "dataset", label: "Dataset" } : null,
      { key: "model", label: "Model" },
      { key: "task", label: "Task" },
      { key: "outcome", label: "Outcome" },
      isCross ? { key: "analysis", label: "Trajectory analysis" } : null,
      { key: "steps", label: "Steps", num: true },
      { key: "cost", label: "Cost", num: true },
    ].filter(Boolean);

    const thead = el("thead");
    const htr = el("tr");
    for (const c of cols) {
      const arrow = sort?.key === c.key ? (sort.direction === "asc" ? " ↑" : " ↓") : " ⇅";
      htr.appendChild(el("th", { class: c.num ? "num" : "" },
        el("button", { onclick: () => { const dir = sort?.key === c.key && sort.direction === "asc" ? "desc" : "asc"; sort = { key: c.key, direction: dir }; renderTable(); } }, [c.label, el("span.muted", {}, arrow)])));
    }
    thead.appendChild(htr);
    table.appendChild(thead);

    const tbody = el("tbody");
    const MAX = 2000;
    const slice = sorted.slice(0, MAX);
    for (const t of slice) {
      const tr = el("tr", { style: { cursor: "pointer" }, onclick: () => navigate(`/data/trials/${encodeURIComponent(t.trial_name)}`) });
      if (isCross) tr.appendChild(el("td.muted", {}, sourceLabel(t.source)));
      tr.appendChild(el("td", {}, modelLabel(t.model, t.reasoning_effort)));
      tr.appendChild(el("td.trunc", { title: t.task_name }, el("a.tasklink", {
        href: `#/data/tasks/${encodeURIComponent(t.task_name)}`,
        onclick: (ev) => ev.stopPropagation(),
      }, t.task_name)));
      tr.appendChild(el("td", {}, el("span", { class: "badge " + (t.passed ? "pass" : t.errored ? "outline" : "fail") }, t.passed ? "pass" : t.errored ? "error" : "fail")));
      if (isCross) tr.appendChild(el("td", {}, critiqueBadge(t.critique)));
      tr.appendChild(el("td.num", {}, fmtInt(t.n_agent_steps)));
      tr.appendChild(el("td.num", {}, fmtCost(t.cost_usd)));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    if (sorted.length > MAX) tableWrap.appendChild(el("div.muted", { style: { padding: ".75rem" } }, `Showing first ${MAX.toLocaleString()} of ${sorted.length.toLocaleString()} — narrow the filters to see the rest.`));
  };
  renderTable();
}

function critiqueBadge(critique) {
  const rating = critique?.rating, tag = critique?.tag;
  if (!rating && !tag) return el("span.muted", {}, "—");
  const cls = rating === "bad" ? "fail" : rating === "good" ? "pass" : "outline";
  return el("span", { class: "badge " + cls, style: { fontFamily: "var(--mono)", textTransform: "none" } }, tag || rating);
}

function compareTrials(a, b, key, dir) {
  const sign = dir === "asc" ? 1 : -1;
  let res;
  if (key === "steps") res = numCmp(a.n_agent_steps, b.n_agent_steps);
  else if (key === "cost") res = numCmp(a.cost_usd, b.cost_usd);
  else res = strCmp(sortField(a, key), sortField(b, key));
  return res !== 0 ? res * sign : strCmp(a.trial_name, b.trial_name);
}
function sortField(r, key) {
  if (key === "dataset") return sourceLabel(r.source);
  if (key === "model") return modelLabel(r.model, r.reasoning_effort);
  if (key === "task") return r.task_name;
  if (key === "outcome") return r.passed ? "1 pass" : r.errored ? "3 error" : "2 fail";
  if (key === "analysis") return `${r.critique?.rating ?? ""} ${r.critique?.tag ?? ""}`;
  return "";
}
function strCmp(a, b) { return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" }); }
function numCmp(a, b) { if (a == null && b == null) return 0; if (a == null) return 1; if (b == null) return -1; return a - b; }

function field(label, control) { return el("div.field", {}, [el("span.field-label", {}, label), control]); }
function select(value, options, onChange) {
  return el("select", { onchange: (e) => onChange(e.target.value) },
    options.map((o) => el("option", { value: o.value, selected: o.value === value }, o.label)));
}

// ================= trial detail =================
export async function renderTrialDetail(mount, params) {
  const trialName = params.id;
  const root = el("div.wrap-narrow");
  mount.appendChild(root);
  root.appendChild(el("button.back-btn", { onclick: () => history.back() }, "← Back"));
  const status = el("div.state", {}, "Loading trial…");
  root.appendChild(status);

  let trialsData, release;
  try { [trialsData, release] = await Promise.all([loadArtifact("trials"), loadArtifact("release")]); }
  catch (e) { status.textContent = `Couldn't load trial: ${e}`; return; }

  const trial = (trialsData.rows || []).find((t) => t.trial_name === trialName);
  if (!trial) { status.textContent = "Trial not found in the index."; return; }
  status.remove();

  root.appendChild(el("div.detail-head", {}, [
    el("p.eyebrow", {}, sourceLabel(trial.source)),
    el("h1", {}, modelLabel(trial.model, trial.reasoning_effort)),
    el("div.detail-meta", {}, [
      el("span", {}, el("span", { class: "badge " + (trial.passed ? "pass" : trial.errored ? "outline" : "fail") }, trial.passed ? "pass" : trial.errored ? "error" : "fail")),
      el("span", {}, [el("span.muted", {}, "task "), el("a.mono", { href: `#/data/tasks/${encodeURIComponent(trial.task_name)}` }, trial.task_name)]),
      el("span.mono", {}, trial.trial_name),
    ]),
  ]));

  // metric strip
  root.appendChild(el("dl.specs", { style: { marginTop: "1.5rem" } }, [
    el("dt", {}, "Steps"), el("dd.tabular", {}, fmtInt(trial.n_agent_steps)),
    el("dt", {}, "Cost"), el("dd.tabular", {}, fmtCost4(trial.cost_usd)),
    el("dt", {}, "Input tokens"), el("dd.tabular", {}, fmtInt(trial.n_input_tokens)),
    el("dt", {}, "Output tokens"), el("dd.tabular", {}, fmtInt(trial.n_output_tokens)),
    el("dt", {}, "Peak context"), el("dd.tabular", {}, fmtInt(trial.peak_context_tokens)),
    trial.trial_duration_seconds ? el("dt", {}, "Duration") : null,
    trial.trial_duration_seconds ? el("dd", {}, fmtDuration(trial.trial_duration_seconds)) : null,
    el("dt", {}, "Provider"), el("dd", {}, trial.provider || "—"),
    el("dt", {}, "Harness"), el("dd", {}, trial.harness || "—"),
  ]));

  if (trial.critique) {
    root.appendChild(el("hr.sep"));
    root.appendChild(el("p.section-label", {}, "Trajectory analysis"));
    root.appendChild(el("div", {}, [
      el("p", {}, [critiqueBadge(trial.critique), " ", el("span.muted", {}, `by ${trial.critique.model || "—"}`)]),
      el("p", { style: { whiteSpace: "pre-wrap", lineHeight: 1.6 } }, trial.critique.feedback || ""),
    ]));
  }

  // trajectory
  root.appendChild(el("hr.sep"));
  root.appendChild(el("p.section-label", {}, "Trajectory"));
  const trajHost = el("div", {}, el("div.state", {}, "Loading trajectory…"));
  root.appendChild(trajHost);

  if (trial.has_trajectory !== false) {
    const base = release.artifact_base_url.replace(/\/$/, "");
    const pat = release.artifact_patterns.trajectory.replace("{trial_name}", encodeURIComponent(trialName));
    try {
      const res = await fetch(`${base}/${pat}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const traj = await res.json();
      clear(trajHost);
      trajHost.appendChild(renderTrajectory(traj));
    } catch (e) {
      clear(trajHost);
      trajHost.appendChild(el("p.state", {}, `Trajectory artifact unavailable (${e}). Path: ${pat}`));
    }
  } else {
    clear(trajHost);
    trajHost.appendChild(el("p.state", {}, "No trajectory recorded for this trial."));
  }
}

function renderTrajectory(traj) {
  const steps = traj.steps || [];
  const wrap = el("div");
  if (traj.agent) wrap.appendChild(el("p.muted", { style: { fontSize: "12px", marginBottom: "1rem" } },
    `${traj.agent.name || ""} ${traj.agent.version || ""} · ${traj.agent.model_name || ""} · ${steps.length} steps`));
  for (const s of steps) {
    if (s.source === "system") continue;
    const step = el("div.traj-step");
    step.appendChild(el("div.step-head", {}, [
      el("span.src", {}, s.source || "step"),
      s.step_id != null ? el("span.muted", {}, `#${s.step_id}`) : null,
    ]));
    const sbody = el("div.step-body");
    if (s.reasoning_content) sbody.appendChild(el("div.reasoning", {}, s.reasoning_content));
    if (s.message) sbody.appendChild(el("div.msg", {}, s.message));
    if (Array.isArray(s.tool_calls)) {
      for (const tc of s.tool_calls) {
        const args = tc.arguments && typeof tc.arguments === "object"
          ? (tc.arguments.command ?? JSON.stringify(tc.arguments, null, 2)) : String(tc.arguments ?? "");
        sbody.appendChild(el("div.toolcall", {}, [
          el("div.fn", {}, `▸ ${tc.function_name || "tool"}`),
          el("pre.code", { html: escapeHtml(args) }),
        ]));
      }
    }
    if (s.observation?.results) {
      for (const r of s.observation.results) {
        const content = typeof r.content === "string" ? r.content : JSON.stringify(r.content, null, 2);
        const truncated = content.length > 4000 ? content.slice(0, 4000) + "\n… (truncated)" : content;
        sbody.appendChild(el("pre.code", { style: { marginTop: ".4rem", opacity: ".85" }, html: escapeHtml(truncated) }));
      }
    }
    step.appendChild(sbody);
    wrap.appendChild(step);
  }
  return wrap;
}

