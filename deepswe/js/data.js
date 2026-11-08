// ============================================================
// data.js — artifact loading + helper functions
// Faithful reimplementation of the original DeepSWE helpers.
// ============================================================

export const SOURCE_DEEPSWE = "deep-swe";
export const SOURCE_CROSSBENCH = "cross-benchmark-analysis";
const EFFORT_SEP = "::effort=";

// ---- in-memory artifact cache (fetch once, reuse) ----
const _cache = new Map();

/** Resolve an artifact key to a URL. Keys without "/" map to /artifacts/<key>.json */
function artifactUrl(key) {
  return key.startsWith("/") || key.includes("/") || key.endsWith(".json")
    ? `artifacts/${key.replace(/^artifacts\//, "")}`
    : `artifacts/${key}.json`;
}

/** Fetch + cache a JSON artifact. `onProgress({loaded,total})` optional. */
export async function loadArtifact(key, onProgress) {
  if (_cache.has(key)) return _cache.get(key);
  const url = artifactUrl(key);
  const promise = fetchJsonWithProgress(url, onProgress);
  _cache.set(key, promise);
  try {
    const data = await promise;
    _cache.set(key, Promise.resolve(data));
    return data;
  } catch (e) {
    _cache.delete(key);
    throw e;
  }
}

async function fetchJsonWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  if (!onProgress || !res.body) return res.json();
  const lenHeader = res.headers.get("Content-Length");
  const total = lenHeader != null ? Number.parseInt(lenHeader, 10) : null;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  onProgress({ loaded: 0, total: Number.isFinite(total) ? total : null });
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) { chunks.push(value); loaded += value.byteLength; onProgress({ loaded, total: Number.isFinite(total) ? total : null }); }
  }
  const dec = new TextDecoder("utf-8");
  let txt = "";
  for (const c of chunks) txt += dec.decode(c, { stream: true });
  txt += dec.decode();
  return JSON.parse(txt);
}

// ---- model / effort label helpers ----

/** Clean a raw model id into a display name (strip provider prefix, fix versions). */
export function cleanModelName(m) {
  return m
    .replace(/^[a-z0-9-]+\//, "")
    .replace(/(\d)-(\d)(?=-|$)/g, "$1.$2")
    .replace(/(\d)\.(\d)\.(\d)(?=-|$)/g, "$1.$2.$3")
    .replace(/-preview$/, "");
}

/** Composite key uniquely identifying a (model, effort) pair. */
export function modelKey(model, effort) {
  return `${model}${EFFORT_SEP}${effort ?? ""}`;
}

/** Human label: "claude-opus-4.7 [max]" or just "claude-opus-4.7". */
export function modelLabel(model, effort) {
  const n = cleanModelName(model);
  return effort ? `${n} [${effort}]` : n;
}

/** Does a selected model key match a row's (model, effort)? */
export function modelMatches(selectedKey, model, effort) {
  return selectedKey.includes(EFFORT_SEP) ? selectedKey === modelKey(model, effort) : selectedKey === model;
}

// ---- source labels ----
const SOURCE_LABELS = { swebenchpro: "SWE-Bench Pro", "deep-swe": "DeepSWE" };
export function sourceLabel(s) {
  return s ? (SOURCE_LABELS[s] ?? s) : "—";
}

// ---- value formatters ----
export function fmtCost(v) {
  if (v == null || Number.isNaN(v)) return "—";
  if (v < 0.01) return "<$0.01";
  return `$${v.toFixed(2)}`;
}
export function fmtCost4(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `$${v.toFixed(4)}`;
}
export function fmtInt(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-US");
}
export function fmtDuration(sec) {
  const t = Math.round(sec);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}
export function fmtTokensK(v) {
  if (v == null) return "—";
  return `${Math.round(v / 1000)}k`;
}

// ============================================================
// Heatmap aggregation — replicates the original pivot engine
// ============================================================

// Stats that can color the heatmap.
export const STATS = [
  { value: "pass_rate", label: "Pass rate" },
  { value: "avg_cost_usd", label: "Avg cost" },
  { value: "total_cost_usd", label: "Total cost" },
  { value: "avg_steps", label: "Avg steps" },
  { value: "avg_input_tokens", label: "Avg input tokens" },
  { value: "avg_output_tokens", label: "Avg output tokens" },
  { value: "avg_peak_context_tokens", label: "Avg peak context" },
  { value: "avg_duration_seconds", label: "Avg duration" },
  { value: "n_trials", label: "Trials" },
  { value: "n_errors", label: "Errors" },
];

export const TRIALS_FILTERS = [
  { value: "all", label: "All trials" },
  { value: "non_errored", label: "Exclude errored" },
  { value: "successful", label: "Only passed" },
];

/** Dimension descriptor for a trial along a given axis. */
function dimDescriptor(trial, dim, effortMode, taskTitles) {
  switch (dim) {
    case "model":
      return effortMode
        ? { key: modelKey(trial.model, trial.reasoning_effort), label: modelLabel(trial.model, trial.reasoning_effort) }
        : { key: trial.model, label: cleanModelName(trial.model) };
    case "task":
      return { key: trial.task_name, label: taskTitles?.get(trial.task_name) ?? trial.task_name };
    case "dataset":
      return { key: trial.source, label: sourceLabel(trial.source) };
  }
}

function newCell(rowKey, columnKey, model, task, source) {
  return {
    rowKey, columnKey, model, task, source,
    n_trials: 0, n_passed: 0, n_errors: 0,
    costSum: 0, costCount: 0, stepsSum: 0, stepsCount: 0,
    inputSum: 0, inputCount: 0, outputSum: 0, outputCount: 0,
    peakSum: 0, peakCount: 0, durationSum: 0, durationCount: 0,
  };
}
function accAdd(cell, k, kc, v) { if (v != null && !Number.isNaN(v)) { cell[k] += v; cell[kc] += 1; } }
function accTrial(cell, t) {
  cell.n_trials += 1;
  if (t.passed) cell.n_passed += 1;
  if (t.errored) cell.n_errors += 1;
  accAdd(cell, "costSum", "costCount", t.cost_usd);
  accAdd(cell, "stepsSum", "stepsCount", t.n_agent_steps);
  accAdd(cell, "inputSum", "inputCount", t.n_input_tokens);
  accAdd(cell, "outputSum", "outputCount", t.n_output_tokens);
  accAdd(cell, "peakSum", "peakCount", t.peak_context_tokens);
  accAdd(cell, "durationSum", "durationCount", t.trial_duration_seconds);
}
function mean(sum, count) { return count > 0 ? sum / count : null; }
function finalizeCell(c) {
  const denom = c.n_trials - c.n_errors;
  return {
    rowKey: c.rowKey, columnKey: c.columnKey,
    n_trials: c.n_trials, n_passed: c.n_passed, n_errors: c.n_errors,
    pass_rate: denom > 0 ? c.n_passed / denom : null,
    avg_cost_usd: mean(c.costSum, c.costCount),
    total_cost_usd: c.costCount > 0 ? c.costSum : null,
    avg_steps: mean(c.stepsSum, c.stepsCount),
    avg_input_tokens: mean(c.inputSum, c.inputCount),
    avg_output_tokens: mean(c.outputSum, c.outputCount),
    avg_peak_context_tokens: mean(c.peakSum, c.peakCount),
    avg_duration_seconds: mean(c.durationSum, c.durationCount),
    model: c.model, task: c.task, source: c.source,
  };
}

/** Read a stat value out of a finalized cell. */
export function statValue(cell, stat) {
  return cell[stat] ?? null;
}

/** Format a stat value for display. */
export function fmtStat(cell, stat) {
  const v = statValue(cell, stat);
  if (v == null) return "—";
  switch (stat) {
    case "pass_rate": return `${Math.round(v * 100)}%`;
    case "avg_cost_usd":
    case "total_cost_usd": return fmtCost(v);
    case "avg_duration_seconds": return fmtDuration(v);
    case "avg_steps": return v.toFixed(v < 10 ? 1 : 0).replace(/\.0$/, "");
    case "avg_input_tokens":
    case "avg_output_tokens":
    case "avg_peak_context_tokens": return fmtInt(Math.round(v));
    default: return fmtInt(v);
  }
}

/**
 * Pivot trials into a {rows, columns, cells, rowSummaries} grid.
 * rowBy/columnBy ∈ {"model","task","dataset"}. Sorting is by pass-rate desc.
 */
export function pivot(trials, rowBy, columnBy, trialsFilter, effortMode, taskTitles) {
  const cells = new Map();
  const summaries = new Map();
  const rowDims = new Map();
  const colDims = new Map();

  for (const t of trials) {
    if (trialsFilter === "non_errored" && t.errored) continue;
    if (trialsFilter === "successful" && !t.passed) continue;

    const rd = dimDescriptor(t, rowBy, effortMode, taskTitles);
    const cd = dimDescriptor(t, columnBy, effortMode, taskTitles);
    rowDims.set(rd.key, rd);
    colDims.set(cd.key, cd);

    const model = rowBy === "model" || columnBy === "model" ? (effortMode ? modelKey(t.model, t.reasoning_effort) : t.model) : null;
    const task = rowBy === "task" || columnBy === "task" ? t.task_name : null;
    const source = rowBy === "dataset" || columnBy === "dataset" ? t.source : null;

    const ck = `${rd.key}\u0000${cd.key}`;
    let cell = cells.get(ck);
    if (!cell) { cell = newCell(rd.key, cd.key, model, task, source); cells.set(ck, cell); }
    accTrial(cell, t);

    let sum = summaries.get(rd.key);
    if (!sum) {
      sum = newCell(rd.key, "",
        rowBy === "model" ? (effortMode ? modelKey(t.model, t.reasoning_effort) : t.model) : null,
        rowBy === "task" ? t.task_name : null,
        rowBy === "dataset" ? t.source : null);
      summaries.set(rd.key, sum);
    }
    accTrial(sum, t);
  }

  const finalCells = new Map();
  for (const [k, v] of cells) finalCells.set(k, finalizeCell(v));
  const finalSummaries = new Map();
  for (const [k, v] of summaries) finalSummaries.set(k, finalizeCell(v));

  // sort rows & columns by overall pass-rate (desc), then label
  const passRateOf = (key, axis) => {
    let p = 0, d = 0;
    for (const c of finalCells.values()) {
      if (axis === "row" ? c.rowKey === key : c.columnKey === key) { p += c.n_passed; d += c.n_trials - c.n_errors; }
    }
    return d > 0 ? p / d : -1;
  };
  const rows = Array.from(rowDims.values()).sort((a, b) => {
    const d = passRateOf(b.key, "row") - passRateOf(a.key, "row");
    return d !== 0 ? d : a.label.localeCompare(b.label);
  });
  const columns = Array.from(colDims.values()).sort((a, b) => {
    const d = passRateOf(b.key, "column") - passRateOf(a.key, "column");
    return d !== 0 ? d : a.label.localeCompare(b.label);
  });

  return { rows, columns, cells: finalCells, rowSummaries: finalSummaries };
}

/** Min/range of a stat across a cell collection (for color scaling). */
export function statRange(cellMap, stat) {
  const vals = [];
  for (const c of cellMap.values()) {
    const v = statValue(c, stat);
    if (v != null) vals.push(v);
  }
  const min = vals.length ? Math.min(...vals) : 0;
  const max = vals.length ? Math.max(...vals) : 1;
  return { min, range: Math.max(max - min, 1e-9) };
}

/** Background color for a normalized [0,1] value, matching the original. */
export function cellColor(norm) {
  const pct = (norm * 88 + 8).toFixed(1);
  return `color-mix(in oklch, var(--foreground) ${pct}%, transparent)`;
}

