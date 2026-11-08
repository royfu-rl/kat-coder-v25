// ============================================================
// view_heatmap.js — pivot heatmap (tasks × models)
// ============================================================
import { el, clear, setQuery, parseQuery, navigate } from "./dom.js";
import {
  loadArtifact, SOURCE_DEEPSWE, SOURCE_CROSSBENCH,
  pivot, statValue, fmtStat, statRange, cellColor,
  STATS, TRIALS_FILTERS,
} from "./data.js";
import { dataHeader } from "./view_common.js";

export async function renderHeatmap(mount) {
  const q = parseQuery();
  const source = q.source === SOURCE_CROSSBENCH ? SOURCE_CROSSBENCH : SOURCE_DEEPSWE;
  const isCross = source === SOURCE_CROSSBENCH;

  const pivoted = q.pivot === "1";
  const stat = q.hm_stat || "pass_rate";
  const trialsFilter = q.hm_trials || "all";
  const effortMode = (q.hm_model || "effort") === "effort";
  const taskGrouping = isCross ? (q.hm_group || "task") : "task";

  // axis assignment: default rows=task(group), cols=model; pivot swaps them
  const rowBy = pivoted ? "model" : taskGrouping;
  const columnBy = pivoted ? taskGrouping : "model";

  const container = el("div", { style: { padding: "2rem 0 3rem" } });
  mount.appendChild(container);

  const head = el("div.wrap", {}, []);
  head.appendChild(dataHeader("heatmap"));
  container.appendChild(head);

  const body = el("div.wrap-wide");
  container.appendChild(body);

  const loadingBar = el("div.spinner-bar", {}, [
    el("div", {}, "Loading trial index…"),
    el("div.track", {}, el("div.fill")),
  ]);
  body.appendChild(loadingBar);

  let trialsData, tasksData;
  try {
    [trialsData, tasksData] = await Promise.all([loadArtifact("trials"), loadArtifact("tasks")]);
  } catch (e) {
    clear(body);
    body.appendChild(el("div.state", {}, `Couldn't load the trial index: ${e}`));
    return;
  }
  loadingBar.remove();

  const taskTitles = new Map((tasksData.rows || []).map((r) => [r.id, r.problem_title]));
  const trials = (trialsData.rows || []).filter((t) =>
    isCross ? t.eval_scope === "cross-bench" : t.eval_scope === "full" && t.source === SOURCE_DEEPSWE
  );

  // ---- controls ----
  const controls = el("div.controls-row");
  body.appendChild(controls);

  const sourceOptions = [
    { value: SOURCE_DEEPSWE, label: "DeepSWE" },
    { value: SOURCE_CROSSBENCH, label: "Cross-benchmark analysis" },
  ];
  controls.appendChild(makeSelect("Source", source, sourceOptions, (v) =>
    setQuery({ source: v === SOURCE_DEEPSWE ? null : v, hm_group: null }, { replace: false })
  ));

  controls.appendChild(makeSelect("Color by", stat, STATS, (v) =>
    setQuery({ hm_stat: v === "pass_rate" ? null : v })
  ));
  controls.appendChild(makeSelect("Trials", trialsFilter, TRIALS_FILTERS, (v) =>
    setQuery({ hm_trials: v === "all" ? null : v })
  ));
  controls.appendChild(makeSelect("Models", effortMode ? "effort" : "base",
    [{ value: "effort", label: "Split effort" }, { value: "base", label: "Base model" }],
    (v) => setQuery({ hm_model: v === "effort" ? null : v })
  ));
  if (isCross) {
    controls.appendChild(makeSelect("Group rows by", taskGrouping,
      [{ value: "task", label: "Per task" }, { value: "dataset", label: "Per dataset" }],
      (v) => setQuery({ hm_group: v === "task" ? null : v })
    ));
  }

  const pivotBtn = el("button.pivot-btn", { onclick: () => setQuery({ pivot: pivoted ? null : "1" }) },
    "⇄ Pivot (rows ⇄ cols)");
  const pivotField = el("div.field", {}, [el("span.field-label", {}, " "), pivotBtn]);
  controls.appendChild(pivotField);

  body.appendChild(el("div.muted", { style: { marginTop: ".75rem", fontSize: "12px" } },
    `${trials.length.toLocaleString()} trials · click a cell to open the matching trials`));

  // ---- grid ----
  const { rows, columns, cells, rowSummaries } = pivot(trials, rowBy, columnBy, trialsFilter, effortMode, taskTitles);
  const { min, range } = statRange(cells, stat);
  const sumRange = statRange(rowSummaries, stat);

  if (rows.length === 0 || columns.length === 0) {
    body.appendChild(el("div.state", {}, "No data for this selection."));
    return;
  }

  const showNumbers = (q.hm_display || "numbers") === "numbers";
  // display toggle
  const displayToggle = el("div.toolbar-toggle", { style: { marginTop: "1rem" } }, [
    el("button", { class: showNumbers ? "active" : "", onclick: () => setQuery({ hm_display: null }) }, "Numbers"),
    el("button", { class: !showNumbers ? "active" : "", onclick: () => setQuery({ hm_display: "color" }) }, "Color only"),
  ]);
  body.appendChild(displayToggle);

  const scroll = el("div.hm-scroll");
  const table = el("table.hm-table");
  scroll.appendChild(table);
  body.appendChild(scroll);

  // header row
  const thead = el("thead");
  const htr = el("tr");
  htr.appendChild(el("th.hm-corner", {}, `${rowBy} ⟍ ${columnBy}`));
  htr.appendChild(el("th.hm-colhead.hm-summary-col", {}, el("div.lbl", {}, "All (row)")));
  for (const col of columns) {
    const family = inferFamily(col.key);
    const dot = el("span.family-dot", { "data-family": family });
    htr.appendChild(el("th.hm-colhead", { title: col.label }, [dot, el("div.lbl", {}, col.label)]));
  }
  thead.appendChild(htr);
  table.appendChild(thead);

  // body rows
  const tbody = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    const rowClick = () => navigateToCell({ rowBy, columnBy, isCross, model: rowBy === "model" ? row.key : null, task: rowBy === "task" ? row.key : null });
    tr.appendChild(el("th.hm-rowhead", { title: row.label, onclick: rowClick }, el("span.rh-inner", {}, row.label)));

    // summary cell
    const sum = rowSummaries.get(row.key);
    tr.appendChild(makeCell(sum, stat, sumRange.min, sumRange.range, showNumbers, true, null));

    for (const col of columns) {
      const cell = cells.get(`${row.key}\u0000${col.key}`);
      const onClick = cell ? () => navigateToCell({
        rowBy, columnBy, isCross,
        model: rowBy === "model" ? row.key : columnBy === "model" ? col.key : null,
        task: rowBy === "task" ? row.key : columnBy === "task" ? col.key : null,
        source: rowBy === "dataset" ? row.key : columnBy === "dataset" ? col.key : null,
      }) : null;
      tr.appendChild(makeCell(cell, stat, min, range, showNumbers, false, onClick));
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

function makeCell(cell, stat, min, range, showNumbers, isSummary, onClick) {
  if (!cell) return el("td.hm-cell.empty");
  const v = statValue(cell, stat);
  const norm = v != null ? (v - min) / range : 0;
  const td = el("td.hm-cell" + (isSummary ? ".hm-summary-col" : ""), {
    title: `${fmtStat(cell, stat)} · ${cell.n_passed}/${cell.n_trials - cell.n_errors} passed · ${cell.n_trials} trials`,
    onclick: onClick || undefined,
    style: { cursor: onClick ? "pointer" : "default" },
  });
  const box = el("div.cellbox", {
    style: {
      backgroundColor: v != null ? cellColor(norm) : "transparent",
      // flip text to the background color on dark cells so numbers stay readable
      color: v != null && norm > 0.55 ? "var(--background)" : "var(--foreground)",
    },
  }, showNumbers ? fmtStat(cell, stat) : "");
  td.appendChild(box);
  return td;
}

function navigateToCell({ rowBy, columnBy, isCross, model, task, source }) {
  const params = {};
  if (isCross) {
    params.source = SOURCE_CROSSBENCH;
    if (source) params.dataset = source === SOURCE_DEEPSWE ? "deep-swe" : source;
  }
  if (model) params.model = model;
  if (task) params.q = task;
  const qs = new URLSearchParams(params).toString();
  navigate("/data/trials" + (qs ? "?" + qs : ""));
}

function makeSelect(label, value, options, onChange) {
  const sel = el("select", { onchange: (e) => onChange(e.target.value) },
    options.map((o) => el("option", { value: o.value, selected: o.value === value }, o.label)));
  return el("div.field", {}, [el("span.field-label", {}, label), sel]);
}

/** Infer the model provider family from a model key string. */
function inferFamily(key) {
  const k = key.toLowerCase();
  if (k.includes("gpt") || k.includes("o1") || k.includes("o3") || k.includes("o4") || k.includes("openai")) return "openai";
  if (k.includes("claude") || k.includes("anthropic") || k.includes("fable") || k.includes("sonnet") || k.includes("opus") || k.includes("haiku")) return "anthropic";
  if (k.includes("gemini") || k.includes("google") || k.includes("bard")) return "google";
  if (k.includes("grok") || k.includes("xai")) return "xai";
  if (k.includes("llama") || k.includes("meta")) return "meta";
  if (k.includes("qwen") || k.includes("alibaba")) return "alibaba";
  if (k.includes("mistral") || k.includes("codestral") || k.includes("mixtral")) return "mistral";
  if (k.includes("deepseek")) return "deepseek";
  if (k.includes("glm") || k.includes("zhipu")) return "zhipu";
  if (k.includes("kimi") || k.includes("moonshot")) return "moonshot";
  if (k.includes("minimax") || k.includes("muse")) return "minimax";
  if (k.includes("xiaomi")) return "xiaomi";
  if (k.includes("cursor")) return "cursor";
  return "other";
}

