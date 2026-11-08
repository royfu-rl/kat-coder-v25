// ============================================================
// view_common.js — shared page header + data tabs
// ============================================================
import { el } from "./dom.js";

const TABS = [
  { id: "heatmap", label: "Heatmap", to: "/data" },
  { id: "tasks", label: "Tasks", to: "/data/tasks" },
  { id: "trials", label: "Trials", to: "/data/trials" },
];

/** The "Browse the data" header block + tab nav, shared by all three views. */
export function dataHeader(active) {
  return el("div", {}, [
    el("p.eyebrow", {}, "Data"),
    el("h1.title", {}, "Browse the data"),
    el("p.lede", {}, "Outcomes from every (task, model) rollout. Pivot the heatmap, scan the task catalog, or open the trial table."),
    el("nav.tabs", {}, TABS.map((t) =>
      el("a", { class: "tab" + (t.id === active ? " active" : ""), href: "#" + t.to }, t.label)
    )),
  ]);
}

