// ============================================================
// app.js — entry point: theme, router, view dispatch
// ============================================================
import { route, startRouter, clear } from "./dom.js";
import { loadArtifact } from "./data.js";
import { renderHeatmap } from "./view_heatmap.js";
import { renderTasks, renderTaskDetail } from "./view_tasks.js";
import { renderTrials, renderTrialDetail } from "./view_trials.js";

// ---- theme ----
const THEME_KEY = "mds-theme";
function applyTheme(t) {
  document.documentElement.classList.toggle("dark", t === "dark");
  localStorage.setItem(THEME_KEY, t);
}
applyTheme(localStorage.getItem(THEME_KEY) ||
  (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
document.getElementById("theme-toggle").addEventListener("click", () => {
  applyTheme(document.documentElement.classList.contains("dark") ? "light" : "dark");
});

// ---- summary-driven branding (optional) ----
loadArtifact("summary").then((s) => {
  if (s?.name) {
    document.getElementById("brand-name").textContent = s.name;
    document.title = `${s.name} · data explorer`;
  }
  if (s?.counts) {
    const c = s.counts;
    document.getElementById("footer-text").textContent =
      `${s.name || "AutoEval"} — ${c.tasks ?? "?"} tasks · ${c.trials ?? "?"} trials`;
  }
}).catch(() => {});

// ---- routes ----
route("/data", renderHeatmap);
route("/data/tasks", renderTasks);
route("/data/tasks/:id", renderTaskDetail);
route("/data/trials", renderTrials);
route("/data/trials/:id", renderTrialDetail);

const viewEl = document.getElementById("view");
startRouter(async (handler, params) => {
  // remember focus so query-param re-renders don't kick the user out of the search box
  const active = document.activeElement;
  const focusId = active && active.id ? active.id : null;
  const caret = active && "selectionStart" in active ? active.selectionStart : null;

  clear(viewEl);
  if (!focusId) window.scrollTo(0, 0);
  try {
    await handler(viewEl, params);
  } catch (e) {
    console.error(e);
    clear(viewEl);
    const div = document.createElement("div");
    div.className = "wrap state";
    div.textContent = `Something went wrong: ${e}`;
    viewEl.appendChild(div);
  }
  // restore focus to the search box after a query-param re-render
  if (focusId) {
    const restored = document.getElementById(focusId);
    if (restored) {
      restored.focus();
      if (caret != null && "setSelectionRange" in restored) {
        try { restored.setSelectionRange(caret, caret); } catch {}
      }
    }
  }
});

