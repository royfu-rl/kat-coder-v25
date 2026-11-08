// ============================================================
// view_tasks.js — task catalog grid + task detail page
// ============================================================
import { el, clear, parseQuery, setQuery, navigate, renderMarkdown, renderDiff, escapeHtml } from "./dom.js";
import { loadArtifact, modelLabel, SOURCE_DEEPSWE } from "./data.js";
import { dataHeader } from "./view_common.js";

export async function renderTasks(mount) {
  const q = parseQuery();
  const search = q.q || "";
  const lang = q.lang || "all";

  const root = el("div.wrap");
  root.appendChild(dataHeader("tasks"));
  mount.appendChild(root);

  const status = el("div.muted", { style: { marginTop: "1.5rem", fontSize: "12px" } }, "Loading tasks…");
  root.appendChild(status);

  let data;
  try { data = await loadArtifact("tasks"); }
  catch (e) { status.textContent = `Couldn't load tasks: ${e}`; return; }

  status.textContent = `${data.n_tasks ?? data.rows.length} tasks · each is a stable entry with instructions, repo, and verifier`;

  const languages = Array.from(new Set(data.rows.map((r) => r.language))).sort();

  // controls
  const bar = el("div", { style: { marginTop: "1rem", display: "flex", flexWrap: "wrap", gap: ".5rem", alignItems: "center" } });
  const searchInput = el("input", { id: "tasks-search", type: "search", placeholder: "Search title, repo, id…", value: search, style: { maxWidth: "24rem" } });
  let debounce;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => setQuery({ q: searchInput.value || null }), 200);
  });
  const langSelect = el("select", { style: { height: "2.25rem", width: "auto" }, onchange: (e) => setQuery({ lang: e.target.value === "all" ? null : e.target.value }) }, [
    el("option", { value: "all", selected: lang === "all" }, "All languages"),
    ...languages.map((l) => el("option", { value: l, selected: l === lang }, l)),
  ]);
  bar.appendChild(searchInput);
  bar.appendChild(langSelect);
  const countSpan = el("span.count-right");
  bar.appendChild(countSpan);
  root.appendChild(bar);

  const grid = el("div.task-grid");
  root.appendChild(grid);

  const e = search.trim().toLowerCase();
  const filtered = data.rows
    .filter((r) => lang === "all" || r.language === lang)
    .filter((r) => (e ? [r.problem_title, r.repository, r.id].join(" ").toLowerCase().includes(e) : true));

  countSpan.textContent = `${filtered.length} of ${data.rows.length}`;

  if (filtered.length === 0) {
    grid.appendChild(el("p.state", {}, "No tasks match these filters."));
    return;
  }
  for (const t of filtered) {
    grid.appendChild(el("a.task-card", { href: `#/data/tasks/${encodeURIComponent(t.id)}`, "data-lang": t.language || "" }, [
      el("h3", { title: t.problem_title }, t.problem_title),
      t.display_description ? el("p.desc", {}, t.display_description) : null,
      el("div.meta", {}, [
        el("span.repo", { title: t.repository }, t.repository || "—"),
        el("span.lang", { "data-lang": t.language || "" }, t.language || ""),
      ]),
    ]));
  }
}

// ---------------- task detail ----------------
const LANG_BY_EXT = (p) =>
  p.endsWith(".patch") ? "diff" : p.endsWith(".toml") ? "toml" : p.endsWith(".md") ? "markdown" :
  p.endsWith(".sh") ? "bash" : p.endsWith("Dockerfile") ? "dockerfile" : "text";

export async function renderTaskDetail(mount, params) {
  const taskId = params.id;
  const root = el("div.wrap-narrow");
  mount.appendChild(root);
  root.appendChild(el("button.back-btn", { onclick: () => history.back() }, "← Back"));
  const status = el("div.state", {}, "Loading task…");
  root.appendChild(status);

  let detail, tasksIndex, trialsData;
  try {
    [detail, tasksIndex, trialsData] = await Promise.all([
      loadArtifact(`tasks/${taskId}.json`),
      loadArtifact("tasks"),
      loadArtifact("trials"),
    ]);
  } catch (e) { status.textContent = `Couldn't load task: ${e}`; return; }
  status.remove();

  const meta = (tasksIndex.rows || []).find((r) => r.id === taskId) || null;
  const matchingTrials = (trialsData.rows || []).filter(
    (t) => t.task_name === taskId && t.source === SOURCE_DEEPSWE && t.eval_scope !== "cross-bench"
  );
  const fileMap = new Map(detail.files.map((f) => [f.path, f]));
  const instruction = fileMap.get("instruction.md")?.content ?? null;

  // header
  root.appendChild(el("div.detail-head", {}, [
    el("p.eyebrow", {}, meta?.language || "task"),
    el("h1", {}, meta?.problem_title || taskId),
    el("div.detail-meta", {}, [
      meta?.repository ? el("span", {}, [el("span.mono", {}, meta.repository)]) : null,
      meta?.base_commit_hash ? el("span", {}, [el("span.muted", {}, "base "), el("span.mono", {}, meta.base_commit_hash.slice(0, 10))]) : null,
      el("span.mono", {}, taskId),
    ]),
  ]));

  // sub-tabs
  const passCount = matchingTrials.filter((t) => t.passed).length;
  const diffStat = (path) => {
    const c = fileMap.get(path)?.content;
    if (!c) return null;
    let a = 0, d = 0;
    for (const ln of c.split("\n")) { if (ln.startsWith("+") && !ln.startsWith("+++")) a++; else if (ln.startsWith("-") && !ln.startsWith("---")) d++; }
    return { a, d };
  };
  const testStat = diffStat("tests/test.patch");
  const solStat = diffStat("solution/solution.patch");

  const tabDefs = [
    { id: "trials", label: "Trials", hint: matchingTrials.length ? `${passCount}/${matchingTrials.length}` : null },
    { id: "verifier", label: "Verifier", hint: testStat ? `+${testStat.a}` : null },
    { id: "solution", label: "Solution", hint: solStat ? `+${solStat.a} −${solStat.d}` : null },
    { id: "environment", label: "Environment", hint: null },
    { id: "config", label: "task.toml", hint: null },
  ];
  let activeTab = "trials";
  const tabBar = el("div.subtabs");
  const panel = el("div", { style: { paddingTop: "1.5rem" } });
  const renderTab = () => {
    clear(tabBar); clear(panel);
    for (const t of tabDefs) {
      tabBar.appendChild(el("button", {
        class: "subtab" + (t.id === activeTab ? " active" : ""),
        onclick: () => { activeTab = t.id; renderTab(); },
      }, [t.label, t.hint ? el("span.hint", {}, t.hint) : null]));
    }
    panel.appendChild(renderTaskTab(activeTab, { fileMap, matchingTrials }));
  };
  renderTab();
  root.appendChild(tabBar);
  root.appendChild(panel);

  // instruction at the bottom
  if (instruction) {
    root.appendChild(el("hr.sep"));
    root.appendChild(el("p.section-label", {}, "Instruction"));
    root.appendChild(renderMarkdown(instruction));
  }
}

function renderTaskTab(tab, { fileMap, matchingTrials }) {
  if (tab === "trials") return renderTaskTrials(matchingTrials);
  if (tab === "verifier") return renderFiles(fileMap, ["tests/test.sh", "tests/test.patch"]);
  if (tab === "solution") return renderFiles(fileMap, ["solution/solve.sh", "solution/solution.patch"]);
  if (tab === "environment") return renderEnvironment(fileMap);
  if (tab === "config") return renderFiles(fileMap, ["task.toml"]);
  return el("div");
}

function renderTaskTrials(trials) {
  if (trials.length === 0) return el("p.state", {}, "No trials reference this task in the published index.");
  // group by (model, effort)
  const groups = new Map();
  for (const t of trials) {
    const key = modelLabel(t.model, t.reasoning_effort);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(t);
  }
  const items = Array.from(groups.entries())
    .map(([label, rollouts]) => ({ label, rollouts, pass: rollouts.filter((r) => r.passed).length }))
    .sort((a, b) => (b.pass / b.rollouts.length) - (a.pass / a.rollouts.length) || a.label.localeCompare(b.label));

  return el("ul", { style: { listStyle: "none", padding: 0, margin: 0 } }, items.map((g) =>
    el("li", { style: { display: "grid", gridTemplateColumns: "minmax(11rem,auto) 1fr auto", gap: ".5rem 1rem", alignItems: "center", padding: ".6rem 0", borderBottom: "1px solid var(--border)" } }, [
      el("span", { style: { fontWeight: 500 } }, g.label),
      el("div", { style: { display: "flex", flexWrap: "wrap", gap: ".25rem" } }, g.rollouts.map((r) =>
        el("a", { href: `#/data/trials/${encodeURIComponent(r.trial_name)}`, title: r.trial_name },
          el("span", { class: "badge " + (r.passed ? "pass" : r.errored ? "outline" : "fail") }, r.passed ? "pass" : r.errored ? "error" : "fail")))),
      el("span.muted.tabular", { style: { justifySelf: "end", fontSize: "11px" } }, `${g.pass}/${g.rollouts.length}`),
    ])
  ));
}

function renderEnvironment(fileMap) {
  const dockerfile = fileMap.get("environment/Dockerfile")?.content || "";
  let baseImage = "—", installLine = "—";
  for (const raw of dockerfile.split("\n")) {
    const ln = raw.trim();
    if (baseImage === "—" && /^FROM\s+/i.test(ln)) baseImage = ln.replace(/^FROM\s+/i, "");
    else if (installLine === "—" && /^RUN\s+/i.test(ln) && !/git\s+clone/.test(ln)) installLine = ln.replace(/^RUN\s+/i, "");
  }
  return el("div", {}, [
    el("dl.specs", {}, [
      el("dt", {}, "Base image"), el("dd.mono", {}, baseImage),
      el("dt", {}, "Install"), el("dd.mono", {}, installLine),
      el("dt", {}, "Resources"), el("dd", {}, "2 CPU · 8 GB RAM · 20 GB disk · no internet"),
    ]),
    renderFiles(fileMap, ["environment/Dockerfile"]),
  ]);
}

function renderFiles(fileMap, paths) {
  const wrap = el("div", { style: { display: "flex", flexDirection: "column", gap: "1.5rem" } });
  let any = false;
  for (const path of paths) {
    const f = fileMap.get(path);
    if (!f || f.content == null) continue;
    any = true;
    const lines = f.content.split("\n").length;
    const body = el("div", { style: { marginTop: ".5rem" } },
      (f.language === "diff" || path.endsWith(".patch")) ? renderDiff(f.content) : el("pre.code", { html: escapeHtml(f.content) }));
    let open = true;
    const toggle = el("button.file-toggle", {}, [
      el("span", {}, "▾ "), el("span", {}, path), el("span.lines", {}, `${lines} lines`),
    ]);
    toggle.addEventListener("click", () => { open = !open; body.style.display = open ? "" : "none"; toggle.firstChild.textContent = open ? "▾ " : "▸ "; });
    wrap.appendChild(el("div.file-block", {}, [toggle, body]));
  }
  if (!any) wrap.appendChild(el("p.state", {}, `${paths.join(", ")} not bundled for this task.`));
  return wrap;
}

