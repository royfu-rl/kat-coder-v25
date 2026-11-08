// ============================================================
// dom.js — tiny DOM helpers + lightweight markdown / diff render
// ============================================================

/** Create an element. el("div.cls#id", {attr}, [children|string]) */
export function el(spec, attrs, children) {
  let tag = "div", id = null;
  const classes = [];
  spec.replace(/([.#]?[^.#]+)/g, (m) => {
    if (m[0] === ".") classes.push(m.slice(1));
    else if (m[0] === "#") id = m.slice(1);
    else tag = m;
  });
  const node = document.createElement(tag);
  if (id) node.id = id;
  if (classes.length) node.className = classes.join(" ");
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "class") node.className += " " + v;
      else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === "html") node.innerHTML = v;
      else if (k in node && k !== "list") { try { node[k] = v; } catch { node.setAttribute(k, v); } }
      else node.setAttribute(k, v);
    }
  }
  appendChildren(node, children);
  return node;
}

function appendChildren(node, children) {
  if (children == null) return;
  if (Array.isArray(children)) { for (const c of children) appendChildren(node, c); return; }
  if (children instanceof Node) { node.appendChild(children); return; }
  node.appendChild(document.createTextNode(String(children)));
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Render a unified diff / patch into a <pre> with +/- coloring. */
export function renderDiff(text) {
  const lines = text.split("\n");
  const html = lines.map((ln) => {
    const e = escapeHtml(ln);
    if (ln.startsWith("+") && !ln.startsWith("+++")) return `<span class="add">${e}</span>`;
    if (ln.startsWith("-") && !ln.startsWith("---")) return `<span class="del">${e}</span>`;
    if (ln.startsWith("@@") || ln.startsWith("diff ") || ln.startsWith("index ")) return `<span class="hunk">${e}</span>`;
    return e;
  }).join("\n");
  return el("pre.code", { html });
}

/** Minimal, safe markdown → HTML (headings, code, lists, bold/italic, links). */
export function renderMarkdown(md) {
  const out = [];
  const lines = md.split("\n");
  let i = 0;
  let inCode = false, codeBuf = [], codeLang = "";
  let listType = null, listBuf = [];

  const flushList = () => {
    if (listType) { out.push(`<${listType}>${listBuf.join("")}</${listType}>`); listBuf = []; listType = null; }
  };
  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  while (i < lines.length) {
    const ln = lines[i];
    const fence = ln.match(/^```(\w*)/);
    if (fence) {
      if (!inCode) { flushList(); inCode = true; codeLang = fence[1]; codeBuf = []; }
      else { out.push(`<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`); inCode = false; }
      i++; continue;
    }
    if (inCode) { codeBuf.push(ln); i++; continue; }

    const h = ln.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }

    const ul = ln.match(/^\s*[-*]\s+(.*)$/);
    const ol = ln.match(/^\s*\d+\.\s+(.*)$/);
    if (ul) { if (listType !== "ul") { flushList(); listType = "ul"; } listBuf.push(`<li>${inline(ul[1])}</li>`); i++; continue; }
    if (ol) { if (listType !== "ol") { flushList(); listType = "ol"; } listBuf.push(`<li>${inline(ol[1])}</li>`); i++; continue; }

    if (ln.trim() === "") { flushList(); i++; continue; }

    flushList();
    out.push(`<p>${inline(ln)}</p>`);
    i++;
  }
  flushList();
  if (inCode) out.push(`<pre><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
  return el("div.instruction", { html: out.join("\n") });
}

// ---- hash router ----
const routes = [];
export function route(pattern, handler) {
  // pattern like "/data/tasks/:id"
  const keys = [];
  const rx = new RegExp("^" + pattern.replace(/:[^/]+/g, (m) => { keys.push(m.slice(1)); return "([^/]+)"; }) + "$");
  routes.push({ rx, keys, handler });
}

export function currentPath() {
  const h = location.hash.replace(/^#/, "");
  return h || "/data";
}
export function parseQuery() {
  const h = location.hash.replace(/^#/, "");
  const qi = h.indexOf("?");
  const params = {};
  if (qi >= 0) new URLSearchParams(h.slice(qi + 1)).forEach((v, k) => { params[k] = v; });
  return params;
}
export function pathOnly() {
  const h = currentPath();
  const qi = h.indexOf("?");
  return qi >= 0 ? h.slice(0, qi) : h;
}

/** Update query params in the URL without adding history noise (replace). */
export function setQuery(updates, { replace = true } = {}) {
  const params = parseQuery();
  for (const [k, v] of Object.entries(updates)) {
    if (v == null || v === "") delete params[k];
    else params[k] = v;
  }
  const qs = new URLSearchParams(params).toString();
  const newHash = "#" + pathOnly() + (qs ? "?" + qs : "");
  if (replace) history.replaceState(null, "", newHash);
  else location.hash = newHash;
  if (replace) window.dispatchEvent(new HashChangeEvent("hashchange"));
}

export function navigate(path) { location.hash = "#" + path; }

export function startRouter(render) {
  const dispatch = () => {
    const path = pathOnly();
    for (const r of routes) {
      const m = path.match(r.rx);
      if (m) {
        const params = {};
        r.keys.forEach((k, idx) => { params[k] = decodeURIComponent(m[idx + 1]); });
        render(r.handler, params);
        return;
      }
    }
    render(routes[0].handler, {});
  };
  window.addEventListener("hashchange", dispatch);
  dispatch();
}

