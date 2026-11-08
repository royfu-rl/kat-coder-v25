#!/usr/bin/env python3
"""Build a single self-contained index.html from code.md + sub-docs.

Usage:  python3 build.py
Output: index.html  (double-click to open, no server needed)

Editing workflow: edit any markdown file listed in DOCS_SPEC below, then re-run.

Reference handling (automatic):
  * A markdown link whose href resolves to a LOCAL .md file (relative path,
    absolute filesystem path, or a directory containing README/index/SKILL.md)
    is pulled into the page as a loadable sub-doc and the link is rewritten to
    an in-page `#doc:<id>` switch — clicking it renders that markdown inline,
    no full-page navigation, no dead links. Discovery is transitive (BFS): a
    referenced doc's own local .md links are included too.
  * Out-of-tree docs are VENDORED: their .md source and every local image they
    use are copied into code/vendor/<id>/, and image paths are rewritten to the
    copies. After a build, code/ has no dependency on files outside it — the
    whole folder can be zipped and moved. (Rebuilt fresh each run.)
  * External links (http/https/mailto) are left untouched; the template opens
    them in a new tab.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import deque
from urllib.parse import unquote

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(ROOT, "vendor")   # copies of out-of-tree docs + their images

# Each entry: (id, label, markdown-path relative to ROOT, is_tab, parent-id)
#   is_tab=True  -> shown in the top tab bar
#   is_tab=False -> loadable sub-doc, reachable only via in-page links; `parent`
#                   keeps that tab highlighted and adds a "← back" breadcrumb.
DOCS_SPEC: list[tuple[str, str, str, bool, str | None]] = [
    ("overview",         "Overview",              "overview.md",                            True,  None),
    ("env",              "Environment",           "env/env.md",                             True,  None),
    ("infra",            "Infra",                 "infra/infra.md",                         True,  None),
    ("infra-distribute", "Distributed RL Infra",  "infra/distribute-rl/distribute-rl.md",   False, "infra"),
    ("infra-blackbox",   "Black-Box RL",          "infra/black-box-rl/black-box-rl.md",     False, "infra"),
    ("infra-sandbox",    "Sandbox",               "infra/sandbox/sandbox.md",               False, "infra"),
    ("alg",              "Algorithm",             "alg/alg.md",                             True,  None),
    ("eval",             "Evaluation",            "eval/eval.md",                           True,  None),
]

ARXIV_URL = "https://arxiv.org/abs/2607.05471"
TRY_URL   = "https://streamlake.com/product/kat-coder"
TITLE     = "KAT-Coder V2.5"
SUBTITLE  = "Scaling Agentic RL for Real-World Software Engineering"

MAX_AUTO_DOCS = 60  # safety cap on transitively-included external docs

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_SRC_RE   = re.compile(r'(src\s*=\s*)"([^"]+)"')
_MDIMG_RE = re.compile(r'(!\[[^\]]*\]\()([^)\s]+)(\))')
# markdown link [text](href) — but NOT an image ![text](href)
_LINK_RE  = re.compile(r'(?<!\!)\[([^\]]*)\]\(([^)\s]+)\)')

_DIR_INDEX_NAMES = ("README.md", "index.md", "index.zh.md", "SKILL.md")

# ---------------------------------------------------------------------------
# Image path helpers
# ---------------------------------------------------------------------------

def _copy_into(src_abs: str, dest_abs: str) -> bool:
    """Copy src_abs → dest_abs, creating parent dirs as needed. Returns True on success."""
    if not os.path.isfile(src_abs):
        return False
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    shutil.copy2(src_abs, dest_abs)
    return True


def _rewrite_img(path_val: str, md_dir: str, vendor_rel: str | None) -> str:
    """Rewrite one image path to a ROOT-relative path that resolves offline.

    In-tree docs: return a path relative to ROOT (http/abs left as-is).
    Vendored docs (out-of-tree): copy the image into vendor/<id>/ and point at
    the copy, so the code/ folder has no external dependency after a build.
    """
    if re.match(r'^(https?:|data:|#)', path_val):
        return path_val
    raw = unquote(path_val)
    if os.path.isabs(raw):
        src_abs  = os.path.normpath(raw)
        subpath  = os.path.basename(raw)
    else:
        src_abs  = os.path.normpath(os.path.join(md_dir, raw))
        subpath  = raw

    if vendor_rel is None:          # in-tree doc
        if path_val.startswith('/'):
            return path_val
        # Always emit forward slashes: this path goes into an HTML/URL context,
        # and os.path.relpath yields backslashes on Windows, which browsers
        # won't resolve as path separators.
        return os.path.relpath(src_abs, ROOT).replace(os.sep, '/')

    # vendored doc: copy the asset under vendor/<id>/ so code/ is self-contained
    subpath  = subpath.lstrip('./').replace('../', '')
    dest_abs = os.path.join(ROOT, vendor_rel, subpath)
    if _copy_into(src_abs, dest_abs):
        return os.path.relpath(dest_abs, ROOT).replace(os.sep, '/')
    return path_val     # missing source: leave untouched


def rewrite_images(md: str, md_dir: str, vendor_rel: str | None = None) -> str:
    """Rewrite all image src/href references in `md` to ROOT-relative paths."""
    md = _SRC_RE.sub(
        lambda m: f'{m.group(1)}"{_rewrite_img(m.group(2), md_dir, vendor_rel)}"', md
    )
    md = _MDIMG_RE.sub(
        lambda m: f'{m.group(1)}{_rewrite_img(m.group(2), md_dir, vendor_rel)}{m.group(3)}', md
    )
    return md

# ---------------------------------------------------------------------------
# Local-path resolution helpers
# ---------------------------------------------------------------------------

def _is_external(href: str) -> bool:
    return bool(re.match(r'^(https?:|mailto:|tel:|data:)', href))


def _pick_md_from_dir(directory: str) -> str | None:
    """Return the best index .md file inside a directory, or None."""
    for name in _DIR_INDEX_NAMES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".md"):
            return os.path.join(directory, filename)
    return None


def _resolve_local_md(href: str, base_dir: str) -> str | None:
    """Resolve a local href to an existing .md path, or return None.

    Handles relative paths, absolute filesystem paths, %20-encoded spaces,
    a trailing #fragment, and directories (→ README/index/SKILL.md inside).
    """
    path_part = unquote(href.split("#")[0])
    if not path_part:
        return None
    abs_path = path_part if os.path.isabs(path_part) else os.path.join(base_dir, path_part)
    abs_path = os.path.normpath(abs_path)
    if os.path.isdir(abs_path):
        return _pick_md_from_dir(abs_path)
    if abs_path.endswith(".md") and os.path.isfile(abs_path):
        return abs_path
    return None


def _slug(text: str, fallback: str) -> str:
    """Turn arbitrary text into a URL-safe lowercase slug."""
    s = re.sub(r'[^a-z0-9]+', '-', (text or "").lower()).strip('-')
    return s or fallback


def _rewrite_local_asset(href: str, base_dir: str) -> str | None:
    """Rewrite a local NON-md link (source file / dir / .mdx) to a ROOT-relative path.

    Doc links resolve relative to the .md they live in, but the SPA loads every
    doc into a single index.html at ROOT, so a browser resolves all relative
    hrefs against ROOT — dropping the doc's own subdir prefix and breaking the
    link. We rewrite such links to a ROOT-relative path (forward slashes) so
    they resolve correctly when the site is served over HTTP.

    Preserves a trailing #fragment (e.g. #L157 line anchors). Returns None when
    the target doesn't exist inside ROOT (external/missing → caller leaves it).
    """
    path_part, _, frag = href.partition("#")
    path_part = unquote(path_part)
    if not path_part:
        return None
    abs_path = path_part if os.path.isabs(path_part) else os.path.join(base_dir, path_part)
    abs_path = os.path.normpath(abs_path)
    # Must live inside ROOT and actually exist (file or directory).
    if not abs_path.startswith(ROOT + os.sep) or not os.path.exists(abs_path):
        return None
    rel = os.path.relpath(abs_path, ROOT).replace(os.sep, '/')
    if os.path.isdir(abs_path) and not rel.endswith('/'):
        rel += '/'
    return rel + (('#' + frag) if frag else '')

# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

def _load_assets() -> dict[str, str]:
    """Read all bundled JS/CSS assets from the assets/ directory.

    Raises FileNotFoundError with a clear message if any asset is missing.
    """
    asset_dir = os.path.join(ROOT, "assets")
    asset_files = {
        "marked_js":          "js/marked.min.js",
        "katex_js":           "js/katex.min.js",
        "katex_autorender_js": "js/katex-auto-render.min.js",
        "katex_css":          "css/katex.selfcontained.css",
    }
    assets: dict[str, str] = {}
    for key, filename in asset_files.items():
        path = os.path.join(asset_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing asset: {path}")
        with open(path, encoding="utf-8") as f:
            assets[key] = f.read()
    return assets

# ---------------------------------------------------------------------------
# Doc-graph construction (BFS)
# ---------------------------------------------------------------------------

def _collect_docs() -> list[dict]:
    """Seed the doc list from DOCS_SPEC, then BFS-expand local .md links.

    Returns an ordered list of doc dicts ready for image-rewriting in the next
    phase. Each dict has the fields:
        id, label, tab, parent, abs, dir, vendor_rel, raw, md (None until phase 3)
    """
    docs: list[dict]      = []
    by_abs: dict[str, dict] = {}
    used_ids: set[str]    = set()

    def _uniq_id(base: str) -> str:
        base = base or "doc"
        candidate, n = base, 1
        while candidate in used_ids:
            n += 1
            candidate = f"{base}-{n}"
        used_ids.add(candidate)
        return candidate

    def _add_doc(
        abs_md: str, doc_id: str, label: str, is_tab: bool, parent: str | None
    ) -> dict:
        with open(abs_md, encoding="utf-8") as f:
            raw = f.read()
        in_tree    = os.path.normpath(abs_md).startswith(ROOT + os.sep)
        vendor_rel = None if in_tree else os.path.join("vendor", doc_id)
        if vendor_rel:      # archive a readable copy of the source for portability
            _copy_into(abs_md, os.path.join(ROOT, vendor_rel, os.path.basename(abs_md)))
        doc = {
            "id": doc_id, "label": label, "tab": is_tab, "parent": parent,
            "abs": abs_md, "dir": os.path.dirname(abs_md), "vendor_rel": vendor_rel,
            "raw": raw, "md": None,
        }
        docs.append(doc)
        by_abs[abs_md] = doc
        return doc

    # Phase 1 — seed from the explicit tab/sub-doc spec
    queue: deque[dict] = deque()
    for doc_id, label, rel, is_tab, parent in DOCS_SPEC:
        full = os.path.normpath(os.path.join(ROOT, rel))
        if not os.path.isfile(full):
            print(f"  [skip] {rel} (missing)")
            continue
        used_ids.add(doc_id)
        queue.append(_add_doc(full, doc_id, label, is_tab, parent))

    # Phase 2 — BFS: rewrite local .md links to #doc:<id>, pulling new docs in
    auto_count = 0

    def _make_link_rewriter(doc: dict):
        """Return a regex replacement function bound to `doc`."""
        nonlocal auto_count

        def _repl(m: re.Match) -> str:
            nonlocal auto_count
            text, href = m.group(1), m.group(2)
            if _is_external(href) or href.startswith("#"):
                return m.group(0)                       # external / plain anchor: leave
            target = _resolve_local_md(href, doc["dir"])
            if not target:
                # Not a local .md: if it points at a real in-tree source file or
                # dir, rewrite to a ROOT-relative path so it resolves in the SPA.
                asset = _rewrite_local_asset(href, doc["dir"])
                if asset and asset != href:
                    return f'[{text}]({asset})'
                return m.group(0)                       # external / missing: leave
            existing = by_abs.get(target)
            if existing:
                return f'[{text}](#doc:{existing["id"]})'
            if auto_count >= MAX_AUTO_DOCS:
                print(f"  [warn] MAX_AUTO_DOCS reached, skipping {target}")
                return m.group(0)
            auto_count += 1
            new_id = _uniq_id(_slug(text, os.path.splitext(os.path.basename(target))[0]))
            # keep the nearest tab ancestor highlighted; add breadcrumb → this doc
            new_doc = _add_doc(
                target, new_id,
                (text or os.path.basename(target)).strip(),
                False, doc["id"],
            )
            queue.append(new_doc)
            print(f"  [ref]  {os.path.relpath(target, ROOT):48s} -> #{new_id}  (from {doc['id']})")
            return f'[{text}](#doc:{new_id})'

        return _repl

    while queue:
        doc = queue.popleft()
        doc["raw"] = _LINK_RE.sub(_make_link_rewriter(doc), doc["raw"])

    return docs, auto_count

# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _render_html(
    template: str,
    assets: dict[str, str],
    payload: list[dict],
) -> str:
    """Fill the template with assets and doc payload, return the final HTML.

    The payload is embedded as JSON inside a <script type="application/json">
    block (injected by build.py via __DOCS__) to avoid the '</script>' XSS
    hazard that arises from direct string-substitution into a <script> block.
    JSON-encodes ensure_ascii=False for compact CJK output.
    """
    # Escape any literal </script> that might appear in markdown content so
    # the browser's HTML parser never sees a premature end-of-script tag.
    docs_json = json.dumps(payload, ensure_ascii=False).replace("</script>", r"<\/script>")

    return (
        template
        .replace("/*__MARKED__*/",   assets["marked_js"])
        .replace("/*__KATEX_JS__*/", assets["katex_js"] + "\n" + assets["katex_autorender_js"])
        .replace("/*__KATEX_CSS__*/", assets["katex_css"])
        .replace("__DOCS__",    docs_json)
        .replace("__ARXIV__",   ARXIV_URL)
        .replace("__TRY__",     TRY_URL)
        .replace("__TITLE__",   TITLE)
        .replace("__SUBTITLE__", SUBTITLE)
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Rebuild vendor dir from scratch each run so stale copies never linger.
    if os.path.isdir(VENDOR):
        shutil.rmtree(VENDOR)

    # 1) Collect all docs via BFS link-expansion
    docs, auto_count = _collect_docs()

    # 2) Finalize: rewrite image paths (done after BFS so final markdown is stable).
    #    For vendored (out-of-tree) docs this also copies their images into vendor/.
    vendored = 0
    for doc in docs:
        doc["md"] = rewrite_images(doc["raw"], doc["dir"], doc["vendor_rel"])
        kind = "tab" if doc["tab"] else f"sub of {doc['parent']}"
        if doc["vendor_rel"] is None:
            print(f"  [ok]   {os.path.relpath(doc['abs'], ROOT):48s} -> #{doc['id']}  ({kind})")
        else:
            vendored += 1
            print(f"  [vendor] {doc['vendor_rel']:46s} <- {doc['abs']}")

    # 3) Strip build-only fields before serializing to the browser payload
    payload = [{k: d[k] for k in ("id", "label", "tab", "parent", "md")} for d in docs]

    # 4) Load assets and template, render final HTML
    assets   = _load_assets()
    with open(os.path.join(ROOT, "template.html"), encoding="utf-8") as f:
        template = f.read()

    html = _render_html(template, assets, payload)

    out = os.path.join(ROOT, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(
        f"\n  {len(docs)} docs ({auto_count} auto-included, {vendored} vendored into vendor/) "
        f"— wrote {out}  ({len(html) // 1024} KB)"
    )


if __name__ == "__main__":
    main()

