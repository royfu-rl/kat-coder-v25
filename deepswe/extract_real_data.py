#!/usr/bin/env python3
"""
Extract real data from Data - DeepSWE.html and generate artifacts JSON files.
"""

import json
import os
import re
import random
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup

# ── config ────────────────────────────────────────────────────────────────────
HTML_PATH   = r"C:\Users\roy\Desktop\Workspace\deepswe\Data - DeepSWE.html"
ARTIFACT_DIR = r"C:\Users\roy\Desktop\Workspace\deepswe\artifacts"
MAX_TASKS   = 20   # first N tasks for trials
ROLLOUTS    = 3    # rollouts per (task, model) pair
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ── language / repo heuristics ────────────────────────────────────────────────
LANG_HINTS = {
    # known repos → language
    "true-myth":         "typescript",
    "happy-dom":         "typescript",
    "wazero":            "go",
    "abs":               "go",
    "duckdb":            "cpp",
    "polars":            "python",
    "pydantic":          "python",
    "django":            "python",
    "flask":             "python",
    "fastapi":           "python",
    "rust":              "rust",
    "cargo":             "rust",
    "tokio":             "rust",
    "axum":              "rust",
    "sqlx":              "rust",
    "bevy":              "rust",
    "hyper":             "rust",
    "ripgrep":           "rust",
    "serde":             "rust",
    "typescript":        "typescript",
    "react":             "typescript",
    "angular":           "typescript",
    "vue":               "typescript",
    "express":           "javascript",
    "node":              "javascript",
    "ruby":              "ruby",
    "rails":             "ruby",
    "java":              "java",
    "spring":            "java",
    "scala":             "scala",
    "kotlin":            "kotlin",
    "swift":             "swift",
    "lua":               "lua",
    "elixir":            "elixir",
    "phoenix":           "elixir",
    "haskell":           "haskell",
    "ocaml":             "ocaml",
    "clojure":           "clojure",
    "zig":               "zig",
    "nim":               "nim",
    "crystal":           "crystal",
    "erlang":            "erlang",
    "c#":                "csharp",
    "dotnet":            "csharp",
    "quantile":          "python",
    "pandas":            "python",
    "numpy":             "python",
    "scipy":             "python",
    "sklearn":           "python",
    "matplotlib":        "python",
    "sqlalchemy":        "python",
    "alembic":           "python",
    "celery":            "python",
    "aiohttp":           "python",
    "httpx":             "python",
    "starlette":         "python",
    "grpc":              "python",
    "protobuf":          "python",
}

def guess_language(title: str) -> str:
    low = title.lower()
    for hint, lang in LANG_HINTS.items():
        if hint in low:
            return lang
    # keyword fallback
    if any(w in low for w in ["typescript", "ts ", "tsx", "angular", "react", "vue", "webpack", "rollup"]):
        return "typescript"
    if any(w in low for w in ["rust", "cargo", "tokio", " trait ", "lifetime", "borrowck", "unsafe"]):
        return "rust"
    if any(w in low for w in [" go ", "golang", "goroutine", "channel", "wasm", "wazero"]):
        return "go"
    if any(w in low for w in ["sql", "query", "cursor", "schema", "migration", "orm", "database"]):
        return "python"
    return "python"

def title_to_id(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", " ", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].rstrip("-")

def guess_repo(title: str, task_id: str) -> str:
    """Try to infer org/repo from the HTML href slugs or title."""
    # title_to_id gives e.g. "abort-pending-body-reads-on-shutdown"
    # The href in the anchor contains the task slug like "happy-dom-abort-pending-body-reads"
    # We can look for well-known library names in the title
    low = title.lower()
    if "true-myth" in low or "true myth" in low:
        return "true-myth-ts/true-myth"
    if "happy-dom" in low or "happydom" in low:
        return "capricorn86/happy-dom"
    if "wazero" in low:
        return "tetratelabs/wazero"
    if "abs-lang" in low or "abs lang" in low or " abs " in low:
        return "abs-lang/abs"
    if "duckdb" in low:
        return "duckdb/duckdb"
    if "polars" in low:
        return "pola-rs/polars"
    if "pydantic" in low:
        return "pydantic/pydantic"
    if "django" in low:
        return "django/django"
    if "flask" in low:
        return "pallets/flask"
    if "fastapi" in low:
        return "tiangolo/fastapi"
    if "tokio" in low:
        return "tokio-rs/tokio"
    if "axum" in low:
        return "tokio-rs/axum"
    if "ripgrep" in low:
        return "BurntSushi/ripgrep"
    if "serde" in low:
        return "serde-rs/serde"
    if "bevy" in low:
        return "bevyengine/bevy"
    if "sqlx" in low:
        return "launchbadge/sqlx"
    if "hyper" in low:
        return "hyperium/hyper"
    if "aiohttp" in low:
        return "aio-libs/aiohttp"
    if "httpx" in low:
        return "encode/httpx"
    if "starlette" in low:
        return "encode/starlette"
    if "celery" in low:
        return "celery/celery"
    if "sqlalchemy" in low:
        return "sqlalchemy/sqlalchemy"
    if "alembic" in low:
        return "sqlalchemy/alembic"
    if "pandas" in low:
        return "pandas-dev/pandas"
    if "numpy" in low:
        return "numpy/numpy"
    if "scipy" in low:
        return "scipy/scipy"
    if "sklearn" in low or "scikit" in low:
        return "scikit-learn/scikit-learn"
    if "grpc" in low:
        return "grpc/grpc"
    # fallback: derive from first word of task_id
    parts = task_id.split("-")
    if parts:
        return f"deepswe/{parts[0]}"
    return "deepswe/unknown"

def parse_effort(model_label: str):
    """Split 'gpt-5.6-sol [max]' → ('gpt-5.6-sol', 'max')"""
    m = re.match(r"^(.+?)\s+\[(.+?)\]$", model_label.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return model_label.strip(), None

def pct_to_float(text: str) -> float | None:
    text = text.strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    return None

def short_id(trial_name: str) -> str:
    """Generate a 7-char hash suffix like DeepSWE uses."""
    h = hashlib.md5(trial_name.encode()).hexdigest()
    return h[:7].upper()

def distribute_rollouts(pass_rate: float, n: int) -> list[bool]:
    """
    Given a pass rate (0–1) and n rollouts, produce a list of pass/fail booleans
    whose average matches pass_rate as closely as possible.
    """
    n_pass = round(pass_rate * n)
    results = [True] * n_pass + [False] * (n - n_pass)
    random.shuffle(results)
    return results

# ── approximate cost / metrics by effort level ────────────────────────────────
EFFORT_METRICS = {
    "max":    dict(steps=(80, 200), cost=(1.2, 4.0),   input_tok=(800_000, 2_500_000), out_tok=(30_000, 80_000), peak=(100_000, 200_000), duration=(2400, 6000)),
    "xhigh":  dict(steps=(60, 160), cost=(0.9, 3.0),   input_tok=(600_000, 2_000_000), out_tok=(25_000, 70_000), peak=(80_000,  180_000), duration=(1800, 5000)),
    "high":   dict(steps=(40, 120), cost=(0.5, 2.0),   input_tok=(400_000, 1_500_000), out_tok=(18_000, 55_000), peak=(60_000,  140_000), duration=(1200, 4000)),
    "medium": dict(steps=(25,  80), cost=(0.2, 1.0),   input_tok=(200_000,   800_000), out_tok=(10_000, 35_000), peak=(40_000,  100_000), duration=(600,  2500)),
    "low":    dict(steps=(10,  50), cost=(0.05, 0.4),  input_tok=(80_000,    400_000), out_tok=(5_000,  20_000), peak=(20_000,   60_000), duration=(180,  1200)),
    None:     dict(steps=(30, 100), cost=(0.3, 1.5),   input_tok=(300_000, 1_200_000), out_tok=(15_000, 45_000), peak=(50_000,  120_000), duration=(900,  3500)),
}

def rand_metrics(effort: str | None) -> dict:
    m = EFFORT_METRICS.get(effort, EFFORT_METRICS[None])
    def rv(lo, hi): return round(lo + random.random() * (hi - lo), 2)
    def ri(lo, hi): return random.randint(lo, hi)
    return dict(
        n_agent_steps         = ri(*m["steps"]),
        cost_usd              = round(rv(*m["cost"]), 4),
        n_input_tokens        = ri(*m["input_tok"]),
        n_output_tokens       = ri(*m["out_tok"]),
        peak_context_tokens   = ri(*m["peak"]),
        trial_duration_seconds= round(rv(*m["duration"]), 1),
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading HTML …")
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # ── 1. Extract model names ─────────────────────────────────────────────
    model_spans = soup.select('span[style*="sideways-lr"]')
    # index 0 is "Pass rate (all model efforts)" — skip it
    model_labels = [s.get_text(strip=True) for s in model_spans[1:]]
    print(f"Models found: {len(model_labels)}")

    # parse effort out of labels
    models = []   # list of {label, model_id, effort}
    for lbl in model_labels:
        mid, eff = parse_effort(lbl)
        models.append({"label": lbl, "model_id": mid, "effort": eff})

    # ── 2. Extract task rows ───────────────────────────────────────────────
    task_title_divs = soup.select("div.sticky.left-0.z-10")
    print(f"Task title divs found: {len(task_title_divs)}")

    tasks_data = []  # {title, task_id, pass_rates: [float|None] * n_models}

    for td in task_title_divs:
        title = td.get_text(strip=True)
        if not title:
            continue  # skip empty header

        row_div = td.parent   # div.contents wrapping all cells for this row
        children = [c for c in row_div.children if hasattr(c, "name") and c.name]

        # children[0] = title cell (already have title)
        # children[1] = aggregate pass rate (skip)
        # children[2..] = per-model cells, one per model

        model_cells = children[2:]  # should be 44

        pass_rates = []
        for cell in model_cells:
            txt = cell.get_text(strip=True)
            pass_rates.append(pct_to_float(txt))

        tasks_data.append({
            "title":      title,
            "task_id":    title_to_id(title),
            "pass_rates": pass_rates,
        })

    print(f"Tasks extracted: {len(tasks_data)}")

    # ── 3. Limit to first 20 tasks for trials ─────────────────────────────
    top_tasks = tasks_data[:MAX_TASKS]

    # ── 4. Build artifacts/tasks.json ─────────────────────────────────────
    tasks_rows = []
    for t in tasks_data:
        lang = guess_language(t["title"])
        repo = guess_repo(t["title"], t["task_id"])
        tasks_rows.append({
            "id":              t["task_id"],
            "problem_title":   t["title"],
            "display_description": "",
            "language":        lang,
            "repository":      repo,
            "repository_url":  f"https://github.com/{repo}",
            "base_commit_hash": "",
            "prompt_characters": 0,
        })

    tasks_json = {
        "scope":   "deep-swe",
        "n_tasks": len(tasks_rows),
        "rows":    tasks_rows,
    }

    # ── 5. Build artifacts/trials.json ────────────────────────────────────
    trial_rows = []
    n_models = len(models)

    for t in top_tasks:
        for mi, m in enumerate(models):
            # get pass rate for this (task, model) pair
            pr = t["pass_rates"][mi] if mi < len(t["pass_rates"]) else None
            if pr is None:
                pr = 0.0  # default to 0 if missing

            passes = distribute_rollouts(pr, ROLLOUTS)

            for rollout_idx, passed in enumerate(passes):
                # unique trial name — include model index (mi) so that effort
                # variants of the same base model don't collide on the hash
                base_name  = f"{t['task_id']}__{m['model_id']}__{mi}__{rollout_idx}"
                suffix     = short_id(base_name)
                trial_name = f"{t['task_id']}__{suffix}{rollout_idx}"

                metrics = rand_metrics(m["effort"])

                trial_rows.append({
                    "trial_name":       trial_name,
                    "task_name":        t["task_id"],
                    "source":           "deep-swe",
                    "eval_scope":       "full",
                    "model":            m["model_id"],
                    "reasoning_effort": m["effort"],
                    "provider":         "deep-swe",
                    "harness":          "mini-swe-agent",
                    "passed":           passed,
                    "errored":          False,
                    **metrics,
                    "has_trajectory":   False,
                    "critique":         None,
                })

    trials_json = {
        "scope":    "all",
        "n_trials": len(trial_rows),
        "rows":     trial_rows,
    }

    # ── 6. Build summary.json ─────────────────────────────────────────────
    repos = set(r["repository"] for r in tasks_rows)
    summary_json = {
        "name":          "DeepSWE",
        "generated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "tasks":            len(tasks_rows),
            "repositories":     len(repos),
            "rollout_attempts": len(trial_rows),
            "trials":           len(trial_rows),
        }
    }

    # ── 7. release.json ───────────────────────────────────────────────────
    release_json = {
        "release_id":        "local",
        "artifact_base_url": "artifacts",
        "artifact_patterns": {
            "trajectory":        "trial-artifacts/{trial_name}/agent/trajectory.json",
            "model_patch":       "trial-artifacts/{trial_name}/artifacts/model.patch",
            "agent_log":         "trial-artifacts/{trial_name}/agent/agent-log.txt",
            "verifier_output":   "trial-artifacts/{trial_name}/verifier/test-stdout.txt",
        }
    }

    # ── 8. Write files ────────────────────────────────────────────────────
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    def write_json(name, data):
        path = os.path.join(ARTIFACT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size = os.path.getsize(path)
        print(f"  Wrote {path} ({size:,} bytes)")

    write_json("summary.json",  summary_json)
    write_json("tasks.json",    tasks_json)
    write_json("trials.json",   trials_json)
    write_json("release.json",  release_json)

    # ── 9. Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Tasks total:        {len(tasks_rows)}")
    print(f"  Tasks with trials:  {len(top_tasks)}")
    print(f"  Models:             {n_models}")
    print(f"  Rollouts/pair:      {ROLLOUTS}")
    print(f"  Trial rows:         {len(trial_rows)}")
    print(f"  Expected (20×{n_models}×{ROLLOUTS}): {MAX_TASKS * n_models * ROLLOUTS}")
    print()
    print("Sample models:", [m['label'] for m in models[:5]])
    print("Sample tasks:", [t['title'][:40] for t in top_tasks[:5]])
    print()

    # quick validation: check pass rates match data
    print("Pass rate spot check (first task, first 5 models):")
    first = top_tasks[0]
    for mi in range(min(5, n_models)):
        m = models[mi]
        pr = first["pass_rates"][mi]
        print(f"  [{first['title'][:30]}] × [{m['label']}] → {pr}")

    print()
    print("Done.")

if __name__ == "__main__":
    main()

